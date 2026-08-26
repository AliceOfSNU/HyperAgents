"""Shared data layer behind the memory_append/memory_search/memory_link
tools -- kept out of agent/tools/ deliberately, since load_tools() globs
every *.py file there and requires each one to expose
tool_info/tool_function (anything else makes it raise).

Storage is two separate files at the repo root:
- memory.jsonl -- one note per line, human/agent-readable (this is the
  file the meta-agent is told it's free to `cat` directly).
- memory_embeddings.jsonl -- one {"id", "embedding"} per line, the
  search index. Kept separate so a plain `cat memory.jsonl` never dumps
  a 1536-float vector into the agent's own context.

Notes are immutable once written, with one narrow, deliberate exception:
add_links() may patch the `links` field of any note appended during the
*current* meta-agent session (tracked via mark_session_start(), called
once at the start of MetaAgent.forward() -- see SESSION_MARKER_PATH).
That's "finish notes you created earlier this run", not A-Mem-style
evolution where a later generation can reach back and rewrite an old
note's own meaning -- which was rejected specifically because it
destroys the audit trail of which generation actually believed what
(see memory: dgm_h_map_elites_parent_selection's sibling discussion).
Once a generation's run ends, all of its notes -- and their links --
are frozen like everything else; the boundary is "this run", not "the
single most recent note", so writing several notes in one session
doesn't force linking each one before appending the next.

Links are stored one-directional (the newer note points at older ones
it relates to) but shown bidirectionally: find_backlinks() computes,
on every read, which other notes point at a given note, purely by
scanning already-loaded notes -- nothing about the older note's own
stored line ever changes. This is the same bidirectional-traversal
benefit A-Mem gets from its "box" concept and its evolution step
(which mutates the neighbor's own record to acknowledge the new link),
without the mutation.
"""
import json
import math
import re
from pathlib import Path

import litellm

_REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = _REPO_ROOT / "memory.jsonl"
EMBEDDINGS_PATH = _REPO_ROOT / "memory_embeddings.jsonl"
# Deliberately outside the repo -- this is per-container-run scratch
# state, not something that should ever show up as an untracked file
# in the meta-agent's own model_patch.diff.
SESSION_MARKER_PATH = Path("/tmp/.memory_session_start")

EMBEDDING_MODEL = "text-embedding-3-small"

NOTE_TYPES = ["bugfix", "prompt", "tools", "loop", "meta", "task", "findings", "other"]
COMMON_RELATIONS = ["resolves", "develops", "contradicts"]

CANDIDATE_K = 5

_ID_RE = re.compile(r"^m(\d+)$")


def load_notes():
    """All notes in memory.jsonl, oldest first. Tolerant of the old
    3-field-only shape (title/description/content, no id/type/etc) --
    every caller must use .get() with defaults, never assume a field
    is present."""
    if not MEMORY_PATH.exists():
        return []
    notes = []
    for line in MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            notes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return notes


def _next_id(notes):
    max_n = 0
    for n in notes:
        m = _ID_RE.match(str(n.get("id", "")))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"m{max_n + 1}"


def _embedding_text(note):
    """The single string that gets embedded -- concatenated structured
    fields + free text, same pattern as A-Mem's own
    concat(content, keywords, tags, context). Deliberately excludes
    `links`: they're structural, not semantic, and (for a brand-new
    note) usually don't exist yet at the moment this is first computed
    anyway -- append_note calls this before links are ever decided."""
    parts = [
        f"[{note.get('type', '')}] {note.get('title', '')}",
        f"Files: {', '.join(note.get('files') or [])}",
        f"Generations: {note.get('about_generations', '')}",
        note.get("description", ""),
        note.get("content", ""),
    ]
    return "\n".join(p for p in parts if p.strip())


def compute_embedding(text):
    """Returns the embedding vector, or None on any failure -- an
    embedding-API hiccup should never block the actual memory write,
    which is the operation that matters. A note saved without an
    embedding just won't surface in memory_search until backfilled."""
    try:
        resp = litellm.embedding(model=EMBEDDING_MODEL, input=[text])
        return resp["data"][0]["embedding"]
    except Exception:
        return None


def _load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        return {}
    out = {}
    for line in EMBEDDINGS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            out[rec["id"]] = rec["embedding"]
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def _append_embedding(note_id, embedding):
    with open(EMBEDDINGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": note_id, "embedding": embedding}) + "\n")


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _top_k_by_vector(vec, embeddings, notes_by_id, k):
    scored = sorted(
        ((note_id, _cosine(vec, e)) for note_id, e in embeddings.items()),
        key=lambda pair: pair[1], reverse=True,
    )[:k]
    return [
        {**notes_by_id[note_id], "similarity": round(score, 4)}
        for note_id, score in scored
        if note_id in notes_by_id
    ]


def find_backlinks(note_id, notes=None):
    """Notes that link TO note_id -- computed fresh on every call by
    scanning already-loaded notes, never stored. This is what makes
    linking bidirectional in effect (you can see what points at a
    note) without ever rewriting that note's own line."""
    notes = load_notes() if notes is None else notes
    backlinks = []
    for n in notes:
        for link in (n.get("links") or []):
            link_id = link.get("id") if isinstance(link, dict) else link
            if link_id == note_id:
                backlinks.append({
                    "id": n.get("id"),
                    "title": n.get("title"),
                    "relation": link.get("relation") if isinstance(link, dict) else None,
                })
    return backlinks


def append_note(title, description, content, type, files=None, about_generations=None):
    """Writes the note immediately (durable in one shot -- nothing is
    ever staged/held back waiting on a follow-up call that might not
    come), then searches existing notes using this note's own real
    content as the query. Returns (note, embedded_ok, candidates) --
    candidates are for the caller to optionally hand to add_links via
    memory_link; nothing here decides that automatically."""
    notes = load_notes()
    note = {
        "id": _next_id(notes),
        "title": title,
        "description": description,
        "content": content,
        "type": type,
        "files": files or [],
        "about_generations": about_generations,
        "links": [],
    }
    embedding = compute_embedding(_embedding_text(note))

    candidates = []
    if embedding is not None:
        embeddings = _load_embeddings()
        notes_by_id = {n["id"]: n for n in notes if "id" in n}
        candidates = _top_k_by_vector(embedding, embeddings, notes_by_id, CANDIDATE_K)

    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(note, ensure_ascii=False) + "\n")
    if embedding is not None:
        _append_embedding(note["id"], embedding)

    return note, embedding is not None, candidates


def mark_session_start():
    """Call once, at the very start of a meta-agent run (before any
    memory_append calls), to record how many notes already existed --
    everything appended from this point on is "this session" and stays
    linkable via add_links() for the rest of the run, no matter how
    many more notes get appended after it. Safe to call even if
    memory.jsonl doesn't exist yet (records 0)."""
    count = len(load_notes())
    SESSION_MARKER_PATH.write_text(str(count), encoding="utf-8")


def _session_start_index():
    """Line-index (0-based) of the first note considered part of the
    current session. Falls back to "only the last line" -- the old,
    most conservative behavior -- if the marker is missing or corrupt,
    rather than silently treating the whole file as in-session."""
    try:
        return int(SESSION_MARKER_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def add_links(note_id, links):
    """Patches ONLY the `links` field of a note appended during the
    current session (see mark_session_start()), in place -- rejects
    any note_id appended in an earlier session. Every other line, and
    every other field of this one, is preserved byte-for-byte."""
    if not MEMORY_PATH.exists():
        return False, "memory.jsonl doesn't exist yet -- nothing to link."
    raw_lines = MEMORY_PATH.read_text(encoding="utf-8").splitlines()
    nonblank_idxs = [i for i, line in enumerate(raw_lines) if line.strip()]
    if not nonblank_idxs:
        return False, "memory.jsonl is empty -- nothing to link."

    session_start = _session_start_index()
    if session_start is None:
        # No valid marker -- fall back to last-line-only.
        allowed_idxs = nonblank_idxs[-1:]
    else:
        allowed_idxs = [i for i in nonblank_idxs if i >= session_start]

    target_idx = None
    for i in allowed_idxs:
        try:
            if json.loads(raw_lines[i]).get("id") == note_id:
                target_idx = i
                break
        except json.JSONDecodeError:
            continue

    if target_idx is None:
        return False, (
            f"{note_id!r} isn't a note from this session (or its line isn't valid JSON) -- "
            "add_links only works on notes you've appended since this run started. "
            "Notes from earlier generations are immutable."
        )

    note = json.loads(raw_lines[target_idx])
    note["links"] = links
    raw_lines[target_idx] = json.dumps(note, ensure_ascii=False)
    MEMORY_PATH.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return True, None


def search(query_text, k=5):
    """Ad-hoc lookup: up to k notes most similar to query_text, highest
    similarity first, each annotated with its backlinks. Notes with no
    stored embedding (embedding-API failure at write time, or written
    before this feature existed) are silently excluded, not errored on."""
    embeddings = _load_embeddings()
    if not embeddings:
        return []
    query_vec = compute_embedding(query_text)
    if query_vec is None:
        return []
    notes = load_notes()
    notes_by_id = {n["id"]: n for n in notes if "id" in n}
    results = _top_k_by_vector(query_vec, embeddings, notes_by_id, k)
    for r in results:
        r["backlinks"] = find_backlinks(r["id"], notes=notes)
    return results
