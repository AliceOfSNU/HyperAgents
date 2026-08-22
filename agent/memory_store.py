"""Shared data layer behind the memory_append/memory_search tools --
kept out of agent/tools/ deliberately, since load_tools() globs every
*.py file there and requires each one to expose tool_info/tool_function
(anything else makes it raise, per its own docstring-adjacent behavior).

Storage is two separate files at the repo root:
- memory.jsonl -- one note per line, human/agent-readable (this is the
  file the meta-agent is told it's free to `cat` directly), append-only,
  never rewritten in place.
- memory_embeddings.jsonl -- one {"id", "embedding"} per line, the
  search index. Kept separate so a plain `cat memory.jsonl` never dumps
  a 1536-float vector into the agent's own context.

Notes are immutable once written (no A-Mem-style "evolution" -- see
memory: dgm_h_map_elites_parent_selection's sibling discussion for why
that was deliberately rejected: silently rewriting an existing note
destroys the audit trail of which generation actually believed what).
"""
import json
import math
import re
from pathlib import Path

import litellm

_REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = _REPO_ROOT / "memory.jsonl"
EMBEDDINGS_PATH = _REPO_ROOT / "memory_embeddings.jsonl"

EMBEDDING_MODEL = "text-embedding-3-small"

NOTE_TYPES = ["bugfix", "prompt", "tools", "loop", "meta", "task", "findings", "other"]

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
    `links`: they don't exist yet at search time for a note being
    composed (that's the whole point of the two-turn write flow), and
    they're structural, not semantic -- once a note is found, its
    `links` field is right there to follow directly, no need to fold
    it into the vector too."""
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


def append_note(title, description, content, type, files=None, about_generations=None, links=None):
    notes = load_notes()
    note = {
        "id": _next_id(notes),
        "title": title,
        "description": description,
        "content": content,
        "type": type,
        "files": files or [],
        "about_generations": about_generations,
        "links": links or [],
    }
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(note, ensure_ascii=False) + "\n")

    embedding = compute_embedding(_embedding_text(note))
    if embedding is not None:
        _append_embedding(note["id"], embedding)

    return note, embedding is not None


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search(query_text, k=5):
    """Returns up to k notes most similar to query_text, highest
    similarity first. Notes with no stored embedding (embedding-API
    failure at write time, or written before this feature existed)
    are silently excluded, not errored on."""
    embeddings = _load_embeddings()
    if not embeddings:
        return []
    query_vec = compute_embedding(query_text)
    if query_vec is None:
        return []
    scored = sorted(
        ((note_id, _cosine(query_vec, vec)) for note_id, vec in embeddings.items()),
        key=lambda pair: pair[1], reverse=True,
    )[:k]
    notes_by_id = {n["id"]: n for n in load_notes() if "id" in n}
    return [
        {**notes_by_id[note_id], "similarity": round(score, 4)}
        for note_id, score in scored
        if note_id in notes_by_id
    ]
