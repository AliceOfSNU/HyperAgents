from agent.memory_store import NOTE_TYPES, append_note


def tool_info():
    return {
        "name": "memory_append",
        "description": """Append one note to memory.jsonl at the repo root -- a persistent, append-only, immutable log that survives across generations the same way any other file you edit does (through your own patch chain). Notes are never rewritten once written (with one narrow exception: memory_link can attach links to the note you *just* created, see below).

This writes the note immediately -- nothing is held back waiting on a follow-up call. It also searches existing notes using this note's own real content (no separate draft/query needed) and returns candidates in the result. If any are genuinely related, call memory_link with the id this call returns and the candidate ids you want to connect; if none are relevant, there's nothing else to do.

Fields:
- title: a concise identifier for the strategy or reasoning pattern
- description: a one-sentence summary of it
- content: the actual distilled reasoning steps, decision rationale, or operational insight - keep it compact.
- type: one of bugfix, prompt, tools, loop, meta, task, findings, other -- `meta` for the meta-agent's own harness/infra, `task` for the task-facing agent's, `findings` for an observation that isn't itself a change (e.g. "gen_2's aggressive notes regressed X")
- files: which code file(s) this note is about, if any
- about_generations: optional, your own belief about which generation(s) this note concerns (e.g. "gen_3") -- not verified, just your note-taking convention

There is no separate retrieval step required to read memory back in bulk -- memory.jsonl is a plain file, readable with bash or the editor tool. Use memory_search when you want notes related to something specific instead of reading everything.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Concise identifier for the strategy or reasoning pattern."},
                "description": {"type": "string", "description": "One-sentence summary of the note."},
                "content": {"type": "string", "description": "The distilled reasoning steps, decision rationale, or operational insight."},
                "type": {"type": "string", "enum": NOTE_TYPES, "description": "Category of this note."},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Relevant code file path(s), if any."},
                "about_generations": {"type": "string", "description": "Optional -- which generation(s) this note is about, e.g. 'gen_3'."},
            },
            "required": ["title", "description", "content", "type"],
        },
    }


def tool_function(title, description, content, type, files=None, about_generations=None):
    try:
        note, embedded, candidates = append_note(
            title=title, description=description, content=content, type=type,
            files=files, about_generations=about_generations,
        )
    except Exception as e:
        return f"Error: could not append to memory: {e}"

    lines = [f"Appended {note['id']}: {title!r}"]
    if not embedded:
        lines.append("(embedding failed -- saved, but won't turn up in memory_search until re-embedded)")
    if candidates:
        lines.append("")
        lines.append(f"Related notes found -- call memory_link(note_id={note['id']!r}, links=[...]) if any are relevant:")
        for c in candidates:
            lines.append(
                f"  [{c['id']}] ({c['similarity']}) type={c.get('type', '?')}: "
                f"{c['title']} -- {c['description']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("description")
    parser.add_argument("content")
    parser.add_argument("type", choices=NOTE_TYPES)
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--about-generations", default=None)
    args = parser.parse_args()
    print(tool_function(
        args.title, args.description, args.content, args.type,
        files=args.files, about_generations=args.about_generations,
    ))
