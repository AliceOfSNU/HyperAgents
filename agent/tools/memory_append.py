from agent.memory_store import NOTE_TYPES, append_note


def tool_info():
    return {
        "name": "memory_append",
        "description": """Append one note to memory.jsonl at the repo root -- a persistent, append-only, immutable log that survives across generations the same way any other file you edit does (through your own patch chain). Notes are never rewritten once written; if your understanding changes, write a new note rather than editing an old one.

Fields:
- title: a concise identifier for the strategy or reasoning pattern
- description: a one-sentence summary of it
- content: the actual distilled reasoning steps, decision rationale, or operational insight
- type: one of bugfix, prompt, tools, loop, meta, task, findings, other -- `meta` for the meta-agent's own harness/infra, `task` for the task-facing agent's, `findings` for an observation that isn't itself a change (e.g. "gen_2's aggressive notes regressed X")
- files: which code file(s) this note is about, if any
- about_generations: optional, your own belief about which generation(s) this note concerns (e.g. "gen_3") -- not verified, just your note-taking convention
- links: optional, ids of related earlier notes -- typically informed by a prior memory_search call rather than guessed. Call memory_search first with a draft of what you're about to write to see if anything's already related before deciding this.

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
                "links": {"type": "array", "items": {"type": "string"}, "description": "Optional -- ids of related earlier notes (e.g. from a prior memory_search call)."},
            },
            "required": ["title", "description", "content", "type"],
        },
    }


def tool_function(title, description, content, type, files=None, about_generations=None, links=None):
    try:
        note, embedded = append_note(
            title=title, description=description, content=content, type=type,
            files=files, about_generations=about_generations, links=links,
        )
        suffix = "" if embedded else " (embedding failed -- saved, but won't turn up in memory_search until re-embedded)"
        return f"Appended {note['id']}: {title!r}{suffix}"
    except Exception as e:
        return f"Error: could not append to memory: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("description")
    parser.add_argument("content")
    parser.add_argument("type", choices=NOTE_TYPES)
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--about-generations", default=None)
    parser.add_argument("--links", nargs="*", default=None)
    args = parser.parse_args()
    print(tool_function(
        args.title, args.description, args.content, args.type,
        files=args.files, about_generations=args.about_generations, links=args.links,
    ))
