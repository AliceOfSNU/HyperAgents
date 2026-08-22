from agent.memory_store import search


def tool_info():
    return {
        "name": "memory_search",
        "description": """Find notes in memory.jsonl related to some text, ranked by semantic similarity (embedding cosine similarity -- no LLM call, cheap and fast). Two uses:

1. Before writing a new note with memory_append, search with a draft of what you're about to write -- the results are candidates for that note's `links` field. Whether any of them are actually related enough to link is your judgment call, not automatic.
2. Ad-hoc lookup -- search for whatever you want to know if you already have notes about, instead of reading memory.jsonl in full.

Returns each matching note's full content plus its `links`, so you can follow the graph structure (a matched note's own linked notes) by eye without a second search.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for -- a draft note, a question, a topic, anything."},
                "k": {"type": "integer", "description": "Max number of results, default 5."},
            },
            "required": ["query"],
        },
    }


def tool_function(query, k=5):
    try:
        results = search(query, k=k)
    except Exception as e:
        return f"Error: could not search memory: {e}"

    if not results:
        return "No matching notes found (memory may be empty, or no notes have embeddings yet)."

    lines = []
    for r in results:
        lines.append(
            f"[{r['id']}] ({r['similarity']}) type={r.get('type', '?')} "
            f"files={r.get('files') or []} gens={r.get('about_generations')}\n"
            f"  {r['title']}: {r['description']}\n"
            f"  content: {r['content']}\n"
            f"  links: {r.get('links') or []}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python memory_search.py <query> [k]")
    else:
        print(tool_function(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5))
