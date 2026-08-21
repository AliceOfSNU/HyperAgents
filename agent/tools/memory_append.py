import json
from pathlib import Path

MEMORY_PATH = Path(__file__).resolve().parents[2] / "memory.jsonl"


def tool_info():
    return {
        "name": "memory_append",
        "description": """Append one memory item to memory.jsonl at the repo root -- a persistent, append-only log that survives across generations the same way any other file you edit does (through your own patch chain).

Each item has three fields:
- title: a concise identifier for the strategy or reasoning pattern
- description: a one-sentence summary of it
- content: the actual distilled reasoning steps, decision rationale, or operational insight

This is the only built-in memory tool -- there is no retrieval tool. Reading memory.jsonl (whether to, when, what's relevant, how) is entirely your call; it's a plain file, readable with bash or the editor tool like any other.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Concise identifier for the strategy or reasoning pattern."},
                "description": {"type": "string", "description": "One-sentence summary of the memory item."},
                "content": {"type": "string", "description": "The distilled reasoning steps, decision rationale, or operational insight."},
            },
            "required": ["title", "description", "content"],
        },
    }


def tool_function(title, description, content):
    try:
        line = json.dumps({"title": title, "description": description, "content": content}, ensure_ascii=False)
        with open(MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return f"Appended to {MEMORY_PATH.name}: {title!r}"
    except Exception as e:
        return f"Error: could not append to memory: {e}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python memory_append.py <title> <description> <content>")
    else:
        print(tool_function(sys.argv[1], sys.argv[2], sys.argv[3]))
