from agent.memory_store import COMMON_RELATIONS, add_links


def tool_info():
    return {
        "name": "memory_link",
        "description": f"""Attach relation-typed links from a note to related earlier notes, using the candidates memory_append's own result just surfaced.

Only works on the single most recently appended note -- not a general "edit any note's links" tool. Notes are otherwise immutable; this exists to let you finish the note you just created, not to revise history.

Common relation labels: {', '.join(COMMON_RELATIONS)} -- or any other short (1-2 word) description that fits better.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "The note to attach links to -- must be the id memory_append just returned."},
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Id of the related earlier note."},
                            "relation": {"type": "string", "description": f"How it relates, e.g. {', '.join(COMMON_RELATIONS)}, or a short custom label."},
                        },
                        "required": ["id", "relation"],
                    },
                    "description": "Related earlier notes and how each one relates.",
                },
            },
            "required": ["note_id", "links"],
        },
    }


def tool_function(note_id, links):
    try:
        ok, error = add_links(note_id, links)
    except Exception as e:
        return f"Error: could not add links: {e}"
    if not ok:
        return f"Error: {error}"
    return f"Linked {note_id} to {len(links)} note(s)."


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("note_id")
    parser.add_argument("links_json", help='JSON list, e.g. \'[{"id":"m3","relation":"develops"}]\'')
    args = parser.parse_args()
    print(tool_function(args.note_id, json.loads(args.links_json)))
