from agent.memory_store import COMMON_RELATIONS, add_links


def tool_info():
    return {
        "name": "memory_link",
        "description": f"""Attach relation-typed links from a note to related earlier notes, using the candidates memory_append's own result surfaced when you wrote it.

Works on any note you've appended so far THIS run -- not just the very last one, so writing several notes before circling back to link them is fine. It does NOT work on notes from earlier generations/sessions -- not a general "edit any note's links" tool. Notes are otherwise immutable; this exists to let you finish notes you created this run, not to revise history. If you write multiple notes in one session, try to decide on links (or explicitly "none apply") for each of them before you finish, since the window closes once this run ends.

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
