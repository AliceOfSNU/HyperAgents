def tool_info():
    return {
        "name": "read_pdf",
        "description": """Extract the text content of a local PDF file.

Use this for PDFs. Plain text/markdown files can just be read with
the bash tool (cat/sed); this tool exists because PDF is a binary format.

* `path` must be a local file path (e.g. one you found via `ls`), not a URL -- use fetch_url first if you only have a link.
* Output is truncated if very long; request a `page_range` (e.g. "1-10") to
  read a specific slice of a long paper instead of everything at once.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path to the PDF file."},
                "page_range": {
                    "type": "string",
                    "description": "Optional 1-indexed page range, e.g. '1-10' or '5'. Omit for the whole document (subject to truncation).",
                },
            },
            "required": ["path"],
        },
    }


MAX_CHARS = 20000


def _parse_page_range(page_range, num_pages):
    if not page_range:
        return range(num_pages)
    try:
        if "-" in page_range:
            start, end = page_range.split("-", 1)
            start_idx = max(int(start) - 1, 0)
            end_idx = min(int(end), num_pages)
        else:
            start_idx = max(int(page_range) - 1, 0)
            end_idx = min(start_idx + 1, num_pages)
        return range(start_idx, end_idx)
    except ValueError:
        return range(num_pages)


def tool_function(path, page_range=None):
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Error: pypdf is not installed in this environment."

    try:
        reader = PdfReader(path)
    except Exception as e:
        return f"Error: could not open '{path}' as a PDF: {e}"

    num_pages = len(reader.pages)
    indices = _parse_page_range(page_range, num_pages)

    chunks = []
    for i in indices:
        try:
            chunks.append(f"--- page {i + 1} ---\n{reader.pages[i].extract_text() or ''}")
        except Exception as e:
            chunks.append(f"--- page {i + 1}: extraction failed ({e}) ---")

    text = "\n\n".join(chunks)
    truncated = ""
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        truncated = f"\n\n[... truncated; document has {num_pages} pages total, request a page_range to read more ...]"
    return f"({num_pages} pages total)\n\n{text}{truncated}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python read_pdf.py <path> [page_range]")
    else:
        print(tool_function(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
