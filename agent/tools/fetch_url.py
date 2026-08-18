import io
import re


def tool_info():
    return {
        "name": "fetch_url",
        "description": """Fetch a URL and return its text content.

Use this for links a human overseeing this run has left for you (e.g. in the
steering directory, or in a PR comment/description) -- arxiv abstract pages,
GitHub READMEs, blog posts, etc. Handles plain text/HTML (tags stripped down
to readable text) and PDF links (arxiv /pdf/ URLs, direct .pdf links) the
same way read_pdf does for local files.

Unlike the bash tool, this genuinely does reach the real internet -- only
fetch URLs you were actually given or that a page you were given links to,
not arbitrary search queries.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
            },
            "required": ["url"],
        },
    }


MAX_CHARS = 20000


def _strip_html(html):
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def tool_function(url):
    try:
        import requests
    except ImportError:
        return "Error: requests is not installed in this environment."

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (compatible; dgm-h-agent)"})
        resp.raise_for_status()
    except Exception as e:
        return f"Error: could not fetch '{url}': {e}"

    content_type = resp.headers.get("Content-Type", "")

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n\n".join(
                f"--- page {i + 1} ---\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)
            )
            prefix = f"(PDF, {len(reader.pages)} pages total)\n\n"
        except Exception as e:
            return f"Error: fetched '{url}' but could not parse it as a PDF: {e}"
    elif "html" in content_type or resp.text.lstrip().lower().startswith("<!doctype html") or "<html" in resp.text[:1000].lower():
        text = _strip_html(resp.text)
        prefix = ""
    else:
        text = resp.text
        prefix = ""

    truncated = ""
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        truncated = "\n\n[... truncated ...]"
    return f"{prefix}{text}{truncated}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fetch_url.py <url>")
    else:
        print(tool_function(sys.argv[1]))
