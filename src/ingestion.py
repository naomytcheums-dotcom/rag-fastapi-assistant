"""
Ingestion pipeline for the FastAPI documentation RAG assistant.

Reads every Markdown file under data/raw/, resolves FastAPI's custom
"{* path *}" code-snippet includes (which point to files outside the
markdown itself, under the sibling `fastapi/docs_src/` folder), strips
HTML/admonition markup, and writes a clean, structured JSON file to
data/processed/fastapi_docs.json for the next pipeline stage (chunking
and embedding).
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# The FastAPI repo was cloned as a sibling of the project folder:
# ~/Downloads/fastapi  and  ~/Downloads/rag-fastapi-assistant
#
# `{* ../../docs_src/x.py *}` snippet paths are NOT real filesystem-relative
# paths from the including .md file: FastAPI's mkdocs-macros plugin resolves
# them against a fixed base, and empirically every single include in the
# docs (502/502) uses the literal "../../docs_src/..." prefix regardless of
# how deeply nested the .md file is. So we strip the leading "../" segments
# and resolve the remainder (always starting with "docs_src/") against the
# repo root directly.
FASTAPI_REPO_ROOT = PROJECT_ROOT.parent / "fastapi"
LEADING_DOTDOT_RE = re.compile(r"^(\.\./)+")

# The include can carry a trailing highlight-lines hint, e.g.
# "{* ../../docs_src/x.py hl[2,6] *}" -- only the path (first token)
# matters for ingestion, so everything up to the closing "*}" is
# consumed non-greedily and discarded.
#
# `\s+` (mandatory whitespace) after "{*" is deliberate: it's what tells a
# real include apart from Python dict-unpacking syntax like "{**old_dict,
# ...}" that shows up verbatim in the docs' prose/code -- that text starts
# with "{*" too but always has another "*" right after, never whitespace.
# The pattern is also deliberately confined to a single line (no DOTALL):
# a false match on "{*" would otherwise search for the next "*}" anywhere
# later in the file and swallow everything in between.
SNIPPET_INCLUDE_RE = re.compile(r"\{\*\s+(\S+).*?\*\}")
ADMONITION_OPEN_RE = re.compile(r"^///\s*(\w+).*$", re.MULTILINE)
ADMONITION_CLOSE_RE = re.compile(r"^///\s*$", re.MULTILINE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HEADING_ANCHOR_RE = re.compile(r"\{\s*#[\w-]+\s*\}")
MULTI_BLANK_RE = re.compile(r"\n{3,}")

LANG_BY_EXTENSION = {
    ".py": "python",
    ".sh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".env": "bash",
}


def resolve_snippet_includes(content):
    """Replace `{* relative/path *}` markers with the actual referenced file
    content, wrapped as a fenced code block. Falls back to a placeholder if
    the referenced file can't be found (keeps ingestion resilient)."""

    def replace(match):
        raw_path = match.group(1)
        repo_relative_path = LEADING_DOTDOT_RE.sub("", raw_path)
        resolved = (FASTAPI_REPO_ROOT / repo_relative_path).resolve()

        if not resolved.is_file():
            return f"[missing code snippet: {raw_path}]"

        snippet = resolved.read_text(encoding="utf-8").strip()
        lang = LANG_BY_EXTENSION.get(resolved.suffix, "")
        return f"```{lang}\n{snippet}\n```"

    return SNIPPET_INCLUDE_RE.sub(replace, content)


def strip_admonitions(content):
    """Turn `/// tip\n...\n///` blocks into `**TIP:** ...` so the semantic
    meaning survives without the mkdocs-material-specific syntax."""

    def open_replace(match):
        label = match.group(1).upper()
        return f"**{label}:**"

    content = ADMONITION_OPEN_RE.sub(open_replace, content)
    content = ADMONITION_CLOSE_RE.sub("", content)
    return content


def clean_markdown(content):
    content = resolve_snippet_includes(content)
    content = strip_admonitions(content)
    content = HTML_TAG_RE.sub("", content)
    content = HEADING_ANCHOR_RE.sub("", content)
    content = MULTI_BLANK_RE.sub("\n\n", content)
    lines = [line.rstrip() for line in content.splitlines()]
    return "\n".join(lines).strip()


def extract_title(content, fallback):
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if not match:
        return fallback
    title = HEADING_ANCHOR_RE.sub("", match.group(1)).strip()
    return title or fallback


def process_document(md_file_path, doc_id):
    raw_content = md_file_path.read_text(encoding="utf-8")
    cleaned = clean_markdown(raw_content)
    relative_path = md_file_path.relative_to(RAW_DIR)
    section = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"

    return {
        "id": doc_id,
        "title": extract_title(cleaned, fallback=md_file_path.stem),
        "section": section,
        "path": relative_path.as_posix(),
        "content": cleaned,
        "char_count": len(cleaned),
    }


def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Raw docs folder not found: {RAW_DIR}")

    md_files = sorted(RAW_DIR.rglob("*.md"))
    if not md_files:
        raise SystemExit(f"No markdown files found under {RAW_DIR}")

    docs = []
    for doc_id, md_file in enumerate(md_files):
        doc = process_document(md_file, doc_id)
        docs.append(doc)
        print(f"processed [{doc['section']}] {doc['path']} ({doc['char_count']} chars)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "fastapi_docs.json"
    output_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")

    total_chars = sum(d["char_count"] for d in docs)
    print(f"\nDone: {len(docs)} documents -> {output_path}")
    print(f"Total characters: {total_chars:,}")


if __name__ == "__main__":
    main()
