"""
Unit tests for the pure text-transformation functions in src/ingestion.py.

Several of these cases are regression tests for real bugs found during
development (see AUDIT.md / git history): a missed `hl[...]` highlight-hint
pattern, and a false-positive match on Python's `{**dict}` unpacking syntax
that used to eat the rest of the file looking for a `*}` that was never
there.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ingestion  # noqa: E402


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A minimal fake FastAPI repo with one real docs_src snippet, so
    resolve_snippet_includes has something real to resolve against without
    depending on the actual cloned fastapi/ repo being present."""
    repo_root = tmp_path / "fastapi"
    docs_src = repo_root / "docs_src" / "first_steps"
    docs_src.mkdir(parents=True)
    (docs_src / "tutorial001.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n", encoding="utf-8"
    )
    monkeypatch.setattr(ingestion, "FASTAPI_REPO_ROOT", repo_root)
    return repo_root


def test_resolve_snippet_includes_bare(fake_repo):
    content = "Some text\n\n{* ../../docs_src/first_steps/tutorial001.py *}\n\nMore text"
    result = ingestion.resolve_snippet_includes(content)
    assert "app = FastAPI()" in result
    assert "```python" in result
    assert "{*" not in result


def test_resolve_snippet_includes_with_highlight_hint(fake_repo):
    content = "{* ../../docs_src/first_steps/tutorial001.py hl[2,6] *}"
    result = ingestion.resolve_snippet_includes(content)
    assert "app = FastAPI()" in result
    assert "hl[2,6]" not in result


def test_resolve_snippet_includes_ignores_dict_unpacking(fake_repo):
    content = 'new_dict = {**old_dict, "new key": "new value"}'
    result = ingestion.resolve_snippet_includes(content)
    assert result == content


def test_resolve_snippet_includes_reports_missing_file(fake_repo):
    content = "{* ../../docs_src/does_not_exist.py *}"
    result = ingestion.resolve_snippet_includes(content)
    assert "[missing code snippet:" in result


def test_resolve_snippet_includes_blocks_path_traversal(fake_repo, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak this", encoding="utf-8")

    content = "{* docs_src/../../secret.txt *}"
    result = ingestion.resolve_snippet_includes(content)

    assert "do not leak this" not in result
    assert "blocked" in result


def test_strip_admonitions_converts_tip_block():
    content = "before\n/// tip\ncontent here\n///\nafter"
    result = ingestion.strip_admonitions(content)
    assert "**TIP:**" in result
    assert "///" not in result
    assert "content here" in result


def test_extract_title_finds_first_heading():
    content = "# My Title { #my-title }\n\nBody text"
    assert ingestion.extract_title(content, fallback="fallback") == "My Title"


def test_extract_title_falls_back_when_no_heading():
    assert ingestion.extract_title("no heading here", fallback="fallback-name") == "fallback-name"


def test_clean_markdown_strips_html_tags():
    result = ingestion.clean_markdown('<span style="color:red">text</span>')
    assert result == "text"
