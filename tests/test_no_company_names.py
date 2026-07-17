"""Guard: the artifact stays generic. No company/product names anywhere in it.

This study is deliberately company-agnostic: it is grounded in a real, generic
finding (a model that hallucinated out-of-distribution instead of abstaining),
not in any company's product. This test fails loudly if a company or product
name ever leaks into the repo.

Word boundaries matter: a forbidden bare word must not trip on benign words that
merely contain it as a substring.
"""

from __future__ import annotations

import re
from pathlib import Path

# Whole-word company/product names that must never appear in the repo's files.
# These are representative third-party names; none of them
# belong in a generic study artifact.
FORBIDDEN = (
    "placeterma",
    "placetermb",
    "placetermc",
    "placetermd",
    "placeterme",
    "placetermf",
)
FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN) + r")\b", re.IGNORECASE)

ROOT = Path(__file__).resolve().parent.parent

# Only scan text we author; skip VCS and tool caches.
SKIP_DIRS = {".git", ".ruff_cache", ".pytest_cache", ".venv", "__pycache__"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".cfg", ".ini"}


def _iter_text_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def test_no_forbidden_company_names_in_repo() -> None:
    # This test file names the forbidden strings; exclude only itself.
    self_path = Path(__file__).resolve()
    offenders: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == self_path:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_RE.search(text):
            hits = sorted({m.group(0).lower() for m in FORBIDDEN_RE.finditer(text)})
            offenders.append(f"{path.relative_to(ROOT)} contains {hits}")
    assert not offenders, "company names leaked into the artifact: " + "; ".join(offenders)
