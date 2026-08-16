#!/usr/bin/env python3
"""
common/textnorm.py — Shared text normalisation for the POI-index pipeline.

Single source of truth for how POI names and queries are normalised.
Both halves of the project depend on this module:

  pipeline/build_index.py   — builds name_index keys with normalize_text()
  assistant/index_tools.py  — looks up queries with the same normalize_text()

The Cloudflare data-prep Worker and the Android/iOS offline runtimes MUST
reimplement these two functions byte-for-byte so that a name indexed at
build time matches the same name searched at runtime:

  - NFKD decomposition, then drop combining marks (diacritic-insensitive)
  - replace every run of non-word characters with a single space
  - lowercase, collapse whitespace

`tokenize` is the word-splitting companion used by find_poi_by_name().
"""

from __future__ import annotations

import re
import unicodedata

_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace.

    Diacritic stripping makes 'Vázquez' and 'Vazquez' compare equal,
    which is what users type in search boxes.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = _NON_WORD_RE.sub(" ", stripped)
    return " ".join(stripped.lower().split())


def tokenize(text: str) -> list[str]:
    """Split normalised text into tokens (words)."""
    return normalize_text(text).split()
