"""
tests/test_index_tools.py — Index schema v2 and tool-layer regression tests.

Covers the PageIndex-inspired improvements:
  - sections > 30 POIs carry per-type `groups` (key_items pattern)
  - get_poi / format_poi accept comma-separated ids (batch fetch)
  - common/textnorm.py normalisation invariants

Run with:
    cd /path/to/pageindex-demo
    .venv/bin/python -m pytest tests/test_index_tools.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))                     # common.*
sys.path.insert(0, str(PROJECT_ROOT / "assistant"))       # index_tools

from index_tools import (
    load_index,
    format_poi,
    format_section,
    get_poi,
    get_pois,
)
from common.textnorm import normalize_text, tokenize

INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda_en.json"


@pytest.fixture(scope="module")
def index():
    if not INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {INDEX_FILE}")
    return load_index(INDEX_FILE)


# ── Text normalisation ───────────────────────────────────────────────────────

class TestTextNorm:
    """The Cloudflare and mobile ports must reproduce these exact outputs."""

    def test_strips_diacritics(self):
        assert normalize_text("Vázquez de Molina") == "vazquez de molina"

    def test_punctuation_becomes_space(self):
        assert normalize_text("Sacra Capilla d'El Salvador") == \
            "sacra capilla d el salvador"

    def test_collapses_whitespace_and_case(self):
        assert normalize_text("  PLAZA\n de\tANDALUCÍA ") == "plaza de andalucia"

    def test_empty_input(self):
        assert normalize_text("") == ""
        assert tokenize("") == []

    def test_tokenize(self):
        assert tokenize("Casa de las Torres") == ["casa", "de", "las", "torres"]


# ── Schema v2: section groups ────────────────────────────────────────────────

class TestSectionGroups:
    """Sections with > 30 POIs must carry a consistent per-type group map."""

    def test_schema_version_is_2(self, index):
        assert index["meta"]["schema_version"] == 2

    def test_large_sections_have_groups(self, index):
        grouped = {s["section_id"] for s in index["sections"] if s.get("groups")}
        # On the Úbeda corpus exactly four sections exceed GROUP_MIN_POIS
        # (30): shopping (66), health-and-beauty (48),
        # civil-and-historical-monuments (44), gastronomy (40).
        # Accommodation (30) stays flat — the threshold is strict (> 30).
        assert grouped == {
            "civil-and-historical-monuments",
            "gastronomy",
            "shopping",
            "health-and-beauty",
        }

    def test_small_sections_stay_flat(self, index):
        for s in index["sections"]:
            if len(s["poi_ids"]) <= 30:
                assert "groups" not in s, f"{s['section_id']} should be flat"

    def test_groups_partition_section(self, index):
        for s in index["sections"]:
            groups = s.get("groups")
            if not groups:
                continue
            merged = [pid for g in groups for pid in g["poi_ids"]]
            assert sorted(merged) == sorted(s["poi_ids"]), (
                f"{s['section_id']}: groups do not partition the section"
            )
            # group ids unique + well-formed
            ids = [g["group_id"] for g in groups]
            assert len(ids) == len(set(ids))
            for gid in ids:
                assert gid.startswith(s["section_id"] + "--")

    def test_groups_sorted_best_first(self, index):
        """Each group's first POI is its best by (interest, zoom, name);
        group order follows the best POI of each group."""
        pois = index["pois"]

        def key(pid):
            p = pois[pid]
            return (p.get("interest_level") or 99,
                    p.get("zoom_level") or 99,
                    p.get("normalized_name") or "")

        for s in index["sections"]:
            for g in s.get("groups") or []:
                keys = [key(pid) for pid in g["poi_ids"]]
                assert keys == sorted(keys)
            groups = s.get("groups") or []
            if groups:
                firsts = [key(g["poi_ids"][0]) for g in groups]
                assert firsts == sorted(firsts)

    def test_format_section_renders_group_map(self, index):
        out = format_section(index, "shopping", sort="interest", limit=50)
        assert "Groups in this section" in out
        assert "shopping--store" in out or "shopping--shoppingcenter" in out
        assert "filter_pois(type=" in out  # drill-down instruction

    def test_flat_section_has_no_group_block(self, index):
        out = format_section(index, "museums-and-culture")
        assert "Groups in this section" not in out


# ── Batch get_poi ────────────────────────────────────────────────────────────

class TestBatchGetPoi:
    def test_single_id_unchanged(self, index):
        out = format_poi(index, "poi/5155")
        assert out.startswith("# ")
        assert "---" not in out.split("\n")[0]
        assert "\n---\n" not in out

    def test_batch_returns_all_records(self, index):
        out = format_poi(index, "poi/5155,poi/65804")
        assert "Úbeda - Heritage City" in out
        assert "Southern Renaissance Route" in out
        assert "\n\n---\n\n" in out

    def test_batch_accepts_bare_numbers_and_spaces(self, index):
        out = format_poi(index, "5155, 65804")
        assert "Úbeda - Heritage City" in out
        assert "Southern Renaissance Route" in out

    def test_batch_unknown_id_inline_error(self, index):
        out = format_poi(index, "poi/5155,poi/99999999")
        assert "Úbeda - Heritage City" in out
        assert "[ERROR] POI 'poi/99999999' not found." in out

    def test_get_pois_returns_list(self, index):
        res = get_pois(index, ["poi/5155", "nope", "65804"])
        assert len(res) == 3
        assert res[0]["poi_id"] == "poi/5155"
        assert res[1] is None
        assert res[2]["poi_id"] == "poi/65804"

    def test_get_poi_none_safe(self, index):
        assert get_poi(index, None) is None
