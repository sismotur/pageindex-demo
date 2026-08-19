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
    extract_poi_tags,
    format_find_poi_by_name,
    format_filter_pois,
    format_poi,
    format_search_pois,
    format_section,
    format_sections_overview,
    get_poi,
    get_pois,
    poi_uri,
    search_pois,
    sanitize_poi_tags,
    sanitize_tourist_answer,
    strip_poi_tags,
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


# ── Schema v3: section groups + evidence search ─────────────────────────────

class TestSectionGroups:
    """Sections with > 30 POIs must carry a consistent per-type group map."""

    def test_schema_version_is_3(self, index):
        assert index["meta"]["schema_version"] == 3

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
        assert "Browse groups:" in out
        assert "shopping--store" not in out  # no raw group ids
        assert "filter_pois(type=" not in out  # no tool internals

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


# ── Context reduction (LLM-side token budget) ────────────────────────────────

class TestContextReduction:
    """The LLM never sees the raw index — only these renderings."""

    def test_poi_output_has_no_media_lines(self, index):
        """Media URLs are for the app UI, not the model (~13% of tokens)."""
        out = format_poi(index, "poi/30117")
        assert "**Images**" not in out
        assert "**Audio guides**" not in out
        assert "**Documents**" not in out
        # …but the content the model needs is still there
        assert "+34953750345" in out
        assert "description" not in out.lower() or len(out) > 200

    def test_find_full_detail_appends_full_record(self, index):
        brief = format_find_poi_by_name(index, "ariza bridge", detail="brief")
        full = format_find_poi_by_name(index, "ariza bridge", detail="full")
        assert "Best match, full record:" not in brief
        assert "Best match, full record:" in full
        # the appended record carries real per-POI facts (Ariza Bridge: 1562)
        assert "1562" in full

    def test_find_full_detail_no_matches(self, index):
        out = format_find_poi_by_name(index, "zzznopezzz", detail="full")
        assert "[INFO] No POI matches" in out
        assert "Best match" not in out

    def test_grouped_section_default_limit_is_20(self, index):
        out = format_section(index, "shopping")          # no limit passed
        preview_lines = [l for l in out.splitlines()
                         if l.startswith("  <poi ")]
        assert len(preview_lines) == 20
        assert "more (raise --limit" in out               # truncation note

    def test_flat_section_default_limit_keeps_all(self, index):
        # museums-and-culture has 5 POIs, no groups → flat default 50
        out = format_section(index, "museums-and-culture")
        preview_lines = [l for l in out.splitlines()
                         if l.startswith("  <poi ")]
        assert len(preview_lines) == 5
        assert "more (raise --limit" not in out

    def test_explicit_limit_overrides_default(self, index):
        out = format_section(index, "shopping", limit=5)
        preview_lines = [l for l in out.splitlines()
                         if l.startswith("  <poi ")]
        assert len(preview_lines) == 5

    def test_sections_overview_drops_top_interests(self, index):
        out = format_sections_overview(index)
        assert "Top interests" not in out
        assert "Notable:" in out          # key items stay
        assert "POIs" in out               # counts stay

    def test_list_tool_output_is_tag_ready_and_hides_raw_metadata(self, index):
        out = format_filter_pois(
            index, type="Restaurant", section_id="gastronomy", limit=2,
        )
        assert "<poi id=" in out and " type=" in out
        assert "[poi/" not in out
        assert "Filter {" not in out
        assert "Interest Level:" not in out
        assert "Type:" not in out


class TestEvidenceSearch:
    """Compound requests require same-record evidence, not category aliases."""

    def test_search_terms_present_and_sorted(self, index):
        terms = index["facets"]["search_terms"]
        assert "olive" in terms
        assert list(terms) == sorted(terms)
        assert terms["olive"] == sorted(terms["olive"])

    def test_compound_query_has_no_false_direct_match(self, index):
        # The catalog has oil-related places and restaurants, but does not
        # establish that one restaurant serves olive-oil cuisine.
        matches = search_pois(
            index, "olive oil restaurant", section_id="gastronomy",
        )
        assert matches == []

    def test_individual_evidence_search_returns_oil_places(self, index):
        matches = search_pois(index, "olive oil", section_id="gastronomy")
        assert matches
        assert all("olive" in item["matched_terms"] for item in matches)
        assert any("olive" in item["evidence"].lower() for item in matches)

    def test_plural_search_matches_restaurant_category(self, index):
        singular = search_pois(index, "restaurant", section_id="gastronomy")
        plural = search_pois(index, "restaurants", section_id="gastronomy")
        assert singular
        assert {item["poi"]["poi_id"] for item in singular} == {
            item["poi"]["poi_id"] for item in plural
        }

    def test_evidence_formatter_no_internal_filter_leak(self, index):
        out = format_search_pois(
            index, "olive oil restaurant", section_id="gastronomy",
        )
        assert "No place record explicitly mentions all of:" in out
        assert "filter_pois" not in out
        assert "type=" not in out

    def test_evidence_formatter_tag_ready_results(self, index):
        out = format_search_pois(index, "olive oil", section_id="gastronomy")
        assert "<poi id=" in out
        assert " type=OilMill>" in out


# ── POI tags in answers (app deep links) ─────────────────────────────────────
# The model wraps POI mentions as <poi id=5155>Church of San Nicolás</poi>;
# the app parser catches the tag, shows the inner text, and opens the POI
# by bare numeric id (PointOfInterestActivity "poiId" extra).

class TestPoiTags:
    def test_poi_uri_format(self):
        assert poi_uri("ubeda", "poi/5155") == \
            "https://inventrip.com/ubeda/object/5155"
        assert poi_uri("ubeda", "5155") == \
            "https://inventrip.com/ubeda/object/5155"   # bare id tolerated

    def test_parse_single_tag_no_type(self, index):
        ans = "Visit <poi id=5155>Úbeda - Heritage City</poi> first."
        refs = extract_poi_tags(ans, index)
        assert len(refs) == 1
        r = refs[0]
        assert r["poi_id"] == "poi/5155"
        assert r["text"] == "Úbeda - Heritage City"
        assert r["known"] is True
        assert r["name"] == "Úbeda - Heritage City"
        assert r["uri"] == "https://inventrip.com/ubeda/object/5155"
        # type_code falls back from the index when not in the tag
        assert r["type_code"] is not None

    def test_parse_tag_with_type_attribute(self, index):
        ans = "<poi id=5155 type=WorldHeritageSite>Úbeda</poi>"
        refs = extract_poi_tags(ans, index)
        assert refs[0]["type_code"] == "WorldHeritageSite"

    def test_parse_tag_type_before_id(self, index):
        ans = "<poi type=OilMill id=36694>Almazara</poi>"
        refs = extract_poi_tags(ans, index)
        assert refs[0]["poi_id"] == "poi/36694"
        assert refs[0]["type_code"] == "OilMill"

    def test_parse_multiple_in_order_and_dedupe(self, index):
        ans = ("<poi id=30117>Dean Ortega Palace</poi>, then "
               "<poi id=5155>Heritage City</poi>, and again "
               "<poi id=30117>the Parador</poi>.")
        refs = extract_poi_tags(ans, index)
        assert [r["poi_id"] for r in refs] == ["poi/30117", "poi/5155"]

    def test_parser_lenient_on_prefix_and_quotes(self, index):
        ans = ('<poi id="poi/5155">a</poi> <poi id="5155">b</poi> '
               '<poi id=poi/5155>c</poi>')
        refs = extract_poi_tags(ans, index)
        # all three forms resolve to the same POI → deduped to one ref
        assert len(refs) == 1
        assert refs[0]["poi_id"] == "poi/5155"
        assert refs[0]["known"] is True

    def test_unknown_id_marked_not_known(self, index):
        refs = extract_poi_tags("<poi id=99999999>Ghost</poi>", index)
        assert len(refs) == 1
        assert refs[0]["known"] is False
        assert refs[0]["text"] == "Ghost"      # inner text survives
        assert "uri" not in refs[0]            # no link for unknown ids

    def test_sanitize_unknown_tag_to_plain_text(self, index):
        answer = "Visit <poi id=99999999 type=Restaurant>Ghost Place</poi>."
        assert sanitize_poi_tags(answer, index) == "Visit Ghost Place."

    def test_sanitize_known_tag_canonicalizes_type(self, index):
        answer = "<poi id=36694 type=Restaurant>Short label</poi>"
        assert sanitize_poi_tags(answer, index) == \
            "<poi id=36694 type=OilMill>Short label</poi>"

    def test_sanitize_known_empty_tag_expands_name(self, index):
        sanitized = sanitize_poi_tags("Visit <poi id=36694/>.", index)
        assert "<poi id=36694 type=OilMill>" in sanitized
        assert "ALMAZARA BALTASAR LARA Y CÍA." in sanitized

    def test_tourist_answer_sanitizer_removes_catalog_language(self, index):
        answer = (
            "I found 2 POIs and one point of interest: "
            "<poi id=99999999>Ghost</poi>."
        )
        sanitized = sanitize_tourist_answer(answer, index)
        assert sanitized == "I found 2 places and one place: Ghost."

    def test_no_tags_returns_empty(self, index):
        assert extract_poi_tags("no tags here", index) == []
        assert extract_poi_tags("", index) == []

    def test_strip_tags_to_visible_text(self):
        ans = ("Start at <poi id=5155>Úbeda - Heritage City</poi>, "
               "then <poi id=30117>the Parador</poi>.")
        assert strip_poi_tags(ans) == \
            "Start at Úbeda - Heritage City, then the Parador."

    def test_strip_leaves_malformed_tags_readable(self):
        ans = "see <poi id=5155 the church"   # unclosed → no match
        assert strip_poi_tags(ans) == ans

    def test_inner_text_with_markup_and_accent(self, index):
        ans = "<poi id=5155>**Úbeda** – *Heritage* City</poi>!"
        refs = extract_poi_tags(ans, index)
        assert refs[0]["text"] == "**Úbeda** – *Heritage* City"
        assert strip_poi_tags(ans) == "**Úbeda** – *Heritage* City!"
