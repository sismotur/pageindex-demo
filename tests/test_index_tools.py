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
    format_history_followup,
    format_poi,
    format_search_pois,
    format_search_trips,
    format_search_paths,
    format_section,
    format_sections_overview,
    format_trip,
    format_trip_choice_offer,
    format_path,
    get_poi,
    get_pois,
    get_trip,
    get_path,
    poi_uri,
    resolve_history_selection,
    resolve_sole_recent_source,
    resolve_trip_query,
    search_pois,
    search_trips,
    search_paths,
    sanitize_poi_tags,
    sanitize_tourist_answer,
    strip_poi_tags,
)
from common.textnorm import normalize_text, tokenize
from run_eval import (
    COMPACTED_TOOL_RESULT,
    MAX_TOOL_HISTORY_CHARS,
    MAX_TOOL_RESULT_CHARS,
    bound_tool_result,
    compact_tool_history,
    execute_tool,
    grounding_failure_message,
    is_physical_route_request,
    requires_current_turn_grounding,
    requires_trip_detail,
)

INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda_en.json"
SPANISH_INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda_es.json"


@pytest.fixture(scope="module")
def index():
    if not INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {INDEX_FILE}")
    return load_index(INDEX_FILE)


@pytest.fixture(scope="module")
def spanish_index():
    if not SPANISH_INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {SPANISH_INDEX_FILE}")
    return load_index(SPANISH_INDEX_FILE)


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


class TestRouteIntent:
    """Physical-route control must not trigger on ordinary tourist questions."""

    @pytest.mark.parametrize("question", [
        "Can you suggest a walking route in Úbeda?",
        "¿Puedes sugerirme una ruta a pie en Úbeda?",
        "¿Hay senderos para caminar?",
        "Vorrei un percorso in bicicletta",
    ])
    def test_detects_physical_route_requests(self, question):
        assert is_physical_route_request(question)

    def test_spanish_breakfast_question_is_not_a_route(self):
        # Regression for `se` falsely matching Croatian `setnja` through
        # symmetric prefix matching, which caused a 14-round chat loop.
        assert not is_physical_route_request(
            "se puedec desayunar aceite autenr"
        )


class TestStrictGrounding:
    def test_social_messages_do_not_require_retrieval(self):
        assert not requires_current_turn_grounding("hola")
        assert not requires_current_turn_grounding("Thanks")
        assert requires_current_turn_grounding("hoteles cerca del ayuntamiento")

    @pytest.mark.parametrize("question", [
        "dame un plan de cosas que ver en dos días",
        "show me a weekend itinerary",
        "dame los detalles del recorrido",
    ])
    def test_plan_detail_intent(self, question):
        assert requires_trip_detail(question)

    def test_failure_message_follows_index_language(self, index):
        spanish = dict(index)
        spanish["meta"] = dict(index["meta"], lang="es")
        assert grounding_failure_message(spanish).startswith("No he podido")

    def test_resolves_unique_trip_selection_from_history(self, spanish_index):
        from index_tools import resolve_history_selection
        history = [{
            "role": "assistant",
            "content": '<trip id=4457>Qué No Perderte</trip>',
        }]
        selection = resolve_history_selection(
            "No Perderte", history, spanish_index
        )
        assert selection == {
            "kind": "trip",
            "id": "trip/4457",
            "label": "Qué No Perderte",
        }

    def test_resolves_unique_poi_selection_from_history(self, index):
        from index_tools import resolve_history_selection
        history = [{
            "role": "assistant",
            "content": '<poi id=36026 type=TouristAttraction>Ariza Bridge</poi>',
        }]
        selection = resolve_history_selection("Ariza", history, index)
        assert selection["kind"] == "poi"
        assert selection["id"] == "poi/36026"

    def test_unknown_history_tag_cannot_be_selected(self, index):
        from index_tools import resolve_history_selection
        history = [{
            "role": "assistant",
            "content": '<trip id=999999>Imaginary Weekend</trip>',
        }]
        assert resolve_history_selection("Weekend", history, index) is None

    def test_direct_trip_title_overrides_route_word(self, spanish_index):
        selection = resolve_trip_query(
            "quiero las rutas por úbeda", spanish_index
        )
        assert selection == {
            "kind": "trip",
            "id": "trip/4420",
            "label": "RUTAS POR ÚBEDA",
        }


# ── Schema v3: section groups + evidence search ─────────────────────────────

class TestSectionGroups:
    """Sections with > 30 POIs must carry a consistent per-type group map."""

    def test_schema_version_is_6(self, index):
        assert index["meta"]["schema_version"] == 6

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
        assert "refine with filters or a name search" in out

    def test_flat_section_default_limit_keeps_all(self, index):
        # museums-and-culture has 5 POIs, no groups → flat default 50
        out = format_section(index, "museums-and-culture")
        preview_lines = [l for l in out.splitlines()
                         if l.startswith("  <poi ")]
        assert len(preview_lines) == 5
        assert "refine with filters or a name search" not in out

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
        assert plural
        # Plural also matches visitor descriptions that mention restaurants,
        # so ranking can include non-restaurant venues. It must still
        # resolve real Restaurant records through the singular variant.
        assert "poi/65817" in {item["poi"]["poi_id"] for item in plural}

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

# ── Curated trip suggestions and physical paths ─────────────────────────────


class TestRagDataSchemaCompat:
    """Forward-compat with inventrip-rag-data index shape changes.

    Routes live only under paths (get_trip falls back). Editorial trips may
    carry is_route=false. New sections appear only when the destination has
    matching POIs; tools must resolve them by id/title without hardcoding.
    Optional top-level type_display must not break loaders/formatters.
    """

    def test_path_only_route_resolves_via_get_trip(self, index):
        synthetic = dict(index)
        synthetic["paths"] = list(synthetic.get("paths") or []) + [{
            "itinerary_id": "trip/88001",
            "path_id": "trip/88001",
            "kind": "path",
            "source_type": "Path",
            "name": "Compat Cliff Path",
            "description": "Coastal walk.",
            "url": "",
            "is_route": True,
            "steps": [],
        }]
        synthetic["trips"] = [
            t for t in (synthetic.get("trips") or [])
            if t.get("itinerary_id") != "trip/88001"
        ]
        item = get_trip(synthetic, "88001")
        assert item is not None
        assert item["is_route"] is True
        assert item["kind"] == "path"
        out = format_trip(synthetic, "88001")
        assert "Compat Cliff Path" in out
        assert search_trips(synthetic, "compat cliff path") == []

    def test_editorial_trip_is_route_false_still_formats(self, index):
        synthetic = dict(index)
        trip = {
            "itinerary_id": "trip/88002",
            "trip_id": "trip/88002",
            "kind": "trip",
            "source_type": "TouristTrip",
            "name": "Compat Day Plan",
            "description": "A themed day.",
            "url": "",
            "is_route": False,
            "steps": [{
                "position": 1,
                "title": "Morning",
                "poi_ids": [],
                "unresolved_poi_names": [],
            }],
        }
        synthetic["trips"] = list(synthetic.get("trips") or []) + [trip]
        out = format_trip(synthetic, "88002")
        assert out.startswith("# <trip id=88002>Compat Day Plan</trip>")
        assert "1. Morning" in out

    def test_new_section_ids_resolve_and_overview_lists_them(self, index):
        synthetic = dict(index)
        synthetic["sections"] = list(synthetic.get("sections") or []) + [
            {
                "section_id": "emergency-services",
                "title": "Emergency Services",
                "summary": "2 POIs.",
                "poi_ids": [],
            },
            {
                "section_id": "transport-and-access",
                "title": "Transport and Access",
                "summary": "3 POIs.",
                "poi_ids": [],
            },
        ]
        from index_tools import find_section, format_sections_overview
        assert find_section(synthetic, "emergency-services")["title"] == "Emergency Services"
        assert find_section(synthetic, "Transport and Access")["section_id"] == "transport-and-access"
        overview = format_sections_overview(synthetic)
        assert "[emergency-services] Emergency Services" in overview
        assert "[transport-and-access] Transport and Access" in overview

    def test_optional_type_display_top_level_ignored_safely(self, index):
        synthetic = dict(index)
        synthetic["type_display"] = {"Museum": "Museum"}
        # load path is dict already; formatters must not require the key
        synthetic.pop("type_display", None)
        from index_tools import format_sections_overview
        assert "SECTIONS:" in format_sections_overview(synthetic)


class TestCuratedItineraries:
    """Trips are suggestions; paths are physical routes from /v120/paths."""

    def test_schema_v6_has_trips_and_paths(self, index):
        assert index["meta"]["schema_version"] == 6
        assert len(index["trips"]) >= 29
        # Úbeda currently advertises no /paths route IDs. Empty is valid.
        assert index["paths"] == []

    def test_trip_steps_keep_order_and_resolution(self, index):
        trip = get_trip(index, "trip/4407")
        assert trip is not None
        assert trip["kind"] == "trip"
        assert trip["name"] == "Savor Úbeda"
        assert [step["position"] for step in trip["steps"]] == \
            list(range(1, len(trip["steps"]) + 1))
        assert any(step["poi_ids"] for step in trip["steps"])
        for step in trip["steps"]:
            for poi_id in step["poi_ids"]:
                assert poi_id in index["pois"]

    def test_trip_search_returns_trip_tags_not_path_tags(self, index):
        out = format_search_trips(index, "savor", limit=3)
        assert "<trip id=4407>Savor Úbeda</trip>" in out
        assert "<path " not in out

    def test_trip_detail_has_tagged_ordered_poi_stops(self, index):
        out = format_trip(index, "4407")  # bare numeric id accepted
        assert out.startswith("# <trip id=4407>Savor Úbeda</trip>")
        assert "1. Restaurants" in out
        assert "<poi id=" in out
        assert "[poi/" not in out

    def test_missing_path_is_truthful(self, index):
        assert search_paths(index, "walking") == []
        assert format_search_paths(index, "walking") == \
            "No curated routes matched this request."
        assert format_path(index, "path/999") == \
            "That physical route is not available."

    def test_synthetic_path_not_found_via_search_trips(self, index):
        """A route is found only via search_paths(), never search_trips().

        It still renders as a <trip id=…> tag (source API routes are trip
        records). get_trip() falls back to paths so tag follow-up works
        without dual-listing the JSON record under trips.
        """
        synthetic = dict(index)
        synthetic["paths"] = [{
            "itinerary_id": "trip/9001",
            "path_id": "trip/9001",
            "kind": "path",
            "source_type": "Path",
            "name": "Riverwalk9001 Walking Route",
            "description": "A walking route beside the river.",
            "url": "",
            "is_route": True,
            "steps": [{
                "position": 1,
                "title": "Start",
                "poi_ids": ["poi/36026"],
                "unresolved_poi_names": ["River viewpoint"],
            }],
        }]
        assert search_trips(synthetic, "riverwalk9001") == []
        paths = search_paths(synthetic, "riverwalk9001")
        assert [path["itinerary_id"] for path in paths] == ["trip/9001"]
        assert get_trip(synthetic, "9001") is paths[0]
        out = format_path(synthetic, "9001")
        assert out.startswith("# <trip id=9001>Riverwalk9001 Walking Route</trip>")
        assert "A walking route beside the river." in out
        # A route's real stops still show (unlike a degenerate header step
        # with no items — see test_route_reached_via_get_trip_shows_stops_
        # without_numbering), but never the day-by-day "N. Title" numbering
        # format_trip() uses for editorial trips, and never an unresolved
        # source label (may be stale/foreign-language).
        assert "1. Start" not in out
        assert "Start" in out
        assert "<poi id=36026" in out
        assert "River viewpoint" not in out

    def test_route_reached_via_get_trip_shows_stops_without_numbering(self, index):
        """Regression: a path-only route opened via get_trip/format_trip
        must render like format_path — real stops, no day-by-day numbering,
        and a step title that repeats the route name is dropped.
        """
        synthetic = dict(index)
        synthetic["paths"] = list(synthetic.get("paths") or []) + [{
            "itinerary_id": "trip/9003",
            "path_id": "trip/9003",
            "kind": "path",
            "source_type": "Path",
            "name": "Riverwalk9003 Walking Route",
            "description": "A walking route beside the river.",
            "url": "",
            "is_route": True,
            "steps": [
                {"position": 1, "title": "Arroyomolinos", "poi_ids": []},
                {"position": 2, "title": "Riverwalk9003 Walking Route",
                 "poi_ids": []},
                {"position": 3, "title": "Principales molinos:",
                 "poi_ids": ["poi/36026"]},
            ],
        }]
        out = format_trip(synthetic, "9003")
        assert out.startswith("# <trip id=9003>Riverwalk9003 Walking Route</trip>")
        assert "A walking route beside the river." in out
        # The real stop shows, tag-ready.
        assert "<poi id=36026" in out
        assert "Principales molinos:" in out
        # The two empty header steps (no items/poi_ids) are dropped
        # entirely — including the one that just repeats the route's own
        # name — and nothing is ever numbered like a day-by-day plan.
        assert "Arroyomolinos" not in out
        assert "1. " not in out
        assert "2. " not in out
        assert "3. " not in out

    def test_unresolved_waypoints_are_not_rendered_to_tourists(self, index):
        synthetic = dict(index)
        synthetic["trips"] = [{
            "itinerary_id": "trip/9002",
            "trip_id": "trip/9002",
            "kind": "trip",
            "source_type": "TouristTrip",
            "name": "Local Discovery",
            "description": "A local suggestion.",
            "url": "",
            "steps": [{
                "position": 1,
                "title": "Meet",
                "poi_ids": [],
                "unresolved_poi_names": ["Uncatalogued meeting point"],
            }],
        }]
        assert "Uncatalogued meeting point" not in format_trip(synthetic, "9002")


class TestSoleRecentSource:
    """Anchor for generic plan/detail follow-ups ("give me the itinerary")."""

    def test_resolves_the_one_trip_shown(self, index):
        history = [{
            "role": "assistant",
            "content": "# <trip id=4407>Savor Úbeda</trip>\n\n...",
        }]
        assert resolve_sole_recent_source(history, index) == {
            "kind": "trip", "id": "trip/4407", "label": "Savor Úbeda",
        }

    def test_returns_none_when_nothing_was_shown(self, index):
        history = [{"role": "assistant", "content": "No tools were called."}]
        assert resolve_sole_recent_source(history, index) is None

    def test_returns_none_when_ambiguous(self, index):
        history = [{
            "role": "assistant",
            "content": (
                "<trip id=4407>Savor Úbeda</trip> "
                "<trip id=4413>To Rest...</trip>"
            ),
        }]
        assert resolve_sole_recent_source(history, index) is None


class TestNestedItems:
    """Schema v6: itineraries preserve folder/POI hierarchy end-to-end.

    The extraction/building half of this hierarchy (extract_itinerary_items,
    _resolve_items, _resolve_itinerary_steps) now lives in the sibling
    inventrip-rag-data repo's tests/test_build_index.py; these tests cover
    the rendering half against an already-built index.
    """

    def test_renderer_indents_nested_folders(self, spanish_index):
        rendered = format_trip(spanish_index, "trip/4444")
        # First folder line at depth 1, its child POIs at depth 2.
        assert "   - 1.1 Plaza Vázquez de Molina" in rendered
        assert "      - <poi id=30536" in rendered

    def test_spanish_trip_source_ids_link_but_absent_stop_stays_hidden(self, spanish_index):
        trip = get_trip(spanish_index, "trip/4444")
        first_step = trip["steps"][0]
        # Schema v6 keeps subfolder labels in a flat list for tools that
        # do not walk the nested tree (Android search, etc.).
        assert "1.1 Plaza Vázquez de Molina" in first_step["subfolders"]
        lodging = next(step for step in trip["steps"] if step["title"].startswith("4."))
        resolved = {item["source_name"]: item for item in lodging["poi_resolutions"]}
        # The API now carries a stable poi id per stop, so the localized
        # Spanish name resolves via source_id rather than a cross-language
        # alias.
        assert resolved["Hotel Yit El Postigo"]["poi_id"] == "poi/30459"
        assert resolved["Hotel Yit El Postigo"]["resolution"] == "source_id"
        # CR La Casería de Tito still has no matching POI in the ES corpus,
        # so it stays QA-only and out of visitor output.
        assert "CR La Casería de Tito" in lodging["unresolved_poi_names"]

        rendered = format_trip(spanish_index, "trip/4444")
        # Nested folder header at depth 1 (three-space indent) followed by
        # its child POIs at depth 2 (six-space indent).
        assert "   - 1.1 Plaza Vázquez de Molina" in rendered
        assert "<poi id=30459 type=Hotel>Hotel Yit El Postigo</poi>" in rendered
        assert "CR La Casería de Tito" not in rendered


class TestTripChoiceOffer:
    """Broad plan requests fan out to a 2–3 curated trip choice, not one auto-pick."""

    def test_offer_lists_up_to_three_tagged_candidates(self, spanish_index):
        matches = search_trips(spanish_index, "plan dos días", limit=3)
        assert len(matches) >= 2
        offer = format_trip_choice_offer(spanish_index, matches[:3])
        # Localized lead line for Spanish
        assert offer.startswith(
            "He encontrado varias sugerencias que podrían encajar"
        )
        # Each candidate rendered as a validated <trip> tag
        import re
        ids = re.findall(r"<trip id=(\d+)>", offer)
        assert len(ids) == len(matches[:3]) >= 2
        # Headline POI names come from resolved child POIs
        assert "Destacan:" in offer
        assert "Dime el nombre o el número" in offer

    def test_offer_falls_back_to_english_for_unknown_lang(self, spanish_index):
        # Clone the index and pretend the runtime is set to Japanese.
        idx = dict(spanish_index)
        idx["meta"] = dict(spanish_index["meta"], lang="ja")
        matches = search_trips(spanish_index, "plan dos días", limit=2)
        offer = format_trip_choice_offer(idx, matches[:2])
        assert offer.startswith("Here are a few curated trips")
        assert "Highlights:" in offer

    def test_offer_omits_period_after_ellipsis(self, spanish_index):
        matches = search_trips(spanish_index, "plan dos días", limit=3)
        offer = format_trip_choice_offer(spanish_index, matches[:3])
        # Never render "…." (period immediately after ellipsis).
        assert "…." not in offer

    def test_bare_numeric_id_selects_previously_shown_trip(self, spanish_index):
        history = [{
            "role": "assistant",
            "content": (
                'Opciones:\n'
                '  - <trip id=4444>BIENVENIDO A ÚBEDA - Tarjeta NFC</trip>\n'
                '  - <trip id=4457>Qué No Perderte</trip>'
            ),
        }]
        selection = resolve_history_selection("4457", history, spanish_index)
        assert selection == {
            "kind": "trip",
            "id": "trip/4457",
            "label": "Qué No Perderte",
        }

    def test_bare_numeric_id_matching_nothing_falls_through(self, spanish_index):
        history = [{
            "role": "assistant",
            "content": '<trip id=4457>Qué No Perderte</trip>',
        }]
        # A number that does not match any shown tag id.
        assert resolve_history_selection("99999", history, spanish_index) is None


class TestHistoryFollowup:
    """When the model refuses to call tools on a broad follow-up, the
    runtime filters previously shown POIs by the visitor's topic words
    and answers directly instead of returning a safe failure.
    """

    @pytest.fixture
    def trip_history(self, spanish_index):
        # Simulate the trip detail we would have just shown.
        return [{"role": "assistant",
                  "content": format_trip(spanish_index, "trip/4444")}]

    def test_tapas_followup_surfaces_tapas_zone(self, spanish_index, trip_history):
        out = format_history_followup(
            spanish_index, "entonces cuál tiene tapas", trip_history
        )
        assert out, "expected at least one match for a 'tapas' follow-up"
        # The tapas zones in the trip are BarOrPub; at least one should
        # appear in the fallback answer.
        assert "<poi id=35708" in out or "<poi id=35695" in out \
            or "<poi id=35702" in out
        # The fallback never invents POIs that were not in history.
        import re
        offered_ids = set(re.findall(r"<poi id=(\d+)", trip_history[0]["content"]))
        answered_ids = set(re.findall(r"<poi id=(\d+)", out))
        assert answered_ids <= offered_ids

    def test_restaurantes_followup_surfaces_restaurant_pois(self, spanish_index, trip_history):
        out = format_history_followup(
            spanish_index, "dame más información sobre los restaurantes",
            trip_history,
        )
        assert out, "expected at least one match for a restaurantes follow-up"
        # There should be at least three Restaurant/BarOrPub picks.
        import re
        assert len(re.findall(r"<poi id=", out)) >= 3

    def test_no_history_returns_empty(self, spanish_index):
        # A conversation with only a system prompt has no shown POIs.
        history = [{"role": "system", "content": "seed"}]
        assert format_history_followup(
            spanish_index, "tapas", history
        ) == ""

    def test_off_topic_followup_returns_empty(self, spanish_index, trip_history):
        # No POI in trip/4444 matches an unrelated 3+ char token like
        # "submarino"; the fallback is empty so the runtime falls back
        # to the localized safe failure.
        assert format_history_followup(
            spanish_index, "submarino", trip_history
        ) == ""


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

    def test_tourist_answer_sanitizer_preserves_known_poi_tag(self, index):
        answer = "POIs include <poi id=36694 type=OilMill>Almazara</poi>."
        sanitized = sanitize_tourist_answer(answer, index)
        assert sanitized.startswith("places include ")
        assert "<poi id=36694 type=OilMill>Almazara</poi>" in sanitized
        assert "<place " not in sanitized

    def test_interactive_output_has_no_redundant_link_footer(self):
        """The <poi>/<trip>/<path> tags are the only link carriers in chat
        output; the interactive print path must not emit a secondary URL
        footer built from extract_poi_tags on the streamed answer.

        We inspect the source of run_interactive so the regression test
        stays deterministic (no LLM call).
        """
        import inspect
        from chat_demo import run_interactive
        src = inspect.getsource(run_interactive)
        # The removed footer built a semicolon-joined `links: ...` line.
        assert " links: " not in src
        # And relied on extract_poi_tags applied to the streamed answer.
        assert 'extract_poi_tags(result["answer"]' not in src

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
