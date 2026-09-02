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
from types import SimpleNamespace

import litellm
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
    GROUNDING_RECOVERY_INSTRUCTION,
    LOOP_REPEAT_CACHE_STUB,
    LOOP_REPEAT_INSTRUCTION,
    LOOP_REPEAT_STUB,
    MAX_ANSWER_TOKENS,
    MAX_TOOL_HISTORY_CHARS,
    MAX_TOOL_RESULT_CHARS,
    backstop_named_poi_ids,
    backstop_named_pois,
    bound_tool_result,
    chant_repeat_prefix,
    compact_tool_history,
    execute_reask_fallback,
    execute_tool,
    final_answer_needs_recovery,
    grounding_failure_message,
    is_brush_off_answer,
    is_designation_question,
    is_physical_route_request,
    is_promise_to_search_later,
    is_pure_reask,
    is_repeat_of_previous_answer,
    is_repeat_tool_call,
    question_names_known_poi,
    reask_fallback_applies,
    requires_current_turn_grounding,
    requires_trip_detail,
    run_agentic_loop,
    tool_call_key,
    tool_result_is_usable,
    validate_tool_call,
)

INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda" / "en.json"
SPANISH_INDEX_FILE = PROJECT_ROOT / "indexes" / "ubeda" / "es.json"


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
    def test_every_message_goes_through_grounding_decision(self):
        # No hardcoded social list to translate: greetings take the
        # recovery instruction's small-talk branch like any other turn.
        assert requires_current_turn_grounding("hola")
        assert requires_current_turn_grounding("Thanks")
        assert requires_current_turn_grounding("hoteles cerca del ayuntamiento")
        assert not requires_current_turn_grounding("")
        assert not requires_current_turn_grounding("   ")

    @pytest.mark.parametrize("question", [
        "dame un plan de cosas que ver en dos días",
        "show me a weekend itinerary",
        "dame los detalles del recorrido",
    ])
    def test_plan_detail_intent(self, question):
        assert requires_trip_detail(question)

    @pytest.mark.parametrize("question", [
        "Is there a planetarium?",
        "planetary museum nearby",
        "What is the plant museum?",
    ])
    def test_plan_prefix_does_not_false_trigger_trip_detail(self, question):
        # "planetarium".startswith("plan") used to force curated-trip mode.
        assert not requires_trip_detail(question)

    def test_failure_message_follows_index_language(self, index):
        spanish = dict(index)
        spanish["meta"] = dict(index["meta"], lang="es")
        assert grounding_failure_message(spanish).startswith("No he podido")

    def test_recovery_instruction_offers_generic_and_specific_paths(self):
        # The model self-classifies the question: generic overview turns
        # may be answered from the preloaded overview/catalogue (never
        # with a failure message), while specific questions must still
        # retrieve.  Guard against a regression to the old unconditional
        # retrieval demand, which made small models fail broad questions
        # like "¿Qué puedo ver?".
        text = GROUNDING_RECOVERY_INSTRUCTION.lower()
        assert "never" in text and "failure message" in text
        assert "find_poi_by_name" in text
        # Small talk is handled by the model, not a translated list.
        assert "small talk" in text
        # A short confirmation answering the assistant's own offer must
        # trigger retrieval of the offered places, not another question.
        assert "confirmation" in text and "not small talk" in text

    def test_named_poi_probe_detects_named_place(self, spanish_index):
        # The language-agnostic backstop: a question explicitly naming a
        # POI from the destination's name_index is detected without any
        # keyword lists.
        name_index = spanish_index.get("name_index") or {}
        multi = next(n for n in name_index if " " in n and len(n) >= 8)
        assert question_names_known_poi(
            f"¿En qué año se construyó {multi}?", spanish_index
        ) == multi

    def test_named_poi_probe_ignores_generic_questions(self, spanish_index):
        assert question_names_known_poi("¿Qué puedo ver?", spanish_index) is None
        assert question_names_known_poi("What is there to do?", spanish_index) is None

    def test_backstop_probes_previous_offer_after_confirmation(
            self, spanish_index):
        # "sí" names nothing itself, but it confirms the assistant's
        # previous offer, which named known places — the backstop must
        # resolve those names so their records are fetched.
        name_index = spanish_index.get("name_index") or {}
        multi = next(n for n in name_index if " " in n and len(n) >= 8)
        offer = f"¿Te gustaría saber más sobre {multi}?"
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "historia"},
            {"role": "assistant", "content": offer},
            {"role": "user", "content": "sí"},
        ]
        assert backstop_named_pois("sí", messages, spanish_index) == [multi]

    def test_backstop_ignores_long_replies_and_non_offers(self, spanish_index):
        name_index = spanish_index.get("name_index") or {}
        multi = next(n for n in name_index if " " in n and len(n) >= 8)
        # Previous assistant turn was not a question → no offer to confirm.
        messages = [
            {"role": "user", "content": "historia"},
            {"role": "assistant", "content": f"Te recomiendo {multi}."},
            {"role": "user", "content": "sí"},
        ]
        assert backstop_named_pois("sí", messages, spanish_index) == []
        # A long visitor message is a new question, not a confirmation —
        # it must stand on its own words.
        messages[-1] = {
            "role": "user",
            "content": "sí pero también quiero ver museos y iglesias",
        }
        assert backstop_named_pois(
            "sí pero también quiero ver museos y iglesias",
            messages, spanish_index,
        ) == []

    def test_failed_get_poi_result_is_not_usable(self):
        assert not tool_result_is_usable(
            "[ERROR] POI 'poi/1' not found. Use find_poi_by_name()."
        )
        assert not tool_result_is_usable(
            "[ERROR] POI 'poi/1' not found.\n\n---\n\n"
            "[ERROR] POI 'poi/19' not found."
        )
        assert tool_result_is_usable(
            "<poi id=5148>Sacra Capilla del Salvador</poi>\n- **Address**: x"
        )
        assert tool_result_is_usable(
            "[ERROR] POI 'poi/1' not found.\n\n---\n\n"
            "Sacra Capilla del Salvador\n- **Address**: x"
        )

    def test_c01_english_chapel_backstop_resolves_salvador(self, index):
        # C01 turn 2: English visitor phrasing of a Spanish catalogue name
        # must still resolve via the focus-phrase + fuzzy path.
        q = (
            "You mentioned the Plaza Vázquez de Molina — what specific "
            "monuments surround it? Tell me about the Sacred Chapel of "
            "El Salvador."
        )
        names = backstop_named_pois(q, [], index)
        ids = backstop_named_poi_ids(q, [], index)
        assert "sacra capilla del salvador" in names
        assert "poi/5148" in ids

    def test_promise_to_search_later_is_brush_off(self):
        decline = (
            "I do not have the full description for the Sacra Capilla del "
            "Salvador at this moment. Would you like me to try searching "
            "for more information on it?"
        )
        assert is_promise_to_search_later(decline)
        assert is_brush_off_answer(decline)
        assert is_pure_reask(decline)
        assert not is_brush_off_answer(
            "The <poi id=5148>Sacra Capilla del Salvador</poi> is a "
            "Renaissance chapel on Plaza Vázquez de Molina."
        )

    def test_hallucinated_poi_ids_trigger_named_backstop(
            self, index, monkeypatch):
        """C01 reproduction: model invents poi/1 + poi/19, then declines.

        Failed get_poi must not mark the turn grounded; the named-POI
        backstop must inject the real chapel record and the next model
        answer is what the visitor sees.
        """
        from types import SimpleNamespace

        q = (
            "You mentioned the Plaza Vázquez de Molina — what specific "
            "monuments surround it? Tell me about the Sacred Chapel of "
            "El Salvador."
        )
        decline = (
            "I do not have the full description for the Sacra Capilla del "
            "Salvador at this moment. Would you like me to try searching "
            "for more information on it?"
        )
        final = (
            "The <poi id=5148 type=PlaceOfWorship>Sacra Capilla del "
            "Salvador</poi> is a 16th-century Renaissance chapel."
        )
        captured: list = []

        def fake(*args, **kwargs):
            captured.append(kwargs)
            n = len(captured)
            if n == 1:
                # Hallucinated ids — the C01 failure mode.
                tcs = [
                    SimpleNamespace(
                        id="c1",
                        function=SimpleNamespace(
                            name="get_poi",
                            arguments='{"poi_id": "poi/1"}',
                        ),
                    ),
                    SimpleNamespace(
                        id="c2",
                        function=SimpleNamespace(
                            name="get_poi",
                            arguments='{"poi_id": "poi/19"}',
                        ),
                    ),
                ]
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content="", tool_calls=tcs))],
                    usage=None,
                )
            if n == 2:
                # Decline after the failed lookups.
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content=decline, tool_calls=None))],
                    usage=None,
                )
            # Answer from the injected backstop records.
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=final, tool_calls=None))],
                usage=None,
            )

        monkeypatch.setattr(litellm, "completion", fake)
        result = run_agentic_loop(
            q, "You are a tourism assistant.", index, "", "fake-model", {},
        )
        assert "5148" in result["answer"] or "Salvador" in result["answer"]
        assert result["answer"] == final or "Salvador" in result["answer"]
        assert any(
            c.get("automatic") and c.get("tool") == "get_poi"
            for c in result["tool_calls"]
        )
        # The hallucinated calls ran but must not leave grounded stuck on
        # errors alone — automatic backstop proves recovery fired.
        auto_ids = [
            (c.get("args") or {}).get("poi_id", "")
            for c in result["tool_calls"]
            if c.get("automatic") and c.get("tool") == "get_poi"
        ]
        assert any("5148" in (pid or "") for pid in auto_ids)

    def test_place_alias_and_bold_dangling_tags_repaired(self, index):
        # Small models sometimes write <place id=…> or leave the tag
        # dangling after a bold label; both must become canonical poi tags.
        poi = next(iter((index.get("pois") or {}).values()))
        bare = str(poi["poi_id"]).split("/", 1)[-1]
        out = sanitize_tourist_answer(
            f"**{poi['name']}** <place id={bare} type=Museum>:", index,
        )
        assert "<place" not in out and "**" not in out
        assert f"<poi id={bare}" in out and poi["name"] in out
        out = sanitize_tourist_answer(
            f"<place id={bare}>Text</place>", index,
        )
        assert out == f"<poi id={bare} type={poi['display_type']}>Text</poi>" \
            or (f"<poi id={bare}" in out and "Text</poi>" in out
                and "place" not in out)

    @pytest.mark.parametrize("text", [
        "¿Qué te gustaría saber ahora?",
        "¿Te interesa la historia o la gastronomía?",
        "What would you like to know?",
    ])
    def test_pure_reask_detected(self, text):
        assert is_pure_reask(text)

    @pytest.mark.parametrize("text", [
        "",   # empty
        "¡Hola! ¿En qué puedo ayudarte?",          # greeting warmth
        "¡Dime qué necesitas!",                     # exclamation only
        "El castillo data del siglo XIII. <poi id=1>X</poi>",  # tagged
        "x" * 301 + "?",                            # long, has content
    ])
    def test_pure_reask_rejects_real_answers(self, text):
        assert not is_pure_reask(text)

    def test_reask_fallback_applies_requires_prior_turn(self):
        # The gate now covers only the repeat-answer branch (a repeat can
        # only exist with a prior turn); bare re-asks are intercepted on
        # any turn.  The function itself still reports whether an earlier
        # visitor turn exists.
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "¿Qué deseas?"},
        ]
        assert not reask_fallback_applies(messages, "Hola")
        messages += [{"role": "user", "content": "sí"}]
        assert reask_fallback_applies(messages, "sí")

    def test_reask_fallback_retrieves_content(self, spanish_index):
        from index_tools import format_sections_overview
        sections_text = format_sections_overview(spanish_index)
        # A content-free confirmation falls back to indispensable highlights.
        result, _, tool, args = execute_reask_fallback(
            "sí", spanish_index, sections_text, {},
        )
        assert tool == "filter_pois"
        assert args["interest_level"] == 1
        assert "<poi" in result
        # A topical question gets an evidence search first.
        result, _, tool, _ = execute_reask_fallback(
            "historia", spanish_index, sections_text, {},
        )
        assert tool in {"search_pois", "filter_pois"}
        assert result
        # A query with no evidence at all still lands on highlights.
        result, _, tool, _ = execute_reask_fallback(
            "zzzzzz", spanish_index, sections_text, {},
        )
        assert tool == "filter_pois"
        assert "<poi" in result

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
        # Live catalogue no longer includes "RUTAS POR ÚBEDA"; any unique
        # multi-word trip title must still beat bare route-intent matching.
        trips = [
            item for item in (spanish_index.get("trips") or [])
            if len((item.get("name") or "").split()) >= 2
        ]
        assert trips
        target = next(
            (item for item in trips
             if "perderte" in (item.get("name") or "").lower()
             or "ruta" in (item.get("name") or "").lower()),
            trips[0],
        )
        selection = resolve_trip_query((target.get("name") or "").lower(),
                                       spanish_index)
        assert selection == {
            "kind": "trip",
            "id": target["itinerary_id"],
            "label": target.get("name") or "",
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
        # museums-and-culture stays flat (≤30 POIs); default limit 50 shows all.
        sec = next(
            s for s in index["sections"]
            if s["section_id"] == "museums-and-culture"
        )
        assert not sec.get("groups")
        out = format_section(index, "museums-and-culture")
        preview_lines = [l for l in out.splitlines()
                         if l.startswith("  <poi ")]
        assert len(preview_lines) == len(sec["poi_ids"])
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
        # Live tourist-destinations trip list for Úbeda (13 ids as of 2026-08-31).
        assert len(index["trips"]) >= 10
        # Paths come from destination route ids; empty or non-empty both valid.
        assert isinstance(index.get("paths"), list)

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
        # Live trips currently ship flat steps; keep nested rendering covered
        # with a synthetic schema-v6 tree using a known ES POI.
        synthetic = dict(spanish_index)
        synthetic["trips"] = list(synthetic.get("trips") or []) + [{
            "itinerary_id": "trip/88044",
            "trip_id": "trip/88044",
            "kind": "trip",
            "source_type": "TouristTrip",
            "name": "Nested Fixture Trip",
            "description": "",
            "url": "",
            "is_route": False,
            "steps": [{
                "position": 1,
                "title": "1. Centro",
                "items": [{
                    "kind": "folder",
                    "name": "1.1 Plaza Vázquez de Molina",
                    "items": [{
                        "kind": "poi",
                        "poi_id": "poi/30536",
                        "source_name": "Plaza Vázquez de Molina",
                        "resolution": "source_id",
                    }],
                }],
                "poi_ids": ["poi/30536"],
                "poi_resolutions": [{
                    "poi_id": "poi/30536",
                    "source_name": "Plaza Vázquez de Molina",
                    "resolution": "source_id",
                }],
                "subfolders": ["1.1 Plaza Vázquez de Molina"],
                "unresolved_poi_names": [],
            }],
        }]
        rendered = format_trip(synthetic, "trip/88044")
        assert "   - 1.1 Plaza Vázquez de Molina" in rendered
        assert "      - <poi id=30536" in rendered

    def test_spanish_trip_source_ids_link_but_absent_stop_stays_hidden(self, spanish_index):
        synthetic = dict(spanish_index)
        synthetic["trips"] = list(synthetic.get("trips") or []) + [{
            "itinerary_id": "trip/88045",
            "trip_id": "trip/88045",
            "kind": "trip",
            "source_type": "TouristTrip",
            "name": "Lodging Fixture Trip",
            "description": "",
            "url": "",
            "is_route": False,
            "steps": [
                {
                    "position": 1,
                    "title": "1. Plaza",
                    "items": [{
                        "kind": "folder",
                        "name": "1.1 Plaza Vázquez de Molina",
                        "items": [{
                            "kind": "poi",
                            "poi_id": "poi/30536",
                            "source_name": "Plaza Vázquez de Molina",
                            "resolution": "source_id",
                        }],
                    }],
                    "poi_ids": ["poi/30536"],
                    "poi_resolutions": [{
                        "poi_id": "poi/30536",
                        "source_name": "Plaza Vázquez de Molina",
                        "resolution": "source_id",
                    }],
                    "subfolders": ["1.1 Plaza Vázquez de Molina"],
                    "unresolved_poi_names": [],
                },
                {
                    "position": 4,
                    "title": "4. Alojamiento",
                    "items": [
                        {
                            "kind": "poi",
                            "poi_id": "poi/30459",
                            "source_name": "Hotel Yit El Postigo",
                            "resolution": "source_id",
                        },
                        {"kind": "unresolved", "name": "CR La Casería de Tito"},
                    ],
                    "poi_ids": ["poi/30459"],
                    "poi_resolutions": [{
                        "poi_id": "poi/30459",
                        "source_name": "Hotel Yit El Postigo",
                        "resolution": "source_id",
                    }],
                    "subfolders": [],
                    "unresolved_poi_names": ["CR La Casería de Tito"],
                },
            ],
        }]
        trip = get_trip(synthetic, "trip/88045")
        first_step = trip["steps"][0]
        assert "1.1 Plaza Vázquez de Molina" in first_step["subfolders"]
        lodging = next(step for step in trip["steps"] if step["title"].startswith("4."))
        resolved = {item["source_name"]: item for item in lodging["poi_resolutions"]}
        assert resolved["Hotel Yit El Postigo"]["poi_id"] == "poi/30459"
        assert resolved["Hotel Yit El Postigo"]["resolution"] == "source_id"
        assert "CR La Casería de Tito" in lodging["unresolved_poi_names"]

        rendered = format_trip(synthetic, "trip/88045")
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
        # Clone the index and pretend the runtime is set to a language
        # outside the supported set.
        idx = dict(spanish_index)
        idx["meta"] = dict(spanish_index["meta"], lang="xx")
        matches = search_trips(spanish_index, "plan dos días", limit=2)
        offer = format_trip_choice_offer(idx, matches[:2])
        assert offer.startswith("Here are a few curated trips")

    def test_offer_is_localized_for_supported_languages(self, spanish_index):
        # Every supported language has its own strings (tests/test_i18n.py
        # guards completeness); spot-check Japanese here.
        idx = dict(spanish_index)
        idx["meta"] = dict(spanish_index["meta"], lang="ja")
        matches = search_trips(spanish_index, "plan dos días", limit=2)
        offer = format_trip_choice_offer(idx, matches[:2])
        assert offer.startswith("ご希望に合いそうなおすすめプラン")
        assert "見どころ:" in offer

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
                '  - <trip id=4407>Savor Úbeda</trip>\n'
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
        # trip/4444 left the live catalogue; Savor Úbeda still has food stops.
        return [{"role": "assistant",
                  "content": format_trip(spanish_index, "trip/4407")}]

    def test_tapas_followup_surfaces_tapas_zone(self, spanish_index, trip_history):
        out = format_history_followup(
            spanish_index, "entonces cuál tiene tapas", trip_history
        )
        assert out, "expected at least one match for a 'tapas' follow-up"
        assert "<poi id=" in out
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


# ── Tool-call loop detection ──────────────────────────────────────────────

class TestToolCallLoopBreak:
    """The repeat detector blocks the 3rd identical (tool, args) call in a
    turn, corrects the model once, and aborts on any further repeat so the
    tail recovery forces a final answer instead of burning all rounds."""

    def test_key_ignores_argument_order(self):
        k1 = tool_call_key("filter_pois", {"limit": 5, "type": "Museum"})
        k2 = tool_call_key("filter_pois", {"type": "Museum", "limit": 5})
        assert k1 == k2

    def test_third_identical_call_is_a_repeat(self):
        keys = []
        key = tool_call_key("get_poi", {"poi_id": "poi/5155"})
        for _ in range(2):
            keys.append(key)
            assert not is_repeat_tool_call(keys)
        keys.append(key)
        assert is_repeat_tool_call(keys)

    def test_alternating_repeat_detected(self):
        a = tool_call_key("find_poi_by_name", {"query": "castillo"})
        b = tool_call_key("get_section", {"section_id": "museums"})
        keys = [a, b, a, b]
        assert not is_repeat_tool_call(keys)
        keys.append(a)   # A B A B A — the third A trips the counter
        assert is_repeat_tool_call(keys)

    def test_different_arguments_are_not_a_repeat(self):
        keys = [
            tool_call_key("get_poi", {"poi_id": "poi/1"}),
            tool_call_key("get_poi", {"poi_id": "poi/2"}),
            tool_call_key("get_poi", {"poi_id": "poi/1"}),
        ]
        assert not is_repeat_tool_call(keys)

    def test_correction_instruction_names_the_tool(self):
        text = LOOP_REPEAT_INSTRUCTION.format(tool="get_poi")
        assert "get_poi" in text
        assert "Do not repeat" in text

    @staticmethod
    def _looping_completion(captured: list, answer_text: str):
        """Fake litellm.completion: re-issues the same find_poi_by_name
        call on every loop round, then answers on the tail recovery call
        (5th call overall: 4 loop rounds + recovery)."""
        def fake(*args, **kwargs):
            captured.append(list(kwargs.get("messages") or []))
            if len(captured) <= 4:
                tc = SimpleNamespace(
                    id=f"call_{len(captured)}",
                    function=SimpleNamespace(
                        name="find_poi_by_name",
                        arguments=json.dumps({"query": "castillo"}),
                    ),
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content=None, tool_calls=[tc]))],
                    usage=None,
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=answer_text, tool_calls=None))],
                usage=None,
            )
        return fake

    def test_two_strike_break_forces_answer(self, index, monkeypatch):
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            self._looping_completion(captured, "Recovered answer text."),
        )
        result = run_agentic_loop(
            "Tell me about the castle", "You are a tourism assistant.",
            index, "", "fake-model", {},
        )
        assert result["answer"] == "Recovered answer text."
        assert result["rounds"] == 4
        blocked = [c for c in result["tool_calls"] if c.get("loop_blocked")]
        real    = [c for c in result["tool_calls"] if not c.get("loop_blocked")]
        assert len(blocked) == 2   # strike 1 + strike 2
        assert len(real) == 2      # call 1 ran; call 2 got the cache stub
        # Only the first occurrence actually executed: the second was
        # served the repeat cache stub without re-running the lookup.
        executed = [c for c in result["tool_calls"]
                    if not c.get("loop_blocked") and not c.get("repeat_cached")]
        assert len(executed) == 1
        assert real[1]["repeat_cached"]
        assert real[1]["result_preview"] == LOOP_REPEAT_CACHE_STUB
        assert all(c["result_preview"] == LOOP_REPEAT_STUB for c in blocked)
        # The one-shot correction was injected after the first blocked call.
        assert any(
            "Do not repeat it" in (m.get("content") or "")
            for messages in captured for m in messages
            if m.get("role") == "user"
        )

    def test_run_turn_blocks_and_recovers(self, index, monkeypatch):
        from chat_demo import run_turn
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            self._looping_completion(captured, "Recovered chat answer."),
        )
        messages = [{"role": "system", "content": "You are a tourism assistant."}]
        result = run_turn("Tell me about the castle", messages, index, "",
                          "fake-model", {})
        assert result["answer"] == "Recovered chat answer."
        blocked = [c for c in result["tool_calls"] if c.get("loop_blocked")]
        assert len(blocked) == 2


# ── Tool-call validation feedback ─────────────────────────────────────────

class TestValidateToolCall:
    """Invalid calls are rejected with an explanatory [ERROR] tool result
    instead of executing on silently defaulted arguments."""

    def test_valid_call_passes(self):
        assert validate_tool_call("get_poi", {"poi_id": "poi/5155"}) is None

    def test_unknown_tool_lists_valid_names(self):
        err = validate_tool_call("get_castle", {})
        assert "Unknown tool" in err
        assert "get_poi" in err and "find_poi_by_name" in err

    def test_malformed_json_reported(self):
        err = validate_tool_call("find_poi_by_name", None, '{"query": ')
        assert "invalid JSON" in err
        assert "find_poi_by_name" in err

    def test_missing_required_argument(self):
        err = validate_tool_call("get_poi", {})
        assert "missing required argument 'poi_id'" in err

    def test_empty_required_argument(self):
        err = validate_tool_call("find_poi_by_name", {"query": ""})
        assert "missing required argument 'query'" in err

    def test_enum_violation(self):
        err = validate_tool_call(
            "get_section", {"section_id": "gastronomy", "sort": "random"})
        assert "'sort'" in err and "interest" in err

    def test_enum_case_variant_accepted(self):
        # The tools normalise case themselves; rejecting 'Full' would
        # churn a round for no benefit.
        assert validate_tool_call(
            "find_poi_by_name", {"query": "x", "detail": "Full"}) is None

    def test_integer_type_mismatch(self):
        err = validate_tool_call("filter_pois", {"interest_level": "high"})
        assert "'interest_level' must be an integer" in err

    def test_boolean_type_mismatch(self):
        err = validate_tool_call("filter_pois", {"indispensable": "yes"})
        assert "'indispensable' must be a boolean" in err

    def test_extra_arguments_ignored(self):
        assert validate_tool_call(
            "get_poi", {"poi_id": "poi/5155", "foo": 1}) is None

    def test_invalid_call_gets_error_then_recovers(self, index, monkeypatch):
        """A malformed first call produces an error tool result (never
        executes); the model's corrected retry executes normally."""
        calls: list = []

        def fake(*args, **kwargs):
            calls.append(list(kwargs.get("messages") or []))
            if len(calls) == 1:
                tc = SimpleNamespace(id="c1", function=SimpleNamespace(
                    name="get_poi", arguments='{"poi_id": '))  # malformed
            elif len(calls) == 2:
                tc = SimpleNamespace(id="c2", function=SimpleNamespace(
                    name="get_poi", arguments='{"poi_id": "poi/5155"}'))
            else:
                tc = None
            if tc is None:
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="Castle answer.",
                                            tool_calls=None))], usage=None)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tc]))],
                usage=None)

        monkeypatch.setattr(litellm, "completion", fake)
        result = run_agentic_loop(
            "Tell me about the castle", "You are a tourism assistant.",
            index, "", "fake-model", {},
        )
        assert result["answer"] == "Castle answer."
        entries = result["tool_calls"]
        assert entries[0].get("invalid_args")
        assert "invalid JSON" in entries[0]["result_preview"]
        assert entries[1]["args"] == {"poi_id": "poi/5155"}
        assert not entries[1].get("invalid_args")
        # The malformed call never executed: exactly one real lookup ran.
        assert sum(1 for c in entries if not c.get("invalid_args")) == 1

    def test_second_identical_call_served_from_cache_stub(
            self, index, monkeypatch):
        """The 2nd occurrence of an identical (tool, args) call does not
        re-execute: while the original result is still in context it gets
        a short cache stub; only the 3rd is blocked outright."""
        calls: list = []

        def fake(*args, **kwargs):
            calls.append(list(kwargs.get("messages") or []))
            if len(calls) <= 2:
                tc = SimpleNamespace(
                    id=f"call_{len(calls)}",
                    function=SimpleNamespace(
                        name="find_poi_by_name",
                        arguments=json.dumps({"query": "castillo"}),
                    ),
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content=None, tool_calls=[tc]))],
                    usage=None,
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="Done.", tool_calls=None))],
                usage=None,
            )

        monkeypatch.setattr(litellm, "completion", fake)
        result = run_agentic_loop(
            "Tell me about the castle", "You are a tourism assistant.",
            index, "", "fake-model", {},
        )
        assert result["answer"] == "Done."
        entries = result["tool_calls"]
        assert len(entries) == 2
        assert not entries[0].get("repeat_cached")
        assert entries[1].get("repeat_cached")
        assert entries[1]["result_preview"] == LOOP_REPEAT_CACHE_STUB


# ── Streaming content-chant guard ─────────────────────────────────────────

# 50 chars exactly: with the chant unit aligned to CHANT_CHUNK_SIZE, the
# first exact occurrence of the tail chunk sits at the run start, so the
# trim offset is exact in these tests.
CHANT_UNIT = "the old walls the old walls the old walls the old "
assert len(CHANT_UNIT) == 50


class TestChantGuard:
    """chant_repeat_prefix detects a degenerating stream (the tail chunk
    repeated >= CHANT_MAX_REPEATS times in the recent window) and returns
    the offset where the chant run began; ordinary text is untouched."""

    def test_short_text_untouched(self):
        text = "Short answer."
        assert chant_repeat_prefix(text) == len(text)

    def test_normal_long_answer_untouched(self):
        text = " ".join(f"word{i}" for i in range(600))
        assert chant_repeat_prefix(text) == len(text)

    def test_below_threshold_repetition_untouched(self):
        text = "Intro. " + CHANT_UNIT * 4   # 4 < CHANT_MAX_REPEATS
        assert chant_repeat_prefix(text) == len(text)

    def test_chant_after_intro_trims_to_intro(self):
        intro = "Here is a real answer about the castle. "
        text = intro + CHANT_UNIT * 12
        assert chant_repeat_prefix(text) == len(intro)

    def test_chant_from_first_token_trims_to_zero(self):
        text = CHANT_UNIT * 12
        assert chant_repeat_prefix(text) == 0

    def test_chant_at_exact_threshold_detected(self):
        text = CHANT_UNIT * 6   # 300 chars, exactly chunk_size * repeats
        assert chant_repeat_prefix(text) == 0

    @staticmethod
    def _chanting_stream(captured: list, pieces: list[str]):
        """Fake litellm.completion: stream= True yields the content pieces
        as chunks; a non-streaming call answers plainly (unused here)."""
        def fake(*args, **kwargs):
            captured.append(kwargs)
            if kwargs.get("stream"):
                def gen():
                    for piece in pieces:
                        yield SimpleNamespace(choices=[SimpleNamespace(
                            delta=SimpleNamespace(content=piece,
                                                  tool_calls=None))])
                return gen()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="Recovered.", tool_calls=None))],
                usage=None,
            )
        return fake

    def test_stream_chant_serves_trimmed_prefix(self, index, monkeypatch):
        from chat_demo import run_turn
        intro = "Here is a real answer about the castle. "
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            self._chanting_stream(captured, [intro] + [CHANT_UNIT] * 12),
        )
        messages = [{"role": "system", "content": "You are a tourism assistant."}]
        result = run_turn("What can I see?", messages, index, "",
                          "fake-model", {}, stream=True)
        assert result["answer"] == intro.strip()
        assert result["rounds"] == 1
        assert result["chant_truncated"] is True
        assert len(captured) == 1   # no re-generation after the chant

    def test_stream_chant_without_prefix_fails_safe(self, index, monkeypatch):
        from chat_demo import run_turn
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            self._chanting_stream(captured, [CHANT_UNIT] * 12),
        )
        messages = [{"role": "system", "content": "You are a tourism assistant."}]
        result = run_turn("Tell me about the castle", messages, index, "",
                          "fake-model", {}, stream=True)
        assert result["answer"] == grounding_failure_message(index)
        assert result["chant_truncated"] is True
        assert len(captured) == 1


# ── Repeat-answer guard ───────────────────────────────────────────────────

class TestRepeatAnswerGuard:
    """A small model at temperature=0 can re-emit its previous reply
    instead of answering the new visitor message.  The duplicate is a
    brush-off: intercept it with the same deterministic retrieval as a
    repeated clarifying question instead of serving it."""

    PREV = "Hello! How can I help you today?"

    def _history(self) -> list[dict]:
        return [
            {"role": "system", "content": "You are a tourism assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": self.PREV},
        ]

    def test_exact_repeat_detected(self):
        assert is_repeat_of_previous_answer(self.PREV, self._history())

    def test_repeat_detected_despite_case_and_punctuation(self):
        assert is_repeat_of_previous_answer(
            "hello, how can I help you today", self._history())

    def test_different_text_is_not_a_repeat(self):
        assert not is_repeat_of_previous_answer(
            "The castle dates from the 13th century.", self._history())

    def test_empty_answer_is_not_a_repeat(self):
        assert not is_repeat_of_previous_answer("", self._history())

    def test_first_turn_is_never_a_repeat(self):
        messages = [{"role": "system", "content": "sys"}]
        assert not is_repeat_of_previous_answer(self.PREV, messages)

    def test_referent_is_the_previous_turn_not_the_current_draft(self):
        # The current turn's own earlier-round draft sits between the
        # question and the new answer; the repeat referent must remain
        # the answer to the PREVIOUS visitor message.
        messages = self._history() + [
            {"role": "user", "content": "What can I see?"},
            {"role": "assistant", "content": "A genuine first attempt."},
            {"role": "user", "content": "(recovery instruction)"},
        ]
        assert is_repeat_of_previous_answer(
            self.PREV, messages, "What can I see?")
        assert not is_repeat_of_previous_answer(
            "A genuine first attempt.", messages, "What can I see?")

    def test_repeated_answer_triggers_deterministic_retrieval(
            self, index, monkeypatch):
        """Turn 2 parrots the greeting verbatim after a genuine first
        attempt: the runtime must not serve the duplicate; it retrieves
        content (reask fallback) and the model's next answer is served
        instead.  The middle reply being different from the parroted
        greeting reproduces the observed E2B dynamics exactly."""
        from chat_demo import run_turn
        captured: list = []
        replies = ["A genuine first attempt.", self.PREV,
                   "Real overview answer."]

        def fake(*args, **kwargs):
            captured.append(kwargs)
            content = replies[min(len(captured), len(replies)) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=content, tool_calls=None))],
                usage=None,
            )

        monkeypatch.setattr(litellm, "completion", fake)
        result = run_turn("What can I see?", self._history(), index, "",
                          "fake-model", {})
        assert result["answer"] == "Real overview answer."
        assert result["chant_truncated"] is False
        # Two brush-off replies, then the answer composed from the
        # deterministically retrieved records.
        assert len(captured) == 3
        assert any(c.get("automatic") and c["tool"] in (
            "search_pois", "filter_pois") for c in result["tool_calls"])
        # Every generation call carries the answer-length cap.
        assert all(c.get("max_tokens") == MAX_ANSWER_TOKENS
                   for c in captured)

    def test_parrot_after_fallback_serves_deterministic_content(
            self, index, monkeypatch):
        """Residual: model parrots the greeting AGAIN after reask fallback
        injects records.  Must not serve the greeting — present the
        deterministic lookup result with no further LLM round.
        """
        from chat_demo import run_turn
        captured: list = []
        # 1) ungrounded first attempt → recovery instruction
        # 2) parrot previous greeting → reask fallback inject
        # 3) parrot AGAIN → deterministic present (no 4th LLM call)
        replies = ["A genuine first attempt.", self.PREV, self.PREV]

        def fake(*args, **kwargs):
            captured.append(kwargs)
            content = replies[min(len(captured), len(replies)) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=content, tool_calls=None))],
                usage=None,
            )

        monkeypatch.setattr(litellm, "completion", fake)
        result = run_turn("What can I see?", self._history(), index, "",
                          "fake-model", {})
        assert result["answer"] != self.PREV
        assert self.PREV not in (result["answer"] or "")
        assert len(captured) == 3   # no 4th generation after 2nd parrot
        assert any(
            c.get("automatic") and c.get("deterministic_present")
            for c in result["tool_calls"]
        )
        assert final_answer_needs_recovery(
            self.PREV, self._history() + [
                {"role": "user", "content": "What can I see?"},
            ], "What can I see?")


# ── Designation-question guard ────────────────────────────────────────────

def _text_only_completion(captured: list, replies: list[str]):
    """Fake litellm.completion answering each call with the next text
    reply (never a tool call); the last reply repeats if exhausted."""
    def fake(*args, **kwargs):
        captured.append(kwargs)
        content = replies[min(len(captured), len(replies)) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=content, tool_calls=None))],
            usage=None,
        )
    return fake


class TestDesignationGuard:
    """Designation/status questions (UNESCO, World Heritage…) name no POI,
    so small models misclassify them as generic overview questions and
    answer from the catalogue without the factual anchors.  One forced
    search_pois evidence lookup fires when the model answers without
    retrieving — same shape as the physical-route guard."""

    @pytest.mark.parametrize("question", [
        "Why is Úbeda a UNESCO World Heritage City?",
        "¿Por qué es Úbeda Patrimonio de la Humanidad?",
        "Perché Úbeda è patrimonio dell'umanità?",
        "Pourquoi Úbeda est-elle au patrimoine mondial?",
        "Warum ist Úbeda Weltkulturerbe?",
    ])
    def test_designation_question_detected(self, question):
        assert is_designation_question(question)

    @pytest.mark.parametrize("question", [
        "What can I see?",
        "¿Qué puedo ver?",
        "Tell me about the castle",
        "Where can I eat traditional food?",
    ])
    def test_generic_questions_not_designation(self, question):
        assert not is_designation_question(question)

    def test_forced_lookup_supplies_the_facts(self, index, monkeypatch):
        """Q01 reproduction: the model's first answer is a generic
        catalogue overview (no tools); the runtime forces one evidence
        search and the model's next answer carries the facts."""
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            _text_only_completion(captured, [
                "Úbeda is an artistic and monumental city with many "
                "churches and palaces.",
                "Úbeda was declared a World Heritage City in 2003 for "
                "its Renaissance architecture.",
            ]),
        )
        result = run_agentic_loop(
            "Why is Úbeda a UNESCO World Heritage City?",
            "You are a tourism assistant.", index, "", "fake-model", {},
        )
        assert result["answer"] == (
            "Úbeda was declared a World Heritage City in 2003 for "
            "its Renaissance architecture.")
        assert result["grounded"]
        auto = [c for c in result["tool_calls"] if c.get("automatic")]
        assert len(auto) == 1 and auto[0]["tool"] == "search_pois"
        # The forced lookup queries the designation proper noun, never
        # the full visitor question (all-tokens evidence search would
        # miss every record).
        assert auto[0]["args"]["query"] == "unesco"
        assert result["rounds"] == 2

    def test_first_turn_bare_reask_triggers_fallback(self, index, monkeypatch):
        """Q20 reproduction: a bare clarifying question is never an
        acceptable final answer — even on the very first turn, the
        runtime retrieves content deterministically and the model's
        next answer is served instead."""
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            _text_only_completion(captured, [
                "What would you like to know?",
                "What would you like to know?",
                "Úbeda pairs a UNESCO-listed Renaissance core with a "
                "living pottery tradition.",
            ]),
        )
        result = run_agentic_loop(
            "What makes Úbeda different from other Spanish cities as a "
            "tourist destination?",
            "You are a tourism assistant.", index, "", "fake-model", {},
        )
        assert result["answer"] == (
            "Úbeda pairs a UNESCO-listed Renaissance core with a "
            "living pottery tradition.")
        auto = [c for c in result["tool_calls"] if c.get("automatic")]
        assert len(auto) == 1
        assert auto[0]["tool"] in {"search_pois", "filter_pois"}
        assert len(captured) == 3

    def test_warm_greeting_stays_on_the_trust_path(self, index, monkeypatch):
        """Greetings must NOT trigger retrieval: a warm greeting reply
        carries \"!\" markers, so is_pure_reask rejects it and the
        answer is served as before."""
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            _text_only_completion(captured, [
                "¡Hola! ¿En qué puedo ayudarte hoy?",
                "¡Hola! ¿En qué puedo ayudarte hoy?",
            ]),
        )
        result = run_agentic_loop(
            "Hola", "You are a tourism assistant.", index, "",
            "fake-model", {},
        )
        assert result["answer"] == "¡Hola! ¿En qué puedo ayudarte hoy?"
        assert result["tool_calls"] == []   # no retrieval for a greeting


# ── Decode-time tool forcing (tool_choice) ──────────────────────────────────

class TestForcedToolChoice:
    """The two instruction turns that DEMAND a tool call carry tool_choice
    for runtimes that honor it (LiteRT ToolChoice; oMLX ignores it).
    Every other turn stays 'auto' — most instructions have legitimate
    no-tool outcomes (small talk, overview answers, \"answer from these
    results\" follow-ups)."""

    def test_complementary_instruction_forces_required(self, index,
                                                       monkeypatch):
        captured: list = []

        def fake(*args, **kwargs):
            captured.append(kwargs)
            n = len(captured)
            tc = None
            if n == 1:
                tc = SimpleNamespace(id="c1", function=SimpleNamespace(
                    name="search_pois",      # misses → direct_evidence_missing
                    arguments=json.dumps({"query": "zzz qqq"})))
            elif n == 3:
                tc = SimpleNamespace(id="c3", function=SimpleNamespace(
                    name="filter_pois",
                    arguments=json.dumps({"interest_level": 1})))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=None if tc else "Answer text.",
                    tool_calls=[tc] if tc else None))],
                usage=None,
            )

        monkeypatch.setattr(litellm, "completion", fake)
        # Compound question with no trip-plan intent ("planetarium" would
        # false-positive: it prefix-matches the plan term "plan").
        run_agentic_loop("olive oil restaurant zzzqqq",
                         "You are a tourism assistant.", index, "",
                         "fake-model", {})
        assert len(captured) == 4
        assert captured[0]["tool_choice"] == "auto"
        assert captured[1]["tool_choice"] == "auto"
        # The call right after COMPLEMENTARY_SEARCH_INSTRUCTION is forced.
        assert captured[2]["tool_choice"] == "required"
        assert captured[3]["tool_choice"] == "auto"   # cleared after use

    def test_trip_detail_instruction_forces_get_trip(self, index,
                                                     monkeypatch):
        captured: list = []
        trip_id = (index.get("trips") or [{}])[0].get("itinerary_id", "")

        def fake(*args, **kwargs):
            captured.append(kwargs)
            n = len(captured)
            tc = None
            if n == 1:
                tc = SimpleNamespace(id="c1", function=SimpleNamespace(
                    name="search_trips",       # real matches → has_results
                    arguments=json.dumps({"query": "Úbeda"})))
            elif n == 3:
                tc = SimpleNamespace(id="c3", function=SimpleNamespace(
                    name="get_trip",
                    arguments=json.dumps({"trip_id": trip_id})))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=None if tc else "Hold on a moment.",
                    tool_calls=[tc] if tc else None))],
                usage=None,
            )

        monkeypatch.setattr(litellm, "completion", fake)
        # Plan intent via "detalles" with zero trip-text token matches,
        # so the deterministic trip-offer shortcut stays out of the way.
        run_agentic_loop("detalles zzzqqq",
                         "You are a tourism assistant.", index, "",
                         "fake-model", {})
        assert len(captured) == 3
        assert captured[0]["tool_choice"] == "auto"
        assert captured[1]["tool_choice"] == "auto"
        # The call right after TRIP_DETAIL_REQUIRED_INSTRUCTION names the tool.
        assert captured[2]["tool_choice"] == {
            "type": "function", "function": {"name": "get_trip"}}

    def test_grounding_recovery_stays_auto(self, index, monkeypatch):
        """The recovery instruction has legitimate no-tool outcomes
        (small talk, generic overview), so it must never force a call."""
        captured: list = []
        monkeypatch.setattr(
            litellm, "completion",
            _text_only_completion(captured, ["Some catalogue answer."]),
        )
        result = run_agentic_loop("What is there to do?",
                                  "You are a tourism assistant.", index, "",
                                  "fake-model", {})
        assert result["answer"] == "Some catalogue answer."
        assert len(captured) == 2   # initial answer + one grounding retry
        assert all(c["tool_choice"] == "auto" for c in captured)
