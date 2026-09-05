"""Golden and paper-rule tests for the query-hierarchy constructor (paper Sec. 3.1).

The golden file ``data/query_construction_expected.json`` is self-contained: it
holds 40 captions with the query list, the collated tensors and the splitter
intermediates each of them must produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyper3_clip.data.query_hierarchy import (
    QUERY_WEIGHTS,
    build_query_hierarchy,
    extract_lightweight_phrases,
    split_sentences,
)

FIXTURES_PATH = Path(__file__).resolve().parent / "data" / "query_construction_expected.json"

#: The fixture dump names the root type ``caption_root``; the public API calls
#: that kind ``caption``.  Every other name is shared.
_FIXTURE_KIND_TO_PUBLIC = {
    "caption_root": "caption",
    "sentence": "sentence",
    "text_box": "text_box",
    "phrase": "phrase",
}


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _kind_map(payload: dict) -> dict[int, str]:
    return {int(code): _FIXTURE_KIND_TO_PUBLIC[name] for code, name in payload["type_codes"].items()}


def _tuples(queries) -> list[tuple]:
    return [(q.text, q.kind, q.parent, q.weight, q.source_part) for q in queries]


def _expected(entries, kinds) -> list[tuple]:
    return [(e["text"], kinds[e["type"]], e["parent"], e["weight"], e["source_part"]) for e in entries]


# --- golden fixtures: exact equality against the recorded construction ------


def test_fixture_file_shape(fixtures: dict) -> None:
    assert len(fixtures["fixtures"]) == 40
    assert fixtures["settings"]["max_sentences"] == 5
    assert fixtures["settings"]["max_phrases"] == 30
    assert fixtures["settings"]["max_queries_per_image"] == 6


def test_golden_full_matches_reference(fixtures: dict) -> None:
    """Pre-truncation construction matches the recorded ``full`` entries exactly."""
    kinds = _kind_map(fixtures)
    for fixture in fixtures["fixtures"]:
        got = build_query_hierarchy(fixture["caption"], list(fixture["part_texts"]), max_queries=None)
        assert _tuples(got) == _expected(fixture["full"], kinds), fixture["name"]


def test_golden_paper_matches_reference(fixtures: dict) -> None:
    """Post-truncation (``max_queries=6``, the paper setting) matches ``paper``."""
    kinds = _kind_map(fixtures)
    for fixture in fixtures["fixtures"]:
        got = build_query_hierarchy(fixture["caption"], list(fixture["part_texts"]), max_queries=6)
        assert _tuples(got) == _expected(fixture["paper"], kinds), fixture["name"]


def test_golden_collated_metadata(fixtures: dict) -> None:
    """Texts, parents, weights and source parts match the collated batch tensors.

    Each fixture is dumped as a single-image batch, so the collator's global
    offsets are all zero and the local indices are the global ones.
    """
    for fixture in fixtures["fixtures"]:
        got = build_query_hierarchy(fixture["caption"], list(fixture["part_texts"]), max_queries=6)
        collated = fixture["collated"]
        assert [q.text for q in got] == collated["query_texts"], fixture["name"]
        assert [q.parent for q in got] == collated["query_parent"], fixture["name"]
        assert [q.weight for q in got] == collated["query_weight"], fixture["name"]
        assert [q.source_part for q in got] == collated["query_source_part"], fixture["name"]


def test_golden_helpers_match_reference(fixtures: dict) -> None:
    """The sentence splitter and phrase extractor match the recorded intermediates."""
    for fixture in fixtures["fixtures"]:
        assert split_sentences(fixture["caption"]) == fixture["split_sentences"], fixture["name"]
        assert extract_lightweight_phrases(fixture["caption"]) == fixture["extracted_phrases"], fixture["name"]


# --- readable unit tests stating the paper rules explicitly ------------------


def test_weights_table() -> None:
    assert QUERY_WEIGHTS == {"caption": 0.0, "sentence": 1.0, "text_box": 0.75, "phrase": 0.5}


def test_caption_root_is_first_and_parentless() -> None:
    queries = build_query_hierarchy("a red bicycle", [])
    assert _tuples(queries) == [("a red bicycle", "caption", -1, 0.0, -1)]


def test_sentence_split_on_terminal_punctuation_with_caption_parent() -> None:
    queries = build_query_hierarchy("A dog runs. A cat sleeps.", [])
    sentences = [q for q in queries if q.kind == "sentence"]
    # terminal punctuation is kept; every sentence's parent is the caption root
    assert [q.text for q in sentences] == ["A dog runs.", "A cat sleeps."]
    assert [q.parent for q in sentences] == [0, 0]
    assert [q.weight for q in sentences] == [1.0, 1.0]


def test_sentence_cap_applies_to_raw_fragments_before_dedupe() -> None:
    queries = build_query_hierarchy("One. Two. Three. Four. Five. Six. Seven.", [])
    assert [q.text for q in queries if q.kind == "sentence"] == ["One.", "Two.", "Three.", "Four.", "Five."]


def test_single_sentence_caption_yields_no_sentence_query() -> None:
    """The lone sentence equals the caption, so the dedupe drops it."""
    queries = build_query_hierarchy("a red bicycle", [])
    assert [q.kind for q in queries] == ["caption"]


def test_connector_split_and_phrase_parent_falls_back_to_caption() -> None:
    queries = build_query_hierarchy("a cat and a dog", [])
    phrases = [q for q in queries if q.kind == "phrase"]
    assert [q.text for q in phrases] == ["a cat", "a dog"]
    # no sentence query exists, so phrases hang off the caption root
    assert all(q.parent == 0 and q.weight == 0.5 for q in phrases)


def test_phrase_parent_is_first_accepted_sentence() -> None:
    queries = build_query_hierarchy("A dog runs. A cat sleeps.", [])
    phrases = [q for q in queries if q.kind == "phrase"]
    assert phrases and all(q.parent == 1 for q in phrases)
    assert queries[1].kind == "sentence"


def test_phrase_length_bounds_two_to_eight_words() -> None:
    assert extract_lightweight_phrases("one two three") == ["one two three"]
    assert extract_lightweight_phrases("one") == []
    assert extract_lightweight_phrases("a and b") == []  # both chunks have one word
    eight = "one two three four five six seven eight"
    assert extract_lightweight_phrases(eight) == [eight]


def test_phrase_windows_of_six_words_every_four() -> None:
    text = "a very long uninterrupted description of a busy city street scene in the late afternoon light"
    words = text.split()
    assert len(words) > 8
    expected = [" ".join(words[start : start + 6]) for start in range(0, len(words) - 1, 4)]
    expected = [phrase for phrase in expected if len(phrase.split()) >= 2]
    assert extract_lightweight_phrases(text) == expected
    # windows of 6 starting every 4 overlap by 2 words
    assert expected[0].split()[4:] == expected[1].split()[:2]


def test_three_character_minimum() -> None:
    """``"a b c"`` is 5 characters, but its phrase chunks are all single words."""
    assert build_query_hierarchy("a b c", [])[0].text == "a b c"
    assert len(build_query_hierarchy("a b c", [])) == 1
    # a two-character box text is below the 3-character floor and is dropped
    assert [q.kind for q in build_query_hierarchy("a b c", ["ab"])] == ["caption"]


def test_case_insensitive_dedupe() -> None:
    queries = build_query_hierarchy("A Red Car", ["a red car", "A RED CAR", "red bench"])
    # the first three candidates all casefold to the caption's key
    assert _tuples(queries) == [
        ("A Red Car", "caption", -1, 0.0, -1),
        ("red bench", "text_box", 0, 0.75, 2),
    ]


def test_text_box_source_part_stays_true_index() -> None:
    """Rejected boxes do not shift the surviving boxes' ``source_part``."""
    queries = build_query_hierarchy("an empty room with white walls", ["a", "", "b c", "d e"])
    boxes = [q for q in queries if q.kind == "text_box"]
    assert [(q.text, q.source_part) for q in boxes] == [("b c", 2), ("d e", 3)]


def test_text_boxes_precede_phrases_and_truncation_keeps_order() -> None:
    caption = "a cat and a dog and a bird and a fish"
    queries = build_query_hierarchy(caption, ["box one", "box two"], max_queries=6)
    assert [q.kind for q in queries] == ["caption", "text_box", "text_box", "phrase", "phrase", "phrase"]
    assert len(queries) == 6


def test_use_text_boxes_false_drops_box_queries() -> None:
    queries = build_query_hierarchy("a cat and a dog", ["box one"], use_text_boxes=False)
    assert not [q for q in queries if q.kind == "text_box"]


def test_rejected_caption_root_falls_back_to_index_zero() -> None:
    """Whitespace-only captions leave the first surviving query parented to itself."""
    queries = build_query_hierarchy("   \n  ", ["a box"])
    assert _tuples(queries) == [("a box", "text_box", 0, 0.75, 0)]
