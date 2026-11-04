import pytest

from src.source_spans import build_source_units, resolve_source_ranges


def unit_for_text(lookup, chunks, text):
    return next(unit_id for unit_id, (chunk, start, end) in lookup.items() if chunks[chunk][start:end] == text)


def test_unit_ids_follow_physical_order_without_overlapping_clause_candidates():
    chunks = ["Paris then Paris", "Rome."]
    display, lookup = build_source_units(chunks)
    assert display == [[[0, "Paris"], [1, "then"], [2, "Paris"]], [[3, "Rome"], [4, "."]]]
    assert list(lookup) == list(range(5))
    for index in range(len(chunks)):
        positions = [(start, end) for chunk, start, end in lookup.values() if chunk == index]
        assert all(left_end <= right_start for (_, left_end), (right_start, _) in zip(positions, positions[1:]))


def test_global_range_includes_all_intermediate_chunks_and_preserves_occurrence():
    chunks = ["Paris then Paris", "", "Café\n上海", " \n", "Berlin then Paris"]
    _, lookup = build_source_units(chunks)
    assert resolve_source_ranges([{"first": 2, "last": 7}], chunks,
                                 ["first", "empty", "middle", "blank", "last"], lookup) == (
        [{"chunk_id": "first", "quote": "Paris", "occurrence": 1},
         {"chunk_id": "middle", "quote": "Café\n上海", "occurrence": 0},
         {"chunk_id": "last", "quote": "Berlin then Paris", "occurrence": 0}],
        ["first", "middle", "last"],
    )


def test_ranges_preserve_unicode_original_whitespace_and_exact_offsets():
    chunks = ["  Café\t上海，Paris  Paris."]
    display, lookup = build_source_units(chunks)
    assert [text for _, text in display[0][-6:]] == ["Café", "上海", "，", "Paris", "Paris", "."]
    assert (0, 2, 6) in lookup.values()
    assert (0, 17, 22) in lookup.values()
    assert [unit_id for unit_id, _ in display[0]] == list(range(len(lookup)))
    for units in display:
        for unit_id, text in units:
            chunk, start, end = lookup[unit_id]
            assert type(unit_id) is int
            assert text == chunks[chunk][start:end]
    first = unit_for_text(lookup, chunks, "Café")
    last = unit_for_text(lookup, chunks, "Paris")
    assert resolve_source_ranges([{"first": first, "last": last}], chunks, ["unicode"], lookup) == (
        [{"chunk_id": "unicode", "quote": "Café\t上海，Paris", "occurrence": 0}], ["unicode"],
    )


def test_repeated_word_range_selects_the_second_occurrence_across_chunks():
    chunks = ["Paris then Paris", "Paris then Paris"]
    _, lookup = build_source_units(chunks)
    ranges = [{"first": 5, "last": 5}, {"first": 0, "last": 0}, {"first": 3, "last": 3}]
    assert resolve_source_ranges(ranges, chunks, ["first", "second"], lookup) == (
        [{"chunk_id": "first", "quote": "Paris", "occurrence": 0},
         {"chunk_id": "second", "quote": "Paris", "occurrence": 0},
         {"chunk_id": "second", "quote": "Paris", "occurrence": 1}], ["first", "second"],
    )
    assert resolve_source_ranges([{"first": 2, "last": 2}], chunks, ["first", "second"], lookup) == (
        [{"chunk_id": "first", "quote": "Paris", "occurrence": 1}], ["first"],
    )


@pytest.mark.parametrize("ranges", [
    None, {}, [], [None], [True], [0], [1.0], ["1"], [[1, 1]],
    [{"first": True, "last": 1}], [{"first": 1, "last": False}],
    [{"first": "1", "last": 1}], [{"first": 1, "last": 1.0}],
    [{"first": -1, "last": 1}], [{"first": 1, "last": -1}],
    [{"first": 999, "last": 1}], [{"first": 1, "last": 999}],
    [{"first": 1}], [{"last": 1}], [{"first": 1, "last": 1, "extra": 1}],
    [{"first": 1, "last": 1}] * 2,
    [{"first": 2, "last": 1}],  # Backwards in character order.
    [{"first": 3, "last": 1}],  # Backwards across different chunks.
    [{"first": 2, "last": 0}],
])
def test_ranges_reject_invalid_unknown_reversed_or_duplicate_endpoints(ranges):
    chunks = ["Paris then Paris", "Berlin"]
    _, lookup = build_source_units(chunks)
    with pytest.raises(ValueError):
        resolve_source_ranges(ranges, chunks, ["c0", "c1"], lookup)


def test_generated_ids_and_physical_positions_have_the_same_range_direction():
    chunks = ["Intro. Paris then Rome."]
    _, lookup = build_source_units(chunks)
    first = unit_for_text(lookup, chunks, "Paris")
    last = unit_for_text(lookup, chunks, "Rome")
    assert first < last
    assert resolve_source_ranges([{"first": first, "last": last}], chunks, ["c"], lookup) == (
        [{"chunk_id": "c", "quote": "Paris then Rome", "occurrence": 0}], ["c"],
    )
    with pytest.raises(ValueError):
        resolve_source_ranges([{"first": last, "last": first}], chunks, ["c"], lookup)


def test_continuous_range_explicitly_includes_all_intervening_words_and_punctuation():
    chunks = ["I prefer short replies; use English."]
    _, lookup = build_source_units(chunks)
    first = unit_for_text(lookup, chunks, "I")
    last = unit_for_text(lookup, chunks, "use")
    assert resolve_source_ranges([{"first": first, "last": last}], chunks, ["c"], lookup) == (
        [{"chunk_id": "c", "quote": "I prefer short replies; use", "occurrence": 0}], ["c"],
    )


def test_separate_ranges_do_not_fill_unselected_words_and_punctuation():
    chunks = ["I prefer short replies; use English."]
    _, lookup = build_source_units(chunks)
    selected = [unit_for_text(lookup, chunks, text) for text in ["I", "replies", "use"]]
    ranges = [{"first": unit_id, "last": unit_id} for unit_id in selected]
    assert resolve_source_ranges(ranges, chunks, ["c"], lookup) == (
        [{"chunk_id": "c", "quote": "I", "occurrence": 0},
         {"chunk_id": "c", "quote": "replies", "occurrence": 0},
         {"chunk_id": "c", "quote": "use", "occurrence": 0}], ["c"],
    )


def test_adjacent_ranges_merge_only_across_original_whitespace():
    chunks = ["I\tprefer\nshort replies; use English."]
    _, lookup = build_source_units(chunks)
    selected = [unit_for_text(lookup, chunks, text) for text in ["replies", "I", "prefer", "short"]]
    assert resolve_source_ranges([{"first": value, "last": value} for value in selected], chunks, ["c"], lookup) == (
        [{"chunk_id": "c", "quote": "I\tprefer\nshort replies", "occurrence": 0}], ["c"],
    )


def test_overlapping_ranges_merge_without_duplicate_quotes():
    chunks = ["I live in Paris."]
    _, lookup = build_source_units(chunks)
    assert resolve_source_ranges([{"first": 0, "last": 3}, {"first": 1, "last": 4}], chunks, ["c"], lookup) == (
        [{"chunk_id": "c", "quote": "I live in Paris.", "occurrence": 0}], ["c"],
    )


def test_unrepresentable_range_occurrence_rejects_without_atomic_fallback():
    chunks = ["a a a"]
    _, lookup = build_source_units(chunks)
    with pytest.raises(ValueError):
        resolve_source_ranges([{"first": 1, "last": 2}], chunks, ["c"], lookup)
    assert resolve_source_ranges([{"first": 0, "last": 1}], chunks, ["c"], lookup) == (
        [{"chunk_id": "c", "quote": "a a", "occurrence": 0}], ["c"],
    )


def test_whitespace_only_chunks_have_no_selectable_ranges():
    chunks = ["", " \n\t"]
    display, lookup = build_source_units(chunks)
    assert display == [[], []]
    assert lookup == {}
    with pytest.raises(ValueError):
        resolve_source_ranges([{"first": 0, "last": 0}], chunks, ["a", "b"], lookup)


def test_word_range_keeps_global_preference_separate_from_project_rule():
    chunks = ["I prefer short replies; this project's replies must be in English."]
    display, lookup = build_source_units(chunks)
    assert display[0][:4] == [[0, "I"], [1, "prefer"], [2, "short"], [3, "replies"]]
    assert resolve_source_ranges([{"first": 0, "last": 3}], chunks, ["c"], lookup) == (
        [{"chunk_id": "c", "quote": "I prefer short replies", "occurrence": 0}], ["c"],
    )


def test_integer_ids_are_dense_across_words_and_empty_chunks():
    chunks = ["Paris.", "", " \n", "Rome."]
    display, lookup = build_source_units(chunks)
    assert display == [
        [[0, "Paris"], [1, "."]], [], [],
        [[2, "Rome"], [3, "."]],
    ]
    ids = [unit_id for units in display for unit_id, _ in units]
    assert ids == list(range(len(lookup)))
    assert lookup[2] == (3, 0, 4)
    assert resolve_source_ranges([{"first": 2, "last": 3}, {"first": 0, "last": 1}],
                                 chunks, ["first", "blank", "white", "second"], lookup) == (
        [{"chunk_id": "first", "quote": "Paris.", "occurrence": 0},
         {"chunk_id": "second", "quote": "Rome.", "occurrence": 0}], ["first", "second"],
    )
    with pytest.raises(ValueError):
        resolve_source_ranges([{"first": 0, "last": len(lookup)}], chunks, ["first", "blank", "white", "second"], lookup)
