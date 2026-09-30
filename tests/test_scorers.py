import json

import pytest

from aml_memory import evaluation


class FakeJudge:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system, prompt, json_response=True):
        self.calls.append((system, prompt))
        return self.response


@pytest.fixture
def clbench_example():
    return {
        "question": "How should I present the proposed project?",
        "sessions": [
            {"id": "s1", "messages": [
                {"role": "user", "content": "Please cover the budget and schedule."},
                {"role": "assistant", "content": "The budget is 500 dollars."},
            ]},
            {"id": "s2", "messages": [
                {"role": "user", "content": "The deadline is next Friday."},
            ]},
        ],
    }


BEAM_PROTOCOL = "Question: <question>\nRubrics: <rubric_item>\nAnswer: <llm_response>\n## OUTPUT FORMAT:\nOld format"
BEAM_BATCH = 'Return {"scores":[{"index":0,"score":1,"reason":"why"}]}'


@pytest.mark.parametrize("answer, rubric, verdict", [
    ("She resides in the French capital.", "The user lives in Paris.", 1),
    ("I don't know.", "The spouse is not mentioned.", 1),
    ("I don't know.", "The user lives in Paris.", 0),
    ("The spouse is Bob.", "The spouse is not mentioned.", 0),
])
def test_beam_uses_semantic_judge_for_facts_and_abstention(answer, rubric, verdict):
    judge = FakeJudge(json.dumps({"scores": [{"index": 0, "score": verdict, "reason": "Semantic verdict."}]}))
    score, details = evaluation.score_beam(answer, [rubric], judge, "Where does the user live?", BEAM_PROTOCOL, BEAM_BATCH)
    assert score == verdict
    assert details == [{"index": 0, "rubric": rubric, "score": float(verdict), "reason": "Semantic verdict."}]
    assert len(judge.calls) == 1
    assert answer in judge.calls[0][1] and rubric in judge.calls[0][1]


def test_beam_orders_scores_and_supports_partial_compliance():
    judge = FakeJudge(json.dumps({"scores": [
        {"index": 1, "score": 0.5, "reason": "Incomplete schedule."},
        {"index": 0, "score": 1, "reason": "Correct budget."},
    ]}))
    score, details = evaluation.score_beam("$500 and next week", ["Budget", "Schedule"], judge, "Plan?", BEAM_PROTOCOL, BEAM_BATCH)
    assert score == 0.75
    assert [item["index"] for item in details] == [0, 1]


@pytest.mark.parametrize("rubrics", [[], None, "Paris", [""], [" "], [None], [42], [{"rubric": "Paris"}]])
def test_beam_rejects_bad_rubrics_without_calling_judge(rubrics):
    judge = FakeJudge("must not be called")
    with pytest.raises(ValueError):
        evaluation.score_beam("An answer", rubrics, judge, "Question?", BEAM_PROTOCOL, BEAM_BATCH)
    assert judge.calls == []


@pytest.mark.parametrize("response", [
    "not JSON", "null", "[]", "{}", '{"scores":[]}',
    '{"scores":[{"index":0,"score":1,"reason":"ok"}]}',
    '{"scores":[{"index":0,"score":1,"reason":"ok"},{"index":0,"score":1,"reason":"ok"}]}',
    *[json.dumps({"scores": [{"index": 0, "score": invalid, "reason": "ok"},
                             {"index": 1, "score": 1, "reason": "ok"}]}) for invalid in [True, "1", -1, 0.7, 2, None]],
    *[json.dumps({"scores": [{"index": invalid, "score": 1, "reason": "ok"},
                             {"index": 1, "score": 1, "reason": "ok"}]}) for invalid in [True, "0", -1, 2, 0.0]],
    *[json.dumps({"scores": [{"index": 0, "score": 1, "reason": invalid},
                             {"index": 1, "score": 1, "reason": "ok"}]}) for invalid in [None, "", " ", 1]],
])
def test_beam_rejects_incomplete_or_invalid_judge_scores(response):
    judge = FakeJudge(response)
    with pytest.raises(ValueError):
        evaluation.score_beam("An answer", ["Budget", "Schedule"], judge, "Plan?", BEAM_PROTOCOL, BEAM_BATCH)
    assert len(judge.calls) == 1


def test_clbench_uses_judge_for_semantic_paraphrase(clbench_example):
    answer = "1. Spend $500.\n2. Finish before the end of next week."
    rubrics = [
        {"rubric": "Provides the budget and deadline in a slide deck."},
        {"criteria": "Uses a structured format."},
    ]
    judge = FakeJudge(json.dumps({"rubric_satisfied": [True, True]}))
    assert evaluation.score_clbench(answer, rubrics, judge, clbench_example) == (
        True, 1.0, [True, True],
    )
    assert len(judge.calls) == 1
    system, prompt = judge.calls[0]
    assert system.strip()
    # Check the submitted source material without requiring a particular prompt layout.
    try:
        prompt_material = json.dumps(json.loads(prompt), ensure_ascii=False)
    except json.JSONDecodeError:
        prompt_material = prompt
    for text in [
        clbench_example["question"],
        *[message["content"] for session in clbench_example["sessions"] for message in session["messages"]],
        *[item.get("rubric") or item.get("criteria") for item in rubrics],
    ]:
        assert text in prompt_material
    assert answer in prompt_material or json.dumps(answer)[1:-1] in prompt_material
    assert "user" in prompt_material and "assistant" in prompt_material


def test_clbench_structure_does_not_override_mixed_constraints(clbench_example):
    answer = "1. Spend $900.\n2. Finish in December."
    rubric = "Uses a slide deck listing a budget of 500 dollars and a deadline next Friday."
    judge = FakeJudge('{"rubric_satisfied": [false]}')
    assert evaluation.score_clbench(answer, [rubric], judge, clbench_example) == (
        False, 0.0, [False],
    )
    assert len(judge.calls) == 1


@pytest.mark.parametrize("verdicts, expected_full, expected_fraction", [
    ([True, True, True], True, 1.0),
    ([True, False, True], False, 2 / 3),
    ([False, False, False], False, 0.0),
])
def test_clbench_aggregates_judge_verdicts(clbench_example, verdicts, expected_full, expected_fraction):
    judge = FakeJudge(json.dumps({"rubric_satisfied": verdicts}))
    full, fraction, satisfied = evaluation.score_clbench(
        "budget schedule structure", ["budget", "schedule", "structure"], judge, clbench_example,
    )
    assert full is expected_full
    assert fraction == pytest.approx(expected_fraction)
    assert satisfied == verdicts
    assert len(judge.calls) == 1


@pytest.mark.parametrize("rubrics", [[], None])
def test_clbench_empty_rubrics_skip_judge(clbench_example, rubrics):
    judge = FakeJudge("This response should never be read")
    assert evaluation.score_clbench("Any answer", rubrics, judge, clbench_example) == (True, 1.0, [])
    assert judge.calls == []


@pytest.mark.parametrize("rubric", ["", " \n\t ", {}, None, 42, []])
def test_clbench_empty_or_unsupported_statements_are_unsatisfied(clbench_example, rubric):
    judge = FakeJudge('{"rubric_satisfied": [true]}')
    assert evaluation.score_clbench("An answer", [rubric], judge, clbench_example) == (
        False, 0.0, [False],
    )
    assert len(judge.calls) == 1


def test_clbench_empty_statements_preserve_verdict_indices(clbench_example):
    rubrics = ["Includes the budget.", "  ", {"criteria": "Includes the deadline."}, {}]
    judge = FakeJudge('{"rubric_satisfied": [true, true, true, true]}')
    assert evaluation.score_clbench("An answer", rubrics, judge, clbench_example) == (
        False, 0.5, [True, False, True, False],
    )
    assert len(judge.calls) == 1


@pytest.mark.parametrize("response", [
    "not JSON",
    "null",
    "[]",
    "true",
    "{}",
    '{"rubric_satisfied": null}',
    '{"rubric_satisfied": true}',
    '{"rubric_satisfied": "true"}',
    '{"rubric_satisfied": []}',
    '{"rubric_satisfied": [true, false]}',
    '{"rubric_satisfied": ["true"]}',
    '{"rubric_satisfied": [1]}',
    '{"rubric_satisfied": [0]}',
    '{"rubric_satisfied": [null]}',
])
def test_clbench_rejects_malformed_judge_verdicts(clbench_example, response):
    judge = FakeJudge(response)
    with pytest.raises(ValueError):
        evaluation.score_clbench("A response", ["A requirement"], judge, clbench_example)
    assert len(judge.calls) == 1
