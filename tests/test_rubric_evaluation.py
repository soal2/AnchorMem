import hashlib
import json

import pytest

from src import evaluation
from src.config import EvaluationConfig, load_config
from src.contracts import Evidence, SearchResponse


def _run_rubric_evaluation(tmp_path, monkeypatch, dataset, answer, rubrics, verdict=None):
    config, config_path = load_config("configs/evaluation/locomo-dev.yaml", EvaluationConfig)
    data_path = tmp_path / "data.json"
    data_path.write_text("[]")
    config = config.model_copy(update={
        "dataset": dataset,
        "data_path": str(data_path),
        "data_sha256": evaluation.sha256(data_path),
        "contract_repository_path": str(tmp_path / "contract"),
        "output_dir": str(tmp_path / "runs"),
        "answer": config.answer.model_copy(update={"name": "test-answer"}),
        "judge": config.judge.model_copy(update={"name": "test-judge"}),
    })
    if dataset == "beam":
        pipeline = tmp_path / "contract/data/beam/pipeline.py"
        pipeline.parent.mkdir(parents=True)
        pipeline.write_text(
            'ANSWER_GENERATION_FOR_RAG = "CONTEXT: <context>\\nQUESTION: <question>"\n'
            'UNIFIED_LLM_JUDGE_BASE_PROMPT = "<question>\\n<rubric_item>\\n<llm_response>\\n## OUTPUT FORMAT:\\nold"\n'
            'BATCH_OUTPUT_FORMAT = \'Return {"scores":[]}\'\n'
            'raise RuntimeError("contract source must not execute")\n'
        )
    example = {
        "id": "q1",
        "group": "conversation-1",
        "category": "abstention" if dataset == "beam" else "clbench",
        "question": "Please propose a plan that follows my preferences.",
        "answers": [],
        "rubrics": rubrics,
        "sessions": [{"id": "session-1", "messages": [
            {"role": "user", "content": "I prefer a concise plan with a weekly schedule."},
            {"role": "assistant", "content": "I will keep that preference in mind."},
        ]}],
    }
    monkeypatch.setattr(evaluation, "load_examples", lambda dataset, path, seed: [example])
    monkeypatch.setattr(evaluation, "_commit", lambda path: config.contract_commit)
    clock = {"seconds": 0.0}
    monkeypatch.setattr(evaluation, "perf_counter", lambda: clock["seconds"])
    answer_calls = []
    judge_calls = []

    class FakeLLM:
        def __init__(self, model_config):
            self.name = model_config.name

        def complete(self, system, prompt, json_response=True):
            if self.name == "test-answer":
                answer_calls.append((system, prompt, json_response))
                clock["seconds"] += 2.0
                assert json_response is False
                return answer
            assert self.name == "test-judge"
            judge_calls.append((system, prompt, json_response))
            clock["seconds"] += 5.0
            assert json_response is True
            return verdict

    class FakeEngine:
        def __init__(self, *args):
            pass

        def add(self, request):
            pass

        def search(self, request):
            return SearchResponse(data=[Evidence(id="e1", content="Retrieved context only.", score=1.0)])

    monkeypatch.setattr(evaluation, "OpenAILLM", FakeLLM)
    monkeypatch.setattr(evaluation, "OpenAIEmbedder", lambda config: object())
    monkeypatch.setattr(evaluation, "MemoryEngine", FakeEngine)
    run_dir = evaluation.evaluate(config, config_path)
    return run_dir, example, answer_calls, judge_calls


@pytest.mark.parametrize("satisfied, label, score", [
    ([True, True], "CORRECT", 1.0),
    ([True, False], "WRONG", 0.5),
])
def test_clbench_evaluation_uses_semantic_judge_and_records_rubric_results(
    tmp_path, monkeypatch, satisfied, label, score,
):
    answer = "1. Define the milestones.\n2. Review progress each Friday."
    rubrics = [
        {"rubric": "The answer presents a slide deck about the plan."},
        {"criteria": "The answer gives a feasible schedule."},
    ]
    run_dir, example, answer_calls, judge_calls = _run_rubric_evaluation(
        tmp_path, monkeypatch, "clbench", answer, rubrics,
        json.dumps({"rubric_satisfied": satisfied}),
    )

    assert len(answer_calls) == len(judge_calls) == 1
    judge_system, judge_prompt, _ = judge_calls[0]
    assert example["question"] in judge_prompt
    for message in example["sessions"][0]["messages"]:
        assert message["content"] in judge_prompt
    for rubric in rubrics:
        assert next(iter(rubric.values())) in judge_prompt
    assert answer in judge_prompt or json.dumps(answer)[1:-1] in judge_prompt

    prediction = json.loads((run_dir / "predictions.jsonl").read_text())
    assert prediction["answer"] == answer
    assert prediction["label"] == label
    assert prediction["rubric_score"] == score
    assert prediction["rubric_satisfied"] == satisfied
    assert all(type(value) is bool for value in prediction["rubric_satisfied"])
    assert prediction["answer_seconds"] == 2.0
    assert prediction["judge_seconds"] == 5.0
    assert (run_dir / "failures.jsonl").read_text() == ""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["judge_prompt_sha256"] == hashlib.sha256(judge_system.encode()).hexdigest()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["evaluated"] == 1
    assert metrics["correct"] == int(label == "CORRECT")


@pytest.mark.parametrize("verdict", [
    "not valid JSON",
    '{"rubric_satisfied": ["true"]}',
    '{"rubric_satisfied": []}',
])
def test_clbench_invalid_judge_response_is_recorded_as_judge_failure(tmp_path, monkeypatch, verdict):
    run_dir, _, answer_calls, judge_calls = _run_rubric_evaluation(
        tmp_path, monkeypatch, "clbench", "A feasible weekly plan.",
        ["The answer gives a feasible schedule."], verdict,
    )

    assert len(answer_calls) == len(judge_calls) == 1
    assert (run_dir / "predictions.jsonl").read_text() == ""
    failure = json.loads((run_dir / "failures.jsonl").read_text())
    assert failure["category"] == "judge_failure"
    assert failure["question_category"] == "clbench"
    assert failure["answer_seconds"] == 2.0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["evaluated"] == 0
    assert metrics["failures"] == {"judge_failure": 1}


@pytest.mark.parametrize("scores, label, expected_mean", [
    ([1, 1], "CORRECT", 1.0), ([1, 0.5], "WRONG", 0.75), ([0, 0], "WRONG", 0.0),
])
def test_beam_evaluation_uses_protocol_judge_and_records_partial_scores(tmp_path, monkeypatch, scores, label, expected_mean):
    rubrics = ["The user prefers concise plans.", "The response proposes a weekly schedule."]
    verdict = json.dumps({"scores": [{"index": index, "score": score, "reason": "Semantic assessment."}
                                     for index, score in enumerate(scores)]})
    run_dir, _, answer_calls, judge_calls = _run_rubric_evaluation(
        tmp_path, monkeypatch, "beam", "A short plan reviewed each Friday.", rubrics, verdict,
    )

    assert len(answer_calls) == len(judge_calls) == 1
    assert all(rubric not in answer_calls[0][1] for rubric in rubrics)
    assert all(rubric in judge_calls[0][1] for rubric in rubrics)
    prediction = json.loads((run_dir / "predictions.jsonl").read_text())
    assert prediction["label"] == label
    assert prediction["beam_score"] == expected_mean
    assert [item["score"] for item in prediction["rubric_scores"]] == scores
    assert all(item["reason"] for item in prediction["rubric_scores"])
    assert prediction["answer_seconds"] == 2.0
    assert prediction["judge_seconds"] == 5.0
    assert (run_dir / "failures.jsonl").read_text() == ""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["requested"] == metrics["evaluated"] == 1
    assert metrics["correct"] == int(label == "CORRECT")
    assert metrics["mean_rubric_score"] == expected_mean
    assert metrics["completion_rate"] == 1.0
    assert metrics["failed"] == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["scoring_protocol"] == "beam-rubric-v1"
    assert manifest["answer_prompt_sha256"] and manifest["judge_prompt_sha256"]


@pytest.mark.parametrize("rubrics", [[], [""], [" "], [None]])
def test_beam_invalid_rubrics_fail_preflight(tmp_path, monkeypatch, rubrics):
    with pytest.raises(ValueError, match="rubric"):
        _run_rubric_evaluation(tmp_path, monkeypatch, "beam", "Any answer", rubrics)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("verdict", ["not JSON", '{"scores":[]}', '{"scores":[{"index":0,"score":true,"reason":"ok"}]}'])
def test_beam_judge_failures_are_not_scored(tmp_path, monkeypatch, verdict):
    run_dir, _, _, judge_calls = _run_rubric_evaluation(
        tmp_path, monkeypatch, "beam", "I don't know.", ["The requested fact is not mentioned."], verdict,
    )
    assert len(judge_calls) == 1
    assert (run_dir / "predictions.jsonl").read_text() == ""
    failure = json.loads((run_dir / "failures.jsonl").read_text())
    assert failure["category"] == "judge_failure"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["requested"] == metrics["failed"] == 1
    assert metrics["evaluated"] == 0
    assert metrics["accuracy"] is None and metrics["mean_rubric_score"] is None
    assert metrics["completion_rate"] == 0
