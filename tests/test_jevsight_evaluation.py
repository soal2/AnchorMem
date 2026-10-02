import importlib.util
import json
from pathlib import Path

import pytest

from src import evaluation
from src.config import EvaluationConfig, load_config
from src.contracts import Evidence, SearchResponse


def _selection_config(tmp_path, questions, **selection_updates):
    config, _ = load_config("configs/evaluation/longmemeval-dev.yaml", EvaluationConfig)
    path = tmp_path / "evaluation.yaml"
    selection = {"dataset": config.dataset, "split": "dev", "seed": config.seed,
                 "data_sha256": config.data_sha256, "questions": questions, **selection_updates}
    (tmp_path / "questions.json").write_text(json.dumps(selection))
    return config.model_copy(update={"question_ids_path": "questions.json", "sample_limit": len(questions)}), path


def test_frozen_selection_preserves_ids_order_and_group_identity(tmp_path):
    examples = [{"id": "q1", "group": "g", "category": "temporal"},
                {"id": "q2", "group": "g", "category": "update"}]
    config, path = _selection_config(tmp_path, list(reversed(examples)))
    assert evaluation.select_examples(examples, config, path) == list(reversed(examples))


@pytest.mark.parametrize("questions,updates", [
    ([{"id": "missing", "group": "g", "category": "temporal"}], {}),
    ([{"id": "q1", "group": "wrong", "category": "temporal"}], {}),
    ([{"id": "q1", "group": "g", "category": 2}], {}),
    ([{"id": "q1", "group": "g", "category": "temporal"}] * 2, {}),
    ([{"id": "q1", "group": "g", "category": "temporal"}], {"data_sha256": "wrong"}),
])
def test_frozen_selection_rejects_modified_identity(tmp_path, questions, updates):
    examples = [{"id": "q1", "group": "g", "category": "temporal"}]
    config, path = _selection_config(tmp_path, questions, **updates)
    with pytest.raises(ValueError):
        evaluation.select_examples(examples, config, path)


def test_model_usage_retains_missing_tokens_and_credits_without_estimating_usd():
    result = evaluation._call_summary([
        {"evaluation_stage": "ingestion", "latency_seconds": 2.0, "success": True,
         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.02}},
        {"evaluation_stage": "ingestion", "latency_seconds": 3.0, "success": False, "usage": None},
    ])["ingestion"]
    assert result["calls"] == 2 and result["failed_calls"] == 1
    assert result["input_tokens"] == 10 and result["input_tokens_missing_calls"] == 1
    assert result["provider_cost_credits"] == 0.02 and result["provider_cost_credits_missing_calls"] == 1
    assert result["cost_usd"] is None and result["cost_usd_missing_calls"] == 2
    assert result["latency_seconds"] == 5.0


def test_evaluation_snapshots_prompts_telemetry_and_real_group_ingestion(tmp_path, monkeypatch, runner):
    config, config_path = load_config("configs/evaluation/longmemeval-dev.yaml", EvaluationConfig)
    data_path = tmp_path / "data.json"
    data_path.write_text("[]")
    contract_path = tmp_path / "contract"
    pipeline = contract_path / "data/longmemeval-s/pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text('OPEN_ENDED_ANSWER_TEMPLATE = "{{question}} {{speaker_1_memories}}"\n'
                        'ACCURACY_PROMPT = "{question} {gold_answer} {generated_answer}"\n')
    config = config.model_copy(update={"data_path": str(data_path), "data_sha256": evaluation.sha256(data_path),
                                      "contract_repository_path": str(contract_path), "output_dir": str(tmp_path / "runs"),
                                      "sample_limit": 2})
    examples = [{"id": f"q{index}", "group": "history", "category": "single-session-user",
                 "question": "Where?", "question_date": "2023/05/20", "answers": ["Paris"],
                 "sessions": [{"id": "s1", "messages": [{"role": "user", "content": "Paris"}]}]}
                for index in (1, 2)]
    monkeypatch.setattr(evaluation, "load_examples", lambda *args: examples)
    monkeypatch.setattr(evaluation, "_commit", lambda path: config.contract_commit)

    class FakeLLM:
        def __init__(self, model_config, telemetry=None):
            self.telemetry = telemetry

        def complete(self, system, prompt, json_response=True):
            self.telemetry({"stage": "extraction" if system == "extract" else "completion", "model": "configured",
                            "resolved_model": "actual", "latency_seconds": 0.1, "request_id": "provider-id",
                            "success": True, "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
            return '{"label":"CORRECT"}' if json_response else "Paris"

    class FakeEngine:
        def __init__(self, service, service_path, llm, embedder):
            self.llm = llm
            self.requests = set()

        def add(self, request):
            if request.request_id not in self.requests:
                self.llm.complete("extract", "Paris", json_response=False)
                self.requests.add(request.request_id)

        def search(self, request):
            assert request.query == "Where?"  # Question date still belongs only to Answer.
            return SearchResponse(data=[Evidence(id="e1", content="Paris", score=1.0)])

    monkeypatch.setattr(evaluation, "OpenAILLM", FakeLLM)
    monkeypatch.setattr(evaluation, "OpenAIEmbedder", lambda config, telemetry=None: object())
    monkeypatch.setattr(evaluation, "MemoryEngine", FakeEngine)
    run_dir = evaluation.evaluate(config, config_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["code_snapshot_sha256"] == runner._source_hash()
    assert (run_dir / manifest["prompt_files"]["extract"]["path"]).read_text()
    assert (run_dir / "prompts/answer.txt").read_text() == "{{question}} {{speaker_1_memories}}"
    chosen = json.loads((run_dir / "chosen-questions.json").read_text())
    assert [item["id"] for item in chosen] == ["q1", "q2"]
    ingestion = evaluation._jsonl(run_dir / "ingestion.jsonl")
    assert len(ingestion) == 1 and ingestion[0]["success"] is True
    calls = evaluation._jsonl(run_dir / "model-calls.jsonl")
    assert len(calls) == 5
    assert calls[0]["evaluation_stage"] == "ingestion" and calls[0]["stage"] == "extraction"
    assert all(item["resolved_model"] == "actual" for item in calls)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["correct_per_requested"] == 1
    assert metrics["ingestion_groups"]["history"]["successful_requests"] == 1
    assert metrics["model_usage"]["ingestion"]["calls"] == 1
    assert metrics["model_usage"]["answer"]["calls"] == 2
    assert metrics["latency"]["search"]["count"] == 2


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/jevsight_experiments.py"
    spec = importlib.util.spec_from_file_location("jevsight_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_selection_excludes_used_ids_and_reuses_no_persona(runner):
    examples = [{"id": "seen", "group": "p1", "category": "x"},
                {"id": "q1", "group": "p2", "category": "x"},
                {"id": "q2", "group": "p2", "category": "y"},
                {"id": "q3", "group": "p3", "category": "x"}]
    runner.VALIDATION_QUOTAS = {**runner.VALIDATION_QUOTAS, "personamem": 2}
    result = runner.choose_validation("personamem", examples, {"ids": {"seen"}, "groups": {"p1"}})
    assert len(result) == 2 and {item["group"] for item in result} == {"p2", "p3"}
    assert result == runner.choose_validation("personamem", list(reversed(examples)), {"ids": {"seen"}, "groups": {"p1"}})


def test_frozen_files_are_not_overwritten(runner, tmp_path):
    path = tmp_path / "selection.json"
    runner._frozen_write(path, "frozen\n")
    runner._frozen_write(path, "frozen\n")
    with pytest.raises(ValueError, match="refusing to change"):
        runner._frozen_write(path, "different\n")
    assert path.read_text() == "frozen\n"


def _run_artifacts(path, complete=True):
    path.mkdir()
    metrics = {"requested": 2, "evaluated": 2 if complete else 1, "failed": 0 if complete else 1,
               "correct": 1, "primary_metric": "accuracy", "primary_value": 0.5,
               "actual_ingestion_seconds": 1.0, "latency": {"search": {"p95_seconds": 0.1}}}
    (path / "metrics.json").write_text(json.dumps(metrics))
    questions = [{"id": "q1"}, {"id": "q2"}]
    (path / "chosen-questions.json").write_text(json.dumps(questions))
    for filename in ("predictions.jsonl", "retrieval.jsonl"):
        (path / filename).write_text('\n'.join(json.dumps({"id": item["id"], "label": "WRONG"}) for item in questions) + '\n')


def test_resume_requires_full_success_and_exact_prediction_ids(runner, tmp_path):
    good = tmp_path / "good"
    _run_artifacts(good)
    assert runner._complete(good, 2)
    bad = tmp_path / "bad"
    _run_artifacts(bad, complete=False)
    assert not runner._complete(bad, 2)
    (good / "predictions.jsonl").write_text('{"id":"q1"}\n{"id":"q1"}\n')
    assert not runner._complete(good, 2)


def test_report_distinguishes_completed_failed_and_pending(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(runner, "_source_hash", lambda: "current")
    good, bad = tmp_path / "good", tmp_path / "bad"
    _run_artifacts(good)
    _run_artifacts(bad, complete=False)
    jobs = [{"stage": stage, "variant": variant, "dataset": dataset, "question_count": 2}
            for stage in ("dev", "validation") for variant in runner.VARIANTS for dataset in runner.DATASETS]

    def matches(job, code_hash):
        if job["stage"] == "dev" and job["dataset"] == "locomo":
            return [good] if job["variant"] == "a" else ([bad] if job["variant"] == "b" else [])
        return []

    monkeypatch.setattr(runner, "matching_runs", matches)
    result = runner.report({"jobs": jobs, "scope": "local sample only"})
    assert result["status_counts"] == {"executed": 1, "running": 0, "failure": 1, "pending": 38}
    assert all(item["status"] == "pending" and "primary_delta" not in item for item in result["comparisons"])
    assert result["validation_acceptance"]["optimization_status"] == "pending"
    assert result["validation_acceptance"]["jev_status"] == "pending"
    assert "local sample only" in (runner.ARTIFACTS / "report.md").read_text()


def test_report_marks_an_unfinished_attempt_as_running(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(runner, "_source_hash", lambda: "current")
    monkeypatch.setattr(runner, "matching_runs", lambda job, code_hash: [])
    jobs_dir = runner.ARTIFACTS / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "current.json").write_text(json.dumps({
        "stage": "dev", "variant": "d", "dataset": "locomo", "question_count": 2,
        "code_snapshot_sha256": "current", "started_at": "2026-10-01T00:00:00+00:00",
        "returncode": None, "log_path": "artifacts/jobs/current.log"}))
    jobs = [{"stage": stage, "variant": variant, "dataset": dataset, "question_count": 2}
            for stage in ("dev", "validation") for variant in runner.VARIANTS for dataset in runner.DATASETS]
    result = runner.report({"jobs": jobs, "scope": "local sample only"})
    assert result["status_counts"] == {"executed": 0, "running": 1, "failure": 0, "pending": 39}


def test_report_does_not_count_stale_config_attempt_as_current_failure(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(runner, "_source_hash", lambda: "current")
    monkeypatch.setattr(runner, "matching_runs", lambda job, code_hash: [])
    jobs_dir = runner.ARTIFACTS / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "old-config.json").write_text(json.dumps({
        "stage": "dev", "variant": "d", "dataset": "locomo", "question_count": 2,
        "code_snapshot_sha256": "current", "started_at": "2026-10-01T00:00:00+00:00",
        "returncode": 0, "completed_at": "2026-10-01T00:01:00+00:00",
        "log_path": "artifacts/jobs/old-config.log"}))
    jobs = [{"stage": stage, "variant": variant, "dataset": dataset, "question_count": 2}
            for stage in ("dev", "validation") for variant in runner.VARIANTS for dataset in runner.DATASETS]
    result = runner.report({"jobs": jobs, "scope": "local sample only"})
    assert result["status_counts"] == {"executed": 0, "running": 0, "failure": 0, "pending": 40}


def test_resource_delta_requires_complete_usage_and_treats_no_calls_as_zero(runner):
    old = {"model_usage": {"ingestion": {"calls": 1, "latency_seconds": 2,
            "input_tokens": 10, "input_tokens_missing_calls": 0,
            "provider_cost_credits": None, "provider_cost_credits_missing_calls": 1}}}
    new = {"model_usage": {"ingestion": {"calls": 2, "latency_seconds": 5,
            "input_tokens": 30, "input_tokens_missing_calls": 0,
            "provider_cost_credits": 0.1, "provider_cost_credits_missing_calls": 0}},
           "provider_stage_usage": {"decision_jev": {"calls": 1, "latency_seconds": 1,
            "input_tokens": 15, "input_tokens_missing_calls": 0}}}
    result = runner._resource_comparison(old, new)
    assert result["model_usage"]["ingestion"]["delta"]["input_tokens"] == 20
    assert result["model_usage"]["ingestion"]["delta"]["latency_seconds"] == 3
    assert result["model_usage"]["ingestion"]["delta"]["provider_cost_credits"] is None
    assert result["provider_stage_usage"]["decision_jev"]["delta"]["input_tokens"] == 15


def test_validation_acceptance_requires_all_five_and_positive_fixed_time_update_gain(runner):
    rows = {("validation", variant, dataset): {"status": "executed", "metrics": {
                "primary_metric": "mean_rubric_score" if dataset == "beam" else "accuracy", "primary_value": 0.5}}
            for variant in ("a", "c", "d") for dataset in runner.DATASETS}
    comparisons = [{"stage": "validation", "dataset": dataset, "contrast": "D-A", "status": "executed",
                    "time_update": {"gained_correct": int(dataset == "longmemeval"), "lost_correct": 0,
                                    "paired_questions": 2}} for dataset in runner.TIME_UPDATE_CATEGORIES]
    result = runner._validation_acceptance(rows, comparisons)
    assert result["optimization_status"] == result["jev_status"] == "pass"
    assert result["time_update"]["net_correct"] == 1
    assert set(result["time_update"]["categories"]["beam"]) == {"temporal_reasoning", "knowledge_update"}
    comparisons[1]["time_update"]["lost_correct"] = 1
    result = runner._validation_acceptance(rows, comparisons)
    assert result["optimization_status"] == result["jev_status"] == "fail"
    assert result["jev_comparison_status"] == "pass"
    rows[("validation", "d", "beam")]["status"] = "pending"
    result = runner._validation_acceptance(rows, comparisons)
    assert result["optimization_status"] == result["jev_status"] == "pending"


def test_comparison_cases_preserve_predictions_and_beam_partial_scores(runner):
    before = {"q1": {"id": "q1", "category": "temporal_reasoning", "label": "CORRECT", "answer": "old", "beam_score": 1},
              "q2": {"id": "q2", "label": "WRONG", "answer": "partial", "beam_score": 0.5},
              "q3": {"id": "q3", "label": "CORRECT", "answer": "same", "beam_score": 1}}
    after = {**before, "q1": {**before["q1"], "label": "WRONG", "answer": "new", "beam_score": 0},
             "q2": {**before["q2"], "answer": "less", "beam_score": 0}}
    cases = runner._comparison_cases(before, after, "beam")
    assert [item["id"] for item in cases] == ["q1", "q2"]
    assert cases[0]["outcome"] == "lost_correct" and cases[0]["right_prediction"] == "new"
    assert cases[1]["left_primary_score"] == 0.5 and cases[1]["right_primary_score"] == 0


def test_attempt_history_retains_interrupted_old_source_usage(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path / "artifacts")
    path = runner.ARTIFACTS / "runs" / "old-run"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"run_label": "jevsight-dev-d-locomo",
        "code_snapshot_sha256": "old", "created_at": "2026-09-30T16:23:36+00:00"}))
    call = {"evaluation_stage": "ingestion", "stage": "decision_jev", "latency_seconds": 2,
            "success": False, "usage": {"total_tokens": 10, "cost": 0.1}}
    (path / "model-calls.jsonl").write_text(json.dumps(call) + '\n{"truncated":\n')
    (path / "ingestion.jsonl").write_text('{"seconds":3}\n')
    attempt = {"stage": "dev", "variant": "d", "dataset": "locomo", "question_count": 30,
               "started_at": "2026-09-30T16:23:35+00:00", "completed_at": "2026-09-30T16:26:28+00:00",
               "returncode": -15, "code_snapshot_sha256": "old", "interruption_reason": "contract diagnosis"}
    result = runner._attempt_history([attempt], "new")[0]
    assert result["matches_current_source"] is False and result["interruption_reason"] == "contract diagnosis"
    artifact = result["run_artifacts"][0]
    assert artifact["complete_success"] is False
    assert artifact["observed_ingestion_seconds"] == 3
    assert artifact["observed_model_usage"]["ingestion"]["provider_cost_credits"] == 0.1
    assert artifact["observed_model_usage"]["ingestion"]["failed_calls"] == 1
    assert artifact["malformed_records"] == {"model-calls.jsonl": 1}


def test_validation_waits_for_all_twenty_current_development_runs(runner, tmp_path, monkeypatch):
    good = tmp_path / "good"
    _run_artifacts(good)
    jobs = [{"stage": "dev", "variant": variant, "dataset": dataset, "question_count": 2}
            for variant in runner.VARIANTS for dataset in runner.DATASETS]
    monkeypatch.setattr(runner, "matching_runs", lambda job, code_hash:
                        [] if job["variant"] == "d" and job["dataset"] == "clbench" else [good])
    assert not runner.development_complete({"jobs": jobs}, "current")
    monkeypatch.setattr(runner, "matching_runs", lambda job, code_hash: [good])
    assert runner.development_complete({"jobs": jobs}, "current")
    with pytest.raises(ValueError, match="exactly all 20"):
        runner.development_complete({"jobs": jobs[:-1]}, "current")


def test_decision_efficiency_counts_jev_tokens_and_first_ingestion_cost(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls = [
        {"stage": "decision_jev", "evaluation_stage": "ingestion", "latency_seconds": 2,
         "success": True, "usage": {"input_tokens": 8, "output_tokens": 2, "cost": 0.01}},
        {"stage": "embedding", "evaluation_stage": "ingestion", "latency_seconds": 1,
         "success": True, "usage": {"total_tokens": 4, "cost": 0.001}},
    ]
    (run_dir / "model-calls.jsonl").write_text("\n".join(json.dumps(call) for call in calls) + "\n")
    row = {"stage": "dev", "dataset": "locomo", "variant": "d", "status": "executed",
           "run_path": "run", "metrics": {"actual_ingestion_seconds": 5}}
    result = runner._decision_efficiency(row)
    assert result["decision_calls"] == 1
    assert result["decision_seconds_per_call"] == 2
    assert result["decision_total_tokens"] == 10
    assert result["ingestion_total_tokens"] == 14
    assert result["ingestion_provider_credits"] == pytest.approx(0.011)
    assert runner._ratio(result["decision_seconds_per_call"], 4) == 0.5
    calls[0]["usage"].pop("output_tokens")
    (run_dir / "model-calls.jsonl").write_text("\n".join(json.dumps(call) for call in calls) + "\n")
    assert runner._decision_efficiency(row)["decision_total_tokens"] is None
