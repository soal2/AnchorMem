import json
from pathlib import Path

import pytest

from aml_memory import evaluation
from aml_memory.config import EvaluationConfig, load_config
from aml_memory.contracts import Evidence, SearchResponse
from aml_memory.providers import ProviderError


def test_locomo_preserves_source_date_and_explicit_timestamp(tmp_path):
    date = "1:56 pm on 8 May, 2023"
    conversation = {"sample_id": "c1", "sessions": [{"session_index": 1, "date_time": date,
                    "messages": [{"speaker": "Alice", "text": "I left yesterday.", "timestamp": 1683554160000},
                                 {"speaker": "Bob", "text": "Goodbye."}]}]}
    question = {"sample_id": "c1", "qa_id": "q1", "question": "When did Alice leave?", "answer": ["7 May"]}
    (tmp_path / "conversations.jsonl").write_text(json.dumps(conversation) + "\n")
    (tmp_path / "questions.jsonl").write_text(json.dumps(question) + "\n")
    messages = evaluation.load_examples("locomo", tmp_path, 42)[0]["sessions"][0]["messages"]
    assert messages[0]["content"] == f"[Session date: {date}]\nAlice: I left yesterday."
    assert messages[0]["timestamp"] == 1683554160000
    assert "timestamp" not in messages[1]  # No timezone was supplied for the date label.


def test_longmemeval_preserves_session_and_question_dates(tmp_path):
    path = tmp_path / "data.json"
    date = "2023/05/20 (Sat) 02:21"
    question_date = "2023/05/30 (Tue) 23:40"
    record = {"question_id": "q1", "question": "How long ago?", "question_date": question_date,
              "answer": "10 days", "haystack_sessions": [[{"role": "user", "content": "I left today."}]],
              "haystack_dates": [date]}
    path.write_text(json.dumps([record]))
    example = evaluation.load_examples("longmemeval", path, 42)[0]
    message = example["sessions"][0]["messages"][0]
    assert message["content"] == f"[Session date: {date}]\nI left today."
    assert "timestamp" not in message
    prompt = evaluation._answer_prompt("{{question}}\n{{speaker_1_memories}}", example, message["content"])
    assert f"[Question date: {question_date}]\nHow long ago?" in prompt
    assert date in prompt


@pytest.mark.parametrize("fail_search", [False, True])
def test_evaluation_records_snapshot_timing_and_progress(tmp_path, monkeypatch, capsys, fail_search):
    config, config_path = load_config("configs/evaluation/locomo-dev.yaml", EvaluationConfig)
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps([{"question_id": "q1", "question": "Where?", "answer": "Paris",
                                      "question_type": "single-session-user",
                                      "haystack_sessions": [[{"role": "user", "content": "Paris"}]]}]))
    contract_path = tmp_path / "contract"
    pipeline = contract_path / "data" / "longmemeval-s" / "pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text('OPEN_ENDED_ANSWER_TEMPLATE = "{{question}} {{speaker_1_memories}}"\n'
                        'ACCURACY_PROMPT = "{question} {gold_answer} {generated_answer}"\n')
    config = config.model_copy(update={"dataset": "longmemeval", "data_path": str(data_path),
                                      "data_sha256": evaluation.sha256(data_path),
                                      "contract_repository_path": str(contract_path), "output_dir": str(tmp_path / "runs")})
    monkeypatch.setattr(evaluation, "_commit", lambda path: config.contract_commit)

    class FakeLLM:
        def __init__(self, config):
            pass

        def complete(self, system, prompt, json_response=True):
            return '{"label": "CORRECT"}' if json_response else "Paris"

    class FakeEngine:
        def __init__(self, *args):
            pass

        def add(self, request):
            pass

        def search(self, request):
            if fail_search:
                raise RuntimeError("controlled failure")
            return SearchResponse(data=[Evidence(id="e1", content="Paris", score=1.0)])

    monkeypatch.setattr(evaluation, "OpenAILLM", FakeLLM)
    monkeypatch.setattr(evaluation, "OpenAIEmbedder", lambda config: object())
    monkeypatch.setattr(evaluation, "MemoryEngine", FakeEngine)
    run_dir = evaluation.evaluate(config, config_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    snapshot = run_dir / manifest["code_snapshot_path"]
    assert manifest["code_snapshot_sha256"] == evaluation.sha256(snapshot)
    source_root = Path(evaluation.__file__).resolve().parents[1]
    for source in source_root.rglob("*.py"):
        assert (snapshot / source.relative_to(source_root)).read_bytes() == source.read_bytes()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["elapsed_seconds"] >= 0
    if fail_search:
        failure = json.loads((run_dir / "failures.jsonl").read_text())
        assert failure["category"] == "search_failure"
        assert failure["question_category"] == "single-session-user"
        assert failure["elapsed_seconds"] >= 0
        assert metrics["evaluated"] == 0
    else:
        prediction = json.loads((run_dir / "predictions.jsonl").read_text())
        retrieval = json.loads((run_dir / "retrieval.jsonl").read_text())
        assert prediction["group"] == retrieval["group"]
        assert prediction["category"] == "single-session-user"
        assert prediction["search_seconds"] == retrieval["search_seconds"]
        for key in ("ingestion_seconds", "search_seconds", "answer_seconds", "judge_seconds", "elapsed_seconds"):
            assert prediction[key] >= 0
        assert metrics["correct"] == 1
    progress = capsys.readouterr().out
    assert "[1/1] q1" in progress and "started" in progress
    assert f"Evaluation complete: {run_dir}" in progress


@pytest.mark.parametrize("failure_kind", ["network", "malformed_extraction"])
def test_evaluation_preserves_provider_diagnostics_and_ingestion_offset(tmp_path, monkeypatch, failure_kind):
    import httpx
    from openai import APIConnectionError

    config, config_path = load_config("configs/evaluation/longmemeval-dev.yaml", EvaluationConfig)
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps([{"question_id": "q1", "question": "Where?", "answer": "Paris",
        "question_type": "single-session-user", "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "Paris"} for _ in range(41)]]}]))
    contract_path = tmp_path / "contract"
    pipeline = contract_path / "data" / "longmemeval-s" / "pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text('OPEN_ENDED_ANSWER_TEMPLATE = "{{question}} {{speaker_1_memories}}"\n'
                        'ACCURACY_PROMPT = "{question} {gold_answer} {generated_answer}"\n')
    config = config.model_copy(update={"data_path": str(data_path), "data_sha256": evaluation.sha256(data_path),
        "contract_repository_path": str(contract_path), "output_dir": str(tmp_path / "runs")})
    monkeypatch.setattr(evaluation, "_commit", lambda path: config.contract_commit)
    requests = []
    raw_response = '{"facts": ['

    class FakeEngine:
        def __init__(self, *args):
            pass

        def add(self, request):
            requests.append(request)
            if len(requests) == 1:
                return
            if failure_kind == "network":
                try:
                    try:
                        raise httpx.ConnectError("private credential must not be logged")
                    except httpx.ConnectError as cause:
                        raise APIConnectionError(request=httpx.Request("POST", "https://example.test")) from cause
                except APIConnectionError as cause:
                    raise ProviderError("embedding provider failed", {"phase": "embedding", "model": "fake-embedding",
                        "exception_type": "APIConnectionError", "cause_type": "ConnectError",
                        "status_code": 502, "request_id": "provider-request-1"}) from cause
            try:
                json.loads(raw_response)
            except json.JSONDecodeError as cause:
                raise ProviderError("invalid extraction JSON", {"phase": "extraction", "model": "fake-extract",
                    "exception_type": "JSONDecodeError", "finish_reason": "length"}, raw_response) from cause

    monkeypatch.setattr(evaluation, "OpenAILLM", lambda config: object())
    monkeypatch.setattr(evaluation, "OpenAIEmbedder", lambda config: object())
    monkeypatch.setattr(evaluation, "MemoryEngine", FakeEngine)
    run_dir = evaluation.evaluate(config, config_path)
    failure = json.loads((run_dir / "failures.jsonl").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(requests) == 2 and len(requests[0].messages) == 20
    assert failure["category"] == "ingestion_failure"
    assert failure["error_type"] == "ProviderError"
    assert failure["session_id"] == requests[1].session_id
    assert failure["message_batch_offset"] == 20
    assert failure["ingestion_request_id"] == requests[1].request_id
    assert metrics["failed"] == 1 and metrics["evaluated"] == 0
    assert "private credential" not in (run_dir / "failures.jsonl").read_text()
    if failure_kind == "network":
        assert failure["phase"] == "embedding"
        assert failure["exception_type"] == "APIConnectionError"
        assert failure["cause_type"] == "ConnectError"
        assert failure["cause_types"] == ["APIConnectionError", "ConnectError"]
        assert failure["status_code"] == 502 and failure["request_id"] == "provider-request-1"
        assert "raw_response_path" not in failure
    else:
        assert failure["phase"] == "extraction"
        assert failure["exception_type"] == "JSONDecodeError"
        assert failure["cause_types"] == ["JSONDecodeError"]
        assert failure["finish_reason"] == "length"
        diagnostic_path = Path(failure["raw_response_path"])
        assert diagnostic_path.is_relative_to(run_dir / "diagnostics")
        assert diagnostic_path.read_text() == raw_response
        assert "raw_response" not in failure
