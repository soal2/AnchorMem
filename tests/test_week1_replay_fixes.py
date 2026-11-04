"""Offline regressions: pinned Mem0 dates, safe replay, and bounded task retries."""
import json
from types import SimpleNamespace

import httpx
import pytest

from src.week1_baseline_worker import patch_mem0_prompt, GatewayNetwork
from src.week1_gateway import Gateway, GatewayError, backoff_seconds, is_transient
from src.week1_runner import FailurePauses, FAILURE_BURST, legacy_mem0_prompt_date
from src.week1_tasks import TaskStore, retryable_error, permanent_error


@pytest.mark.parametrize("clock_date", ["2026-11-03", "2030-01-01"])
def test_mem0_prompt_uses_manifest_date_and_patches_both_references(monkeypatch, clock_date):
    prompts = SimpleNamespace(FACT_RETRIEVAL_PROMPT=f"prefix Today's date is {clock_date}. suffix {{facts}}")
    utils = SimpleNamespace(FACT_RETRIEVAL_PROMPT=prompts.FACT_RETRIEVAL_PROMPT)
    monkeypatch.setattr("src.week1_baseline_worker.importlib.import_module",
                        lambda name: prompts if name == "mem0.configs.prompts" else utils)
    patch_mem0_prompt({"mem0_prompt_date": "2026-11-03"})
    assert prompts.FACT_RETRIEVAL_PROMPT == utils.FACT_RETRIEVAL_PROMPT == "prefix Today's date is 2026-11-03. suffix {facts}"


def test_legacy_date_uses_run_creation_in_original_timezone():
    assert legacy_mem0_prompt_date("20261103T012529354847Z") == "2026-11-03"
    assert legacy_mem0_prompt_date("20261103T235959354847Z") == "2026-11-04"


@pytest.mark.parametrize("status,message,transient", [
    (403, "model not available in your region", True), (403, "forbidden", False),
    (400, "invalid request", False), (429, "rate limit", True)])
def test_region_refusals_and_invalid_requests(status, message, transient):
    failure = "upstream_region_error" if "region" in message else "upstream_http_error"
    assert is_transient(status, failure) is transient
    assert retryable_error({"message": f"HTTP {status}: {message}"}) is transient
    if status == 403 and transient:
        assert backoff_seconds(1, status) == 30


def test_worker_preserves_region_classification_and_stops_after_error():
    seen = []
    network = GatewayNetwork("http://127.0.0.1:8123", httpx.MockTransport(
        lambda request: seen.append(request) or httpx.Response(403, json={"error": {"message": "region refusal"}})))
    network.client().post("https://api.openai.com/v1/embeddings", json={})
    with pytest.raises(RuntimeError) as error:
        network.check()
    assert retryable_error({"message": str(error.value)})
    with pytest.raises(RuntimeError):
        network.client().post("https://api.openai.com/v1/embeddings", json={})
    assert len(seen) == 1


def checkpoint_gateway(tmp_path):
    gateway = Gateway("dummy-test-key", tmp_path)
    gateway.run_dir.mkdir(exist_ok=True)
    gateway._open_checkpoints()
    gateway.set_context({"task_id": "t", "run_id": "r", "unit_id": "u", "stage": "ingestion", "protocol_sha256": "p"})
    return gateway


@pytest.mark.parametrize("state,cost,request_id,safe", [
    ("failed", 0, None, True), ("failed", None, None, False),
    ("failed", .01, "gen-paid", False), ("completed", 0, None, False),
    ("uncertain", 0, None, False), ("inflight", 0, None, False)])
def test_changed_checkpoint_only_archives_proven_unbilled_failures(tmp_path, state, cost, request_id, safe):
    gateway = checkpoint_gateway(tmp_path)
    original, _ = gateway._checkpoint_call("/path", {"prompt": "yesterday"})
    gateway._checkpoints.execute("INSERT INTO calls(scope,ordinal,request_sha256,call_id,state,reservation_json,completed_json) VALUES (?,?,?,?,?,?,?)",
        (*original, "old-call", state, "{}", json.dumps({"actual_cost_usd": cost, "request_id": request_id, "http_status": 400})))
    gateway._checkpoints.commit()
    gateway._ordinal = 0
    if safe:
        identity, replay = gateway._checkpoint_call("/path", {"prompt": "pinned"})
        assert replay is None and identity[2] != original[2]
        assert gateway._checkpoints.execute("SELECT call_id FROM call_attempts").fetchone()[0] == "old-call"
        assert gateway._checkpoints.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0
    else:
        with pytest.raises(GatewayError, match="request differs"):
            gateway._checkpoint_call("/path", {"prompt": "pinned"})
        assert gateway._checkpoints.execute("SELECT COUNT(*) FROM call_attempts").fetchone()[0] == 0
    gateway._checkpoints.close()


def test_task_store_requeues_region_but_keeps_invalid_and_uncertain_tasks(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    examples = [{"id": str(i), "dataset": "memorybank", "group": str(i), "sessions": []} for i in range(4)]
    store.seed(examples, ["ours", "mem0"], "protocol")
    errors = ["HTTP 400 invalid input", "HTTP 403 region refusal", "week1_call_uncertain", "week1_replay_mismatch"]
    for method in ("ours", "mem0"):
        for i, message in enumerate(errors):
            store.fail(store.build(method, str(i))["task_id"], "ingestion", {"message": message})
    requeued = store.requeue(4)
    assert len(requeued) == 4  # two builds and their blocked questions
    for method in ("ours", "mem0"):
        for i in range(4):
            assert store.build(method, str(i))["state"] == ("pending" if i == 1 else "failed")
    store.close()


def test_permanent_errors_never_pause_and_all_successes_reset_burst():
    streak = FailurePauses()
    assert all(streak.failure(False) == 0 for _ in range(100))
    for _ in range(FAILURE_BURST - 1):
        assert streak.failure(True) == 0
    streak.success()
    assert streak.failure(True) == 0
    waits = [streak.failure(True) for _ in range(1000)]
    assert sum(waits) == 600
    streak.success()
    assert streak.wait(1800) == 0  # success cannot reset the per-run pause allowance


@pytest.mark.parametrize("event", [
    {"actual_cost_usd": None, "http_status": 429},
    {"actual_cost_usd": .001, "http_status": 503},
    {"actual_cost_usd": 0, "http_status": 403, "request_id": "gen-possibly-billed"}])
def test_possibly_billed_errors_are_never_safe_to_resend(event):
    assert not Gateway._safe_to_resend(event)


@pytest.mark.parametrize("status,cost,message,expected_calls", [
    (403, 0, "model not available in your region", 2),
    (403, 0, "forbidden", 1), (400, 0, "invalid input", 1),
    (403, None, "model not available in your region", 1),
    (503, .001, "service failed", 1)])
def test_gateway_retries_only_unbilled_transient_failures(tmp_path, status, cost, message, expected_calls):
    from decimal import Decimal
    gateway = checkpoint_gateway(tmp_path)
    gateway._sleep = lambda seconds: None
    requests = []
    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            result = {"error": {"message": message}}
            if cost is not None:
                result["usage"] = {"cost": cost}
            return httpx.Response(status, json=result)
        return httpx.Response(200, json={"id": "test-success", "choices": [], "usage": {"cost": 0}})
    gateway._client = httpx.Client(transport=httpx.MockTransport(respond))
    gateway.allow_unconfirmed_costs = True
    gateway.pricing_sha256 = "offline"
    gateway._prepare = lambda path, payload: (path, payload, Decimal(".01"), 1)
    try:
        if expected_calls == 2:
            assert gateway._request_serial("/api/v1/chat/completions", {"model": "openai/gpt-4o-mini"})[0] == 200
        else:
            with pytest.raises(GatewayError):
                gateway._request_serial("/api/v1/chat/completions", {"model": "openai/gpt-4o-mini"})
        assert len(requests) == expected_calls
        assert gateway.budget == 20 and gateway.normal_limit == 18
    finally:
        gateway._client.close()
        gateway._checkpoints.close()


@pytest.mark.parametrize("error,expected", [
    ({"message": "HTTP 429 request 400123"}, True),
    ({"message": "request id 503123"}, False),
    ({"status_code": 400, "retryable": True}, False),
    ({"status_code": 403, "message": "region refusal"}, True),
    ({"message": "previous upstream request failed (upstream_region_error); call will not be repeated", "status_code": 403}, False),
    ({"code": "week1_call_uncertain", "retryable": True}, False),
])
def test_task_classification_uses_status_and_never_overrides_unsafe_calls(error, expected):
    assert retryable_error(error) is expected


def test_only_confirmed_permanent_errors_get_permanent_status():
    assert permanent_error({"status_code": 400, "message": "bad request"})
    assert not permanent_error({"message": "unknown worker error"})
    assert not permanent_error({"status_code": 403, "message": "region refusal"})
