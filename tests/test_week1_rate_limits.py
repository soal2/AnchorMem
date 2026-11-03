"""The serial gateway preserves HTTP errors and retries transient ones with bounded backoff."""

import json
from decimal import Decimal

import httpx
import pytest
from openai import OpenAI, RateLimitError

from src import week1_gateway
from src.week1_gateway import Gateway, MAX_CALL_ATTEMPTS, backoff_seconds
from test_week1_gateway import body, completion, events, install


def upstream_error(status, *, cost=0, message="Upstream rejected the request"):
    value = {"error": {"code": status, "message": message}}
    if cost is not None:
        value["usage"] = {"cost": cost}
    return httpx.Response(status, json=value)


def paid_requests(requests):
    return [request for request in requests if request.method == "POST"]


def test_rate_limit_is_retried_with_backoff_and_identical_payload(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        return upstream_error(429) if attempts == 1 else httpx.Response(200, json=completion())

    transport, requests = install(respond)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway.set_context({"method": "ours", "dataset": "locomo", "stage": "ingestion"})
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body(max_tokens=128))
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Done"
        assert waits == [backoff_seconds(1, 429)]
        calls = paid_requests(requests)
        assert len(calls) == 2
        assert json.loads(calls[0].content) == json.loads(calls[1].content)
        ledger = events(gateway)
        reservations = [row for row in ledger if row["event"] == "reservation"]
        completed = [row for row in ledger if row["event"] == "completed"]
        retries = [row for row in ledger if row["event"] == "transient-retry"]
        assert len(reservations) == len(completed) == 2
        assert [row["retry_reserve"] for row in reservations] == [False, False]
        assert [row["http_status"] for row in completed] == [429, 200]
        assert len(retries) == 1
        assert retries[0]["attempt"] == 1
        assert retries[0]["wait_seconds"] == backoff_seconds(1, 429)
        assert retries[0]["context"] == {"method": "ours", "dataset": "locomo", "stage": "ingestion"}
        assert gateway.spent_usd == pytest.approx(.0001)


def test_forbidden_status_is_preserved_without_retry_and_secrets_are_redacted(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    transport, requests = install(lambda request: upstream_error(403, message="private-key cannot use this provider"))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 403
        assert response.headers["x-should-retry"] == "false"
        assert len(paid_requests(requests)) == 1
        assert waits == []
        assert "private-key" not in response.text
        assert "private-key" not in gateway.ledger.read_text()
        assert "private-key" not in (tmp_path / "budget-summary.json").read_text()
        assert not any(row["event"] == "transient-retry" for row in events(gateway))
        assert events(gateway)[-1]["http_status"] == 403
        assert "[REDACTED]" in events(gateway)[-1]["upstream_error"]["message"]


def test_persistent_rate_limit_is_returned_as_429_after_bounded_attempts(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    transport, requests = install(lambda request: upstream_error(429))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 429
        assert response.headers["x-should-retry"] == "false"
        assert waits == [backoff_seconds(n, 429) for n in range(1, MAX_CALL_ATTEMPTS)]
        assert waits == sorted(waits)
        calls = paid_requests(requests)
        assert len(calls) == MAX_CALL_ATTEMPTS
        assert len({call.content for call in calls}) == 1
        ledger = events(gateway)
        assert [row["http_status"] for row in ledger if row["event"] == "completed"] == [429] * MAX_CALL_ATTEMPTS
        assert len([row for row in ledger if row["event"] == "transient-retry"]) == MAX_CALL_ATTEMPTS - 1
        assert len([row for row in ledger if row["event"] == "reservation"]) == MAX_CALL_ATTEMPTS


def test_rate_limit_retry_checks_the_twenty_dollar_hard_gate_before_network(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    transport, requests = install(lambda request: upstream_error(429, cost=None))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway.allow_unconfirmed_costs = True  # Authorized policy is tested separately.
        gateway._spent = Decimal("14.7")
        # This valid embedding batch reserves $3.2768: one fits $18, two exceed $20.
        payload = {"model": "openai/text-embedding-3-small", "input": ["x"] * 20_000}
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/embeddings", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "week1_budget_exhausted"
        assert gateway.summary()["normal_limit_usd"] == 18
        assert gateway.summary()["budget_usd"] == 20
        assert gateway.summary()["reserved_usd"] == pytest.approx(3.2768)
        assert gateway.summary()["committed_upper_usd"] == pytest.approx(17.9768)
        assert len(paid_requests(requests)) == 1
        assert waits == [backoff_seconds(1, 429)]
        ledger = events(gateway)
        assert len([row for row in ledger if row["event"] == "reservation"]) == 1
        assert len([row for row in ledger if row["event"] == "completed"]) == 1
        rejected = [row for row in ledger if row["event"] == "rejected"]
        assert len(rejected) == 1
        assert rejected[0]["retry_reserve"] is False
        assert rejected[0]["code"] == "week1_budget_exhausted"
        first = next(row for row in ledger if row["event"] == "completed")
        assert first["http_status"] == 429
        assert first["actual_cost_usd"] is None
        assert first["cost_status"] == "unknown"
        assert gateway.stop_reason == "week1_budget_exhausted"


def test_authorized_unknown_rate_limit_charge_keeps_full_upper_reservation(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        return upstream_error(429, cost=None) if attempts == 1 else httpx.Response(200, json=completion())

    transport, requests = install(respond)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway.allow_unconfirmed_costs = True
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 200
        completed = [row for row in events(gateway) if row["event"] == "completed"]
        assert completed[0]["actual_cost_usd"] is None
        assert completed[0]["cost_status"] == "unknown"
        assert gateway.summary()["reserved_usd"] == completed[0]["cost_quoted_upper_usd"]
        assert gateway.summary()["spent_usd"] == pytest.approx(.0001)
        assert gateway.stop_reason is None
        assert len(paid_requests(requests)) == 2
        assert waits == [backoff_seconds(1, 429)]
        assert all(request.url.path in {"/api/v1/models", "/api/v1/chat/completions"} for request in requests)


def test_retry_cannot_bypass_an_unauthorized_unknown_cost_stop(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    transport, requests = install(lambda request: upstream_error(429, cost=None))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "week1_cost_unknown"
        assert gateway.stop_reason == "week1_cost_unknown"
        assert gateway.summary()["reserved_usd"] > 0
        assert len(paid_requests(requests)) == 1
        assert waits == []
        assert len([row for row in events(gateway) if row["event"] == "reservation"]) == 1


@pytest.mark.parametrize("status,attempts", [(400, 2), (403, 1), (500, MAX_CALL_ATTEMPTS),
                                             (502, MAX_CALL_ATTEMPTS), (503, MAX_CALL_ATTEMPTS)])
def test_http_errors_preserve_status_and_retry_only_transient_ones(tmp_path, monkeypatch, retry_waits, status, attempts):
    waits = retry_waits
    transport, requests = install(lambda request: upstream_error(status))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == status
        assert response.headers["x-should-retry"] == "false"
        assert len(paid_requests(requests)) == attempts
        assert len(waits) == attempts - 1
        assert len([row for row in events(gateway) if row["event"] == "transient-retry"]) == attempts - 1


def test_transport_timeout_is_not_a_rate_limit_or_a_retry_trigger(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits

    def fail(request):
        raise httpx.ReadTimeout("private-key timeout", request=request)

    transport, requests = install(fail)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway.allow_unconfirmed_costs = True
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 400
        assert len(paid_requests(requests)) == 1
        assert waits == []
        assert events(gateway)[-1]["failure"] == "ReadTimeout"
        assert gateway.summary()["reserved_usd"] > 0
        assert "private-key" not in response.text + gateway.ledger.read_text()
        assert not any(row["event"] == "transient-retry" for row in events(gateway))


def test_invalid_model_content_is_preserved_without_a_transport_retry(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    value = completion(choices=[{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "not JSON",
    }}])
    transport, requests = install(lambda request: httpx.Response(200, json=value))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "not JSON"
        assert len(paid_requests(requests)) == 1
        assert waits == []
        assert not any(row["event"] == "transient-retry" for row in events(gateway))


def test_terminal_429_disables_any_additional_sdk_retry(tmp_path, monkeypatch, retry_waits):
    waits = retry_waits
    transport, requests = install(lambda request: upstream_error(429))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with OpenAI(api_key="dummy", base_url=gateway.base_url, max_retries=5,
                    http_client=httpx.Client(trust_env=False)) as client:
            with pytest.raises(RateLimitError) as raised:
                client.chat.completions.create(**body())
        assert raised.value.status_code == 429
        assert raised.value.response.headers["x-should-retry"] == "false"
        assert len(paid_requests(requests)) == MAX_CALL_ATTEMPTS
        assert len(waits) == MAX_CALL_ATTEMPTS - 1


def test_credit_exhaustion_402_waits_and_recovers_without_holding_a_reservation(tmp_path, monkeypatch, retry_waits):
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return upstream_error(402, cost=None, message="Insufficient credits")
        return httpx.Response(200, json=completion())

    transport, requests = install(respond)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 200
        assert retry_waits == [backoff_seconds(1, 402), backoff_seconds(2, 402)]
        assert gateway.summary()["reserved_usd"] == 0
        assert gateway.stop_reason is None
    with Gateway("private-key", tmp_path, transport=transport) as resumed:
        assert resumed.summary()["reserved_usd"] == 0
        assert resumed.stop_reason is None


def test_connect_error_is_retried_but_read_timeout_is_never_resent(tmp_path, monkeypatch, retry_waits):
    attempts = 0

    def respond(request):
        nonlocal attempts
        if request.url.path.endswith("/chat/completions"):
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=completion())

    transport, requests = install(respond)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway.allow_unconfirmed_costs = True
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.base_url + "/chat/completions", json=body())
        assert response.status_code == 200
        assert len(paid_requests(requests)) == 2
        assert len(retry_waits) == 1
