import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import httpx
import pytest
from openai import BadRequestError, OpenAI

from src.week1_gateway import Gateway, GatewayError, MODEL_PATHS, MODELS_URL


def catalog():
    values = {
        "openai/gpt-4o-mini": (128000, "0.00000015", "0.0000006"),
        "openai/gpt-4o-2024-08-06": (128000, "0.0000025", "0.00001"),
        "openai/text-embedding-3-small": (8192, "0.00000002", "0"),
        "typesafe/jev-1.13": (64000, "0.000000042", "0"),
    }
    return {"data": [{"id": model, "context_length": context,
                       "pricing": {"prompt": prompt, "completion": completion}}
                      for model, (context, prompt, completion) in values.items()]}


def completion(cost=0.0001, **extra):
    return {"id": "gen-test", "model": "openai/gpt-4o-mini-2024-07-18", "object": "chat.completion",
            "created": 0, "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": "Done"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": cost}, **extra}


def body(**extra):
    return {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "private prompt"}], **extra}


def install(handler):
    requests = []

    def respond(request):
        assert request.headers["Authorization"] == "Bearer private-key"
        requests.append(request)
        if str(request.url) == MODELS_URL:
            return httpx.Response(200, json=catalog())
        return handler(request)

    return httpx.MockTransport(respond), requests


def events(gateway):
    return [json.loads(line) for line in gateway.ledger.read_text().splitlines()]


def test_proxy_preserves_structured_payload_and_enforces_frozen_parameters(tmp_path):
    transport, requests = install(lambda request: httpx.Response(200, json=completion()))
    schema = {"type": "json_schema", "json_schema": {"name": "facts", "schema": {"type": "object"}}}
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway.set_context({"run_id": "pilot", "method": "amem", "stage": "build"})
        with OpenAI(api_key="dummy-key", base_url=gateway.base_url, max_retries=0,
                    http_client=httpx.Client(trust_env=False)) as client:
            result = client.chat.completions.create(**body(temperature=.9, max_tokens=5000, response_format=schema))
        assert result.choices[0].message.content == "Done"
        forwarded = json.loads(requests[-1].content)
        assert forwarded["messages"] == body()["messages"]
        assert forwarded["response_format"] == schema
        assert forwarded["temperature"] == 0
        assert forwarded["max_tokens"] == 4096
        assert forwarded["provider"]["require_parameters"] is True
        assert forwarded["provider"]["allow_fallbacks"] is False
        assert forwarded["provider"]["max_price"] == {"prompt": .15, "completion": .6, "request": 0}
        assert gateway.spent_usd == .0001
        completed = events(gateway)[-1]
        assert completed["context"]["run_id"] == "pilot"
        assert completed["cost_status"] == "confirmed"
        assert completed["cost_quoted_upper_usd"] > completed["actual_cost_usd"]
        assert "private-key" not in gateway.ledger.read_text()
        assert "private prompt" not in gateway.ledger.read_text()
        assert json.loads((tmp_path / "pricing.json").read_text())["currency"] == "USD"


def test_normal_eighteen_and_explicit_retry_twenty_caps(tmp_path):
    transport, requests = install(lambda request: httpx.Response(200, json=completion()))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway._spent = Decimal("17.99")
        with httpx.Client(trust_env=False) as client:
            response = client.post(gateway.url + "/v1/chat/completions", json=body())
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "week1_budget_exhausted"
            assert len(requests) == 1  # Only free pricing GET.
            retry = client.post(gateway.url + "/v1/chat/completions", json=body(), headers={"X-Week1-Retry": "1"})
            assert retry.status_code == 200
            gateway._spent = Decimal("19.99")
            stopped = client.post(gateway.url + "/v1/chat/completions", json=body(), headers={"X-Week1-Retry": "1"})
            assert stopped.status_code == 400
            assert len(requests) == 2
            assert gateway.stopped == "week1_budget_exhausted"


def test_unknown_cost_keeps_success_then_stops_with_reservation(tmp_path):
    response = completion()
    response.pop("id")
    response["usage"].pop("cost")
    transport, requests = install(lambda request: httpx.Response(200, json=response))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with httpx.Client(trust_env=False) as client:
            first = client.post(gateway.url + "/api/v1/chat/completions", json=body())
            second = client.post(gateway.url + "/api/v1/chat/completions", json=body())
        assert first.status_code == 200
        assert second.status_code == 400
        assert second.json()["error"]["code"] == "week1_cost_unknown"
        assert len(requests) == 2
        assert gateway.spent_usd == 0
        assert gateway.summary()["reserved_usd"] > 0
        assert events(gateway)[-1]["actual_cost_usd"] is None
        assert events(gateway)[-1]["cost_status"] == "unknown"
    with Gateway("private-key", tmp_path, transport=transport) as resumed:
        assert resumed.stopped == "week1_cost_unknown"
        with pytest.raises(GatewayError, match="stopped"):
            resumed._request("/api/v1/chat/completions", body(), retry=True)


def test_generation_lookup_confirms_dollar_cost(tmp_path):
    response = completion()
    response["usage"].pop("cost")

    def respond(request):
        if request.method == "GET":
            assert request.url.path == "/api/v1/generation"
            assert request.url.params["id"] == "gen-test"
            return httpx.Response(200, json={"data": {"total_cost": .0002, "upstream_inference_cost": 999}})
        return httpx.Response(200, json=response)

    transport, requests = install(respond)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway._request("/api/v1/chat/completions", body())
        assert gateway.spent_usd == .0002
        assert len(requests) == 3
        assert events(gateway)[-1]["cost_source"] == "generation.total_cost"


@pytest.mark.parametrize("cost", [None, True, "invalid", -1, float("inf")])
def test_malformed_billed_cost_never_counts_as_zero(tmp_path, cost):
    transport, _ = install(lambda request: httpx.Response(200, content=json.dumps(completion(cost))))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        if cost == float("inf"):
            with pytest.raises(GatewayError):
                gateway._request("/api/v1/chat/completions", body())
        else:
            gateway._request("/api/v1/chat/completions", body())
        assert gateway.stopped == "week1_cost_unknown"
        assert gateway.summary()["reserved_usd"] > 0
        assert events(gateway)[-1]["actual_cost_usd"] is None


def test_transport_failure_has_no_hidden_retry_and_unknown_charge_stops(tmp_path):
    def fail(request):
        raise httpx.ReadTimeout("failure including private-key and private prompt", request=request)

    transport, requests = install(fail)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with OpenAI(api_key="dummy", base_url=gateway.base_url, max_retries=5,
                    http_client=httpx.Client(trust_env=False)) as client:
            with pytest.raises(BadRequestError, match="upstream request failed"):
                client.chat.completions.create(**body())
        assert len(requests) == 2
        assert gateway.stopped == "week1_cost_unknown"
        assert gateway.summary()["reserved_usd"] > 0
        assert "private-key" not in gateway.ledger.read_text()
        assert "private prompt" not in gateway.ledger.read_text()


@pytest.mark.parametrize("path,payload", [
    ("/api/v1/chat/completions", body(model="openai/gpt-4")),
    ("/api/v1/embeddings", body()),
    ("https://evil.invalid/x", body()),
    ("/api/v1/chat/completions", body(plugins=[{"id": "web"}])),
    ("/api/v1/chat/completions", body(stream=True)),
    ("/api/v1/chat/completions", body(n=2)),
    ("/api/v1/chat/completions", body(messages=[{"role": "user", "content": [{"type": "image_url"}]}])),
])
def test_unbounded_or_wrong_model_requests_rejected_before_paid_call(tmp_path, path, payload):
    transport, requests = install(lambda request: pytest.fail("must not call paid endpoint"))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with pytest.raises(GatewayError):
            gateway._request(path, payload)
        assert len(requests) == 1
        assert gateway.spent_usd == 0


def test_embedding_batch_quote_and_parameter_pinning(tmp_path):
    transport, requests = install(lambda request: httpx.Response(200, json={"data": [], "usage": {"cost": .00001}}))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway._request("/api/v1/embeddings", {"model": "openai/text-embedding-3-small", "input": ["hello", "world"], "dimensions": 1536})
        forwarded = json.loads(requests[-1].content)
        assert forwarded["dimensions"] == 1024
        assert forwarded["encoding_format"] == "float"
        assert events(gateway)[-1]["cost_quoted_upper_usd"] == pytest.approx(8192 * 2 * .00000002)


def test_jev_questions_keep_state_once_and_use_whole_request_quote(tmp_path):
    transport, requests = install(lambda request: httpx.Response(200, json={"answers": {}, "usage": {"cost": .000001}}))
    questions = {"q1": {"type": "choice", "criteria": {"yes": "yes", "no": "no"}},
                 "q2": {"type": "noul", "instructions": "valid?"}}
    payload = {"model": "typesafe/jev-1.13", "state": "shared state", "questions": questions}
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        gateway._request("/api/alpha/decisions", payload)
        assert json.loads(requests[-1].content) == payload
        assert events(gateway)[-1]["cost_quoted_upper_usd"] == pytest.approx(64000 * .000000042)


def test_serial_requests_and_restart_cannot_reset_task_spending(tmp_path):
    active = 0
    peak = 0
    lock = threading.Lock()

    def respond(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.03)
        with lock:
            active -= 1
        return httpx.Response(200, json=completion())

    transport, _ = install(respond)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: gateway._request("/api/v1/chat/completions", body()), range(4)))
        assert len(results) == 4
        assert peak == 1
        assert gateway.spent_usd == .0004
    with Gateway("private-key", tmp_path, transport=transport) as resumed:
        assert resumed.spent_usd == .0004
        resumed._request("/api/v1/chat/completions", body())
        assert resumed.spent_usd == .0005


def test_crashed_pending_reservation_and_corrupt_ledger_fail_closed(tmp_path):
    ledger = tmp_path / "api-ledger.jsonl"
    ledger.write_text(json.dumps({"event": "reservation", "call_id": "interrupted", "cost_quoted_upper_usd": .1}) + "\n")
    transport, requests = install(lambda request: pytest.fail("pending call must stop further spending"))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        assert gateway.stopped == "week1_cost_unknown"
        assert gateway.summary()["reserved_usd"] == .1
        with pytest.raises(GatewayError):
            gateway._request("/api/v1/chat/completions", body())
    ledger.write_text("broken\n")
    with pytest.raises(json.JSONDecodeError):
        with Gateway("private-key", tmp_path, transport=transport):
            pass


def test_one_process_owns_task_ledger(tmp_path):
    transport, _ = install(lambda request: pytest.fail("no paid calls"))
    with Gateway("private-key", tmp_path, transport=transport):
        with pytest.raises(BlockingIOError):
            with Gateway("private-key", tmp_path, transport=transport):
                pass


def test_decision_compacts_only_oversize_json_without_losing_evidence(tmp_path):
    transport, _ = install(lambda request: httpx.Response(200, json=completion()))
    with Gateway('private-key', tmp_path, transport=transport) as gateway:
        state = {'facts': [{'text': 'evidence', 'number': i} for i in range(780)]}
        payload = {'model': 'typesafe/jev-1.13', 'state': json.dumps(state),
                   'questions': {'relation': {'type': 'choice', 'criteria': {'a': 'new', 'b': 'old'}}}}
        original = payload['state']
        _, prepared, _, size = gateway._prepare('/api/alpha/decisions', payload)
        assert json.loads(prepared['state']) == state
        assert prepared['state'] != original
        assert size < 32000
        small = {**payload, 'state': json.dumps({'text': 'keep spacing'})}
        assert gateway._prepare('/api/alpha/decisions', small)[1]['state'] == small['state']
        with pytest.raises(GatewayError, match='conservative input bound'):
            gateway._prepare('/api/alpha/decisions', {**payload, 'state': json.dumps({'text': 'x' * 33000})})
