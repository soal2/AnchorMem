import hashlib
import json
import sqlite3

import httpx
import pytest

from src.week1_gateway import Gateway, GatewayError, MAX_CALL_ATTEMPTS, MODELS_URL


PATH = "/api/v1/chat/completions"


def catalog():
    prices = {
        "openai/gpt-4o-mini": (128000, "0.00000015", "0.0000006"),
        "openai/gpt-4o-2024-08-06": (128000, "0.0000025", "0.00001"),
        "openai/text-embedding-3-small": (8192, "0.00000002", "0"),
        "typesafe/jev-1.13": (64000, "0.000000042", "0"),
    }
    return {"data": [{"id": model, "context_length": size,
                      "pricing": {"prompt": prompt, "completion": output}}
                     for model, (size, prompt, output) in prices.items()]}


def payload(text="complete legal history", **kwargs):
    return {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": text}], **kwargs}


def response(text="saved original answer", cost=0.001):
    return {"id": "gen-frozen", "choices": [{"message": {"role": "assistant", "content": text},
                "finish_reason": "stop", "index": 0}], "usage": {"cost": cost}}


def context(**kwargs):
    return {"run_id": "new-frozen-run", "task_id": "build:ours:history-1", "unit_id": "add:session-1:0",
            "stage": "ingestion", "protocol_sha256": "frozen-protocol-sha", **kwargs}


def transport(handler=None):
    paid = []

    def answer(request):
        if str(request.url) == MODELS_URL:
            return httpx.Response(200, json=catalog())
        assert request.method == "POST", "recovery must not query billing or generation endpoints"
        paid.append(json.loads(request.content))
        return handler(request) if handler else httpx.Response(200, json=response())

    return httpx.MockTransport(answer), paid


def events(directory):
    return [json.loads(line) for line in (directory / "api-ledger.jsonl").read_text().splitlines()]


def permit_unknown(gateway):
    receipt = gateway.run_dir / "authorized-unknown-costs.json"
    receipt.write_text(json.dumps({"scope": "week1-pilot", "user_instruction": "你不需要核查费用，你只需要继续实验即可",
        "budget_usd": 20, "ledger_sha256": hashlib.sha256(gateway.ledger.read_bytes()).hexdigest()}))
    gateway.enable_unconfirmed_cost_policy(receipt)


def test_restart_replays_successful_response_and_each_ordinal_without_upstream_or_duplicate_charge(tmp_path):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        first = gateway._request(PATH, payload("first"))
        second = gateway._request(PATH, payload("second"))
        assert gateway.spent_usd == .002
    with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
        resumed.set_context(context())
        assert resumed._request(PATH, payload("first")) == first
        assert resumed._request(PATH, payload("second")) == second
        assert resumed.spent_usd == .002
    assert len(paid) == 2
    assert [row["event"] for row in events(tmp_path)] == ["reservation", "completed"] * 2


def test_response_durable_before_ledger_completion_and_recovery_repairs_once(tmp_path, monkeypatch):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        def interrupted(_):
            raise RuntimeError("process interrupted before ledger completion")
        monkeypatch.setattr(gateway, "_record_completion", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            gateway._request(PATH, payload())
    assert [row["event"] for row in events(tmp_path)] == ["reservation"]
    for _ in range(2):
        with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
            resumed.set_context(context())
            assert resumed._request(PATH, payload()) == (200, response())
            assert resumed.spent_usd == .001
    assert len(paid) == 1
    assert [row["event"] for row in events(tmp_path)] == ["reservation", "completed"]


def test_inflight_is_uncertain_on_restart_and_only_blocks_its_own_task(tmp_path, monkeypatch):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt
        monkeypatch.setattr(gateway._client, "post", interrupted)
        with pytest.raises(KeyboardInterrupt):
            gateway._request(PATH, payload())
    with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
        permit_unknown(resumed)
        resumed.set_context(context())
        with pytest.raises(GatewayError) as failure:
            resumed._request(PATH, payload())
        assert failure.value.code == "week1_call_uncertain"
        assert resumed.stopped is None
        resumed.set_context(context(task_id="build:rag:history-1"))
        assert resumed._request(PATH, payload())[0] == 200
        assert resumed.summary()["reserved_usd"] > 0
    assert len(paid) == 1


def test_reservation_durable_before_jsonl_and_network_send(tmp_path, monkeypatch):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        def interrupted(_):
            raise RuntimeError("interrupted before reservation JSONL")
        monkeypatch.setattr(gateway, "_record_reservation", interrupted)
        with pytest.raises(RuntimeError, match="reservation JSONL"):
            gateway._request(PATH, payload())
    with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
        resumed.set_context(context())
        with pytest.raises(GatewayError) as failure:
            resumed._request(PATH, payload())
        assert failure.value.code == "week1_call_uncertain"
        assert resumed.summary()["reserved_usd"] > 0
    assert paid == []
    assert [row["event"] for row in events(tmp_path)] == ["reservation"]


def test_task_unit_and_protocol_isolate_equal_payloads(tmp_path):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        for identity in (context(), context(unit_id="add:session-1:20"),
                         context(task_id="build:mem0:history-1"), context(protocol_sha256="refrozen-protocol-sha")):
            gateway.set_context(identity)
            gateway._request(PATH, payload())
        gateway.set_context(context())
        gateway._request(PATH, payload())
    assert len(paid) == 4


def test_prepared_request_hash_is_canonical_but_changed_content_blocks_only_task(tmp_path):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        original = gateway._request(PATH, payload(max_tokens=5000, temperature=.9))
        gateway.set_context(context())
        assert gateway._request(PATH, payload(temperature=0, max_tokens=4096)) == original
        gateway.set_context(context())
        with pytest.raises(GatewayError) as failure:
            gateway._request(PATH, payload("changed source", max_tokens=4096))
        assert failure.value.code == "week1_replay_mismatch"
        assert gateway.stopped is None
        gateway.set_context(context(task_id="qa:rag:another-question"))
        gateway._request(PATH, payload())
    assert len(paid) == 2


@pytest.mark.parametrize("stop", ["week1_budget_exhausted", "week1_cost_unknown"])
def test_saved_success_replays_even_when_new_paid_calls_are_stopped(tmp_path, stop):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        original = gateway._request(PATH, payload())
        gateway.stopped = stop
        gateway.set_context(context())
        assert gateway._request(PATH, payload()) == original
        gateway.set_context(context(task_id="qa:another-task"))
        with pytest.raises(GatewayError) as failure:
            gateway._request(PATH, payload())
        assert failure.value.code == stop
    assert len(paid) == 1


@pytest.mark.parametrize("has_generation_id", [False, True])
def test_success_with_unknown_cost_can_be_replayed_without_generation_lookup(tmp_path, has_generation_id):
    result = response()
    if not has_generation_id:
        result.pop("id")
    result["usage"].pop("cost")
    network, paid = transport(lambda request: httpx.Response(200, json=result))
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        assert gateway._request(PATH, payload()) == (200, result)
        assert gateway.stopped == "week1_cost_unknown"
    with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
        resumed.set_context(context())
        assert resumed._request(PATH, payload()) == (200, result)
        assert resumed.stopped == "week1_cost_unknown"
    assert len(paid) == 1


def test_http_failure_is_durable_and_retries_are_bounded_across_restarts(tmp_path):
    network, paid = transport(lambda request: httpx.Response(429, json={"error": {"message": "retry later"},
                                                                                 "usage": {"cost": 0}}))
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        for _ in range(2):
            gateway.set_context(context())
            with pytest.raises(GatewayError) as failure:
                gateway._request(PATH, payload())
            assert failure.value.status_code == 429
        assert gateway.stopped is None
    assert len(paid) == MAX_CALL_ATTEMPTS
    with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
        resumed.set_context(context())
        with pytest.raises(GatewayError, match="will not be repeated"):
            resumed._request(PATH, payload())
    assert len(paid) == MAX_CALL_ATTEMPTS


def test_checkpointed_transient_failure_is_resent_on_resume(tmp_path, monkeypatch):
    from src import week1_gateway
    monkeypatch.setattr(week1_gateway, "MAX_CALL_ATTEMPTS", 2)
    state = {"fail": True}

    def respond(request):
        if state["fail"]:
            return httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
        return httpx.Response(200, json=response())

    network, paid = transport(respond)
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        with pytest.raises(GatewayError):
            gateway._request(PATH, payload())
    failed = len(paid)
    assert failed == 2
    monkeypatch.setattr(week1_gateway, "MAX_CALL_ATTEMPTS", 3)
    state["fail"] = False
    with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
        resumed.set_context(context())
        status, result = resumed._request(PATH, payload())
        assert status == 200
        assert resumed.summary()["reserved_usd"] == 0
    assert len(paid) == failed + 1
    with sqlite3.connect(tmp_path / "api-call-checkpoints.sqlite3") as database:
        assert database.execute("SELECT state FROM calls").fetchall() == [("completed",)]
        assert database.execute("SELECT COUNT(*) FROM call_attempts").fetchone()[0] == failed


def test_response_and_diagnostics_never_persist_real_key_or_request_body(tmp_path):
    network, paid = transport(lambda request: httpx.Response(200, json=response(text="real-secret-key")))
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context(task_id="build:real-secret-key"))
        result = gateway._request(PATH, payload("source containing real-secret-key"))
        assert result[1]["choices"][0]["message"]["content"] == "[REDACTED]"
    with sqlite3.connect(tmp_path / "api-call-checkpoints.sqlite3") as database:
        saved = str(database.execute("SELECT * FROM calls").fetchall())
        assert "real-secret-key" not in saved
        assert "source containing" not in saved
    assert "real-secret-key" not in (tmp_path / "api-ledger.jsonl").read_text()


def test_duplicate_completion_ledger_entry_does_not_double_charge_on_restart(tmp_path):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        gateway._request(PATH, payload())
    ledger = tmp_path / "api-ledger.jsonl"
    with ledger.open("a") as stream:
        stream.write(json.dumps(events(tmp_path)[-1]) + "\n")
    with Gateway("real-secret-key", tmp_path, transport=network) as resumed:
        assert resumed.spent_usd == .001
        resumed.set_context(context())
        resumed._request(PATH, payload())
    assert len(paid) == 1


def test_task_call_requires_complete_checkpoint_identity(tmp_path):
    network, _ = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        with pytest.raises(ValueError, match="identifiers"):
            gateway.set_context({"task_id": "build:ours:history-1"})


def test_shared_checkpoint_damage_stops_new_paid_calls(tmp_path):
    network, paid = transport()
    with Gateway("real-secret-key", tmp_path, transport=network) as gateway:
        gateway.set_context(context())
        gateway._checkpoints.execute("DROP TABLE calls")
        gateway._checkpoints.commit()
        with pytest.raises(GatewayError) as failure:
            gateway._request(PATH, payload())
        assert failure.value.code == gateway.stopped == "week1_checkpoint_error"
    assert paid == []


@pytest.mark.parametrize('state,status,failure,message,cost,request_id,safe', [
    ('failed', 403, 'upstream_http_error', 'This model is not available in your region.', None, None, True),
    ('failed', 403, 'upstream_region_error', 'This model is not available in your region.', None, None, True),
    ('uncertain', None, 'ConnectError', None, None, None, True),
    ('failed', 403, 'upstream_region_error', 'region provider error', None, None, False),
    ('failed', 403, 'upstream_http_error', 'This model is not available in your region.', .01, None, False),
    ('failed', 403, 'upstream_http_error', 'This model is not available in your region.', None, 'gen-id', False),
    ('uncertain', None, 'RemoteProtocolError', None, None, None, False),
    ('uncertain', None, 'ReadTimeout', None, None, None, False),
    ('uncertain', None, 'ConnectError', None, .01, None, False),
])
@pytest.mark.parametrize('changed', [False, True])
def test_legacy_checkpoint_refusals_and_uncertain_calls_are_isolated(
        tmp_path, state, status, failure, message, cost, request_id, safe, changed):
    """Real shared-DB shape: unknown costs, old failure label, NULL response.

    Only synthetic records are written; no running artifacts are used by tests.
    """
    network, paid = transport()
    with Gateway('dummy-test-key', tmp_path, transport=network) as gateway:
        gateway.set_context(context(task_id='build:mem0:cn:焦彦'))
        path, body, _, _ = gateway._prepare(PATH, payload())
        identity, _ = gateway._checkpoint_call(path, body)
        reservation = {'call_id': 'legacy-call', 'context': dict(gateway._context),
                       'cost_quoted_upper_usd': .02}
        completed = {**reservation, 'actual_cost_usd': cost, 'http_status': status,
                     'failure': failure, 'request_id': request_id, 'cost_status': 'unknown',
                     'stop_reason': None, 'upstream_error': {'code': str(status), 'message': message}
                     if message else None}
        gateway._checkpoints.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)',
            (*identity, 'legacy-call', state, json.dumps(reservation), json.dumps(completed), None, status))
        gateway._checkpoints.commit()
    with Gateway('dummy-test-key', tmp_path, transport=network) as resumed:
        if safe and status == 403:
            assert resumed.stopped is None
            assert resumed.summary()['reserved_usd'] == 0
        permit_unknown(resumed)
        resumed.set_context(context(task_id='build:mem0:cn:焦彦'))
        if safe:
            assert resumed._request(PATH, payload('revised' if changed else 'complete legal history'))[0] == 200
            assert resumed.summary()['reserved_usd'] == (.02 if status is None else 0)
        else:
            with pytest.raises(GatewayError) as error:
                resumed._request(PATH, payload('revised' if changed else 'complete legal history'))
            assert error.value.code == ('week1_replay_mismatch' if changed else
                'week1_call_uncertain' if state == 'uncertain' else 'week1_upstream_error')
            if state == 'uncertain' and not changed:
                assert 'manual review' in str(error.value) and 'only this task' in str(error.value)
        assert resumed.stopped is None
        assert resumed.budget == 20
        resumed.set_context(context(task_id='build:mem0:cn:other'))
        assert resumed._request(PATH, payload())[0] == 200
        archived = resumed._checkpoints.execute('SELECT call_id FROM call_attempts').fetchall()
        assert [row[0] for row in archived] == (['legacy-call'] if safe else [])
    assert len(paid) == (2 if safe else 1)
