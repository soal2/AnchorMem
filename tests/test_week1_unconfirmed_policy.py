"""User-authorized unknown charges retain full reservations and budget limits."""

import hashlib
import json
from decimal import Decimal

import httpx
import pytest

from src.week1_gateway import Gateway, GatewayError
from test_week1_gateway import body, completion, events, install


def authorize(gateway, directory, **changes):
    value = {
        "scope": "week1-pilot",
        "user_instruction": "你不需要核查费用，你只需要继续实验即可",
        "budget_usd": 20,
        "ledger_sha256": hashlib.sha256(gateway.ledger.read_bytes()).hexdigest(),
    }
    value.update(changes)
    path = directory / "user-instruction.json"
    path.write_text(json.dumps(value))
    return path


def timed_out(request):
    raise httpx.ReadTimeout("unconfirmed request", request=request)


def stop_unknown(gateway):
    with pytest.raises(GatewayError):
        gateway._request("/api/v1/chat/completions", body())
    assert gateway.stopped == "week1_cost_unknown"


def test_authorized_policy_preserves_unknown_cost_and_restores_after_restart(tmp_path):
    transport, requests = install(timed_out)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_unknown(gateway)
        assert gateway.summary()["allow_unconfirmed_costs"] is False
        unknown = events(gateway)[-1]
        reservation = gateway.summary()["reserved_usd"]
        path = authorize(gateway, tmp_path)
        before_requests = len(requests)
        gateway.enable_unconfirmed_cost_policy(path)
        assert len(requests) == before_requests  # Authorization makes no API requests.
        assert gateway.stopped is None
        assert gateway.allow_unconfirmed_costs
        assert gateway.spent_usd == 0
        assert gateway.summary()["reserved_usd"] == reservation
        assert gateway.summary()["committed_upper_usd"] == reservation
        assert unknown["actual_cost_usd"] is None
        event = events(gateway)[-1]
        assert event["event"] == "unconfirmed-cost-policy-authorized"
        assert event["receipt_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        gateway._client.close()
        gateway._client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=completion(cost=0.001))))
        gateway._request("/api/v1/chat/completions", body())
        assert gateway.spent_usd == 0.001
        assert gateway.summary()["reserved_usd"] == reservation
    with Gateway("private-key", tmp_path, transport=transport) as resumed:
        assert resumed.allow_unconfirmed_costs
        assert resumed.stopped is None
        assert resumed.spent_usd == 0.001
        assert resumed.summary()["reserved_usd"] == reservation
        assert next(event for event in events(resumed) if event.get("call_id") == unknown["call_id"]
                    and event["event"] == "completed")["actual_cost_usd"] is None


def test_future_timeouts_hold_upper_bounds_stop_at_normal_limit_without_retry(tmp_path):
    transport, requests = install(timed_out)
    with Gateway("private-key", tmp_path, normal_limit=0.05, transport=transport) as gateway:
        stop_unknown(gateway)
        gateway.enable_unconfirmed_cost_policy(authorize(gateway, tmp_path))
        first = gateway.summary()["reserved_usd"]
        with pytest.raises(GatewayError) as failure:
            gateway._request("/api/v1/chat/completions", body())
        assert failure.value.code == "week1_upstream_error"
        assert gateway.stopped is None
        assert gateway.summary()["reserved_usd"] == first * 2
        with pytest.raises(GatewayError) as exhausted:
            gateway._request("/api/v1/chat/completions", body())
        assert exhausted.value.code == "week1_budget_exhausted"
        assert gateway.stopped == "week1_budget_exhausted"
        assert len(requests) == 3  # One free pricing GET and two paid attempts, no retries.
        assert gateway.spent_usd == 0
        assert gateway.summary()["reserved_usd"] <= gateway.normal_limit
    with Gateway("private-key", tmp_path, normal_limit=0.05, transport=transport) as resumed:
        assert resumed.allow_unconfirmed_costs
        assert resumed.stopped == "week1_budget_exhausted"


def test_unknown_reservation_counts_against_hard_limit_even_for_reserve_retry(tmp_path):
    transport, requests = install(timed_out)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_unknown(gateway)
        gateway.enable_unconfirmed_cost_policy(authorize(gateway, tmp_path))
        gateway._spent = Decimal("19.97")
        with pytest.raises(GatewayError) as failure:
            gateway._request("/api/v1/chat/completions", body(), retry=True)
        assert failure.value.code == "week1_budget_exhausted"
        assert len(requests) == 2  # Original unknown request plus pricing GET only.
        assert gateway._spent + sum(gateway._reservations.values()) < gateway.budget


@pytest.mark.parametrize("changes", [
    {"scope": "unrelated-run"},
    {"user_instruction": "继续"},
    {"budget_usd": 21},
    {"budget_usd": True},
    {"ledger_sha256": "0" * 64},
])
def test_policy_receipt_requires_exact_user_instruction_scope_and_current_ledger(tmp_path, changes):
    transport, _ = install(timed_out)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_unknown(gateway)
        before = gateway.ledger.read_bytes()
        reserved = gateway.summary()["reserved_usd"]
        with pytest.raises(ValueError):
            gateway.enable_unconfirmed_cost_policy(authorize(gateway, tmp_path, **changes))
        assert gateway.ledger.read_bytes() == before
        assert not gateway.allow_unconfirmed_costs
        assert gateway.stopped == "week1_cost_unknown"
        assert gateway.summary()["reserved_usd"] == reserved


def test_policy_cannot_authorize_a_different_gateway_budget(tmp_path):
    transport, _ = install(timed_out)
    with Gateway("private-key", tmp_path, budget_usd=10, normal_limit=9, transport=transport) as gateway:
        stop_unknown(gateway)
        with pytest.raises(ValueError):
            gateway.enable_unconfirmed_cost_policy(authorize(gateway, tmp_path, budget_usd=10))
        assert gateway.stopped == "week1_cost_unknown"


def test_enabling_policy_preserves_an_existing_budget_stop(tmp_path):
    transport, requests = install(timed_out)
    with Gateway("private-key", tmp_path, normal_limit=0.01, transport=transport) as gateway:
        with pytest.raises(GatewayError):
            gateway._request("/api/v1/chat/completions", body())
        assert gateway.stopped == "week1_budget_exhausted"
        gateway.enable_unconfirmed_cost_policy(authorize(gateway, tmp_path))
        assert gateway.allow_unconfirmed_costs
        assert gateway.stopped == "week1_budget_exhausted"
        assert len(requests) == 1


def test_receipt_cannot_be_reused_or_enable_policy_after_ledger_changed(tmp_path):
    transport, _ = install(timed_out)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_unknown(gateway)
        path = authorize(gateway, tmp_path)
        gateway._append({"event": "rejected", "code": "week1_invalid_request"})
        with pytest.raises(ValueError):
            gateway.enable_unconfirmed_cost_policy(path)
        gateway.enable_unconfirmed_cost_policy(authorize(gateway, tmp_path))
        with pytest.raises(ValueError):
            gateway.enable_unconfirmed_cost_policy(authorize(gateway, tmp_path))
