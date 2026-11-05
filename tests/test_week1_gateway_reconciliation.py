"""Offline balance reconciliation tests against actual durable gateway events."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.week1_gateway import Gateway, GatewayError, ORIGIN
from test_week1_gateway import body, completion, events, install


def refusal_transport(status=403, generation_id=None, transport_error=httpx.ReadTimeout):
    """One confirmed charge followed by one refusal with unknown billing."""
    paid_requests = 0

    def respond(request):
        nonlocal paid_requests
        if request.method == "GET":
            assert request.url.path == "/api/v1/generation"
            assert request.url.params["id"] == generation_id
            return httpx.Response(404, json={"error": {"message": "generation not available"}})
        paid_requests += 1
        if paid_requests != 2:
            return httpx.Response(200, json=completion(cost=0.001))
        if status is None:
            raise transport_error("unconfirmed transport failure", request=request)
        result = {"error": {"code": status, "message": "refused before generation"}}
        if generation_id is not None:
            result["id"] = generation_id
        return httpx.Response(status, json=result)

    return install(respond)


def stop_after_refusal(gateway):
    gateway._request("/api/v1/chat/completions", body())
    assert gateway.spent_usd == 0.001
    with pytest.raises(GatewayError, match="upstream request failed"):
        gateway._request("/api/v1/chat/completions", body())
    assert gateway.stopped == "week1_cost_unknown"
    assert gateway.summary()["reserved_usd"] > 0
    pending = events(gateway)[-1]
    assert pending["event"] == "completed"
    assert pending["actual_cost_usd"] is None
    return pending


def receipt(gateway, **changes):
    result = {
        "endpoint": ORIGIN + "/api/v1/key",
        "http_status": 200,
        "failed_request_http_status": 403,
        "request_had_generation_id": False,
        "before_remaining_usd": "10.000",
        "after_remaining_usd": "9.999",
        "account_delta_usd": "0.001",
        "ledger_confirmed_usd": "0.001",
        "unexplained_delta_usd": "0",
        "call_id": next(iter(gateway._reservations)),
        "ledger_sha256": hashlib.sha256(gateway.ledger.read_bytes()).hexdigest(),
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    result.update(changes)
    return result


def write_receipt(directory, value):
    path = directory / "balance-receipt.json"
    path.write_text(json.dumps(value) + "\n")
    return path


def assert_rejected_without_changing_ledger(gateway, path):
    before = gateway.ledger.read_bytes()
    reserved = gateway.summary()["reserved_usd"]
    with pytest.raises(ValueError):
        gateway.reconcile_verified_zero(path)
    assert gateway.stopped == "week1_cost_unknown"
    assert gateway.summary()["reserved_usd"] == reserved
    assert gateway.spent_usd == 0.001
    assert gateway.ledger.read_bytes() == before


@pytest.mark.parametrize("numeric_strings", [True, False])
@pytest.mark.parametrize("connect_error", [False, True])
def test_verified_uncharged_failure_reconciliation_survives_restart_and_keeps_spend(
        tmp_path, numeric_strings, connect_error):
    transport, requests = refusal_transport(
        status=None if connect_error else 403, transport_error=httpx.ConnectError)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        failed = stop_after_refusal(gateway)
        value = receipt(gateway)
        if connect_error:
            value.update(failed_request_http_status=None, failed_request_failure="ConnectError")
        if not numeric_strings:
            for field in ("before_remaining_usd", "after_remaining_usd", "account_delta_usd",
                          "ledger_confirmed_usd", "unexplained_delta_usd"):
                value[field] = float(value[field])
        path = write_receipt(tmp_path, value)
        assert failed["http_status"] == (None if connect_error else 403)
        assert failed["request_id"] is None
        before_requests = len(requests)
        gateway.reconcile_verified_zero(path)
        assert len(requests) == before_requests  # Reconciliation itself spends nothing.
        assert gateway.stopped is None
        assert gateway.spent_usd == 0.001
        assert gateway.summary()["reserved_usd"] == 0
        reconciled = events(gateway)[-1]
        assert reconciled["event"] == "reconciled-zero"
        assert reconciled["call_id"] == failed["call_id"]
        assert reconciled["actual_cost_usd"] == 0
        assert reconciled["receipt_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        summary = json.loads((tmp_path / "budget-summary.json").read_text())
        assert summary["spent_usd"] == 0.001
        assert summary["stopped"] is None
        with pytest.raises(ValueError):
            gateway.reconcile_verified_zero(path)  # A receipt cannot be consumed twice.
    with Gateway("private-key", tmp_path, transport=transport) as resumed:
        assert resumed.spent_usd == 0.001
        assert resumed.stopped is None
        assert resumed.summary()["reserved_usd"] == 0
        resumed._request("/api/v1/chat/completions", body())
        assert resumed.spent_usd == 0.002


@pytest.mark.parametrize("transport_error", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError])
def test_connect_error_receipt_cannot_override_another_transport_failure(tmp_path, transport_error):
    transport, _ = refusal_transport(status=None, transport_error=transport_error)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_after_refusal(gateway)
        path = write_receipt(tmp_path, receipt(
            gateway, failed_request_http_status=None, failed_request_failure="ConnectError"))
        assert_rejected_without_changing_ledger(gateway, path)


@pytest.mark.parametrize("changes,missing", [
    ({"failed_request_http_status": 403}, None),
    ({"failed_request_failure": "ReadTimeout"}, None),
    ({}, "failed_request_http_status"),
    ({}, "failed_request_failure"),
    ({"request_had_generation_id": True}, None),
    ({"ledger_sha256": "0" * 64}, None),
    ({"before_remaining_usd": "10.001"}, None),
    ({"audited_at": (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()}, None),
])
def test_connect_error_requires_explicit_matching_failure_and_original_audit_proof(tmp_path, changes, missing):
    transport, _ = refusal_transport(status=None, transport_error=httpx.ConnectError)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_after_refusal(gateway)
        value = receipt(gateway, failed_request_http_status=None, failed_request_failure="ConnectError")
        value.update(changes)
        if missing:
            value.pop(missing)
        path = write_receipt(tmp_path, value)
        assert_rejected_without_changing_ledger(gateway, path)


@pytest.mark.parametrize("missing", [
    "endpoint", "http_status", "failed_request_http_status", "request_had_generation_id",
    "before_remaining_usd", "after_remaining_usd", "account_delta_usd",
    "ledger_confirmed_usd", "unexplained_delta_usd",
    "call_id", "ledger_sha256", "audited_at",
])
def test_incomplete_receipt_cannot_clear_unknown_billing(tmp_path, missing):
    transport, _ = refusal_transport()
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_after_refusal(gateway)
        value = receipt(gateway)
        value.pop(missing)
        path = write_receipt(tmp_path, value)
        assert_rejected_without_changing_ledger(gateway, path)


@pytest.mark.parametrize("changes", [
    {"endpoint": ORIGIN + "/api/v1/credits"},
    {"http_status": 403},
    {"failed_request_http_status": 500},
    {"request_had_generation_id": True},
    {"before_remaining_usd": "10.001"},
    {"after_remaining_usd": "9.998"},
    {"ledger_confirmed_usd": "0.002"},
    {"account_delta_usd": "0.002"},
    {"unexplained_delta_usd": "0.001"},
    {"unexplained_delta_usd": "-0.001"},
    {"after_remaining_usd": "9.998", "account_delta_usd": "0.002", "unexplained_delta_usd": "0.001"},
    {"before_remaining_usd": "NaN"},
    {"after_remaining_usd": "Infinity"},
    {"unexplained_delta_usd": False},
    {"call_id": "unrelated-call"},
    {"ledger_sha256": "0" * 64},
    {"audited_at": (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()},
    {"audited_at": (datetime.now(timezone.utc) + timedelta(minutes=6)).isoformat()},
])
def test_mismatched_or_extra_charge_receipt_cannot_clear_unknown_billing(tmp_path, changes):
    transport, _ = refusal_transport()
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_after_refusal(gateway)
        path = write_receipt(tmp_path, receipt(gateway, **changes))
        assert_rejected_without_changing_ledger(gateway, path)


@pytest.mark.parametrize("status,generation_id", [(500, None), (200, None), (None, None), (403, "gen-refusal")])
def test_receipt_claims_cannot_override_actual_ledger_status_or_generation_id(tmp_path, status, generation_id):
    # Even matching account arithmetic cannot establish a pre-generation 403
    # when the durable provider response says otherwise.
    transport, _ = refusal_transport(status=status, generation_id=generation_id)
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        pending = stop_after_refusal(gateway)
        path = write_receipt(tmp_path, receipt(gateway))
        assert pending["http_status"] == status
        assert pending["request_id"] == generation_id
        assert_rejected_without_changing_ledger(gateway, path)



def test_restore_tolerates_completed_then_reconciled_zero_for_same_call(tmp_path):
    """A ledger that settles a call twice must not KeyError on restart.

    Real run 933deca1… had completed (which cleared the reservation) followed by
    reconciled-zero; hard pop used to crash Gateway before requeue could run.
    """
    ledger = tmp_path / "api-ledger.jsonl"
    call_id = "933deca1-completed-then-reconciled"
    rows = [
        {"event": "reservation", "call_id": call_id, "cost_quoted_upper_usd": 0.01},
        {"event": "completed", "call_id": call_id, "actual_cost_usd": 0.001, "stop_reason": None},
        {"event": "reconciled-zero", "call_id": call_id, "actual_cost_usd": 0,
         "cost_source": "account-balance-reconciliation"},
        {"event": "reservation", "call_id": "still-open", "cost_quoted_upper_usd": 0.02},
        {"event": "completed", "call_id": "uncharged-403", "actual_cost_usd": None,
         "http_status": 403, "stop_reason": None},
        {"event": "reconciled-zero", "call_id": "uncharged-403", "actual_cost_usd": 0},
    ]
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    transport, _ = install(lambda request: pytest.fail("restore must not call paid endpoint"))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        assert gateway.spent_usd == 0.001
        assert call_id not in gateway._reservations
        assert "uncharged-403" not in gateway._reservations
        assert gateway.summary()["reserved_usd"] == 0.02
        assert gateway.stopped == "week1_cost_unknown"


def test_crashed_reservation_without_completed_403_cannot_be_reconciled(tmp_path):
    ledger = tmp_path / "api-ledger.jsonl"
    ledger.write_text(json.dumps({"event": "completed", "call_id": "confirmed",
                                 "actual_cost_usd": 0.001, "stop_reason": None}) + "\n" +
                      json.dumps({"event": "reservation", "call_id": "interrupted",
                                  "cost_quoted_upper_usd": 0.01}) + "\n")
    transport, _ = install(lambda request: pytest.fail("reconciliation must not call paid endpoint"))
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        assert gateway.stopped == "week1_cost_unknown"
        path = write_receipt(tmp_path, receipt(gateway))
        assert_rejected_without_changing_ledger(gateway, path)


def test_receipt_cannot_be_replayed_after_a_second_refusal(tmp_path):
    transport, _ = refusal_transport()
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_after_refusal(gateway)
        path = write_receipt(tmp_path, receipt(gateway))
        original_call = next(iter(gateway._reservations))
        gateway.reconcile_verified_zero(path)
        gateway._client.close()
        gateway._client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(403, json={"error": {"message": "another refusal"}})))
        with pytest.raises(GatewayError):
            gateway._request("/api/v1/chat/completions", body())
        assert next(iter(gateway._reservations)) != original_call
        assert_rejected_without_changing_ledger(gateway, path)


def test_receipt_bound_to_ledger_digest_rejects_append_after_audit(tmp_path):
    transport, _ = refusal_transport()
    with Gateway("private-key", tmp_path, transport=transport) as gateway:
        stop_after_refusal(gateway)
        path = write_receipt(tmp_path, receipt(gateway))
        gateway._append({"event": "rejected", "code": "week1_invalid_request"})
        assert_rejected_without_changing_ledger(gateway, path)
