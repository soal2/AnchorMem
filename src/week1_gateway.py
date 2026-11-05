"""Serial, task-wide OpenRouter budget gate for the first-week experiment.

Only this process holds the real key. Workers use a dummy key with ``base_url``.
Reservations are fsynced before network I/O; unpriced requests stop paid calls
unless the user authorizes keeping their full upper bounds in the budget.
"""

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass

import httpx
import tiktoken


ORIGIN = "https://openrouter.ai"
MODELS_URL = ORIGIN + "/api/v1/models?output_modalities=all"
MODEL_PATHS = {
    "openai/gpt-4o-mini": "/api/v1/chat/completions",
    "openai/gpt-4o-2024-08-06": "/api/v1/chat/completions",
    "openai/text-embedding-3-small": "/api/v1/embeddings",
    "typesafe/jev-1.13": "/api/alpha/decisions",
}
SOURCES = {
    "currency": "https://openrouter.ai/docs/faq#credit-and-billing-systems",
    "actual_cost": "https://openrouter.ai/docs/cookbook/administration/usage-accounting",
    "price_cap": "https://openrouter.ai/docs/guides/routing/provider-selection#max-price",
    "jev_input_budget": "https://docs.typesafe.ai/models",
}


# Transient upstream failures are retried with bounded exponential backoff.
# Invalid requests (400) are permanent; region refusals (403) are transient.
TRANSIENT_STATUSES = frozenset({402, 408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})
TRANSPORT_FAILURES = frozenset({"ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
                                "RemoteProtocolError", "ReadError", "WriteError"})
# Failures that never reached a billable generation.
UNSENT_FAILURES = frozenset({"ConnectError", "ConnectTimeout"})
MAX_CALL_ATTEMPTS = 6
MAX_BAD_REQUEST_ATTEMPTS = 2
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 300.0


def backoff_seconds(attempt, status=None):
    """Delay before retry number ``attempt`` (1-based); credit/rate errors wait longer."""
    base = BACKOFF_BASE_SECONDS * (6 if status in (402, 403, 429) else 1)
    return min(BACKOFF_MAX_SECONDS, base * (2 ** (attempt - 1)))


def is_transient(status, failure):
    if status == 403:
        return failure == "upstream_region_error"
    if status is not None and not 200 <= status < 300:
        return status in TRANSIENT_STATUSES
    # Only connections that never opened are resent; a request that may have
    # reached the provider (timeouts, dropped streams) could be billed twice.
    return failure in UNSENT_FAILURES


def attempt_limit(status):
    return MAX_BAD_REQUEST_ATTEMPTS if status == 400 else MAX_CALL_ATTEMPTS


def uncharged(event):
    """Recognize explicit pre-generation HTTP refusals.

    A generic HTTP/region error is insufficient evidence of zero billing.
    Positive cost or a generation ID always prevents this classification.
    """
    if event.get("actual_cost_usd") not in (None, 0) or event.get("request_id"):
        return False
    error = event.get("upstream_error")
    message = error.get("message", "") if isinstance(error, dict) else ""
    region_refusal = (event.get("http_status") == 403
                      and "model is not available in your region" in str(message).lower())
    return event.get("http_status") == 402 or region_refusal


class GatewayError(ValueError):
    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _money(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("cost must be a finite nonnegative number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("cost must be a finite nonnegative number") from error
    if not result.is_finite() or result < 0:
        raise ValueError("cost must be a finite nonnegative number")
    return result


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Gateway:
    def __init__(self, api_key, run_dir, budget_usd=20, normal_limit=18, *, transport=None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("gateway requires the real API key")
        self._api_key = api_key
        self.run_dir = Path(run_dir)
        self.budget = _money(budget_usd)
        self.normal_limit = _money(normal_limit)
        if self.normal_limit > self.budget:
            raise ValueError("normal limit exceeds hard budget")
        self.ledger = self.run_dir / "api-ledger.jsonl"
        self._transport = transport
        self._lock = threading.RLock()
        self._context = {}
        self._ordinal = 0
        self._checkpoints = None
        self._ledger_reservations = set()
        self._ledger_completed = set()
        self._spent = Decimal(0)
        self._reservations = {}
        self.allow_unconfirmed_costs = False
        self.stopped = None
        self._last_failure = (None, None)
        self._last_resend_safe = False
        self.url = None
        self._server = self._thread = self._client = self._file_lock = None

    @property
    def base_url(self):
        return self.url + "/api/v1" if self.url else None

    @property
    def spent_usd(self):
        return float(self._spent)

    @property
    def stop_reason(self):
        return self.stopped

    def set_context(self, context):
        # Identifiers only: never accept prompts, headers, arbitrary diagnostics.
        allowed = {"run_id", "method", "dataset", "group", "group_id", "question", "question_id", "id", "stage",
                   "task_id", "unit_id", "protocol_sha256"}
        if not isinstance(context, dict) or set(context) - allowed:
            raise ValueError("gateway context accepts experiment identifiers only")
        if "task_id" in context and any(not context.get(key) for key in
                ("run_id", "task_id", "unit_id", "stage", "protocol_sha256")):
            raise ValueError("checkpointed calls require run, task, unit, stage and protocol identifiers")
        with self._lock:
            self._context = {key: self._redact(str(value)) for key, value in context.items()}
            self._ordinal = 0

    def summary(self):
        with self._lock:
            return {"spent_usd": float(self._spent), "reserved_usd": float(sum(self._reservations.values())),
                    "committed_upper_usd": float(self._spent + sum(self._reservations.values())),
                    "budget_usd": float(self.budget), "normal_limit_usd": float(self.normal_limit),
                    "allow_unconfirmed_costs": self.allow_unconfirmed_costs,
                    "stopped": self.stopped, "ledger": str(self.ledger),
                    "pricing": str(getattr(self, "pricing_path", self.run_dir / "pricing.json"))}

    def _redact(self, value):
        return value.replace(self._api_key, "[REDACTED]")

    def _append(self, event):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(self._redact(_encoded(event)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _save_summary(self):
        path = self.run_dir / "budget-summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(_encoded(self.summary()) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _restore(self):
        if not self.ledger.exists():
            return
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)  # A corrupt ledger fails closed, never resets spending.
            if event["event"] == "reservation":
                self._ledger_reservations.add(event["call_id"])
                if event["call_id"] not in self._ledger_completed:
                    self._reservations[event["call_id"]] = _money(event["cost_quoted_upper_usd"])
            elif event["event"] == "completed":
                if event["call_id"] in self._ledger_completed:
                    continue
                self._ledger_completed.add(event["call_id"])
                actual = event.get("actual_cost_usd")
                if actual is not None:
                    self._spent += _money(actual)
                    self._reservations.pop(event["call_id"], None)
                elif uncharged(event):
                    self._reservations.pop(event["call_id"], None)
                self.stopped = self._restored_stop(event.get("stop_reason"))
            elif event["event"] == "rejected" and event["code"] == "week1_budget_exhausted":
                self.stopped = event["code"]
            elif event["event"] == "reconciled-zero":
                # Reservation may already be cleared by a prior completed event.
                self._reservations.pop(event["call_id"], None)
                self.stopped = None
            elif event["event"] == "unconfirmed-cost-policy-authorized":
                self.allow_unconfirmed_costs = True
                if self.stopped == "week1_cost_unknown":
                    self.stopped = None
        if self._reservations and not self.allow_unconfirmed_costs:
            self.stopped = "week1_cost_unknown"

    def _open_checkpoints(self):
        self._checkpoints = sqlite3.connect(self.run_dir / "api-call-checkpoints.sqlite3", check_same_thread=False)
        self._checkpoints.row_factory = sqlite3.Row
        self._checkpoints.execute("PRAGMA synchronous=FULL")
        self._checkpoints.execute("""CREATE TABLE IF NOT EXISTS calls (
            scope TEXT NOT NULL, ordinal INTEGER NOT NULL, request_sha256 TEXT NOT NULL,
            call_id TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
            reservation_json TEXT NOT NULL, completed_json TEXT, response_json TEXT,
            status_code INTEGER, PRIMARY KEY (scope, ordinal))""")
        self._checkpoints.execute("""CREATE TABLE IF NOT EXISTS call_attempts (
            scope TEXT NOT NULL, ordinal INTEGER NOT NULL, request_sha256 TEXT NOT NULL,
            call_id TEXT NOT NULL UNIQUE, state TEXT NOT NULL, status_code INTEGER,
            failure TEXT, superseded_at TEXT NOT NULL)""")
        self._checkpoints.commit()
        # SQLite is committed before JSONL. Repair only locally recorded events;
        # recovery never asks an upstream billing or generation endpoint.
        for row in self._checkpoints.execute("SELECT * FROM calls ORDER BY rowid").fetchall():
            self._record_reservation(json.loads(row["reservation_json"]))
            if row["completed_json"]:
                self._record_completion(json.loads(row["completed_json"]))
        self._checkpoints.execute("UPDATE calls SET state='uncertain' WHERE state='inflight'")
        self._checkpoints.commit()
        if self._reservations and not self.allow_unconfirmed_costs:
            self.stopped = "week1_cost_unknown"

    def _record_reservation(self, event):
        call_id = event["call_id"]
        if call_id in self._ledger_reservations:
            return
        self._append({"event": "reservation", **event})
        self._ledger_reservations.add(call_id)
        if call_id not in self._ledger_completed:
            self._reservations[call_id] = _money(event["cost_quoted_upper_usd"])

    def _record_completion(self, event):
        call_id = event["call_id"]
        if call_id in self._ledger_completed:
            return
        self._append({"event": "completed", **event})
        self._ledger_completed.add(call_id)
        actual = event.get("actual_cost_usd")
        if actual is not None:
            self._spent += _money(actual)
            self._reservations.pop(call_id, None)
        elif uncharged(event):
            self._reservations.pop(call_id, None)
        self.stopped = self._restored_stop(event.get("stop_reason"))

    def _restored_stop(self, reason):
        # An unknown cost no longer blocks once every unknown reservation is settled
        # as uncharged or the user authorized conservative reservations.
        if reason == "week1_cost_unknown" and (self.allow_unconfirmed_costs or not self._reservations):
            return None
        return reason

    def _checkpoint_call(self, path, body):
        if "task_id" not in self._context:
            return None, None
        scope = json.dumps(self._context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ordinal = self._ordinal
        self._ordinal += 1
        request_sha256 = hashlib.sha256(json.dumps({"path": path, "body": body}, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        row = self._checkpoints.execute("SELECT * FROM calls WHERE scope=? AND ordinal=?", (scope, ordinal)).fetchone()
        identity = (scope, ordinal, request_sha256)
        if row is None:
            return identity, None
        if row["request_sha256"] != request_sha256:
            event = json.loads(row["completed_json"] or "{}")
            # A failed HTTP response alone does not prove zero billing.
            safe = self._retryable_state(row, event) and self._safe_to_resend(event)
            if safe:
                self._supersede(row)
                return identity, None
            raise GatewayError("week1_replay_mismatch", "request differs from the frozen call checkpoint")
        if row["state"] == "completed":
            self._record_completion(json.loads(row["completed_json"]))
            return identity, (row["status_code"], json.loads(row["response_json"]))
        if self._may_resend(row):
            self._supersede(row)
            return identity, None
        if row["state"] == "failed":
            raise GatewayError("week1_upstream_error", f"previous upstream request failed ({self._row_failure(row)[1]}); call will not be repeated",
                               status_code=row["status_code"] or 400)
        raise GatewayError("week1_call_uncertain", "previous request may have been sent; automatic resend is prohibited; "
                           "this checkpoint requires manual review and blocks only this task")

    def _attempts(self, scope, ordinal):
        return self._checkpoints.execute("SELECT COUNT(*) FROM call_attempts WHERE scope=? AND ordinal=?",
                                         (scope, ordinal)).fetchone()[0] + 1

    def _row_failure(self, row):
        completed = json.loads(row["completed_json"]) if row["completed_json"] else {}
        status = completed.get("http_status", row["status_code"])
        failure = completed.get("failure")
        if status == 403 and "region" in str(completed.get("upstream_error", "")).lower():
            failure = "upstream_region_error"
        return status, failure

    @staticmethod
    def _safe_to_resend(event):
        return (not event.get("request_id")
                and (event.get("actual_cost_usd") == 0 or uncharged(event)
                     or (event.get("actual_cost_usd") is None and event.get("http_status") is None
                         and event.get("failure") in UNSENT_FAILURES)))

    @staticmethod
    def _retryable_state(row, event):
        # Older versions stored even proven-unsent connection errors as uncertain.
        return row["state"] == "failed" or (row["state"] == "uncertain"
            and event.get("http_status") is None and event.get("failure") in UNSENT_FAILURES)

    def _may_resend(self, row):
        """A checkpointed transient failure may be sent again, within the attempt bound."""
        status, failure = self._row_failure(row)
        # Uncertain (possibly sent) calls are never resent automatically.
        event = json.loads(row["completed_json"] or "{}")
        allowed = (self._retryable_state(row, event) and is_transient(status, failure)
                   and self._safe_to_resend(event))
        return allowed and self._attempts(row["scope"], row["ordinal"]) < attempt_limit(status)

    def _supersede(self, row):
        status, failure = self._row_failure(row)
        self._checkpoints.execute("""INSERT INTO call_attempts
            (scope, ordinal, request_sha256, call_id, state, status_code, failure, superseded_at)
            VALUES (?,?,?,?,?,?,?,?)""", (row["scope"], row["ordinal"], row["request_sha256"], row["call_id"],
            row["state"], status, failure, datetime.now(timezone.utc).isoformat()))
        self._checkpoints.execute("DELETE FROM calls WHERE scope=? AND ordinal=?", (row["scope"], row["ordinal"]))
        self._checkpoints.commit()
        self._append({"event": "call-resend", "superseded_call_id": row["call_id"], "http_status": status,
                      "failure": failure, "attempt": self._attempts(row["scope"], row["ordinal"]),
                      "context": dict(self._context)})

    def _reserve_checkpoint(self, identity, event):
        if identity is not None:
            self._checkpoints.execute("""INSERT INTO calls
                (scope, ordinal, request_sha256, call_id, state, reservation_json)
                VALUES (?, ?, ?, ?, 'inflight', ?)""", (*identity, event["call_id"], self._redact(_encoded(event))))
            self._checkpoints.commit()
        self._record_reservation(event)

    def _complete_checkpoint(self, identity, event, result, returnable):
        if identity is not None:
            status = event["http_status"]
            state = "completed" if returnable and status is not None and 200 <= status < 300 else (
                "uncertain" if status is None else "failed")
            self._checkpoints.execute("""UPDATE calls SET state=?, completed_json=?, response_json=?, status_code=?
                WHERE scope=? AND ordinal=?""", (state, self._redact(_encoded(event)),
                    self._redact(_encoded(result)) if state == "completed" else None,
                    200 if state == "completed" else status, identity[0], identity[1]))
            self._checkpoints.commit()
        self._record_completion(event)

    def enable_unconfirmed_cost_policy(self, receipt_path):
        """Keep unknown calls reserved at their full upper bound on user instruction."""
        with self._lock:
            receipt_path = Path(receipt_path)
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
            instruction = "你不需要核查费用，你只需要继续实验即可"
            if (receipt.get("scope") != "week1-pilot"
                    or receipt.get("user_instruction") != instruction
                    or _money(receipt.get("budget_usd")) != self.budget
                    or self.budget != Decimal(20)
                    or receipt.get("ledger_sha256") != hashlib.sha256(self.ledger.read_bytes()).hexdigest()):
                raise ValueError("unconfirmed-cost policy requires the user instruction and current budget ledger")
            if self.allow_unconfirmed_costs:
                raise ValueError("unconfirmed-cost policy is already authorized")
            self._append({"event": "unconfirmed-cost-policy-authorized", "scope": receipt["scope"],
                          "user_instruction": instruction, "budget_usd": float(self.budget),
                          "receipt": str(receipt_path),
                          "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest()})
            self.allow_unconfirmed_costs = True
            if self.stopped == "week1_cost_unknown":
                self.stopped = None
            self._save_summary()

    def reconcile_verified_zero(self, receipt_path):
        """Settle one refusal or connection failure after an explicit balance audit."""
        with self._lock:
            receipt_path = Path(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            if self.stopped != "week1_cost_unknown" or len(self._reservations) != 1:
                raise ValueError("billing reconciliation requires one unknown request")
            call_id = next(iter(self._reservations))
            ledger_bytes = self.ledger.read_bytes()
            events = [json.loads(line) for line in ledger_bytes.splitlines()]
            completed = next((event for event in reversed(events)
                              if event.get("call_id") == call_id and event["event"] == "completed"), None)
            try:
                audited_at = datetime.fromisoformat(receipt["audited_at"])
                completed_at = datetime.fromisoformat(completed["timestamp"])
                now = datetime.now(timezone.utc)
                delta = _money(receipt["before_remaining_usd"]) - _money(receipt["after_remaining_usd"])
                eligible_failure = (completed["http_status"] == 403
                                    or (completed["http_status"] is None
                                        and completed.get("failure") == "ConnectError"
                                        and receipt.get("failed_request_failure") == "ConnectError"))
                valid = (eligible_failure and not completed.get("request_id")
                         and completed["actual_cost_usd"] is None
                         and receipt["failed_request_http_status"] == completed["http_status"]
                         and receipt["call_id"] == call_id
                         and receipt["ledger_sha256"] == hashlib.sha256(ledger_bytes).hexdigest()
                         and completed_at <= audited_at <= now
                         and (now - audited_at).total_seconds() <= 300
                         and _money(receipt["account_delta_usd"]) == delta == self._spent
                         and _money(receipt["unexplained_delta_usd"]) == 0
                         and _money(receipt["ledger_confirmed_usd"]) == self._spent)
            except (KeyError, TypeError, ValueError):
                valid = False
            if (not valid
                    or receipt.get("http_status") != 200
                    or receipt.get("request_had_generation_id") is not False
                    or receipt.get("endpoint") != ORIGIN + "/api/v1/key"):
                raise ValueError("billing receipt does not prove the single failed request was uncharged")
            self._append({"event": "reconciled-zero", "call_id": call_id, "actual_cost_usd": 0,
                "cost_source": "account-balance-reconciliation", "receipt": str(receipt_path),
                "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()})
            self._reservations.pop(call_id, None)
            self.stopped = None
            self._save_summary()

    def _pricing(self):
        response = self._client.get(MODELS_URL)
        response.raise_for_status()
        catalog = response.json()
        if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
            raise ValueError("invalid public model pricing response")
        self.models = {row["id"]: row for row in catalog["data"]
                       if isinstance(row, dict) and row.get("id") in MODEL_PATHS}
        if set(self.models) != set(MODEL_PATHS):
            raise ValueError("required model missing from fresh public pricing")
        for row in self.models.values():
            pricing = row.get("pricing", {})
            _money(pricing["prompt"])
            _money(pricing["completion"])
            _money(pricing.get("request", 0))
            if type(row.get("context_length")) is not int or row["context_length"] <= 0:
                raise ValueError("model pricing has invalid context length")
        snapshot = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "source": MODELS_URL,
                    "currency": "USD", "documentation": SOURCES, "data": list(self.models.values())}
        content = _encoded(snapshot) + "\n"
        self.pricing_sha256 = hashlib.sha256(content.encode()).hexdigest()
        folder = self.run_dir / "pricing"
        folder.mkdir(exist_ok=True)
        self.pricing_path = folder / (self.pricing_sha256 + ".json")
        self.pricing_path.write_text(content, encoding="utf-8")
        (self.run_dir / "pricing.json").write_text(content, encoding="utf-8")

    def __enter__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._file_lock = (self.run_dir / "gateway.lock").open("a")
        try:
            fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._restore()
            self._open_checkpoints()
            transport = self._transport
            if transport is None:
                proxy = getproxies().get("https")
                if proxy_bypass(urlsplit(ORIGIN).hostname):
                    proxy = None
                transport = httpx.HTTPTransport(proxy=proxy, retries=0)
            self._client = httpx.Client(transport=transport,
                                        timeout=180, follow_redirects=False,
                                        headers={"Authorization": "Bearer " + self._api_key})
            self._pricing()
            gateway = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def do_POST(self):
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length <= 0 or length > 8_000_000:
                            raise GatewayError("week1_invalid_request", "invalid request length")
                        payload = json.loads(self.rfile.read(length))
                        status, result = gateway._request(self.path, payload, self.headers.get("X-Week1-Retry") == "1")
                    except (ValueError, TypeError) as error:
                        status, result = getattr(error, "status_code", 400), {
                            "error": {"code": getattr(error, "code", "week1_invalid_request"),
                                      "message": gateway._redact(str(error)), "type": "invalid_request_error"}}
                    data = _encoded(result).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("x-should-retry", "false")
                    self.end_headers()
                    self.wfile.write(data)

            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self._server.daemon_threads = True
            self.url = "http://127.0.0.1:" + str(self._server.server_port)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            self._save_summary()
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)
        if self._client:
            self._client.close()
        if self._checkpoints:
            self._checkpoints.close()
        if self._file_lock:
            self._file_lock.close()

    def _prepare(self, path, payload):
        path = urlsplit(path).path
        if path.startswith("/v1/"):
            path = "/api" + path
        if not isinstance(payload, dict):
            raise GatewayError("week1_invalid_request", "request body must be an object")
        body = dict(payload)
        model = body.get("model")
        if MODEL_PATHS.get(model) != path or model not in self.models:
            raise GatewayError("week1_unpriced_model", "model or endpoint is not part of this experiment")
        if any(key in body for key in ("models", "route", "plugins", "web_search_options", "service_tier",
                                       "reasoning", "audio", "modalities", "prediction", "transforms")):
            raise GatewayError("week1_unbounded_request", "request enables unpriced routing or extra services")
        if body.get("stream") or body.get("n", 1) != 1:
            raise GatewayError("week1_unbounded_request", "only one non-streaming result is supported")
        row = self.models[model]
        context = row["context_length"]
        pricing = row["pricing"]
        prompt_price = _money(pricing["prompt"])
        completion_price = _money(pricing["completion"])
        request_price = _money(pricing.get("request", 0))
        output = 0
        count = 1
        if path.endswith("chat/completions"):
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise GatewayError("week1_invalid_request", "messages are required")
            for message in messages:
                if not isinstance(message, dict):
                    raise GatewayError("week1_invalid_request", "invalid message")
                content = message.get("content")
                if isinstance(content, list) and any(not isinstance(part, dict) or part.get("type") != "text" for part in content):
                    raise GatewayError("week1_unbounded_request", "only text evidence is priced")
                if content is not None and not isinstance(content, (str, list)):
                    raise GatewayError("week1_invalid_request", "invalid message content")
            output = body.get("max_completion_tokens", body.get("max_tokens", 4096))
            if type(output) is not int or output <= 0:
                raise GatewayError("week1_invalid_request", "max completion tokens must be a positive integer")
            output = min(output, 4096)
            body.pop("max_completion_tokens", None)
            body["max_tokens"] = output
            body["temperature"] = 0
            estimated = len(tiktoken.get_encoding("o200k_base").encode(_encoded(body), disallowed_special=())) + 256 + 16 * len(messages)
            if estimated + output > context:
                raise GatewayError("week1_input_too_long", "input exceeds frozen model context; history will not be truncated")
        elif path.endswith("embeddings"):
            inputs = body.get("input")
            inputs = [inputs] if isinstance(inputs, str) else inputs
            if not isinstance(inputs, list) or not inputs or any(not isinstance(item, str) or not item for item in inputs):
                raise GatewayError("week1_invalid_request", "embedding input must contain nonempty text strings")
            encoding = tiktoken.get_encoding("cl100k_base")
            counts = [len(encoding.encode(item, disallowed_special=())) for item in inputs]
            if any(size > context for size in counts) or sum(counts) > 300_000:
                raise GatewayError("week1_input_too_long", "embedding input exceeds token limit; history will not be truncated")
            count = len(inputs)
            estimated = sum(counts)
            body["dimensions"] = 1024
            body["encoding_format"] = "float"
        else:
            questions = body.get("questions")
            if not isinstance(questions, dict) or not questions or "state" not in body:
                raise GatewayError("week1_invalid_request", "decisions require state and named questions")
            # Jev has a different tokenizer. Byte counts are a conservative local
            # length check; the quote uses the documented whole 64k token budget.
            state_bytes = len(_encoded(body["state"]).encode())
            question_bytes = [len(_encoded(question).encode()) + len(key.encode()) + 64 for key, question in questions.items()]
            estimated = state_bytes + sum(question_bytes) + 256
            if estimated > min(context, 64000) or state_bytes + max(question_bytes) + 256 > 32000:
                raise GatewayError("week1_input_too_long", "decisions exceed conservative input bound; no truncation")
            if completion_price:
                raise GatewayError("week1_unbounded_request", "Jev output pricing changed; no documented output cap")
        if not path.endswith("decisions"):
            provider = dict(body.get("provider") or {})
            provider["allow_fallbacks"] = False
            provider["require_parameters"] = True
            provider["max_price"] = {"prompt": float(prompt_price * 1_000_000),
                                      "completion": float(completion_price * 1_000_000), "request": float(request_price)}
            body["provider"] = provider
        upper = prompt_price * context * count + completion_price * output + request_price
        return path, body, upper, estimated

    def _actual_cost(self, result):
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict) and "cost" in usage:
            return _money(usage["cost"]), "usage.cost"
        if "task_id" in self._context:
            # Task recovery uses only locally persisted responses and the
            # existing conservative reservation policy, never a billing audit.
            raise ValueError("checkpointed response did not supply usage.cost")
        request_id = result.get("id") if isinstance(result, dict) else None
        if isinstance(request_id, str) and request_id:
            response = self._client.get(ORIGIN + "/api/v1/generation", params={"id": request_id})
            response.raise_for_status()
            record = response.json()["data"]
            return _money(record["total_cost"]), "generation.total_cost"
        raise ValueError("no billed cost or generation identifier was supplied")

    def _request(self, path, payload, retry=False):
        with self._lock:
            try:
                return self._request_serial(path, payload, retry)
            except (sqlite3.Error, OSError) as error:
                self.stopped = "week1_checkpoint_error"
                raise GatewayError(self.stopped, "shared call checkpoint could not be read or committed") from error

    def _request_serial(self, path, payload, retry=False):
        with self._lock:
            path, body, upper, estimated = self._prepare(path, payload)
            identity, replay = self._checkpoint_call(path, body)
            if replay is not None:
                return replay
            attempt = self._attempts(*identity[:2]) if identity is not None else 1
            while True:
                try:
                    return self._send(path, body, upper, estimated, identity, retry)
                except GatewayError as error:
                    status, failure = self._last_failure
                    if (getattr(error, "code", None) != "week1_upstream_error" or not is_transient(status, failure)
                            or not self._last_resend_safe or attempt >= attempt_limit(status) or self.stopped):
                        raise
                    wait = backoff_seconds(attempt, status)
                    self._append({"event": "transient-retry", "http_status": status, "failure": failure,
                                  "attempt": attempt, "wait_seconds": wait, "context": dict(self._context)})
                    if identity is not None:
                        row = self._checkpoints.execute("SELECT * FROM calls WHERE scope=? AND ordinal=?",
                                                        identity[:2]).fetchone()
                        if row is not None:
                            self._supersede(row)
                    self._sleep(wait)
                    attempt += 1

    def _sleep(self, seconds):
        time.sleep(seconds)

    def _send(self, path, body, upper, estimated, identity, retry):
        self._last_failure = (None, None)
        self._last_resend_safe = False
        with self._lock:
            if self.stopped and not (retry and self.stopped == "week1_budget_exhausted"):
                raise GatewayError(self.stopped, "paid requests stopped: " + self.stopped)
            ceiling = self.budget if retry else self.normal_limit
            if self._spent + sum(self._reservations.values()) + upper > ceiling:
                self.stopped = "week1_budget_exhausted"
                self._append({"event": "rejected", "code": self.stopped, "context": dict(self._context),
                              "cost_quoted_upper_usd": float(upper), "retry_reserve": retry})
                self._save_summary()
                raise GatewayError(self.stopped, "request cost upper bound would exceed the authorized budget")
            call_id = uuid.uuid4().hex
            event = {"call_id": call_id, "endpoint": path, "model": body["model"],
                     "context": dict(self._context), "cost_quoted_upper_usd": float(upper),
                     "input_tokens_estimated": estimated, "pricing_sha256": self.pricing_sha256,
                     "retry_reserve": retry}
            self._reserve_checkpoint(identity, event)
            started = time.perf_counter()
            actual = None
            result = None
            status = None
            cost_source = None
            failure = None
            returnable = False
            try:
                response = self._client.post(ORIGIN + path, json=body)
                status = response.status_code
                if not 200 <= status < 300:
                    failure = "upstream_http_error"
                result = response.json()
                _encoded(result)  # Reject non-JSON values before writing an invalid proxy response.
                returnable = isinstance(result, dict) and "error" not in result
                if not returnable:
                    failure = "upstream_http_error"
                if (status == 403 and isinstance(result, dict)
                        and "region" in str(result.get("error", "")).lower()):
                    failure = "upstream_region_error"
                actual, cost_source = self._actual_cost(result)
            except Exception as error:
                failure = failure or type(error).__name__  # No exception text: it can contain secrets or prompts.
            if actual is None:
                stop_reason = None if self.allow_unconfirmed_costs else "week1_cost_unknown"
            else:
                stop_reason = self.stopped
                if actual > upper or self._spent + actual > self.budget:
                    stop_reason = "week1_cost_bound_violated"
                elif retry and self._spent + actual < self.normal_limit:
                    stop_reason = None
            usage = result.get("usage") if isinstance(result, dict) else None
            safe_usage = {key: value for key, value in (usage or {}).items()
                          if key in {"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens", "cost"}
                          and type(value) in (int, float) and math.isfinite(value)} if isinstance(usage, dict) else None
            upstream_error = None
            if isinstance(result, dict) and isinstance(result.get("error"), dict):
                upstream_error = {key: self._redact(str(result["error"][key]))[:4000]
                                  for key in ("code", "message") if key in result["error"]}
            completed = {**event, "actual_cost_usd": float(actual) if actual is not None else None,
                          "cost_status": "confirmed" if actual is not None else "unknown", "cost_source": cost_source,
                          "usage": safe_usage, "latency_seconds": time.perf_counter() - started,
                          "http_status": status, "resolved_model": result.get("model") if isinstance(result, dict) else None,
                          "request_id": result.get("id") if isinstance(result, dict) else None,
                          "failure": failure, "upstream_error": upstream_error, "stop_reason": stop_reason}
            self._complete_checkpoint(identity, completed, result, returnable)
            self._save_summary()
            if status is not None and 200 <= status < 300 and returnable:
                # Keep a successful paid response even when its billing is unknown.
                # Later calls stop unless the user authorized conservative reservations.
                return 200, json.loads(self._redact(_encoded(result))) if identity is not None else result
            self._last_failure = (status, failure)
            self._last_resend_safe = self._safe_to_resend(completed)
            raise GatewayError(self.stopped or "week1_upstream_error", f"upstream request failed ({failure})",
                               status_code=status if status is not None and not 200 <= status < 300 else 400)
