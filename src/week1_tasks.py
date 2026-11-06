"""Durable task and stage receipts for the frozen week-one pilot."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def retryable_error(error):
    """Recognize transient failures, including legacy timeout wrappers."""
    if error.get("deadline_exceeded") is True:
        return True
    text = json.dumps(error, ensure_ascii=False).lower()
    if any(marker in text for marker in ("readtimeout", "attemptdeadlineexceeded", "connecterror", "connecttimeout", "week1_call_uncertain")):
        return True
    if any(marker in text for marker in ("uncertain", "replay_mismatch", "badrequest", "invalid", "finish_reason", "will not be repeated")):
        return False
    status = error.get("status_code")
    if status is None:
        match = re.search(r"\b(?:http(?: status)?|error code)\s*[:=]?\s*(\d{3})\b", text)
        status = int(match[1]) if match else None
    if status == 400:
        return False
    if status == 403:
        return "region" in text
    if "retryable" in error:
        return error["retryable"] is True
    return (isinstance(status, int) and 500 <= status < 600) or status in {402, 408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524} or any(
        marker in text for marker in ("connecterror", "connecttimeout"))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def permanent_error(error):
    """Unknown errors need review; do not label them confirmed invalid requests."""
    if retryable_error(error):
        return False
    text = json.dumps(error, ensure_ascii=False).lower()
    return error.get("status_code") in {400, 401, 403, 404, 422} or any(
        marker in text for marker in ("http 400", "http 401", "http 403", "invalid", "badrequest", "finish_reason"))


def history_digest(example):
    return hashlib.sha256(json_text(example["sessions"]).encode()).hexdigest()


def task_key(kind, method, identity):
    return f"{kind}:{method}:" + hashlib.sha256(identity.encode()).hexdigest()


def ingestion_units(example, method):
    if method not in ("ours", "mem0"):
        return [{"id": "full-history", "example": example}]
    return [{"id": f"{session['id']}:{offset}", "session_id": str(session["id"]),
             "offset": offset, "messages": session["messages"][offset:offset + 20]}
            for session in example["sessions"] for offset in range(0, len(session["messages"]), 20)]


class TaskStore:
    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
          PRAGMA foreign_keys=ON;
          PRAGMA journal_mode=WAL;
          PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, kind TEXT NOT NULL, method TEXT NOT NULL,
            dataset TEXT NOT NULL, group_id TEXT NOT NULL, question_id TEXT,
            position INTEGER NOT NULL, protocol_sha256 TEXT NOT NULL,
            history_sha256 TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
            completed_units INTEGER NOT NULL DEFAULT 0, total_units INTEGER NOT NULL DEFAULT 0,
            index_path TEXT, receipt TEXT, error TEXT);
          CREATE TABLE IF NOT EXISTS stages (
            task_id TEXT NOT NULL REFERENCES tasks(task_id), name TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(task_id,name));
          CREATE TABLE IF NOT EXISTS failures (
            event_id INTEGER PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
            stage TEXT NOT NULL, payload TEXT NOT NULL);
        """)

    def close(self):
        self.connection.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        value = dict(row)
        value["group"] = value.pop("group_id")
        value["id"] = value["question_id"]
        for name in ("receipt", "error"):
            value[name] = json.loads(value[name]) if value[name] else None
        return value

    def seed(self, examples, methods, protocol_sha256):
        expected = set()
        with self.connection:
            for position, example in enumerate(examples):
                for method in methods:
                    digest = history_digest(example)
                    for kind, identity in (("build", example["group"]), ("qa", example["id"])):
                        key = task_key(kind, method, identity)
                        expected.add(key)
                        existing = self.get(key)
                        if existing:
                            if (existing["history_sha256"] != digest
                                    or existing["protocol_sha256"] != protocol_sha256
                                    or existing["dataset"] != example["dataset"]):
                                raise ValueError("frozen task history or protocol changed")
                            continue
                        self.connection.execute("""INSERT INTO tasks
                          (task_id,kind,method,dataset,group_id,question_id,position,protocol_sha256,
                           history_sha256,total_units) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                          (key, kind, method, example["dataset"], example["group"],
                           example["id"] if kind == "qa" else None, position, protocol_sha256, digest,
                           len(ingestion_units(example, method)) if kind == "build" else 0))
            if {row["task_id"] for row in self.rows()} != expected:
                raise ValueError("frozen task selection changed")
            self.connection.execute("UPDATE tasks SET state='pending' WHERE state IN ('running','budget-stopped')")

    def get(self, task_id):
        return self.decode(self.connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    def build(self, method, group):
        return self.get(task_key("build", method, group))

    def question(self, method, question_id):
        return self.get(task_key("qa", method, question_id))

    def rows(self, kind=None):
        query = "SELECT * FROM tasks" + (" WHERE kind=?" if kind else "") + " ORDER BY position,method"
        return [self.decode(row) for row in self.connection.execute(query, (kind,) if kind else ())]

    def stage(self, task_id, name):
        row = self.connection.execute("SELECT payload FROM stages WHERE task_id=? AND name=?", (task_id, name)).fetchone()
        return json.loads(row[0]) if row else None

    def start(self, task_id):
        with self.connection:
            self.connection.execute("UPDATE tasks SET state='running' WHERE task_id=? AND state='pending'", (task_id,))

    def commit_build_unit(self, task_id, unit_number, index_path):
        row = self.get(task_id)
        if row["kind"] != "build" or unit_number != row["completed_units"] + 1 or unit_number > row["total_units"]:
            raise ValueError("build checkpoint is not the next frozen unit")
        with self.connection:
            self.connection.execute("UPDATE tasks SET completed_units=?,index_path=?,state='pending' WHERE task_id=?",
                                    (unit_number, str(index_path), task_id))

    def commit_build(self, task_id, receipt):
        row = self.get(task_id)
        if (row["kind"] != "build" or row["completed_units"] != row["total_units"]
                or receipt.get("history_sha256") != row["history_sha256"]
                or receipt.get("history_complete") is not True or not row["index_path"]):
            raise ValueError("full frozen history must be committed before a build receipt")
        with self.connection:
            self.connection.execute("UPDATE tasks SET state='success',receipt=?,error=NULL WHERE task_id=?",
                                    (json_text(receipt), task_id))

    def require_history(self, task_id):
        task = self.get(task_id)
        build = self.build(task["method"], task["group"])
        if (build["state"] != "success" or not build["receipt"]
                or build["receipt"].get("history_complete") is not True
                or build["receipt"].get("history_sha256") != task["history_sha256"]):
            raise ValueError("question cannot use an incomplete or different history")
        return build

    def commit_stage(self, task_id, name, payload, index_path=None):
        build = self.require_history(task_id)
        if name == "score":
            answer = self.stage(task_id, "answer")
            judge = self.stage(task_id, "judge")
            if (not answer or answer.get("finish_reason") != "stop"
                    or (self.get(task_id)["dataset"] in {"longmemeval", "memorybank"}
                        and (not judge or judge.get("finish_reason") != "stop"))):
                raise ValueError("scoring requires a complete persisted answer and official judge response")
        existing = self.stage(task_id, name)
        if existing is not None:
            if existing != payload:
                raise ValueError("completed task stage cannot be overwritten")
            return
        with self.connection:
            self.connection.execute("INSERT INTO stages VALUES (?,?,?)", (task_id, name, json_text(payload)))
            if index_path is not None:
                self.connection.execute("UPDATE tasks SET index_path=? WHERE task_id=?", (str(index_path), build["task_id"]))
            self.connection.execute("UPDATE tasks SET state=? WHERE task_id=?",
                                    ("success" if name == "score" else "pending", task_id))

    def fail(self, task_id, stage, error, budget_stop=None):
        task = self.get(task_id)
        payload = {"method": task["method"], "dataset": task["dataset"], "group": task["group"],
                   "id": task["id"], "task_id": task_id, "stage": stage,
                   "time": datetime.now(timezone.utc).isoformat(),
                   "history_complete": self.build(task["method"], task["group"])["state"] == "success",
                   "budget_stop": budget_stop, **error}
        state = "budget-stopped" if budget_stop else "failed"
        with self.connection:
            self.connection.execute("UPDATE tasks SET state=?,error=? WHERE task_id=?", (state, json_text(payload), task_id))
            self.connection.execute("INSERT INTO failures(task_id,stage,payload) VALUES (?,?,?)",
                                    (task_id, stage, json_text(payload)))
            if task["kind"] == "build" and not budget_stop:
                self.connection.execute("UPDATE tasks SET state='blocked',error=? WHERE kind='qa' AND method=? AND group_id=? AND state='pending'",
                                        (json_text({"dependency": task_id}), task["method"], task["group"]))
        return payload

    def attempts(self, task_id):
        return self.connection.execute("SELECT COUNT(*) FROM failures WHERE task_id=?", (task_id,)).fetchone()[0]

    def recovery_evidence(self, checkpoint_path):
        """Read billing evidence without initializing or modifying the gateway DB."""
        from .week1_gateway import Gateway, is_transient
        result = {}
        if checkpoint_path is None or not Path(checkpoint_path).is_file():
            return result
        connection = sqlite3.connect(Path(checkpoint_path).resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            for row in connection.execute("SELECT * FROM calls"):
                scope = json.loads(row["scope"])
                if scope.get("run_id") != self.path.parent.name:
                    continue
                task_id = scope.get("task_id")
                task = self.get(task_id) if task_id else None
                if not task or task["method"] not in {"ours", "mem0", "memorybank"}:
                    continue
                if row["state"] == "completed":
                    continue  # Successful calls replay from cache.
                event = json.loads(row["completed_json"] or "{}")
                safe = ((Gateway._retryable_state(row, event) and Gateway._safe_to_resend(event, retry_uncertain=True))
                        or (row["state"] in {"uncertain", "inflight"} and not event))
                status = event.get("http_status", row["status_code"])
                region = status == 403 and (event.get("failure") == "upstream_region_error"
                            or "region" in str(event.get("upstream_error", "")).lower())
                unsent = event.get("http_status") is None and event.get("failure") in {"ConnectError", "ConnectTimeout"}
                result.setdefault(task_id, []).append((safe, region or unsent or is_transient(status, event.get("failure"))
                    or (row["state"] in {"uncertain", "inflight"} and not event)))
        finally:
            connection.close()
        return result

    def requeue(self, max_attempts, checkpoint_path=None, dry_run=False):
        """Reclassify legacy failures from checkpoints, retaining audit history.

        dry_run performs no writes. Old non-retryable failures do not consume the
        bounded recovery allowance; new retryable failures do.
        """
        evidence = self.recovery_evidence(checkpoint_path)
        requeued = []
        recoveries = {}
        compactions = set()
        size_recoveries = set()
        for task in self.rows():
            if task["state"] != "failed" or task["method"] not in {"ours", "mem0", "memorybank"}:
                continue
            error = task["error"] or {}
            calls = evidence.get(task["task_id"], [])
            mismatch = "replay_mismatch" in json_text(error).lower()
            recovery = bool(calls) and all(safe and (mismatch or transient)
                                           for safe, transient in calls)
            # Audited local rejection: 48 completed calls precede the unsent
            # request. JSON compaction now fits the unchanged conservative bound.
            # Restrict this legacy migration to the diagnosed receipt, and retain
            # the ordinary total-attempt ceiling (no reset of its two attempts).
            compact_recovery = (
                self.path.parent.name == "20261105T122803255750Z"
                and task["task_id"] == "build:ours:2aac22fc66ecbe0ce4bfbcd2ac79671d9ee19b133f0bd53789e580560c613516"
                and error.get("message") == "Relation decision failed"
                and error.get("raw_response") is None
                and error.get("diagnostics") == {"exception_type": "BadRequestError",
                    "model": "typesafe/jev-1.13", "phase": "decision_jev", "status_code": 400}
                and not calls
                and not self.stage(task["task_id"], "json-compaction-requeue"))
            # One migration per fix version, only for the two audited unsent
            # local size rejections. Permanent provider 400s remain terminal.
            size_failure = (
                self.path.parent.name == "20261105T122803255750Z"
                and task["kind"] == "build" and task["method"] == "ours"
                and task["group"] in {"conv-26", "conv-41"}
                and (error.get("http_response_body") or {}).get("code") == "week1_input_too_long"
                and error.get("status_code") == 400
                and (error.get("diagnostics") or {}).get("model") == "typesafe/jev-1.13"
                and (error.get("diagnostics") or {}).get("phase") == "decision_jev")
            size_recovery = size_failure and not self.stage(task["task_id"], "evidence-table-requeue")
            eligible = size_recovery if size_failure else compact_recovery or recovery or (retryable_error(error) and not calls
                and (error.get("deadline_exceeded") is True or error.get("retryable") is not False or any(marker in json_text(error).lower()
                    for marker in ("readtimeout", "attemptdeadlineexceeded", "week1_call_uncertain", "connecterror", "connecttimeout"))))
            attempts = self.attempts(task["task_id"])
            if recovery and not size_failure:
                attempts = sum(item.get("retryable") is True for item in self.failures()
                               if item.get("task_id") == task["task_id"])
                count = self.stage(task["task_id"], "recovery-requeues") or 0
                attempts = max(attempts, count)
                recoveries[task["task_id"]] = count + 1
            if eligible and attempts < max_attempts:
                requeued.append(task["task_id"])
                if size_recovery:
                    size_recoveries.add(task["task_id"])
                if compact_recovery:
                    compactions.add(task["task_id"])
        for task in self.rows("qa"):
            if task["state"] != "blocked" or task["method"] not in {"ours", "mem0", "memorybank"}:
                continue
            build = self.build(task["method"], task["group"])
            if build and (build["state"] in {"pending", "running", "success"} or build["task_id"] in requeued):
                requeued.append(task["task_id"])
        if not dry_run:
            with self.connection:
                for task_id in requeued:
                    if task_id in size_recoveries:
                        self.connection.execute("INSERT INTO stages VALUES (?,?,?)",
                                                (task_id, "evidence-table-requeue", "true"))
                    if task_id in compactions:
                        self.connection.execute("INSERT INTO stages VALUES (?,?,?)",
                                                (task_id, "json-compaction-requeue", "true"))
                    if task_id in recoveries:
                        self.connection.execute("INSERT OR REPLACE INTO stages VALUES (?,?,?)",
                                                (task_id, "recovery-requeues", json_text(recoveries[task_id])))
                self.connection.executemany("UPDATE tasks SET state='pending',error=NULL WHERE task_id=?",
                                            [(task_id,) for task_id in requeued])
        return requeued

    def failures(self):
        return [json.loads(row[0]) for row in self.connection.execute("SELECT payload FROM failures ORDER BY event_id")]

    def predictions(self):
        result = []
        for task in self.rows("qa"):
            score = self.stage(task["task_id"], "score")
            if score is not None:
                self.require_history(task["task_id"])
                if task["state"] != "success":
                    raise ValueError("scored question has inconsistent task state")
                result.append(score)
        return result

    def counts(self):
        return {kind: {state: sum(row["state"] == state for row in self.rows(kind))
                       for state in ("pending", "running", "success", "failed", "blocked", "budget-stopped")}
                for kind in ("build", "qa")}
