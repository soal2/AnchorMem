"""Durable task and stage receipts for the frozen week-one pilot."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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

    def requeue(self, max_attempts):
        """Return failed tasks with attempts left, and QA blocked on a requeued build, to pending.

        Completed units and stages are kept, so work resumes where it stopped.
        Budget-stopped tasks are left alone. Returns the requeued task ids.
        """
        requeued = []
        with self.connection:
            for row in self.connection.execute("SELECT task_id, kind, method, group_id FROM tasks WHERE state='failed'"
                                               " ORDER BY position").fetchall():
                if self.attempts(row[0]) < max_attempts:
                    self.connection.execute("UPDATE tasks SET state='pending', error=NULL WHERE task_id=?", (row[0],))
                    requeued.append(row[0])
            for row in self.connection.execute("SELECT task_id, method, group_id FROM tasks WHERE state='blocked'"
                                               " AND kind='qa'").fetchall():
                build = self.connection.execute("SELECT state FROM tasks WHERE kind='build' AND method=? AND group_id=?",
                                                (row[1], row[2])).fetchone()
                if build is not None and build[0] in ("pending", "running", "success"):
                    self.connection.execute("UPDATE tasks SET state='pending', error=NULL WHERE task_id=?", (row[0],))
                    requeued.append(row[0])
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
