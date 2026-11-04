"""Recover proven pre-request connection failures without replaying successful work.

This operator leaves the frozen experiment sources, budget ledger and unknown
requests unchanged. Two database commits are deliberately ordered: removing a
failed call before releasing its task is safe even if the operator is interrupted.
"""

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ServiceConfig, load_config
from src import week1_runner as runner
from src.week1_tasks import history_digest, ingestion_units, json_text


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run_config(root):
    config, _ = load_config(root / "configs/service/vikingjev.yaml", ServiceConfig)
    return config.model_copy(update={
        "llm": config.llm.model_copy(update={"retries": 0}),
        "embedding": config.embedding.model_copy(update={"retries": 0}),
        "consolidation": config.consolidation.model_copy(update={
            "decision_model": config.consolidation.decision_model.model_copy(update={"retries": 0})}),
        "directory_selector": config.directory_selector.model_copy(update={"retries": 0})})


@contextmanager
def locked(directory):
    # Match the executor's lock ordering; do not create or rewrite lock files.
    with ExitStack() as stack:
        for path in (directory / "executor.lock", directory.parent / "gateway.lock"):
            handle = stack.enter_context(path.open("r"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("experiment executor or shared gateway is already running") from None
        yield


@contextmanager
def database(path, writable=False):
    if not path.is_file():
        raise ValueError("checkpoint database is missing")
    connection = sqlite3.connect(path.as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def table_rows(connection, table):
    return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def protected_records(tasks, calls, intent):
    excluded_calls = {row["call_id"] for row in intent["calls"]}
    excluded_tasks = {row["task_id"] for row in intent["tasks"]}
    all_calls = table_rows(calls, "calls")
    groups = {
        "other_calls": [row for row in all_calls if row["call_id"] not in excluded_calls],
        "completed_calls": [row for row in all_calls if row["state"] == "completed"],
        "other_tasks": [row for row in table_rows(tasks, "tasks") if row["task_id"] not in excluded_tasks],
        "stages": table_rows(tasks, "stages"),
        "failures": table_rows(tasks, "failures"),
    }
    return {name: {"count": len(rows), "sha256": digest(rows)} for name, rows in groups.items()}


def select_recoverable(tasks, calls, directory, manifest):
    all_tasks = table_rows(tasks, "tasks")
    all_calls = table_rows(calls, "calls")
    stages = {row["task_id"] for row in table_rows(tasks, "stages")}
    by_id = {row["task_id"]: row for row in all_tasks}
    examples = {example["group"]: example for example in runner.load_pilot(directory.parents[2])}
    selected_calls, selected_tasks, builds, questions = [], {}, [], []
    for call in all_calls:
        context = json.loads(call["scope"])
        if context.get("run_id") != directory.name:
            continue
        completed = json.loads(call["completed_json"]) if call["completed_json"] else {}
        if (call["state"] != "uncertain" or completed.get("failure") != "ConnectError"
                or call["ordinal"] != 0 or call["status_code"] is not None
                or completed.get("http_status") is not None or call["response_json"] is not None
                or completed.get("request_id") is not None):
            continue
        task = by_id.get(context.get("task_id"))
        if (not task or task["kind"] != "build" or task["state"] != "failed"
                or task["completed_units"] != 0 or task["index_path"] is not None or task["receipt"] is not None
                or task["task_id"] in stages or task["protocol_sha256"] != manifest["protocol_sha256"]
                or task["history_sha256"] != history_digest(examples[task["group_id"]])):
            continue
        expected_context = {"run_id": directory.name, "method": task["method"], "dataset": task["dataset"],
            "group": task["group_id"], "question_id": "history", "stage": "ingestion", "task_id": task["task_id"],
            "unit_id": ingestion_units(examples[task["group_id"]], task["method"])[0]["id"],
            "protocol_sha256": manifest["protocol_sha256"]}
        reservation = json.loads(call["reservation_json"])
        error = json.loads(task["error"]) if task["error"] else {}
        if (context != expected_context or completed.get("context") != context
                or reservation.get("context") != context or completed.get("call_id") != call["call_id"]
                or reservation.get("call_id") != call["call_id"] or error.get("stage") != "ingestion"
                or error.get("budget_stop") or error.get("history_complete") is not False):
            continue
        task_calls = [row for row in all_calls if json.loads(row["scope"]).get("task_id") == task["task_id"]
                      and json.loads(row["scope"]).get("run_id") == directory.name]
        if len(task_calls) != 1:
            continue
        dependents = [row for row in all_tasks if row["kind"] == "qa" and row["method"] == task["method"]
                      and row["group_id"] == task["group_id"]]
        if not dependents or any(row["state"] != "blocked" or row["task_id"] in stages
                or json.loads(row["error"] or "{}") != {"dependency": task["task_id"]}
                or row["protocol_sha256"] != task["protocol_sha256"]
                or row["history_sha256"] != task["history_sha256"] for row in dependents):
            continue
        selected_calls.append(call)
        builds.append(task["task_id"])
        questions.extend(row["task_id"] for row in dependents)
        selected_tasks.update((row["task_id"], row) for row in [task, *dependents])
    return {"calls": selected_calls, "tasks": list(selected_tasks.values()),
            "build_task_ids": builds, "qa_task_ids": questions}


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def backup_database(connection, path):
    destination = sqlite3.connect(path)
    try:
        connection.backup(destination)
    finally:
        destination.close()
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def validate_current(tasks, calls, intent, resuming):
    if protected_records(tasks, calls, intent) != intent["protected"]:
        raise ValueError("protected records changed; recovery cannot overwrite new work")
    for original in intent["calls"]:
        row = calls.execute("SELECT * FROM calls WHERE scope=? AND ordinal=?", (original["scope"], original["ordinal"])).fetchone()
        if row is None and resuming:
            continue
        if row is None or dict(row) != original:
            raise ValueError("failed call changed or disappeared without a recovery intent")
    for original in intent["tasks"]:
        row = tasks.execute("SELECT * FROM tasks WHERE task_id=?", (original["task_id"],)).fetchone()
        restored = {**original, "state": "pending", "error": None}
        if row is None or (dict(row) != original and not (resuming and dict(row) == restored)):
            raise ValueError("target task changed; recovery cannot overwrite new work")


def _delete_calls(path, intent, resuming):
    with database(path, writable=True) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        for original in intent["calls"]:
            row = connection.execute("SELECT * FROM calls WHERE scope=? AND ordinal=?", (original["scope"], original["ordinal"])).fetchone()
            if row is None and resuming:
                continue
            if row is None or dict(row) != original:
                raise ValueError("failed call changed during recovery")
            connection.execute("DELETE FROM calls WHERE scope=? AND ordinal=? AND call_id=?",
                               (original["scope"], original["ordinal"], original["call_id"]))


def _reset_tasks(path, intent, resuming):
    with database(path, writable=True) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        for original in intent["tasks"]:
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (original["task_id"],)).fetchone()
            restored = {**original, "state": "pending", "error": None}
            if row is not None and dict(row) == restored and resuming:
                continue
            if row is None or dict(row) != original:
                raise ValueError("target task changed during recovery")
            connection.execute("UPDATE tasks SET state='pending',error=NULL WHERE task_id=?", (original["task_id"],))


def read_receipt(path, root, directory, manifest):
    path = path.resolve()
    if path.name != "receipt.json" or path.parent.parent != directory / "transport-recovery":
        raise ValueError("resume receipt must belong to the same run directory")
    receipt = json.loads(path.read_text())
    intent = receipt["intent"]
    if (intent["version"] != 1 or receipt["intent_sha256"] != digest(intent)
            or intent["root"] != str(root) or intent["run_dir"] != str(directory)
            or intent["run_id"] != directory.name or intent["protocol_sha256"] != manifest["protocol_sha256"]
            or intent["manifest_sha256"] != file_digest(directory / "manifest.json")):
        raise ValueError("recovery intent fingerprint does not match the frozen run")
    if set(receipt["backup_sha256"]) != {"tasks.sqlite3", "api-call-checkpoints.sqlite3"}:
        raise ValueError("recovery must retain both database backup fingerprints")
    for name, expected in receipt["backup_sha256"].items():
        if name not in ("tasks.sqlite3", "api-call-checkpoints.sqlite3") or file_digest(path.parent / name) != expected:
            raise ValueError("recovery backup fingerprint changed")
    with database(path.parent / "tasks.sqlite3") as tasks, database(path.parent / "api-call-checkpoints.sqlite3") as calls:
        expected = select_recoverable(tasks, calls, directory, manifest)
        if any(intent[name] != expected[name] for name in ("calls", "tasks", "build_task_ids", "qa_task_ids")):
            raise ValueError("recovery intent does not match the backed-up eligible failures")
    return path, receipt


def recover(root, run_dir, *, apply=False, resume_receipt=None):
    root = Path(root).resolve()
    directory = (root / run_dir).resolve() if not Path(run_dir).is_absolute() else Path(run_dir).resolve()
    if directory.parent != root / "artifacts/week1":
        raise ValueError("run directory must be an immediate child of this experiment's artifacts/week1")
    with locked(directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        runner.verify_resume(root, directory, manifest, run_config(root))
        if manifest["run_id"] != directory.name:
            raise ValueError("run directory and frozen identity disagree")
        recovery_base = directory / "transport-recovery"
        unfinished = [path for path in recovery_base.glob("*/receipt.json")
                      if json.loads(path.read_text()).get("status") != "complete"]
        if not resume_receipt and unfinished:
            raise ValueError("unfinished recovery requires explicit --resume-receipt")
        if resume_receipt:
            path = Path(resume_receipt)
            receipt_path, receipt = read_receipt((root / path) if not path.is_absolute() else path,
                                                root, directory, manifest)
            if any(path.resolve() != receipt_path for path in unfinished):
                raise ValueError("another unfinished recovery must be resolved first")
            intent = receipt["intent"]
        else:
            receipt_path = None
            receipt = None
            intent = None
        task_path = directory / "tasks.sqlite3"
        call_path = directory.parent / "api-call-checkpoints.sqlite3"
        ledger = directory.parent / "api-ledger.jsonl"
        with database(task_path) as tasks, database(call_path) as calls:
            if intent is None:
                intent = {"version": 1, "root": str(root), "run_dir": str(directory), "run_id": directory.name,
                    "protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": file_digest(directory / "manifest.json"),
                    "ledger_sha256": file_digest(ledger), "created_at": datetime.now(timezone.utc).isoformat(),
                    **select_recoverable(tasks, calls, directory, manifest)}
                intent["protected"] = protected_records(tasks, calls, intent)
            validate_current(tasks, calls, intent, bool(resume_receipt))
            if file_digest(ledger) != intent["ledger_sha256"]:
                raise ValueError("shared ledger changed; recovery never resets reservations")
            result = {"applied": False, "calls": len(intent["calls"]), "builds": len(intent["build_task_ids"]),
                "questions": len(intent["qa_task_ids"]), "build_task_ids": intent["build_task_ids"],
                "qa_task_ids": intent["qa_task_ids"], "call_ids": [row["call_id"] for row in intent["calls"]],
                "protocol_sha256": intent["protocol_sha256"], "before": intent["protected"]}
            if not apply or not intent["calls"]:
                return result
            if receipt is None:
                folder = recovery_base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                folder.mkdir(parents=True)
                sync_directory(directory)
                sync_directory(recovery_base)
                for connection, name in ((tasks, "tasks.sqlite3"), (calls, "api-call-checkpoints.sqlite3")):
                    backup_database(connection, folder / name)
                receipt_path = folder / "receipt.json"
                receipt = {"status": "prepared", "intent": intent, "intent_sha256": digest(intent),
                    "tool_sha256": file_digest(Path(__file__).resolve()),
                    "backup_sha256": {name: file_digest(folder / name) for name in ("tasks.sqlite3", "api-call-checkpoints.sqlite3")}}
                save_json(receipt_path, receipt)
        # These commits cannot be jointly atomic in WAL mode. This order keeps
        # tasks blocked until their failed cache entries have been archived.
        _delete_calls(call_path, intent, bool(resume_receipt))
        receipt["status"] = "calls-cleared"
        save_json(receipt_path, receipt)
        _reset_tasks(task_path, intent, bool(resume_receipt))
        with database(task_path) as tasks, database(call_path) as calls:
            validate_current(tasks, calls, intent, True)
            if any(calls.execute("SELECT 1 FROM calls WHERE call_id=?", (row["call_id"],)).fetchone() for row in intent["calls"]):
                raise ValueError("failed connection cache was not fully cleared")
            for original in intent["tasks"]:
                row = tasks.execute("SELECT * FROM tasks WHERE task_id=?", (original["task_id"],)).fetchone()
                if dict(row) != {**original, "state": "pending", "error": None}:
                    raise ValueError("task recovery was not fully committed")
            result["after"] = protected_records(tasks, calls, intent)
        if file_digest(ledger) != intent["ledger_sha256"]:
            raise ValueError("shared ledger changed during recovery")
        result.update(applied=True, receipt=str(receipt_path))
        receipt.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), result=result)
        save_json(receipt_path, receipt)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="back up, persist intent, and recover eligible connection failures")
    parser.add_argument("--resume-receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.run_dir, apply=args.apply, resume_receipt=args.resume_receipt),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
