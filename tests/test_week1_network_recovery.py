import fcntl
import json
import sqlite3
from pathlib import Path

import pytest

from scripts import week1_recover_transport as recovery
from src import week1_runner as runner
from src.week1_tasks import TaskStore, json_text


def rows(path, table):
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path
    for folder in ("src", "prompts", "configs/service", "data/week1/envs"):
        (root / folder).mkdir(parents=True)
    original = Path(__file__).resolve().parents[1]
    (root / "configs/service/vikingjev.yaml").write_bytes((original / "configs/service/vikingjev.yaml").read_bytes())
    for name, text in {"src/example.py": "FIXED = True\n", "prompts/extract.txt": "strict",
                       "uv.lock": "fixed", "pyproject.toml": "project", "data/week1/prepared.json": "{}",
                       "data/week1/envs/mem0-requirements.lock": "mem0 fixed",
                       "data/week1/envs/memorybank-requirements.lock": "memorybank fixed"}.items():
        (root / name).write_text(text)
    examples = [{"dataset": "locomo", "group": group, "id": f"{group}-{number}", "category": 1,
                 "sessions": [{"id": "s1", "messages": [{"role": "user", "content": "full history"}]}]}
                for group in ("connect", "timeout", "inflight", "http", "done") for number in range(2)]
    selections = [{name: example[name] for name in ("dataset", "id", "group", "category")} for example in examples]
    (root / "data/week1/selection.json").write_text(json.dumps({"examples": selections}))
    monkeypatch.setattr(runner, "snapshot_authors", lambda *args: None)
    monkeypatch.setattr(runner, "baseline_sources", lambda *args: {})
    monkeypatch.setattr(runner, "load_pilot", lambda *args: examples)
    config = recovery.run_config(root)
    directory, manifest = runner.freeze_run(root, config)
    store = TaskStore(directory / "tasks.sqlite3")
    store.seed(examples, runner.METHODS, manifest["protocol_sha256"])
    # Only one method participates in this fixture's network incident.
    for group in ("connect", "timeout", "inflight", "http"):
        task = store.build("rag", group)
        store.fail(task["task_id"], "ingestion", {"type": "RuntimeError", "message": "network failure"})
    done = store.build("rag", "done")
    store.commit_build_unit(done["task_id"], 1, "/committed/index")
    store.commit_build(done["task_id"], {"history_complete": True, "history_sha256": done["history_sha256"]})
    question = store.question("rag", "done-0")
    store.commit_stage(question["task_id"], "answer", {"answer": "saved", "finish_reason": "stop"})
    store.commit_stage(question["task_id"], "score", {"score": 0.5})
    tasks = {group: store.build("rag", group)["task_id"] for group in ("connect", "timeout", "inflight", "http", "done")}
    store.close()
    api = directory.parent / "api-call-checkpoints.sqlite3"
    with sqlite3.connect(api) as connection:
        connection.execute("""CREATE TABLE calls (scope TEXT NOT NULL,ordinal INTEGER NOT NULL,
          request_sha256 TEXT NOT NULL,call_id TEXT NOT NULL UNIQUE,state TEXT NOT NULL,
          reservation_json TEXT NOT NULL,completed_json TEXT,response_json TEXT,status_code INTEGER,
          PRIMARY KEY(scope,ordinal))""")
        for group, state, failure, status in (("connect", "uncertain", "ConnectError", None),
                  ("timeout", "uncertain", "ReadTimeout", None), ("inflight", "inflight", None, None),
                  ("http", "failed", "upstream_http_error", 502), ("done", "completed", None, 200)):
            context = {"run_id": directory.name, "task_id": tasks[group], "protocol_sha256": manifest["protocol_sha256"],
                       "method": "rag", "dataset": "locomo", "group": group, "question_id": "history",
                       "stage": "ingestion", "unit_id": "full-history"}
            event = {"call_id": f"call-{group}", "context": context, "failure": failure, "http_status": status,
                     "request_id": "success-id" if state == "completed" else None}
            connection.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)", (json_text(context), 0, f"sha-{group}",
                event["call_id"], state, json_text({"call_id": event["call_id"], "context": context}),
                None if state == "inflight" else json_text(event),
                json_text({"answer": "existing response"}) if state == "completed" else None, status))
    (directory / "executor.lock").touch()
    (directory.parent / "gateway.lock").touch()
    ledger = directory.parent / "api-ledger.jsonl"
    ledger.write_text("ORIGINAL LEDGER MUST NEVER CHANGE\n")
    return root, directory, api, tasks, ledger


def test_default_dry_run_does_not_change_databases_or_ledger(experiment):
    root, directory, api, _, ledger = experiment
    before = {path: path.read_bytes() for path in (api, directory / "tasks.sqlite3", ledger)}
    result = recovery.recover(root, directory)
    assert result["builds"] == 1 and result["questions"] == 2 and result["calls"] == 1
    assert result["applied"] is False
    assert not (directory / "transport-recovery").exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_only_connect_failures_recover_completed_scores_and_unknown_calls_are_unchanged(experiment):
    root, directory, api, tasks, ledger = experiment
    before_calls = rows(api, "calls")
    before_stages = rows(directory / "tasks.sqlite3", "stages")
    before_failures = rows(directory / "tasks.sqlite3", "failures")
    before_success = [row for row in rows(directory / "tasks.sqlite3", "tasks") if row["state"] == "success"]
    result = recovery.recover(root, directory, apply=True)
    assert result["applied"] and result["builds"] == 1 and result["questions"] == 2
    assert rows(api, "calls") == [row for row in before_calls if row["call_id"] != "call-connect"]
    assert rows(directory / "tasks.sqlite3", "stages") == before_stages
    assert rows(directory / "tasks.sqlite3", "failures") == before_failures
    assert [row for row in rows(directory / "tasks.sqlite3", "tasks") if row["state"] == "success"] == before_success
    states = {row["task_id"]: row["state"] for row in rows(directory / "tasks.sqlite3", "tasks")}
    assert states[tasks["connect"]] == "pending"
    assert all(states[tasks[group]] == "failed" for group in ("timeout", "inflight", "http"))
    assert ledger.read_text() == "ORIGINAL LEDGER MUST NEVER CHANGE\n"
    assert result["before"] == result["after"]
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["status"] == "complete"
    assert len(receipt["intent"]["calls"]) == 1
    assert (Path(result["receipt"]).parent / "tasks.sqlite3").is_file()
    assert (Path(result["receipt"]).parent / "api-call-checkpoints.sqlite3").is_file()


@pytest.mark.parametrize("name", ["executor.lock", "gateway.lock"])
def test_running_executor_or_gateway_lock_refuses_recovery(experiment, name):
    root, directory, api, _, _ = experiment
    path = directory / name if name == "executor.lock" else directory.parent / name
    with path.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            recovery.recover(root, directory, apply=True)
    assert len(rows(api, "calls")) == 5
    assert not (directory / "transport-recovery").exists()


def test_changed_snapshot_rejects_recovery_before_mutating_anything(experiment):
    root, directory, api, _, _ = experiment
    (directory / "source/example.py").write_text("CHANGED = True\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        recovery.recover(root, directory, apply=True)
    assert len(rows(api, "calls")) == 5
    assert not (directory / "transport-recovery").exists()


def test_crash_between_commits_requires_explicit_intent_and_is_resumable(experiment, monkeypatch):
    root, directory, api, tasks, _ = experiment
    reset = recovery._reset_tasks
    monkeypatch.setattr(recovery, "_reset_tasks", lambda *args: (_ for _ in ()).throw(RuntimeError("injected crash")))
    with pytest.raises(RuntimeError, match="injected crash"):
        recovery.recover(root, directory, apply=True)
    assert len(rows(api, "calls")) == 4
    assert next(row for row in rows(directory / "tasks.sqlite3", "tasks") if row["task_id"] == tasks["connect"])["state"] == "failed"
    receipt = next((directory / "transport-recovery").glob("*/receipt.json"))
    with pytest.raises(ValueError, match="unfinished recovery"):
        recovery.recover(root, directory, apply=True)
    monkeypatch.setattr(recovery, "_reset_tasks", reset)
    result = recovery.recover(root, directory, apply=True, resume_receipt=receipt)
    assert result["applied"]
    assert next(row for row in rows(directory / "tasks.sqlite3", "tasks") if row["task_id"] == tasks["connect"])["state"] == "pending"
    assert result["before"] == result["after"]


def test_resume_does_not_overwrite_new_stages(experiment, monkeypatch):
    root, directory, _, tasks, _ = experiment
    reset = recovery._reset_tasks
    monkeypatch.setattr(recovery, "_reset_tasks", lambda *args: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        recovery.recover(root, directory, apply=True)
    receipt = next((directory / "transport-recovery").glob("*/receipt.json"))
    with sqlite3.connect(directory / "tasks.sqlite3") as connection:
        connection.execute("INSERT INTO stages VALUES (?,?,?)", (tasks["connect"], "unexpected", "{}"))
    monkeypatch.setattr(recovery, "_reset_tasks", reset)
    with pytest.raises(ValueError, match="protected records changed"):
        recovery.recover(root, directory, apply=True, resume_receipt=receipt)


@pytest.mark.parametrize("reason", ["partial-build", "question-stage", "index-pointer"])
def test_partial_build_and_question_stages_are_not_reset(experiment, reason):
    root, directory, api, tasks, _ = experiment
    with sqlite3.connect(directory / "tasks.sqlite3") as connection:
        if reason == "partial-build":
            connection.execute("UPDATE tasks SET completed_units=1 WHERE task_id=?", (tasks["connect"],))
        elif reason == "index-pointer":
            connection.execute("UPDATE tasks SET index_path='/saved/index' WHERE task_id=?", (tasks["connect"],))
        else:
            question = connection.execute("SELECT task_id FROM tasks WHERE kind='qa' AND method='rag' AND group_id='connect'").fetchone()[0]
            connection.execute("INSERT INTO stages VALUES (?,?,?)", (question, "answer", '{"answer":"already saved"}'))
    result = recovery.recover(root, directory)
    assert result["calls"] == 0
    assert len(rows(api, "calls")) == 5


def test_unknown_timeout_keeps_all_57_successful_responses_in_its_scope(experiment):
    root, directory, api, _, _ = experiment
    original = next(row for row in rows(api, "calls") if row["call_id"] == "call-timeout")
    with sqlite3.connect(api) as connection:
        connection.execute("UPDATE calls SET ordinal=57 WHERE call_id='call-timeout'")
        for ordinal in range(57):
            completed = json.loads(original["completed_json"])
            completed.update(failure=None, http_status=200, call_id=f"saved-{ordinal}")
            saved = {**original, "ordinal": ordinal, "call_id": completed["call_id"], "state": "completed",
                     "completed_json": json_text(completed), "response_json": json_text({"facts": [ordinal]}), "status_code": 200}
            connection.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)", tuple(saved.values()))
    protected = [row for row in rows(api, "calls") if row["call_id"] != "call-connect"]
    result = recovery.recover(root, directory, apply=True)
    assert rows(api, "calls") == protected
    assert result["before"]["completed_calls"]["count"] == 58
    assert result["before"] == result["after"]


def test_run_directory_boundary_is_checked_before_opening_locks(experiment):
    root, _, _, _, _ = experiment
    with pytest.raises(ValueError, match="run directory"):
        recovery.recover(root, root.parent)


def test_production_selection_metadata_does_not_replace_the_frozen_history_loader(experiment):
    root, directory, _, _, _ = experiment
    assert all("sessions" not in row for row in json.loads((directory / "selection.json").read_text())["examples"])
    # LongMemEval selections have neither group nor sessions; both come from the
    # already verified original loader, never from inferred question metadata.
    metadata = json.loads((directory / "selection.json").read_text())
    metadata["examples"].append({"dataset": "longmemeval", "id": "lme-q", "category": "temporal-reasoning"})
    text = json.dumps(metadata)
    (directory / "selection.json").write_text(text)
    (root / "data/week1/selection.json").write_text(text)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["selection_sha256"] = recovery.file_digest(directory / "selection.json")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    result = recovery.recover(root, directory)
    assert result["calls"] == 1 and result["builds"] == 1 and result["questions"] == 2
