import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import tiktoken
import pytest

from src import directory_selector, providers
from src.week1_runner import AuthorRAG, GatewayTransport, clean_error, render_evidence, report, routed_clients
from src import week1_runner as runner
from src.week1_tasks import TaskStore, history_digest


def selection():
    return {"examples": [{"dataset": dataset, "id": f"{dataset}-q{i}", "group": dataset, "category": 1,
                         "sessions": [{"id": "s1", "messages": [{"role": "user", "content": "complete original history"}]}]}
                         for dataset, count in (("locomo", 20), ("longmemeval", 12)) for i in range(count)]}


def authoritative_results(directory, predictions=(), failures=(), methods=None):
    methods = list(methods or runner.METHODS)
    examples = json.loads((directory / "selection.json").read_text())["examples"]
    store = TaskStore(directory / "tasks.sqlite3")
    store.seed(examples, methods, "test-protocol")
    for task in store.rows("build"):
        for number in range(1, task["total_units"] + 1):
            store.commit_build_unit(task["task_id"], number, "/index/complete")
        store.commit_build(task["task_id"], {"history_complete": True, "history_sha256": task["history_sha256"]})
    for prediction in predictions:
        task = store.question(prediction["method"], prediction["id"])
        if not task:
            continue
        store.commit_stage(task["task_id"], "answer", {"answer": "saved", "finish_reason": "stop"})
        if prediction["dataset"] == "longmemeval":
            store.commit_stage(task["task_id"], "judge", {"judge_response": "yes", "finish_reason": "stop"})
        store.commit_stage(task["task_id"], "score", prediction)
    for failure in failures:
        task = next(row for row in store.rows("qa") if row["method"] == failure["method"] and row["dataset"] == failure["dataset"] and row["state"] != "success")
        store.fail(task["task_id"], "answer", failure)
    store.close()


def test_rendered_budget_includes_headers_and_preserves_rank():
    tokenizer = tiktoken.get_encoding("o200k_base")
    records = [{"id": str(i), "content": "evidence " * 200, "score": float(3 - i)} for i in range(3)]
    text, delivered, size = render_evidence(records, tokenizer, 43)
    assert size == len(tokenizer.encode(text)) <= 43
    assert [item["id"] for item in delivered] == ["0"]
    assert delivered[0]["truncated"]
    assert records[0]["content"].startswith(delivered[0]["content"])


def test_author_rag_keeps_full_history_and_reuses_persisted_vectors(tmp_path):
    class Embedder:
        calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[1.0, len(text) / 1000] + [0.0] * 1022 for text in texts]

    embedder = Embedder()
    example = {"sessions": [{"id": "session_1", "messages": [{"role": "user", "content": "early " * 1500},
                                         {"role": "assistant", "content": "FINAL_HISTORY_MARKER"}]}]}
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    memory = AuthorRAG(root, tmp_path, embedder)
    memory.ingest(example)
    assert "FINAL_HISTORY_MARKER" in memory.chunks[-1]
    assert "rag-source/evaluation/src/rag.py" in memory.manager.search.__code__.co_filename
    calls = embedder.calls
    reopened = AuthorRAG(root, tmp_path, embedder)
    reopened.ingest(example)
    assert embedder.calls == calls
    assert np.array_equal(memory.vectors, reopened.vectors)
    assert reopened.search("early")[0]["id"] == "rag-native-context"


def test_transport_intercepts_absolute_decisions_and_removes_real_key():
    transport = GatewayTransport("http://127.0.0.1:12345")
    transport.transport.close()
    observed = []

    def respond(request):
        observed.append(request)
        return httpx.Response(200, json={"ok": True})

    transport.transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as client:
        client.post("https://openrouter.ai/api/alpha/decisions", headers={"authorization": "Bearer secret-real"}, json={})
    assert observed[0].url == "http://127.0.0.1:12345/api/alpha/decisions"
    assert observed[0].headers["authorization"] == "Bearer week1-local"


def test_routing_covers_both_provider_modules_and_disables_retries():
    old_provider, old_selector = providers.OpenAI, directory_selector.OpenAI
    with routed_clients("http://127.0.0.1:12345") as factory:
        assert providers.OpenAI is factory and directory_selector.OpenAI is factory
        client = factory(api_key="real-key", base_url="https://openrouter.ai/api/v1", max_retries=2)
        assert client.max_retries == 0
        assert client.api_key == "week1-local"
        assert str(client.base_url) == "http://127.0.0.1:12345/api/v1/"
    assert providers.OpenAI is old_provider and directory_selector.OpenAI is old_selector


def test_report_does_not_compare_different_completed_question_sets(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test", "methods": list(runner.METHODS), "status": "budget-stopped"}))
    predictions = [{"method": method, "dataset": "locomo", "id": question, "category": 1, "score": score}
                   for method, question, score in [("rag", "locomo-q1", 1), ("mem0", "locomo-q1", 0),
                                                   ("memorybank", "locomo-q2", 1), ("ours", "locomo-q2", 1)]]
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in predictions))
    (tmp_path / "failures.jsonl").write_text("")
    (tmp_path / "selection.json").write_text(json.dumps(selection()))
    authoritative_results(tmp_path, predictions)
    result = report(tmp_path)
    assert result["common_completed"]["locomo"]["count"] == 0
    assert all(score is None for score in result["common_completed"]["locomo"]["scores"].values())
    assert not result["complete"]
    assert result["metrics"]["locomo/rag"]["completion_rate"] == 1 / 20


def test_failure_messages_redact_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-value")
    assert "private-value" not in json.dumps(clean_error(RuntimeError("failed private-value")))


def test_provider_failure_retains_validation_and_raw_output_without_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-value")
    error = providers.ProviderError("invalid private-value", {
        "validation_error": "missing quote private-value", "finish_reason": "stop"},
        '{"facts": ["private-value"]}')
    saved = clean_error(error)
    assert saved["diagnostics"]["validation_error"] == "missing quote [REDACTED]"
    assert saved["diagnostics"]["finish_reason"] == "stop"
    assert saved["raw_response"] == '{"facts": ["[REDACTED]"]}'
    assert "private-value" not in json.dumps(saved)


def test_report_counts_unique_coverage_and_keeps_historical_failure(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test", "methods": list(runner.METHODS), "status": "finished"}))
    frozen = selection()
    (tmp_path / "selection.json").write_text(json.dumps(frozen))
    predictions = [{**item, "method": method, "score": 1} for item in frozen["examples"] for method in runner.METHODS]
    predictions.append(predictions[0])
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in predictions))
    (tmp_path / "failures.jsonl").write_text(json.dumps({"method": "rag", "dataset": "locomo", "type": "old-region-refusal"}))
    authoritative_results(tmp_path, predictions)
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.question("rag", "locomo-q0")
    store.connection.execute("INSERT INTO failures(task_id,stage,payload) VALUES (?,?,?)", (task["task_id"], "old", json.dumps({"method": "rag", "dataset": "locomo", "type": "old-region-refusal"})))
    store.connection.commit()
    store.close()
    result = report(tmp_path)
    assert result["complete"] and not result["not_evaluated"]
    assert result["common_completed"]["locomo"]["count"] == 20
    assert result["metrics"]["locomo/rag"]["evaluated"] == 20
    assert result["metrics"]["locomo/rag"]["failure_records"] == 1
    assert result["metrics"]["locomo/rag"]["categories"]["1"]["score"] == 1
    manifest = {"run_id": "test", "methods": list(runner.METHODS), "status": "finished", "budget": {
        "spent_usd": .4, "reserved_usd": .02, "stopped": None, "allow_unconfirmed_costs": False}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert not report(tmp_path)["complete"]
    manifest["budget"]["allow_unconfirmed_costs"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert report(tmp_path)["complete"]


def test_report_distinguishes_unpriced_calls_from_reconciled_refusals(tmp_path):
    directory = tmp_path / "pilot"
    directory.mkdir()
    for name in ("predictions", "failures"):
        (directory / f"{name}.jsonl").touch()
    (directory / "selection.json").write_text(json.dumps(selection()))
    (directory / "manifest.json").write_text(json.dumps({"run_id": "pilot", "methods": list(runner.METHODS), "status": "blocked-provider-region",
                                                        "blocking_error": "region unavailable"}))
    rows = [{"event": "completed", "call_id": name,
             "context": {"run_id": "pilot", "method": "rag", "stage": "answer"},
             "actual_cost_usd": cost, "http_status": status, "latency_seconds": 2}
            for name, cost, status in (("paid", .01, 200), ("refused", None, 403), ("unknown", None, 500))]
    rows.append({"event": "reconciled-zero", "call_id": "refused"})
    (tmp_path / "api-ledger.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    authoritative_results(directory)
    result = report(directory)
    assert result["api_stages"]["rag/answer"] == {"calls": 3, "api_seconds": 6,
        "confirmed_cost_usd": .01, "unknown_cost_calls": 1, "refused_calls": 1}
    assert len(result["not_evaluated"]) == 128
    assert result["week2_budget_estimate"]["estimated_usd"] is None


def test_report_removed_method_never_counts_as_a_complete_four_system_pilot(tmp_path):
    methods = ["rag", "mem0", "ours"]
    frozen = selection()
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "old-pilot", "status": "results-removed",
        "methods": ["rag", "mem0", "amem", "ours"], "result_methods": methods}))
    (tmp_path / "selection.json").write_text(json.dumps(frozen))
    predictions = [{**item, "method": method, "score": 1}
                   for item in frozen["examples"] for method in [*methods, "amem"]]
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in predictions))
    authoritative_results(tmp_path, predictions, methods=methods)
    result = report(tmp_path)
    assert result["methods"] == methods and result["requested_predictions"] == 96
    assert not result["not_evaluated"] and not result["complete"]
    assert result["common_completed"]["locomo"]["count"] == 20
    assert not any("amem" in key for key in result["metrics"])
    assert report(tmp_path, result_methods=["rag"])["requested_predictions"] == 32


def test_freeze_run_saves_only_current_sources_without_opening_a_gateway(tmp_path, monkeypatch):
    for folder in ("src", "prompts", "data/week1/envs"):
        (tmp_path / folder).mkdir(parents=True)
    for name in ("src/current.py", "prompts/extraction.txt", "uv.lock", "pyproject.toml",
                 "data/week1/prepared.json", "data/week1/envs/mem0-requirements.lock",
                 "data/week1/envs/memorybank-requirements.lock", "data/week1/envs/amem-requirements.lock"):
        (tmp_path / name).write_text(name)
    (tmp_path / "data/week1/selection.json").write_text(json.dumps(selection()))
    snapshots = []
    sources = {"mem0": {"commit": "mem0-fixed"}, "memorybank": {"commit": "memorybank-fixed"}}
    monkeypatch.setattr(runner, "snapshot_authors", lambda *args: snapshots.append(args))
    monkeypatch.setattr(runner, "baseline_sources", lambda root: sources)
    monkeypatch.setattr(runner, "Gateway", lambda *args, **kwargs: pytest.fail("freeze opened a paid gateway"))
    directory, manifest = runner.freeze_run(tmp_path, SimpleNamespace(model_dump=lambda: {"fixed": True}), 20)
    assert snapshots == [(tmp_path, directory)]
    assert json.loads((directory / "manifest.json").read_text()) == manifest
    assert manifest["status"] == "frozen" and manifest["methods"] == ["rag", "mem0", "memorybank", "ours"]
    assert manifest["baseline_sources"] == sources
    assert (directory / "source/current.py").read_text() == "src/current.py"
    assert (directory / "memorybank-requirements.lock").is_file()
    assert not (directory / "amem-requirements.lock").exists()




@pytest.fixture
def execution(tmp_path, monkeypatch):
    from contextlib import contextmanager

    examples = [{"dataset": dataset, "id": f"{dataset}-q{i}", "group": dataset, "category": category,
                 "question": f"question-{i}", "answer": "answer", "source_tokens": 10,
                 "sessions": [{"id": "session_1", "messages": [{"role": "user", "content": f"history {n}"} for n in range(21)]}]}
                for dataset, category in (("locomo", 1), ("longmemeval", "single-session-user")) for i in range(2)]
    for folder in ("src", "prompts", "data/week1/envs", "data/week1/baselines"):
        (tmp_path / folder).mkdir(parents=True)
    for name in ("week1_runner.py", "week1_gateway.py", "week1_tasks.py"):
        (tmp_path / "src" / name).write_text((Path(runner.__file__).parent / name).read_text())
    for name in ("uv.lock", "pyproject.toml", "data/week1/prepared.json", "data/week1/envs/mem0-requirements.lock",
                 "data/week1/envs/memorybank-requirements.lock", "data/week1/baselines/manifest.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "data/week1/selection.json").write_text(json.dumps({"examples": examples}))
    config, _ = runner.load_config(Path(runner.__file__).parents[1] / "configs/service/vikingjev.yaml", runner.ServiceConfig)
    monkeypatch.setattr(runner, "load_config", lambda *args: (config, None))
    monkeypatch.setattr(runner, "preflight", lambda root: None)
    monkeypatch.setattr(runner, "load_pilot", lambda root: examples)
    monkeypatch.setattr(runner, "snapshot_authors", lambda *args: None)
    monkeypatch.setattr(runner, "baseline_sources", lambda root: {"mem0": {}, "memorybank": {}})
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    state = SimpleNamespace(root=tmp_path, examples=examples, calls=[], ingestions=[], searches=[],
                            fail_build=set(), fail_question=set(), fail_stage=set(), interrupt=None,
                            budget_stopped=None, contexts=[], reopened=[], close_error_on_interrupt=False)

    class FakeGateway:
        url = "local"

        def __init__(self, *args, **kwargs):
            self.stopped = state.budget_stopped
            self.context = None

        def __enter__(self):
            state.gateway = self
            return self

        def __exit__(self, *args):
            pass

        def set_context(self, context):
            self.context = context
            state.contexts.append(context)

        def summary(self):
            return {"spent_usd": 0, "reserved_usd": 0, "stopped": self.stopped}

    class FakeMemory:
        def __init__(self, root, path, *args):
            self.path = path
            path.mkdir(parents=True, exist_ok=True)
            self.file = path / "native-state.json"
            self.value = json.loads(self.file.read_text()) if self.file.exists() else {"units": [], "retrievals": []}
            self.group = state.gateway.context["group"]
            state.reopened.append((self.method, self.group, self.path))

        def persist(self):
            self.file.write_text(json.dumps(self.value))

        def ingest(self, example):
            state.ingestions.append((self.method, example['group'], 'full-history'))
            self.value['units'].append('full-history')
            self.persist()
            if (self.method, example['group']) in state.fail_build:
                raise providers.ProviderError('invalid extraction', {'validation_error': 'missing quote'}, 'invalid raw')

        def ingest_unit(self, example, unit):
            state.ingestions.append((self.method, example['group'], unit['id']))
            self.value['units'].append(unit['id'])
            self.persist()
            if (self.method, example['group']) in state.fail_build:
                raise providers.ProviderError('invalid extraction', {'validation_error': 'missing quote'}, 'invalid raw')
            if state.interrupt == ('ingestion', self.method, example['group'], unit['id']):
                state.interrupt = None
                self.cleanup_should_fail = state.close_error_on_interrupt
                raise KeyboardInterrupt

        def search(self, question, *args):
            state.searches.append((self.method, self.group, question))
            self.value['retrievals'].append(question)
            self.persist()
            if state.interrupt == ('search', self.method, self.group, question):
                state.interrupt = None
                self.cleanup_should_fail = state.close_error_on_interrupt
                raise KeyboardInterrupt
            if (self.method, self.group, question) in state.fail_question:
                raise RuntimeError('retrieval failed after native mutation')
            return [{'id': 'memory', 'content': 'complete memory evidence', 'score': 1}]

        def close(self):
            if getattr(self, 'cleanup_should_fail', False):
                raise TimeoutError('cleanup timed out after an interrupt')

    def create(**kwargs):
        context = state.gateway.context
        state.calls.append((context['method'], context['group'], context['question_id'], context['stage']))
        if (context['method'], context['question_id'], context['stage']) in state.fail_stage:
            raise RuntimeError('isolated model stage failure')
        if state.gateway.stopped:
            raise ValueError('budget has stopped new API requests')
        return SimpleNamespace(id='judge-id', choices=[SimpleNamespace(message=SimpleNamespace(content='yes'), finish_reason='stop')])

    @contextmanager
    def clients(url):
        yield lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(runner, 'Gateway', FakeGateway)
    monkeypatch.setattr(runner, 'routed_clients', clients)
    monkeypatch.setattr(runner.providers, 'OpenAIEmbedder', lambda config: None)
    for name, method in (('AuthorRAG', 'rag'), ('NativeMemory', 'ours'), ('BaselineProcess', 'mem0'), ('MemoryBankProcess', 'memorybank')):
        monkeypatch.setattr(runner, name, type(name, (FakeMemory,), {'method': method}))
    monkeypatch.setattr(runner, 'score_locomo', lambda *args: 1)
    monkeypatch.setattr(runner, 'get_lme_judge_prompt', lambda *args: 'official judge prompt')
    return state


def test_independent_build_and_question_failures_continue_other_tasks(execution):
    state = execution
    state.fail_build.add(('ours', 'locomo'))
    state.fail_stage.add(('rag', 'locomo-q0', 'answer'))
    state.fail_stage.add(('mem0', 'longmemeval-q0', 'judge'))
    directory = runner.run(state.root)
    store = TaskStore(directory / 'tasks.sqlite3')
    assert len(store.rows('build')) == 8 and len(store.rows('qa')) == 16
    assert store.build('ours', 'locomo')['state'] == 'failed'
    assert store.question('ours', 'locomo-q0')['state'] == store.question('ours', 'locomo-q1')['state'] == 'blocked'
    assert store.question('ours', 'longmemeval-q1')['state'] == 'success'
    assert store.question('rag', 'locomo-q0')['state'] == 'failed'
    assert store.question('rag', 'locomo-q1')['state'] == 'success'
    assert store.question('mem0', 'longmemeval-q0')['state'] == 'failed'
    assert store.question('mem0', 'longmemeval-q1')['state'] == 'success'
    assert len(store.predictions()) == 12
    assert not any(row['method'] == 'ours' and row['dataset'] == 'locomo' for row in store.predictions())
    assert store.failures()[1]['raw_response'] == 'invalid raw'
    store.close()
    assert json.loads((directory / 'manifest.json').read_text())['status'] == 'incomplete'
    state.calls.clear(); state.searches.clear(); state.ingestions.clear()
    runner.run(state.root, resume_dir=directory)
    assert not state.calls and not state.searches and not state.ingestions
    result = runner.report(directory)
    assert result['task_counts']['qa']['blocked'] == 2 and result['task_counts']['qa']['failed'] == 2
    assert result['common_completed']['locomo']['count'] == 0
    assert result['common_completed']['longmemeval']['ids'] == ['longmemeval-q1']


def test_resume_uses_sqlite_stages_and_ignores_incomplete_jsonl_exports(execution):
    state = execution
    directory = runner.run(state.root)
    assert len(state.searches) == 16 and len(state.calls) == 24
    store = TaskStore(directory / 'tasks.sqlite3')
    task = store.question('ours', 'longmemeval-q1')
    # Crash after judge was saved but before local scoring: no paid stage needs replay.
    store.connection.execute("DELETE FROM stages WHERE task_id=? AND name='score'", (task['task_id'],))
    store.connection.execute("UPDATE tasks SET state='running' WHERE task_id=?", (task['task_id'],))
    store.connection.commit(); store.close()
    for name in ('ingestion', 'predictions', 'answers', 'judges', 'retrieval'):
        (directory / f'{name}.jsonl').write_text('{torn export')
    state.calls.clear(); state.searches.clear(); state.ingestions.clear()
    runner.run(state.root, resume_dir=directory)
    assert not state.calls and not state.searches and not state.ingestions
    assert len(runner.records(directory / 'predictions.jsonl')) == 16
    assert len(runner.records(directory / 'judges.jsonl')) == 8
    contexts = state.contexts
    assert all(context['task_id'] and context['unit_id'] and context['protocol_sha256'] for context in contexts)
    source = state.root / 'src/week1_runner.py'
    source.write_text(source.read_text() + '\n# changed protocol\n')
    with pytest.raises(ValueError, match='protocol changed'):
        runner.run(state.root, resume_dir=directory)


def test_add_checkpoint_resume_keeps_completed_20_message_unit(execution):
    state = execution
    state.interrupt = ('ingestion', 'ours', 'locomo', 'session_1:20')
    directory = runner.run(state.root)
    store = TaskStore(directory / 'tasks.sqlite3')
    task = store.build('ours', 'locomo')
    assert task['completed_units'] == 1 and task['total_units'] == 2
    assert json.loads((Path(task['index_path']) / 'native-state.json').read_text())['units'] == ['session_1:0']
    assert not store.predictions() or all(row['method'] != 'ours' for row in store.predictions())
    store.close()
    assert json.loads((directory / 'manifest.json').read_text())['status'] == 'interrupted'
    state.calls.clear(); state.searches.clear(); state.ingestions.clear()
    runner.run(state.root, resume_dir=directory)
    assert ('ours', 'locomo', 'session_1:0') not in state.ingestions
    assert state.ingestions[0] == ('ours', 'locomo', 'session_1:20')
    store = TaskStore(directory / 'tasks.sqlite3')
    assert len(store.predictions()) == 16
    assert store.build('ours', 'locomo')['receipt']['source_messages'] == 21
    store.close()


def test_memorybank_search_failure_rolls_back_but_answer_failure_keeps_once_only_reinforcement(execution):
    state = execution
    state.fail_question.add(('memorybank', 'locomo', 'question-0'))
    state.fail_stage.add(('memorybank', 'longmemeval-q0', 'answer'))
    directory = runner.run(state.root)
    store = TaskStore(directory / 'tasks.sqlite3')
    failed_search = store.question('memorybank', 'locomo-q0')
    assert store.stage(failed_search['task_id'], 'retrieval') is None
    locomo = store.build('memorybank', 'locomo')
    assert json.loads((Path(locomo['index_path']) / 'native-state.json').read_text())['retrievals'] == ['question-1']
    lme = store.build('memorybank', 'longmemeval')
    assert json.loads((Path(lme['index_path']) / 'native-state.json').read_text())['retrievals'] == ['question-0', 'question-1']
    failed_answer = store.question('memorybank', 'longmemeval-q0')
    assert store.stage(failed_answer['task_id'], 'retrieval') is not None
    assert store.stage(failed_answer['task_id'], 'answer') is None
    store.close()
    state.calls.clear(); state.searches.clear(); state.ingestions.clear()
    runner.run(state.root, resume_dir=directory)
    assert not state.calls and not state.searches and not state.ingestions


def test_budget_stop_still_scores_saved_local_answers_and_judge(execution):
    state = execution
    directory = runner.run(state.root)
    store = TaskStore(directory / 'tasks.sqlite3')
    for task in store.rows('qa'):
        store.connection.execute("DELETE FROM stages WHERE task_id=? AND name='score'", (task['task_id'],))
        store.connection.execute("UPDATE tasks SET state='running' WHERE task_id=?", (task['task_id'],))
    store.connection.commit(); store.close()
    state.budget_stopped = 'week1_budget_exhausted'
    state.calls.clear(); state.searches.clear(); state.ingestions.clear()
    runner.run(state.root, resume_dir=directory)
    assert not state.calls and not state.searches and not state.ingestions
    store = TaskStore(directory / 'tasks.sqlite3')
    assert len(store.predictions()) == 16
    store.close()
    status = json.loads((directory / 'runtime-status.json').read_text())
    assert status['stage'] == 'stopped' and status['progress']['qa']['success'] == 16
    assert status['pid'] > 0


def test_report_rejects_scores_with_removed_full_history_receipt(execution):
    directory = runner.run(execution.root)
    store = TaskStore(directory / 'tasks.sqlite3')
    task = store.build('rag', 'locomo')
    store.connection.execute("UPDATE tasks SET receipt=NULL WHERE task_id=?", (task['task_id'],))
    store.connection.commit(); store.close()
    with pytest.raises(ValueError, match='incomplete or different history'):
        runner.report(directory)


@pytest.mark.parametrize('stage', ['ingestion', 'search'])
def test_interrupt_survives_native_cleanup_timeout_and_stops_later_tasks(execution, stage):
    state = execution
    state.close_error_on_interrupt = True
    state.interrupt = ((stage, 'ours', 'locomo', 'session_1:20') if stage == 'ingestion' else
                       (stage, 'memorybank', 'locomo', 'question-0'))
    directory = runner.run(state.root)
    assert json.loads((directory / 'manifest.json').read_text())['status'] == 'interrupted'
    assert all(dataset == 'locomo' for _, dataset, _ in state.ingestions)
    assert all(dataset == 'locomo' for _, dataset, _ in state.searches)
    assert not any(group == 'longmemeval' for _, group, _, _ in state.calls)
    store = TaskStore(directory / 'tasks.sqlite3')
    assert not store.failures()
    assert all(task['state'] == 'pending' for task in store.rows('build') if task['dataset'] == 'longmemeval')
    store.close()


def test_another_executor_cannot_reset_running_tasks_before_gateway_lock(execution):
    import fcntl

    directory = runner.run(execution.root)
    lock = (directory / 'executor.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    execution.calls.clear(); execution.searches.clear(); execution.ingestions.clear()
    try:
        with pytest.raises(ValueError, match='already running'):
            runner.run(execution.root, resume_dir=directory)
    finally:
        lock.close()
    assert not execution.calls and not execution.searches and not execution.ingestions


def test_locomo_run_uses_selected_methods_cap_and_f1_without_judge(execution, monkeypatch):
    state = execution
    examples = [e for e in state.examples if e['dataset'] == 'locomo']
    monkeypatch.setattr(runner, 'preflight_locomo', lambda *args: None)
    monkeypatch.setattr(runner, 'load_locomo', lambda root, cap=None: examples[:cap])
    monkeypatch.setattr(runner, 'locomo_manifest', lambda root: {'metric': 'Token F1'})
    monkeypatch.setattr(runner, 'load_pilot', lambda *args: pytest.fail('loaded pilot/LME'))
    monkeypatch.setattr(runner, 'preflight_memorybank', lambda *args: pytest.fail('wrong benchmark'))
    directory = runner.run(state.root, benchmark='locomo', methods=['ours', 'memorybank'], max_questions=1)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'finished'
    assert manifest['methods'] == ['ours', 'memorybank']
    assert manifest['requested_per_method'] == 1
    assert manifest['budget_usd'] == 20
    assert len(json.loads((directory / 'selection.json').read_text())['examples']) == 1
    assert {call[3] for call in state.calls} == {'answer'}
    assert {call[0] for call in state.calls} == {'ours', 'memorybank'}
    assert runner.report(directory)['requested_predictions'] == 2
    runner.run(state.root, resume_dir=directory, benchmark='locomo', methods=['ours', 'memorybank'], max_questions=1)
    with pytest.raises(ValueError, match='selection|cap'):
        runner.run(state.root, resume_dir=directory, benchmark='locomo', methods=['ours', 'memorybank'], max_questions=2)


def test_locomo_native_population_and_cap_without_lme(tmp_path, monkeypatch):
    from src import week1_data as data
    import shutil
    root = Path(runner.__file__).parents[1]
    path = tmp_path / 'data/week1/datasets/locomo10.json'
    path.parent.mkdir(parents=True)
    shutil.copyfile(root / 'data/week1/datasets/locomo10.json', path)
    monkeypatch.setattr(data, '_download', lambda *args: pytest.fail('network fetch'))
    full = data.load_locomo(tmp_path)
    assert len(full) == 1540
    assert len({e['group'] for e in full}) == 10
    assert all(e['category'] != 5 for e in full)
    capped = data.load_locomo(tmp_path, 2)
    assert capped == full[:2]
    assert capped[0]['sessions'] == full[0]['sessions']
    with pytest.raises(ValueError, match='positive'):
        data.load_locomo(tmp_path, 0)


def test_locomo_preflight_uses_exported_openai_key_only(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-exported-key')
    monkeypatch.delenv('AML_MEMORY_KEY', raising=False)
    monkeypatch.setattr(runner, 'load_environment', lambda *args: pytest.fail('read dotenv'))
    monkeypatch.setattr(runner, 'locomo_manifest', lambda *args: {})
    monkeypatch.setattr(runner, 'load_locomo', lambda *args: [{'group': 'g'}])
    assert runner.preflight_locomo(tmp_path, ['ours'])['api_key_configured']
    assert runner.active_methods('locomo') == ('ours', 'mem0', 'memorybank')
    for methods in ([], ['rag'], ['ours', 'ours']):
        with pytest.raises(ValueError):
            runner.active_methods('locomo', methods)
