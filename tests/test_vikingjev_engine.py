import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from src.config import ServiceConfig, load_config
from src.contracts import AddRequest, Message, SearchRequest
from src.engine import MemoryEngine


class FactModel:
    def __init__(self):
        self.calls = []
        self.fail = False

    def extract(self, prompt, messages, chunks, temporal=False, project_id=None):
        self.calls.append((messages, chunks))
        if self.fail:
            raise RuntimeError("extraction unavailable")
        return [{"content": "DERIVED FACT: " + chunk["content"], "source_chunk_ids": [chunk["id"]],
                 "source_quotes": [{"chunk_id": chunk["id"], "quote": chunk["content"], "occurrence": 0}],
                 "topic": "entities" if "UNRELATED" in chunk["content"] or "SECRET DISTRACTOR" in chunk["content"] else "preferences",
                 "scope": "default", "parent_index": None, "event_status": None, "conditions": None,
                 "verification": None, "occurred_start": None, "occurred_end": None,
                 "occurred_precision": None, "mentioned_at": None}
                for chunk in chunks]

    def rewrite(self, prompt, jobs):
        return {job["id"]: "DERIVED OBSERVATION" for job in jobs}


class Embedder:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.fail_at = None

    def embed(self, texts):
        self.calls.append(texts)
        if self.fail or len(self.calls) == self.fail_at:
            raise RuntimeError("embedding unavailable")
        return [[1.0] + [0.0] * 1023 for _ in texts]


class Relations:
    def __init__(self):
        self.calls = []
        self.fail = False

    def decide(self, state, criteria):
        self.calls.append((state, criteria))
        if self.fail:
            raise RuntimeError("relation unavailable")
        choice = next((key for key in criteria if key.endswith("_supplement")), "new")
        return {"choice": choice, "probabilities": {key: float(key == choice) for key in criteria}, "model": "fake"}


class Selector:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.choose = lambda item: "relevant"

    def select(self, query, options, candidates):
        self.calls.append((query, options, candidates))
        if self.fail:
            raise RuntimeError("selector unavailable")
        return [{"id": item["id"], "choice": self.choose(item), "model": "fake", "probabilities": {key: float(key == self.choose(item)) for key in ("relevant", "irrelevant", "uncertain")}}
                for item in candidates]


@pytest.fixture
def memory(tmp_path):
    config, path = load_config("configs/service/jevsight-d.yaml", ServiceConfig)
    config = config.model_copy(update={"database_dir": str(tmp_path / "index"),
                                       "directory_selector": config.consolidation.decision_model})
    config.retrieval.directory_routing_threshold = 0
    llm, embedder, relations, selector = FactModel(), Embedder(), Relations(), Selector()
    engine = MemoryEngine(config, path, llm, embedder, decision_model=relations, directory_selector=selector)
    return engine, llm, embedder, relations, selector


def request(identity, text, user="u1", session="s1"):
    return AddRequest(request_id=identity, user_id=user, session_id=session,
                      messages=[Message(role="user", content=text, timestamp=1704067200000)])


def search(engine, user="u1"):
    return engine.search(SearchRequest(query="What do I prefer?", user_id=user, top_k=20))


def counts(engine, user="u1"):
    with engine.store.connect(user) as db:
        return {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in (
            "messages", "chunks", "facts", "fact_sources", "observations", "observation_revisions",
            "observation_sources", "consolidation_decisions", "embeddings", "requests",
            "memory_directories", "message_sources", "chunks_fts")}


def source_files(engine):
    return {path.relative_to(engine.store.root): path.read_bytes()
            for path in engine.store.root.rglob("messages.json")}


def test_directory_sources_and_vectors_are_separate_and_search_returns_original(memory):
    engine, _, _, _, selector = memory
    raw = "My favorite drink is tea, served without sugar."
    engine.add(request("r1", raw))
    with engine.store.connect("u1") as db:
        directory = dict(db.execute("SELECT * FROM memory_directories").fetchone())
        assert db.execute("SELECT content FROM messages").fetchone()[0] == ""
        assert db.execute("SELECT content FROM chunks").fetchone()[0] == ""
        assert {row[0] for row in db.execute("SELECT DISTINCT owner_type FROM embeddings")} == {
            "chunk", "fact", "observation"}
        assert db.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
        assert db.execute("SELECT directory_id FROM message_sources").fetchone()[0] == directory["id"]
    path = engine.store.root / directory["path"]
    assert (path / ".abstract.md").read_text() == directory["abstract"]
    assert (path / ".overview.md").read_text() == directory["overview"]
    source = (path / "messages.json").read_bytes()
    assert hashlib.sha256(source).hexdigest() == directory["source_sha256"]
    assert json.loads(source)["messages"][0]["content"] == raw
    result = search(engine)
    assert [item.content for item in result.data] == [raw]
    assert all("DERIVED" not in item.content for item in result.data)
    assert [t["id"] for t in selector.calls[0][2]] == ["preferences"]


def test_topic_routing_requires_local_evidence_retrieval(memory, monkeypatch):
    engine, _, _, _, _ = memory
    engine.add(request("r1", "I prefer jasmine tea."))
    monkeypatch.setattr(engine, "_vector_rank", lambda db, vector, kind, limit: [])
    monkeypatch.setattr(engine, "_bm25_rank", lambda *args: [])
    assert search(engine).data == []


def test_original_chunk_offsets_and_neighbor_windows_are_reconstructed(memory, monkeypatch):
    engine, _, _, _, _ = memory
    raw = " ".join(f"word{index}" for index in range(1500))
    engine.add(request("r1", raw))
    with engine.store.connect("u1") as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY chunk_seq")]
    assert len(rows) > 4
    expected = "\n".join(raw[max(row["char_start"], rows[index]["char_end"] if index else 0):row["char_end"]]
                         for index, row in enumerate(rows[1:4]))
    monkeypatch.setattr(engine, "_vector_rank", lambda db, vector, kind, limit:
                        [rows[2]["id"]] if kind == "chunk" else [])
    monkeypatch.setattr(engine, "_bm25_rank", lambda *args: [])
    assert [item.content for item in search(engine).data] == [expected]
    assert expected != raw


def test_restart_idempotence_skips_models_and_prior_context_comes_from_original(memory):
    engine, llm, embedder, relations, selector = memory
    first = request("r1", "I prefer green tea.")
    original_response = engine.add(first)
    before = counts(engine), source_files(engine)
    call_counts = len(llm.calls), len(embedder.calls), len(relations.calls), len(selector.calls)
    restarted = MemoryEngine(engine.config, Path("configs/service/jevsight-d.yaml").resolve(), llm, embedder,
                             decision_model=relations, directory_selector=selector)
    assert restarted.add(first) == original_response
    assert (counts(engine), source_files(engine)) == before
    assert (len(llm.calls), len(embedder.calls), len(relations.calls), len(selector.calls)) == call_counts
    restarted.add(request("r2", "I now drink it every morning."))
    assert llm.calls[1][0][0]["content"] == first.messages[0].content
    assert "DERIVED" not in llm.calls[1][0][0]["content"]
    assert "green tea" in "\n".join(item.content for item in search(restarted).data)


def test_unsafe_user_and_session_ids_stay_inside_root_and_users_remain_isolated(memory):
    engine, _, _, _, _ = memory
    user = "../../outside-user"
    session = "/tmp/../../outside-session"
    engine.add(request("../request", "Private tea preference.", user=user, session=session))
    engine.add(request("../request", "Other user's coffee preference.", user="other", session=session))
    with engine.store.connect(user) as db:
        relative = db.execute("SELECT path FROM memory_directories").fetchone()[0]
    path = (engine.store.root / relative).resolve()
    assert path.is_relative_to(engine.store.root.resolve())
    assert "outside-user" not in relative and "outside-session" not in relative
    assert ".." not in Path(relative).parts
    assert [item.content for item in search(engine, user).data] == ["Private tea preference."]
    assert [item.content for item in search(engine, "other").data] == ["Other user's coffee preference."]
    assert search(engine, "missing").data == []


def test_probability_routing_keeps_uncertain_topic_and_excludes_irrelevant_topic(memory):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "SECRET DISTRACTOR: the dishwasher is blue."))
    engine.add(request("r2", "I prefer tea."))
    selector.choose = lambda item: "irrelevant" if item["id"] == "entities" else "uncertain"
    response = search(engine)
    assert [item.content for item in response.data] == ["I prefer tea."]
    assert len(selector.calls[0][2]) == 2


def test_all_irrelevant_topics_return_empty(memory):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "This is unrelated."))
    selector.choose = lambda item: "irrelevant"
    assert search(engine).data == []


def test_selector_failure_is_surfaced_without_unfiltered_fallback(memory):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "I prefer tea."))
    selector.fail = True
    with pytest.raises(RuntimeError, match="selector unavailable"):
        search(engine)


@pytest.mark.parametrize("result", [
    [{"id": "invented", "choice": "relevant"}],
    [],
])
def test_unknown_or_missing_topic_decisions_are_rejected(memory, monkeypatch, result):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "I prefer tea."))
    monkeypatch.setattr(selector, "select", lambda *args: result)
    with pytest.raises(ValueError, match="invalid candidate decisions"):
        search(engine)


def test_unknown_topic_choice_is_rejected(memory):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "I prefer tea."))
    selector.choose = lambda item: "invented"
    with pytest.raises(ValueError, match="invalid candidate decisions"):
        search(engine)


@pytest.mark.parametrize("damage", ["corrupt", "delete"])
def test_missing_or_modified_original_fails_instead_of_using_index_text(memory, damage):
    engine, _, _, _, _ = memory
    engine.add(request("r1", "I prefer tea."))
    path = next(engine.store.root.rglob("messages.json"))
    if damage == "delete":
        path.unlink()
    else:
        path.write_text('{"messages": []}')
    with pytest.raises((ValueError, FileNotFoundError)):
        search(engine)


@pytest.mark.parametrize("phase", ["extraction", "embedding", "relation", "observation_embedding"])
def test_failed_models_preserve_registered_directories_sources_and_observations(memory, phase):
    engine, llm, embedder, relations, _ = memory
    engine.add(request("r1", "I prefer tea."))
    before = counts(engine), source_files(engine)
    if phase == "extraction":
        llm.fail = True
    elif phase == "embedding":
        embedder.fail = True
    elif phase == "relation":
        relations.fail = True
    else:
        embedder.fail_at = len(embedder.calls) + 2
    with pytest.raises(RuntimeError):
        engine.add(request("r2", "Now I prefer coffee."))
    assert (counts(engine), source_files(engine)) == before


def test_failed_sql_insert_rolls_back_published_source_and_all_index_tables(memory):
    engine, _, _, _, _ = memory
    engine.add(request("r1", "I prefer tea."))
    before = counts(engine), source_files(engine)
    with engine.store.connect("u1") as db:
        db.execute("""CREATE TRIGGER reject_directory BEFORE INSERT ON memory_directories
                      BEGIN SELECT RAISE(ABORT, 'forced directory insert error'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="forced directory insert error"):
        engine.add(request("r2", "Now I prefer coffee."))
    assert (counts(engine), source_files(engine)) == before
    assert not list(engine.store.root.rglob(".pending-*"))
    assert [item.content for item in search(engine).data] == ["I prefer tea."]


def test_interruption_after_commit_preserves_sources_and_retry_is_idempotent(memory, monkeypatch):
    engine, llm, embedder, relations, selector = memory
    first = request("r1", "I prefer jasmine tea.")
    commit = engine._commit_add

    def interrupt_after_commit(*args, **kwargs):
        commit(*args, **kwargs)
        raise KeyboardInterrupt("interrupted after commit")

    monkeypatch.setattr(engine, "_commit_add", interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt, match="interrupted after commit"):
        engine.add(first)
    before = counts(engine), source_files(engine)
    assert before[0]["requests"] == before[0]["memory_directories"] == 1
    assert len(before[1]) == 1
    call_counts = len(llm.calls), len(embedder.calls), len(relations.calls), len(selector.calls)
    response = engine.add(first)
    assert response.success and response.request_id == first.request_id
    assert (counts(engine), source_files(engine)) == before
    assert (len(llm.calls), len(embedder.calls), len(relations.calls), len(selector.calls)) == call_counts
    assert [item.content for item in search(engine).data] == [first.messages[0].content]


def test_interruption_before_commit_removes_published_sources_and_leaves_no_records(memory, monkeypatch):
    engine, _, _, _, _ = memory

    def interrupt_before_commit(*args, **kwargs):
        assert len(source_files(engine)) == 1
        raise KeyboardInterrupt("interrupted before commit")

    monkeypatch.setattr(engine, "_commit_add", interrupt_before_commit)
    with pytest.raises(KeyboardInterrupt, match="interrupted before commit"):
        engine.add(request("r1", "I prefer jasmine tea."))
    assert all(count == 0 for count in counts(engine).values())
    assert source_files(engine) == {}
    assert not list(engine.store.root.rglob(".pending-*"))
    assert search(engine).data == []


def test_many_directories_require_only_one_parent_decision_request(memory):
    engine, _, _, _, selector = memory
    for index in range(11):
        engine.add(request(f"r{index}", f"Preference number {index}.", session=f"session-{index}"))
    response = engine.search(SearchRequest(query="Preference", options=["tea", "coffee"], user_id="u1", top_k=20))
    assert len(selector.calls) == 1
    query, options, directories = selector.calls[0]
    assert (query, options) == ("Preference", ["tea", "coffee"])
    assert [item["id"] for item in directories] == ["preferences"]
    assert len(response.data) == 11


def test_jev_precedes_local_retrieval_and_rejected_vectors_are_never_loaded(memory, monkeypatch):
    engine, _, _, relations, selector = memory
    relations.decide = lambda state, criteria: {"choice": "new", "model": "fake",
        "probabilities": {key: float(key == "new") for key in criteria}}
    engine.add(request("keep", "I prefer jasmine tea.", session="keep"))
    engine.add(request("drop", "UNRELATED: car maintenance.", session="drop"))
    for i in range(3):
        engine.add(request(f"extra-{i}", f"Extra tea preference {i}.", session=f"extra-{i}"))
    selector.choose = lambda item: "irrelevant" if item["id"] == "entities" else "relevant"
    with engine.store.connect("u1") as db:
        keep = db.execute("SELECT id FROM memory_directories WHERE overview NOT LIKE '%UNRELATED%'").fetchone()[0]
        drop = db.execute("SELECT id FROM memory_directories WHERE overview LIKE '%UNRELATED%'").fetchone()[0]
        # A global vector load would fail to decode these bytes.
        for kind in ("chunk", "fact", "observation"):
            if kind == "chunk":
                ids = db.execute("SELECT c.id FROM chunks c JOIN message_sources s ON s.message_id=c.message_id WHERE s.directory_id=?", (drop,))
            elif kind == "fact":
                ids = db.execute("SELECT fs.fact_id FROM fact_sources fs JOIN chunks c ON c.id=fs.chunk_id JOIN message_sources s ON s.message_id=c.message_id WHERE s.directory_id=?", (drop,))
            else:
                ids = db.execute("SELECT os.observation_id FROM observation_sources os JOIN fact_sources fs ON fs.fact_id=os.fact_id JOIN chunks c ON c.id=fs.chunk_id JOIN message_sources s ON s.message_id=c.message_id WHERE s.directory_id=?", (drop,))
            identities = [row[0] for row in ids]
            assert identities
            db.executemany("UPDATE embeddings SET vector=? WHERE owner_type=? AND owner_id=?",
                           [(b"bad", kind, identity) for identity in identities])
    calls = []
    original = engine._vector_rank

    def checked(db, vector, kind, limit):
        assert len(selector.calls) == 1
        scoped = {row[0] for row in db.execute("""SELECT DISTINCT s.directory_id FROM search_facts q
            JOIN fact_sources fs ON fs.fact_id=q.id JOIN chunks c ON c.id=fs.chunk_id
            JOIN message_sources s ON s.message_id=c.message_id""")}
        assert keep in scoped and drop not in scoped
        assert len(scoped) == 4
        calls.append(kind)
        return original(db, vector, kind, limit)

    monkeypatch.setattr(engine, "_vector_rank", checked)
    assert "I prefer jasmine tea." in [item.content for item in search(engine).data]
    assert calls == ["chunk", "fact", "observation"]


def test_bm25_scope_is_applied_before_candidate_limit(memory):
    engine, _, _, _, _ = memory
    engine.add(request("keep", "jasmine tea with breakfast", session="keep"))
    engine.add(request("drop", "UNRELATED jasmine jasmine jasmine jasmine", session="drop"))
    with engine.store.connect("u1") as db:
        directory = db.execute("SELECT id FROM memory_directories WHERE overview LIKE '%breakfast%'").fetchone()[0]
        chunk = db.execute("SELECT c.id FROM chunks c JOIN message_sources s ON s.message_id=c.message_id WHERE s.directory_id=?", (directory,)).fetchone()[0]
        engine._prepare_scope(db, SearchRequest(query="jasmine", user_id="u1", top_k=1))
        db.execute("DELETE FROM search_facts WHERE id NOT IN (SELECT id FROM facts WHERE topic='preferences')")
        assert engine._bm25_rank(db, "jasmine", 1) == [chunk]
        db.execute("DELETE FROM search_facts")
        assert engine._bm25_rank(db, "jasmine", 1) == []


def test_rejected_gap_does_not_create_new_neighbors_or_merge_windows(memory, monkeypatch):
    engine, _, _, _, selector = memory
    engine.add(request("first", "I prefer tea."))
    engine.add(request("gap", "UNRELATED message."))
    engine.add(request("last", "I also like coffee."))
    selector.choose = lambda item: "irrelevant" if item["id"] == "entities" else "relevant"
    for i in range(2):
        engine.add(request(f"extra-{i}", f"Extra preference {i}.", session=f"extra-{i}"))
    engine.config.retrieval.seed_count = 1
    with engine.store.connect("u1") as db:
        first = db.execute("SELECT id FROM chunks ORDER BY message_seq LIMIT 1").fetchone()[0]
    def rank(db, vector, kind, limit):
        return [first] if kind == "chunk" else []
    monkeypatch.setattr(engine, "_vector_rank", rank)
    monkeypatch.setattr(engine, "_bm25_rank", lambda *args: [])
    assert [item.content for item in search(engine).data] == ["I prefer tea."]


def test_evaluation_reuses_frozen_index_without_reextracting_or_counting_skips_as_writes(memory, tmp_path, monkeypatch):
    from src import evaluation, directory_selector
    from src.config import EvaluationConfig, resolved_path, write_snapshot

    engine, llm, embedder, relations, selector = memory
    llm.complete = lambda system, prompt, json_response=True: '{"label":"CORRECT"}' if json_response else "Paris"
    monkeypatch.setattr(evaluation, "OpenAILLM", lambda *a, **k: llm)
    monkeypatch.setattr(evaluation, "OpenAIEmbedder", lambda *a, **k: embedder)
    monkeypatch.setattr(evaluation, "OpenAIDecision", lambda *a, **k: relations)
    monkeypatch.setattr(directory_selector, "JevDirectorySelector", lambda *a, **k: selector)
    service = engine.config.model_copy(deep=True)
    origin = Path("configs/service/jevsight-d.yaml").resolve()
    service.facts.prompt_path = str(resolved_path(origin, service.facts.prompt_path))
    service.consolidation.prompt_path = str(resolved_path(origin, service.consolidation.prompt_path))
    service_path = tmp_path / "service.yaml"
    write_snapshot(service, service_path)
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"question_id": "q1", "question": "Where?", "answer": "Paris",
                                "haystack_sessions": [[{"role": "user", "content": "Paris"}]]}]))
    contract = tmp_path / "contract"
    pipeline = contract / "data/longmemeval-s/pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text('OPEN_ENDED_ANSWER_TEMPLATE = "{{question}} {{speaker_1_memories}}"\n'
                        'ACCURACY_PROMPT = "{question} {gold_answer} {generated_answer}"\n')
    config, path = load_config("configs/evaluation/longmemeval-dev.yaml", EvaluationConfig)
    config = config.model_copy(update={"service_config_path": str(service_path), "data_path": str(data),
        "data_sha256": evaluation.sha256(data), "contract_repository_path": str(contract),
        "output_dir": str(tmp_path / "runs")})
    monkeypatch.setattr(evaluation, "_commit", lambda *args: config.contract_commit)
    first = evaluation.evaluate(config, path)
    original_hash = evaluation.sha256(first / "memory-index")
    extraction_calls = len(llm.calls)
    second = evaluation.evaluate(config, path, reuse_index_from=first)
    metrics = json.loads((second / "metrics.json").read_text())
    assert metrics["requested"] == metrics["evaluated"] == 1 and metrics["failed"] == 0
    assert metrics["index_mode"] == "frozen-reuse"
    assert metrics["actual_ingestion_seconds"] == 0 and metrics["ingestion_groups"] == {}
    assert (second / "ingestion.jsonl").read_text() == ""
    assert len(llm.calls) == extraction_calls
    assert evaluation.sha256(first / "memory-index") == original_hash
    manifest = json.loads((first / "manifest.json").read_text())
    manifest["data_sha256"] = "wrong"
    (first / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen index run does not match"):
        evaluation.evaluate(config, path, reuse_index_from=first)


def test_invalid_extracted_topic_rolls_back_everything(memory, monkeypatch):
    engine, llm, _, _, selector = memory
    original = llm.extract
    def invalid_topic(*args, **kwargs):
        entries = original(*args, **kwargs)
        entries[0]["topic"] = "invented"
        return entries
    monkeypatch.setattr(llm, "extract", invalid_topic)
    with pytest.raises(ValueError, match="invalid extracted entry metadata"):
        engine.add(request("r1", "Tea."))
    assert all(value == 0 for value in counts(engine).values())
    assert source_files(engine) == {}
    assert selector.calls == []


def test_entry_topics_and_auxiliary_linkage_need_no_separate_classification_call(memory, monkeypatch):
    engine, llm, _, _, selector = memory
    original = llm.extract
    def extract_with_auxiliary(*args, **kwargs):
        entries = original(*args, **kwargs)
        entries.append({**entries[0], "content": "Use tea during this exercise.",
                        "topic": "requirements", "parent_index": 0})
        return entries
    monkeypatch.setattr(llm, "extract", extract_with_auxiliary)
    req = request("r1", "Tea helps me relax.")
    engine.add(req)
    engine.add(req)
    assert len(llm.calls) == 1 and selector.calls == []
    assert not hasattr(selector, "classify")
    with engine.store.connect("u1") as db:
        entries = [dict(row) for row in db.execute("SELECT id,topic,parent_fact_id FROM facts")]
        assert {entry["topic"] for entry in entries} == {"preferences", "requirements"}
        primary = next(entry for entry in entries if entry["topic"] == "preferences")
        auxiliary = next(entry for entry in entries if entry["topic"] == "requirements")
        assert primary["parent_fact_id"] is None and auxiliary["parent_fact_id"] == primary["id"]
    selector.choose = lambda item: "relevant" if item["id"] == "requirements" else "irrelevant"
    result = search(engine)
    assert len(result.data) == 1 and result.data[0].topic == "requirements"
    assert result.data[0].entry_ids == [auxiliary["id"]]
    assert result.data[0].content == req.messages[0].content


@pytest.mark.parametrize("threshold,expect_routing", [(2, True), (3, False), (4, False)])
def test_vector_threshold_boundary(memory, threshold, expect_routing):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "I prefer tea."))
    engine.config.retrieval.directory_routing_threshold = threshold
    with engine.store.connect("u1") as db:
        engine._prepare_scope(db, SearchRequest(query="tea", user_id="u1", top_k=1))
        assert engine._retrieval_vector_count(db) == 3
    assert [item.content for item in search(engine).data] == ["I prefer tea."]
    assert bool(selector.calls) == expect_routing


def test_small_memory_skips_jev_even_if_selector_unavailable(memory):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "I prefer tea."))
    engine.config.retrieval.directory_routing_threshold = 100
    selector.fail = True
    assert [item.content for item in search(engine).data] == ["I prefer tea."]
    assert not selector.calls


def test_threshold_counts_only_current_user_enabled_vector_routes(memory):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "I prefer tea."))
    engine.add(request("other", "Private other user data.", user="other"))
    engine.config.retrieval.fact_vector_weight = 0
    engine.config.retrieval.directory_routing_threshold = 2
    with engine.store.connect("u1") as db:
        engine._prepare_scope(db, SearchRequest(query="tea", user_id="u1", top_k=1))
        assert engine._retrieval_vector_count(db) == 2
    assert [item.content for item in search(engine).data] == ["I prefer tea."]
    assert not selector.calls


def test_index_threshold_is_query_only_and_growth_crosses_boundary(memory):
    from src.config import index_fingerprint
    engine, _, _, _, selector = memory
    before = index_fingerprint(engine.config, engine.prompt)
    engine.config.retrieval.directory_routing_threshold = 3
    assert index_fingerprint(engine.config, engine.prompt) == before
    engine.add(request("r1", "Tea."))
    search(engine)
    assert not selector.calls
    engine.add(request("r2", "Coffee.", session="s2"))
    search(engine)
    assert len(selector.calls) == 1
