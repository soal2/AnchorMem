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

    def extract(self, prompt, messages, chunks, temporal=False):
        self.calls.append((messages, chunks))
        if self.fail:
            raise RuntimeError("extraction unavailable")
        return [{"content": "DERIVED FACT: " + chunk["content"], "source_chunk_ids": [chunk["id"]]}
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
        return [{"id": item["id"], "choice": self.choose(item), "model": "fake", "probabilities": None}
                for item in candidates]


@pytest.fixture
def memory(tmp_path):
    config, path = load_config("configs/service/jevsight-d.yaml", ServiceConfig)
    config = config.model_copy(update={"database_dir": str(tmp_path / "index"),
                                       "directory_selector": config.consolidation.decision_model})
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
            "chunk", "fact", "observation", "directory"}
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
    assert selector.calls[0][2] == [{key: directory[key] for key in ("id", "path", "abstract", "overview")}]


def test_directory_vector_route_can_recover_evidence_without_other_routes(memory, monkeypatch):
    engine, _, _, _, _ = memory
    engine.add(request("r1", "I prefer jasmine tea."))
    with engine.store.connect("u1") as db:
        identity = db.execute("SELECT id FROM memory_directories").fetchone()[0]
    monkeypatch.setattr(engine, "_vector_rank", lambda db, vector, kind, limit: [identity] if kind == "directory" else [])
    monkeypatch.setattr(engine, "_bm25_rank", lambda *args: [])
    assert [item.content for item in search(engine).data] == ["I prefer jasmine tea."]


def test_original_chunk_offsets_and_neighbor_windows_are_reconstructed(memory, monkeypatch):
    engine, _, _, _, _ = memory
    raw = " ".join(f"word{index}" for index in range(1500))
    engine.add(request("r1", raw))
    with engine.store.connect("u1") as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY chunk_seq")]
    assert len(rows) > 4
    tokens = engine.tokenizer.encode(raw)
    expected = "\n".join(engine.tokenizer.decode(tokens[row["token_start"]:row["token_end"]]) for row in rows[1:4])
    monkeypatch.setattr(engine, "_vector_rank", lambda db, vector, kind, limit: [rows[2]["id"]] if kind == "chunk" else [])
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


def test_irrelevant_directory_cannot_return_through_session_neighbors_and_uncertain_stays(memory):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "SECRET DISTRACTOR: the dishwasher is blue."))
    engine.add(request("r2", "I prefer tea."))
    selector.choose = lambda item: "irrelevant" if "SECRET DISTRACTOR" in item["overview"] else "uncertain"
    response = search(engine)
    assert [item.content for item in response.data] == ["I prefer tea."]
    assert len(selector.calls[0][2]) == 2


def test_all_irrelevant_returns_empty(memory):
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
def test_unknown_or_missing_directory_decisions_are_rejected(memory, monkeypatch, result):
    engine, _, _, _, selector = memory
    engine.add(request("r1", "I prefer tea."))
    monkeypatch.setattr(selector, "select", lambda *args: result)
    with pytest.raises(ValueError, match="invalid candidate decisions"):
        search(engine)


def test_unknown_directory_choice_is_rejected(memory):
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


def test_selection_batches_no_more_than_ten_distinct_directories(memory):
    engine, _, _, _, selector = memory
    for index in range(11):
        engine.add(request(f"r{index}", f"Preference number {index}.", session=f"session-{index}"))
    response = engine.search(SearchRequest(query="Preference", options=["tea", "coffee"], user_id="u1", top_k=20))
    assert len(selector.calls) == 1
    query, options, directories = selector.calls[0]
    assert (query, options) == ("Preference", ["tea", "coffee"])
    assert len(directories) == len({item["id"] for item in directories}) == 10
    assert len(response.data) == 10
