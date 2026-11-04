import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import ServiceConfig, load_config
from src.contracts import AddRequest, Message, SearchRequest
from src.engine import MemoryEngine
from src.service import create_app


def entry(chunk, *, quote=None, topic="entities", scope="default", parent_index=None, occurred=None,
          event_status=None, conditions=None, verification=None, occurrence=0):
    return {"content": quote or chunk["content"], "source_chunk_ids": [chunk["id"]],
            "source_quotes": [{"chunk_id": chunk["id"], "quote": quote or chunk["content"], "occurrence": occurrence}],
            "topic": topic, "scope": scope, "parent_index": parent_index, "event_status": event_status,
            "conditions": conditions, "verification": verification,
            "occurred_start": occurred, "occurred_end": occurred,
            "occurred_precision": "day" if occurred else None, "mentioned_at": None}


class Model:
    def __init__(self):
        self.calls = []
        self.builder = lambda chunks: [entry(c) for c in chunks]

    def extract(self, prompt, messages, chunks, temporal=False, project_id=None):
        self.calls.append((messages, chunks, project_id))
        return self.builder(chunks)

    def rewrite(self, prompt, jobs):
        return {job["id"]: job["previous_text"] + " / " + job["pending_facts"][-1]["content"] for job in jobs}


class Embedder:
    def __init__(self):
        self.calls = []
        self.fail = False

    def embed(self, texts):
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[1.0] + [0.0] * 1023 for _ in texts]


class Relations:
    def __init__(self):
        self.calls = []
        self.relation = "same_matter"

    def decide(self, state, criteria):
        self.calls.append(state)
        choice = next((key for key in criteria if key.endswith("_" + self.relation)), "new")
        return {"choice": choice, "model": "test", "probabilities": {key: float(key == choice) for key in criteria}}


class Selector:
    def __init__(self):
        self.calls = []

    def select(self, query, options, candidates):
        self.calls.append(candidates)
        return [{"id": c["id"], "choice": "relevant", "probabilities": {
            "relevant": 1., "irrelevant": 0., "uncertain": 0.}} for c in candidates]


@pytest.fixture
def memory(tmp_path):
    config, path = load_config("configs/service/vikingjev.yaml", ServiceConfig)
    config = config.model_copy(deep=True)
    config.database_dir = str(tmp_path / "index")
    model, embedder, relations, selector = Model(), Embedder(), Relations(), Selector()
    return MemoryEngine(config, path, model, embedder, relations, selector), model, embedder, relations, selector


def add(engine, text, identity="r", project=None, session="s", user="u"):
    return engine.add(AddRequest(request_id=identity, user_id=user, session_id=session, project_id=project,
                                 messages=[Message(role="user", content=text)]))


def search(engine, project=None, time_range=None, user="u"):
    return engine.search(SearchRequest(query="memory", user_id=user, project_id=project,
                                       event_time_range=time_range, top_k=100)).data


def test_mixed_sentence_scopes_are_isolated_and_auxiliary_is_independent(memory):
    engine, model, _, _, _ = memory
    text = "我偏好中文。项目要求输出英文。"
    model.builder = lambda chunks: [entry(chunks[0], quote="我偏好中文。", topic="preferences"),
                                    entry(chunks[0], quote="项目要求输出英文。", topic="requirements", parent_index=0)]
    add(engine, text, project="A")
    assert [e.content for e in search(engine)] == ["我偏好中文。"]
    assert [e.content for e in search(engine, "B")] == ["我偏好中文。"]
    evidence = search(engine, "A")
    assert {e.content for e in evidence} == {"我偏好中文。", "项目要求输出英文。"}
    requirement = next(e for e in evidence if e.topic == "requirements")
    assert requirement.scope == "project" and requirement.project_id == "A" and requirement.requirement_priority == 2
    with engine.store.connect("u") as db:
        auxiliary = db.execute("SELECT * FROM facts WHERE parent_fact_id IS NOT NULL").fetchone()
        assert auxiliary["id"] in requirement.entry_ids
        db.execute("DELETE FROM embeddings WHERE owner_type='fact' AND owner_id=?", (auxiliary["parent_fact_id"],))
    assert any(e.content == "项目要求输出英文。" for e in search(engine, "A"))
    source = next(engine.store.root.rglob("messages.json"))
    assert json.loads(source.read_text())["messages"][0]["content"] == text


def test_unresolved_entry_kept_without_poisoning_global_add_or_model_calls(memory):
    engine, model, embedder, relations, _ = memory
    text = "I like tea. Project-only: use English."
    model.builder = lambda chunks: [entry(chunks[0], quote="I like tea.", topic="preferences"),
        entry(chunks[0], quote="Project-only: use English.", topic="requirements", scope="project")]
    response = add(engine, text)
    assert response.success and response.unresolved_entry_count == 1 and response.warnings
    assert [e.content for e in search(engine)] == ["I like tea."]
    assert all("Project-only" not in text for batch in embedder.calls for text in batch)
    assert len(relations.calls) == 1
    with engine.store.connect("u") as db:
        pending = db.execute("SELECT * FROM facts WHERE scope='unresolved'").fetchone()
        assert pending and pending["project_id"] is None
        assert db.execute("SELECT 1 FROM embeddings WHERE owner_id=?", (pending["id"],)).fetchone() is None
    add(engine, text, identity="explicit", project="A")
    assert any(e.content == "Project-only: use English." for e in search(engine, "A"))
    with engine.store.connect("u") as db:
        assert db.execute("SELECT COUNT(*) FROM facts WHERE scope='unresolved'").fetchone()[0] == 1


@pytest.mark.parametrize("project", [None, "", "   "])
def test_missing_null_and_blank_project_share_idempotency(memory, project):
    engine, model, embedder, _, _ = memory
    request = AddRequest(request_id="same", user_id="u", session_id="s", messages=[Message(role="user", content="global")])
    first = engine.add(request)
    assert engine.add(request.model_copy(update={"project_id": project}) if project is None else
                      AddRequest(**{**request.model_dump(), "project_id": project})) == first
    assert len(model.calls) == len(embedder.calls) - 1 == 1
    with pytest.raises(FileExistsError):
        engine.add(request.model_copy(update={"project_id": "A"}))


def test_history_context_and_jev_candidates_are_scoped_before_model_input(memory):
    engine, model, _, relations, selector = memory
    add(engine, "secret A", identity="a", project="A")
    add(engine, "shared", identity="global")
    add(engine, "private B", identity="b", project="B")
    context = model.calls[-1][0]
    assert [item["content"] for item in context] == ["shared"]
    assert not relations.calls[-1]["candidate_observations"]
    engine.config.retrieval.directory_routing_threshold = 0
    assert {e.content for e in search(engine, "B")} == {"shared", "private B"}
    assert [c["id"] for c in selector.calls[-1]] == ["entities"]
    assert search(engine, "A", user="another") == []


def test_scope_precedes_vector_count_and_rejected_vectors_are_not_read(memory):
    engine, _, _, _, selector = memory
    for i in range(3):
        add(engine, f"secret A {i}", identity=f"a{i}", project="A", session=f"a{i}")
    add(engine, "B", identity="b", project="B")
    engine.config.retrieval.directory_routing_threshold = 3
    with engine.store.connect("u") as db:
        ids = [row[0] for row in db.execute("SELECT id FROM facts WHERE project_id='A'")]
        for identity in ids:
            db.execute("UPDATE embeddings SET vector=? WHERE owner_type='fact' AND owner_id=?", (b"bad", identity))
            db.execute("UPDATE embeddings SET vector=? WHERE owner_type='chunk' AND owner_id IN (SELECT chunk_id FROM fact_sources WHERE fact_id=?)", (b"bad", identity))
            db.execute("UPDATE embeddings SET vector=? WHERE owner_type='observation' AND owner_id IN (SELECT observation_id FROM observation_sources WHERE fact_id=?)", (b"bad", identity))
    assert [e.content for e in search(engine, "B")] == ["B"]
    assert not selector.calls


def test_requirements_keep_both_sources_and_scope_priority(memory):
    engine, model, _, _, _ = memory
    model.builder = lambda chunks: [entry(c, topic="requirements") for c in chunks]
    add(engine, "Use Chinese.", identity="global")
    add(engine, "Use English.", identity="local", project="A")
    evidence = search(engine, "A")
    assert {(e.content, e.requirement_priority) for e in evidence} == {("Use Chinese.", 1), ("Use English.", 2)}
    assert [(e.content, e.requirement_priority) for e in search(engine, "B")] == [("Use Chinese.", 1)]


def test_same_matter_cross_sessions_and_project_isolation(memory):
    engine, model, _, _, _ = memory
    model.builder = lambda chunks: [entry(c, topic="events", event_status="planned", occurred="2026-10-11") for c in chunks]
    add(engine, "Launch planned", identity="first", project="A", session="one")
    add(engine, "Launch progressed", identity="second", project="A", session="two")
    add(engine, "Launch planned", identity="third", project="B", session="three")
    with engine.store.connect("u") as db:
        a = [dict(r) for r in db.execute("SELECT * FROM facts WHERE project_id='A'")]
        b = db.execute("SELECT * FROM facts WHERE project_id='B'").fetchone()
        assert len(a) == 2 and a[0]["matter_id"] == a[1]["matter_id"] != b["matter_id"]
        assert a[0]["logical_path"] == a[1]["logical_path"]
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3


@pytest.mark.parametrize("time_range,expected", [
    ({"start": "2026-10-10", "end": "2026-10-12"}, {"old", "latest"}),
    ({"end": "2026-10-11"}, {"old"}), ({"start": "2026-10-11"}, {"latest"}),
    ({}, {"old", "latest", "undated"}), (None, {"old", "latest", "undated"}),
])
def test_time_filter_retraces_only_matching_sources_even_in_shared_matter(memory, time_range, expected):
    engine, model, _, relations, _ = memory
    relations.relation = "supplement"
    dates = {"latest": "2026-10-11", "old": "2026-10-10", "undated": None}
    model.builder = lambda chunks: [entry(c, topic="events", occurred=dates[c["content"]]) for c in chunks]
    for text in ("latest", "old", "undated"):
        add(engine, text, identity=text, session=text)
    assert {e.content for e in search(engine, time_range=time_range)} == expected


def test_methods_conditions_and_source_declared_verification_are_retained(memory):
    engine, model, _, _, _ = memory
    model.builder = lambda chunks: [entry(chunks[0], topic="methods", conditions="Only offline", verification="User reports tested")]
    add(engine, "Run this offline", project="A")
    with engine.store.connect("u") as db:
        f = db.execute("SELECT * FROM facts").fetchone()
        assert (f["conditions"], f["verification"]) == ("Only offline", "User reports tested")


@pytest.mark.parametrize("mutation", ["bad_quote", "foreign_source", "bad_parent", "cycle", "bad_topic"])
def test_invalid_extraction_rolls_back_sources_and_records(memory, mutation):
    engine, model, _, _, _ = memory
    def build(chunks):
        value = entry(chunks[0])
        if mutation == "bad_quote": value["source_quotes"][0]["quote"] = "absent"
        if mutation == "foreign_source": value["source_chunk_ids"] = ["old"]
        if mutation == "bad_parent": value["parent_index"] = 5
        if mutation == "bad_topic": value["topic"] = "work"
        if mutation == "cycle":
            return [{**value, "parent_index": 1}, {**value, "parent_index": 0}]
        return [value]
    model.builder = build
    with pytest.raises(ValueError):
        add(engine, "original")
    with engine.store.connect("u") as db:
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    assert not list(engine.store.root.rglob("messages.json"))


def test_failed_embedding_does_not_commit_unresolved_side_entries(memory):
    engine, model, embedder, _, _ = memory
    model.builder = lambda chunks: [entry(chunks[0], quote="global"),
        entry(chunks[0], quote="project", topic="requirements", scope="project")]
    embedder.fail = True
    with pytest.raises(RuntimeError):
        add(engine, "global project")
    with engine.store.connect("u") as db:
        assert db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0


def test_unicode_windows_recover_exact_quotes(memory):
    engine, _, _, _, _ = memory
    engine.config.chunk.max_tokens = 3
    engine.config.chunk.overlap_tokens = 1
    text = "你好世界🙂我们一起测试中文来源。"
    add(engine, text)
    assert all(e.content.replace("\n", "") in text for e in search(engine))


def test_empty_extraction_preserves_raw_without_unclassified_retrieval(memory):
    engine, model, embedder, _, selector = memory
    model.builder = lambda chunks: []
    add(engine, "Hello")
    assert search(engine) == [] and not embedder.calls and not selector.calls
    assert next(engine.store.root.rglob("messages.json")).exists()


def test_unsupported_schema_is_rejected_without_migration(memory):
    engine, _, _, _, _ = memory
    add(engine, "text")
    with engine.store.connect("u") as db:
        db.execute("UPDATE metadata SET value='4' WHERE key='schema_version'")
    with pytest.raises(ValueError, match="new database"):
        search(engine)


def test_http_optional_defaults_and_invalid_explicit_values(memory, monkeypatch):
    engine, model, embedder, relations, selector = memory
    monkeypatch.setenv(engine.config.auth_key_env, "secret")
    client = TestClient(create_app(engine.config, Path("configs/service/vikingjev.yaml").resolve(), model, embedder, relations, selector))
    headers = {"X-Api-Key": "secret"}
    response = client.post("/add", headers=headers, json={"request_id": "r", "user_id": "u", "session_id": "s",
        "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert client.post("/search", headers=headers, json={"query": "hello", "user_id": "u", "top_k": 1,
        "event_time_range": {}}).status_code == 200
    for bad in ({"project_id": 1}, {"event_time_range": {"start": "bad"}},
                {"event_time_range": {"start": "2026-11", "end": "2026-10"}}):
        assert client.post("/search", headers=headers, json={"query": "hello", "user_id": "u", "top_k": 1, **bad}).status_code == 422
