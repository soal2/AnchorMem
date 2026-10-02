from pathlib import Path

import pytest

from src.config import ServiceConfig, index_fingerprint, load_config
from src.contracts import AddRequest, Message, SearchRequest
from src.engine import MemoryEngine
from src.providers import OpenAIDecision


class MemoryLLM:
    fail_rewrite = False

    def __init__(self):
        self.extractions = []
        self.rewrites = []

    def extract(self, prompt, messages, chunks, temporal=False):
        self.extractions.append((messages, chunks, temporal))
        return [{"content": chunk["content"], "source_chunk_ids": [chunk["id"]],
                 **({"occurred_start": "2020-05-01", "occurred_end": "2020-05-31",
                     "occurred_precision": "month", "mentioned_at": "2022-01-01"} if temporal else {})}
                for chunk in chunks]

    def rewrite(self, prompt, jobs):
        self.rewrites.extend(jobs)
        if self.fail_rewrite:
            raise RuntimeError("writer unavailable")
        return {job["id"]: "DERIVED ONLY: preference changed with historical evidence." for job in jobs}


class MemoryEmbedder:
    fail = False

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[1.0] + [0.0] * 1023 for _ in texts]


class Decisions:
    relation = "supplement"
    fail = False

    def __init__(self):
        self.calls = []

    def decide(self, state, criteria):
        self.calls.append((state, criteria))
        if self.fail:
            raise RuntimeError("decision unavailable")
        choice = next((key for key in criteria if key.endswith("_" + self.relation)), "new")
        return {"choice": choice, "probabilities": {key: float(key == choice) for key in criteria}, "model": "fake"}


@pytest.fixture
def memory(tmp_path):
    config, path = load_config("configs/service/jevsight-d.yaml", ServiceConfig)
    config = config.model_copy(update={"database_dir": str(tmp_path / "index")})
    llm, embedder, decisions = MemoryLLM(), MemoryEmbedder(), Decisions()
    engine = MemoryEngine(config, path, llm, embedder, decision_model=decisions)
    return engine, llm, embedder, decisions


def request(identity, text, session="s1", user="u1", timestamp=1704067200000):
    return AddRequest(request_id=identity, user_id=user, session_id=session,
                      messages=[Message(role="user", content=text, timestamp=timestamp)])


def counts(engine):
    with engine.store.connect("u1") as db:
        return {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in
                ("messages", "chunks", "facts", "fact_sources", "observations", "observation_revisions",
                 "observation_sources", "consolidation_decisions", "embeddings", "requests")}


def test_temporal_sources_versions_and_raw_retrieval(memory):
    engine, llm, _, _ = memory
    first = request("r1", "I used to prefer coffee.")
    second = request("r2", "Now I prefer tea.", session="s2", timestamp=1735689600000)
    engine.add(first)
    engine.add(second)
    assert llm.extractions[0][1][0]["role"] == "user"
    assert llm.extractions[0][1][0]["timestamp"] == 1704067200000
    assert llm.extractions[0][1][0]["source_order"] == 0
    with engine.store.connect("u1") as db:
        facts = [dict(row) for row in db.execute("SELECT * FROM facts ORDER BY mentioned_at")]
        assert [row["mentioned_at"] for row in facts] == ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
        assert all(row["occurred_start"] == "2020-05-01" and row["recorded_at"] for row in facts)
        observation = dict(db.execute("SELECT * FROM observations").fetchone())
        assert observation["current_revision"] == 2
        assert db.execute("SELECT count(*) FROM observation_sources WHERE revision=1").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM observation_sources WHERE revision=2").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM consolidation_decisions").fetchone()[0] == 2
    # Only the observation route can find evidence in this controlled test.
    engine._vector_rank = lambda db, vector, kind, limit: [observation["id"]] if kind == "observation" else []
    engine._bm25_rank = lambda db, query, limit: []
    results = engine.search(SearchRequest(query="current preference", user_id="u1", top_k=10)).data
    assert {item.content for item in results} == {first.messages[0].content, second.messages[0].content}
    assert all("DERIVED ONLY" not in item.content for item in results)
    assert engine.search(SearchRequest(query="preference", user_id="another", top_k=10)).data == []


def test_same_batch_updates_one_revision_and_restart_is_idempotent(memory):
    engine, llm, embedder, decisions = memory
    batch = AddRequest(request_id="batch", user_id="u1", session_id="s1", messages=[
        Message(role="user", content="I like coffee."),
        Message(role="user", content="I like coffee with milk."),
        Message(role="user", content="I drink it in the morning."),
    ])
    response = engine.add(batch)
    before = counts(engine)
    assert before["observations"] == before["observation_revisions"] == 1
    assert before["observation_sources"] == 3
    assert len(llm.rewrites) == 1
    restarted = MemoryEngine(engine.config, Path("configs/service/jevsight-d.yaml").resolve(),
                             llm, embedder, decision_model=decisions)
    call_count = len(decisions.calls)
    assert restarted.add(batch) == response
    assert counts(engine) == before
    assert len(decisions.calls) == call_count


@pytest.mark.parametrize("phase", ["decision", "rewrite", "embedding"])
def test_failed_add_preserves_previous_observation_atomically(memory, phase):
    engine, llm, embedder, decisions = memory
    engine.add(request("r1", "I like coffee."))
    before = counts(engine)
    if phase == "decision":
        decisions.fail = True
    elif phase == "rewrite":
        llm.fail_rewrite = True
    else:
        embedder.fail = True
    with pytest.raises(RuntimeError):
        engine.add(request("r2", "Now I prefer tea."))
    assert counts(engine) == before
    decisions.fail = llm.fail_rewrite = embedder.fail = False
    engine.add(request("r2", "Now I prefer tea."))
    assert counts(engine)["requests"] == before["requests"] + 1


def test_duplicate_attaches_sources_without_rewriting(memory):
    engine, llm, _, decisions = memory
    decisions.relation = "duplicate"
    engine.add(request("r1", "I like coffee."))
    engine.add(request("r2", "I like coffee.", session="s2"))
    assert not llm.rewrites
    with engine.store.connect("u1") as db:
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM observation_sources WHERE revision=2").fetchone()[0] == 2


def test_index_rejects_different_decision_model(memory):
    engine, llm, embedder, decisions = memory
    engine.add(request("r1", "I like coffee."))
    decision_config = engine.config.consolidation.decision_model.model_copy(update={"name": "another-model"})
    config = engine.config.model_copy(update={"consolidation": engine.config.consolidation.model_copy(
        update={"decision_model": decision_config})})
    other = MemoryEngine(config, Path("configs/service/jevsight-d.yaml").resolve(), llm, embedder, decision_model=decisions)
    with pytest.raises(ValueError, match="new database"):
        other.search(SearchRequest(query="coffee", user_id="u1", top_k=1))


def test_distinct_event_times_do_not_collapse_identical_fact_text(memory):
    engine, llm, _, _ = memory

    def extract(prompt, messages, chunks, temporal=False):
        return [{"content": "Alice visited Paris.", "source_chunk_ids": [chunks[0]["id"]],
                 "occurred_start": year, "occurred_end": year, "occurred_precision": "year",
                 "mentioned_at": None} for year in ("2020", "2021")]

    llm.extract = extract
    engine.add(request("r1", "Alice visited Paris in 2020 and again in 2021.", timestamp=None))
    with engine.store.connect("u1") as db:
        assert {row[0] for row in db.execute("SELECT occurred_start FROM facts")} == {"2020", "2021"}


def test_changed_relation_instructions_invalidate_existing_index(memory, monkeypatch):
    engine, llm, embedder, decisions = memory
    engine.add(request("r1", "Alice likes coffee."))
    monkeypatch.setattr(OpenAIDecision, "INSTRUCTIONS", "Different relationship policy")
    changed = MemoryEngine(engine.config, Path("configs/service/jevsight-d.yaml").resolve(),
                           llm, embedder, decision_model=decisions)
    with pytest.raises(ValueError, match="new database"):
        changed.add(request("r1", "Alice likes coffee."))


def test_temporal_policy_and_prior_context_are_index_identity(memory):
    engine, _, _, _ = memory
    fingerprint = index_fingerprint(engine.config, engine.prompt, temporal_rules="policy one")
    assert fingerprint != index_fingerprint(engine.config, engine.prompt, temporal_rules="policy two")
    changed = engine.config.model_copy(update={"facts": engine.config.facts.model_copy(
        update={"prior_context_tokens": engine.config.facts.prior_context_tokens + 1})})
    assert fingerprint != index_fingerprint(changed, engine.prompt, temporal_rules="policy one")


def test_storage_failure_rolls_back_versions_sources_and_current_pointer(memory, monkeypatch):
    engine, _, _, _ = memory
    engine.add(request("r1", "Alice likes coffee."))
    before = counts(engine)
    with engine.store.connect("u1") as db:
        previous = [tuple(row) for row in db.execute("SELECT * FROM observations")]
    save = engine.store.save_observations

    def fail_after_write(*args):
        save(*args)
        raise RuntimeError("storage commit interrupted")

    monkeypatch.setattr(engine.store, "save_observations", fail_after_write)
    with pytest.raises(RuntimeError, match="storage commit interrupted"):
        engine.add(request("r2", "Alice now prefers tea."))
    assert counts(engine) == before
    with engine.store.connect("u1") as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM observations")] == previous
