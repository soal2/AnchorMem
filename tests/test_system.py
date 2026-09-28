import copy
import json
import pytest
from fastapi.testclient import TestClient

from aml_memory.api import create_app
from aml_memory.config import ServiceConfig, load_config
from aml_memory.contracts import AddRequest, Message, SearchRequest
from aml_memory.engine import MemoryEngine
from aml_memory.evaluation import _answer_prompt, _contract_prompts, load_examples, split_examples
from aml_memory.storage import user_db_path


class FakeLLM:
    fail = False

    def extract(self, prompt, messages, chunks):
        if self.fail:
            raise RuntimeError("model unavailable")
        return [{"content": chunk["content"], "source_chunk_ids": [chunk["id"]]} for chunk in chunks]


class FakeEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 1024
            vector[0 if "Paris" in text else 1] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def setup(tmp_path):
    config, path = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    config = config.model_copy(update={"database_dir": str(tmp_path)})
    llm = FakeLLM()
    return config, path, llm, FakeEmbedder()


def test_add_search_idempotence_isolation_and_restart(setup):
    config, path, llm, embedder = setup
    engine = MemoryEngine(config, path, llm, embedder)
    first = AddRequest(request_id="r1", user_id="u1", session_id="s1", messages=[Message(role="user", content="Paris is in France", timestamp=1704067200000)])
    response = engine.add(first)
    assert response.success and response.request_id == "r1"
    assert engine.add(first) == response
    with engine.store.connect("u1") as db:
        assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
    with pytest.raises(FileExistsError):
        engine.add(first.model_copy(update={"messages": [Message(role="user", content="changed")]}))
    assert engine.search(SearchRequest(query="Paris", user_id="u2", top_k=1)).data == []
    items = MemoryEngine(config, path, llm, embedder).search(SearchRequest(query="Paris", user_id="u1", top_k=1)).data
    assert len(items) == 1 and "Paris is in France" in items[0].content
    assert items[0].created_at == "2024-01-01T00:00:00Z"
    assert user_db_path(engine.store.root, "u1").exists()


def test_failed_model_does_not_write_and_neighbors_cross_batches(setup):
    config, path, llm, embedder = setup
    engine = MemoryEngine(config, path, llm, embedder)
    llm.fail = True
    first = AddRequest(request_id="r1", user_id="u1", session_id="s1", messages=[Message(role="user", content="Paris is in France")])
    with pytest.raises(RuntimeError):
        engine.add(first)
    llm.fail = False
    engine.add(first)
    engine.add(AddRequest(request_id="r2", user_id="u1", session_id="s1", messages=[Message(role="assistant", content="It has many museums")]))
    result = engine.search(SearchRequest(query="Paris", user_id="u1", top_k=1))
    assert len(result.data) == 1
    assert "Paris is in France" in result.data[0].content
    assert "many museums" in result.data[0].content


def test_api_contract_and_auth(setup, monkeypatch):
    config, path, llm, embedder = setup
    monkeypatch.setenv("AML_MEMORY_KEY", "secret")
    client = TestClient(create_app(config, path, llm, embedder))
    assert client.get("/health").status_code == 200
    payload = {"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": [{"role": "user", "content": "Paris is in France"}]}
    assert client.post("/add", json=payload).status_code == 401
    response = client.post("/add", json=payload, headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json() == {"success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1"}
    found = client.post("/search", json={"query": "Paris", "user_id": "u1", "top_k": 1}, headers={"X-Api-Key": "secret"})
    assert found.status_code == 200 and len(found.json()["data"]) == 1
    payload["messages"][0]["content"] = "changed"
    assert client.post("/add", json=payload, headers={"X-Api-Key": "secret"}).status_code == 409


def test_config_rejects_unknown_and_changed_index(setup):
    config, path, llm, embedder = setup
    data = config.model_dump()
    data["unknown"] = 1
    with pytest.raises(Exception):
        ServiceConfig.model_validate(data)
    changed = copy.deepcopy(config.model_dump())
    changed["chunk"]["overlap_tokens"] = changed["chunk"]["max_tokens"]
    with pytest.raises(Exception):
        ServiceConfig.model_validate(changed)
    engine = MemoryEngine(config, path, llm, embedder)
    engine.add(AddRequest(request_id="r1", user_id="u1", session_id="s1", messages=[Message(role="user", content="Paris")]))
    changed = config.model_copy(update={"chunk": config.chunk.model_copy(update={"max_tokens": 256})})
    with pytest.raises(ValueError, match="new database"):
        MemoryEngine(changed, path, llm, embedder).search(SearchRequest(query="Paris", user_id="u1", top_k=1))
    retriever = config.retrieval.model_copy(update={"neighbor_count": 0})
    MemoryEngine(config.model_copy(update={"retrieval": retriever}), path, llm, embedder).search(SearchRequest(query="Paris", user_id="u1", top_k=1))


def test_evaluation_adapter_and_group_split(tmp_path):
    data = [{"question_id": f"q{i}", "question": "Where?", "answer": "Paris", "question_type": "single-session-user",
             "haystack_sessions": [[{"role": "user", "content": "Paris is in France"}]], "haystack_session_ids": [f"s{i}"]} for i in range(4)]
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    examples = load_examples("longmemeval", path)
    dev = split_examples(examples, "dev", 42, 0.5, None)
    validation = split_examples(examples, "validation", 42, 0.5, None)
    assert {item["id"] for item in dev}.isdisjoint({item["id"] for item in validation})
    assert len(dev) == len(validation) == 2


def test_public_contract_prompts_are_loaded_without_execution(tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text('OPEN_ENDED_ANSWER_TEMPLATE = "Question: {{question}}\\n{{speaker_1_memories}}"\nACCURACY_PROMPT = "Gold: {gold_answer}"\nraise RuntimeError("must not run")\n', encoding="utf-8")
    answer, judge = _contract_prompts(source)
    assert _answer_prompt(answer, {"question": "Where?"}, "Paris") == "Question: Where?\nParis"
    assert judge == "Gold: {gold_answer}"
