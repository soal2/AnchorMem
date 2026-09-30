import json
from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, OpenAI

from src import providers
from src.config import EvaluationConfig, ServiceConfig, load_config, resolved_path
from src.contracts import AddRequest, Message, SearchRequest
from src.engine import MemoryEngine


@pytest.fixture
def mock_openrouter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openrouter-test-key")
    clients = []

    def install(handler):
        requests = []

        def respond(request):
            assert request.url.host == "openrouter.ai"
            assert request.method == "POST"
            assert request.headers["Authorization"] == "Bearer openrouter-test-key"
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            return handler(request.url.path, body)

        def create_client(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            clients.append(client)
            return client

        monkeypatch.setattr(providers, "OpenAI", create_client)
        return requests

    yield install
    for client in clients:
        client.close()


def completion(content, *, finish_reason="stop", refusal=None):
    return httpx.Response(200, json={
        "id": "chat-test", "object": "chat.completion", "created": 0,
        "model": "openai/gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "refusal": refusal},
                     "finish_reason": finish_reason}],
    })


def embeddings(body, values):
    assert body["model"] == "openai/text-embedding-3-small"
    assert body["dimensions"] == 1024
    assert body["encoding_format"] == "float"
    return httpx.Response(200, json={
        "object": "list", "model": body["model"],
        "data": [{"object": "embedding", "index": i, "embedding": [value] + [0.0] * 1023}
                 for i, value in reversed(list(enumerate(values)))],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    })


def test_openrouter_configs_share_provider_and_use_separate_index():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs/service").glob("v1-b*.yaml")):
        config, path = load_config(path, ServiceConfig)
        assert not config.competition_mode
        assert config.llm.name == "openai/gpt-4o-mini"
        assert config.embedding.name == "openai/text-embedding-3-small"
        assert config.embedding.dimensions == 1024
        assert resolved_path(path, config.database_dir) == root / "data/openrouter-index"
        for model in (config.llm, config.embedding):
            assert model.base_url == "https://openrouter.ai/api/v1"
            assert model.api_key_env == "OPENAI_API_KEY"
            assert model.retries == 4
    for dataset in ("locomo", "longmemeval"):
        for split in ("dev", "validation"):
            config, path = load_config(root / f"configs/evaluation/{dataset}-{split}.yaml", EvaluationConfig)
            service, _ = load_config(resolved_path(path, config.service_config_path), ServiceConfig)
            for model in (config.answer, config.judge):
                assert (model.name, model.base_url, model.api_key_env) == (
                    service.llm.name, service.llm.base_url, service.llm.api_key_env,
                )


def test_openrouter_chat_extraction_and_plain_completion(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    chunks = [{"id": "chunk-1", "content": "I live in Paris."},
              {"id": "chunk-2", "content": "I moved here last spring."}]
    prior = [{"role": "user", "content": "My name is Alice."}]
    facts = [{"content": "Alice lives in Paris.", "source_chunk_ids": ["chunk-1"]},
             {"content": "Alice moved to Paris last spring.", "source_chunk_ids": ["chunk-2", "chunk-1"]}]

    def handler(path, body):
        assert path == "/api/v1/chat/completions"
        assert body["model"] == config.llm.name
        assert body["temperature"] == config.llm.temperature
        assert body["max_tokens"] == config.llm.max_output_tokens
        if "response_format" in body:
            response_format = body["response_format"]
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"]["strict"] is True
            schema = response_format["json_schema"]["schema"]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            assert schema["required"] == ["facts"]
            fact_schema = schema["properties"]["facts"]["items"]
            assert fact_schema["additionalProperties"] is False
            assert set(fact_schema["required"]) == {"content", "source_indices"}
            source_schema = fact_schema["properties"]["source_indices"]
            assert source_schema["minItems"] == 1
            assert source_schema["items"] == {"type": "integer", "enum": [0, 1]}
            assert body["provider"]["require_parameters"] is True
            assert body["messages"][0] == {"role": "system", "content": "Extract grounded facts."}
            assert json.loads(body["messages"][1]["content"]) == {
                "prior_messages": prior,
                "new_chunks": [{"index": 0, "content": chunks[0]["content"]},
                               {"index": 1, "content": chunks[1]["content"]}],
            }
            return completion(json.dumps({"facts": [
                {"content": "Alice lives in Paris.", "source_indices": [0]},
                {"content": "Alice moved to Paris last spring.", "source_indices": [1, 0]},
            ]}))
        assert body["messages"] == [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Where?"}]
        return completion("Paris")

    requests = mock_openrouter(handler)
    llm = providers.OpenAILLM(config.llm)
    assert llm.extract("Extract grounded facts.", prior, chunks) == facts
    assert llm.complete("Answer briefly.", "Where?", json_response=False) == "Paris"
    assert len(requests) == 2


def test_openrouter_embeddings_preserve_order_across_batches(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)

    def handler(path, body):
        assert path == "/api/v1/embeddings"
        return embeddings(body, [float(text) for text in body["input"]])

    requests = mock_openrouter(handler)
    embedder = providers.OpenAIEmbedder(config.embedding.model_copy(update={"batch_size": 2}))
    assert embedder.embed([]) == []
    assert requests == []
    vectors = embedder.embed(["1", "2", "3"])
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]
    assert all(len(vector) == 1024 for vector in vectors)
    assert [body["input"] for _, body in requests] == [["1", "2"], ["3"]]


def test_openrouter_add_search_returns_original_evidence(mock_openrouter, tmp_path):
    config, path = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    config = config.model_copy(update={"database_dir": str(tmp_path)})
    original = "I moved to Paris last spring."
    fact = "The user resides in Paris."

    def handler(path, body):
        if path == "/api/v1/chat/completions":
            chunks = json.loads(body["messages"][1]["content"])["new_chunks"]
            assert chunks[0]["content"] == original
            return completion(json.dumps({"facts": [{"content": fact, "source_indices": [chunks[0]["index"]]}]}))
        assert path == "/api/v1/embeddings"
        return embeddings(body, [1.0] * len(body["input"]))

    requests = mock_openrouter(handler)
    engine = MemoryEngine(config, path, providers.OpenAILLM(config.llm), providers.OpenAIEmbedder(config.embedding))
    response = engine.add(AddRequest(request_id="r1", user_id="u1", session_id="s1", messages=[
        Message(role="user", content=original, timestamp=1704067200000),
    ]))
    assert response.success
    result = engine.search(SearchRequest(query="Where does the user live?", user_id="u1", top_k=1))
    assert len(result.data) == 1
    assert result.data[0].content == original
    assert result.data[0].created_at == "2024-01-01T00:00:00Z"
    assert result.data[0].score > 0
    with engine.store.connect("u1") as db:
        assert db.execute("SELECT content FROM facts").fetchone()[0] == fact
        rows = db.execute("SELECT model,dimension,length(vector) FROM embeddings").fetchall()
        assert [tuple(row) for row in rows] == [(config.embedding.name, 1024, 1024 * 4)] * 2
    assert [body["input"] for path, body in requests if path.endswith("/embeddings")] == [
        [original, fact], ["Where does the user live?"],
    ]


def test_openrouter_extraction_allows_no_facts_and_skips_empty_chunks(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: completion('{"facts": []}'))
    llm = providers.OpenAILLM(config.llm)
    assert llm.extract("Extract facts.", [], []) == []
    assert requests == []
    assert llm.extract("Extract facts.", [], [{"id": "chunk-1", "content": "Hello."}]) == []
    assert len(requests) == 1


@pytest.mark.parametrize("raw_response", [
    "not JSON",
    "null",
    "[]",
    "{}",
    '{"facts": null}',
    '{"facts": [], "extra": true}',
    '{"facts": [null]}',
    '{"facts": [{"content": "Paris"}]}',
    '{"facts": [{"source_indices": [0]}]}',
    '{"facts": [{"content": "Paris", "source_indices": [0], "extra": true}]}',
    '{"facts": [{"content": "Paris", "source_chunk_ids": ["chunk-1"]}]}',
    '{"facts": [{"content": "", "source_indices": [0]}]}',
    '{"facts": [{"content": "   ", "source_indices": [0]}]}',
    '{"facts": [{"content": 42, "source_indices": [0]}]}',
    '{"facts": [{"content": "Paris", "source_indices": []}]}',
    '{"facts": [{"content": "Paris", "source_indices": null}]}',
    '{"facts": [{"content": "Paris", "source_indices": [true]}]}',
    '{"facts": [{"content": "Paris", "source_indices": [false]}]}',
    '{"facts": [{"content": "Paris", "source_indices": [-1]}]}',
    '{"facts": [{"content": "Paris", "source_indices": [1]}]}',
    '{"facts": [{"content": "Paris", "source_indices": [0.0]}]}',
    '{"facts": [{"content": "Paris", "source_indices": ["0"]}]}',
])
def test_openrouter_extraction_rejects_invalid_output_without_retry(mock_openrouter, raw_response):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    llm = providers.OpenAILLM(config.llm)
    with pytest.raises(providers.ProviderError) as raised:
        llm.extract("Extract facts.", [], [{"id": "chunk-1", "content": "Paris"}])
    error = raised.value
    assert isinstance(error.__cause__, ValueError)
    assert error.raw_response == raw_response
    assert error.diagnostics["phase"] == "extraction"
    assert error.diagnostics["model"] == config.llm.name
    assert error.diagnostics["exception_type"]
    assert error.diagnostics["validation_error"] == str(error.__cause__)
    assert len(requests) == 1


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_openrouter_extraction_rejects_unfinished_output(mock_openrouter, finish_reason):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    raw_response = '{"facts": [{"content": "Paris", "source_indices": [0]}]}'
    requests = mock_openrouter(lambda path, body: completion(raw_response, finish_reason=finish_reason))
    llm = providers.OpenAILLM(config.llm)
    with pytest.raises(providers.ProviderError) as raised:
        llm.extract("Extract facts.", [], [{"id": "chunk-1", "content": "Paris"}])
    assert raised.value.raw_response == raw_response
    assert raised.value.diagnostics["finish_reason"] == finish_reason
    assert raised.value.__cause__ is not None
    assert raised.value.diagnostics["validation_error"] == "extraction provider did not finish normally"
    assert len(requests) == 1


def test_openrouter_extraction_rejects_refusal(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: completion(None, refusal="Cannot extract these facts."))
    llm = providers.OpenAILLM(config.llm)
    with pytest.raises(providers.ProviderError) as raised:
        llm.extract("Extract facts.", [], [{"id": "chunk-1", "content": "Paris"}])
    assert raised.value.__cause__ is not None
    assert raised.value.diagnostics["phase"] == "extraction"
    assert raised.value.diagnostics["validation_error"] == "extraction provider refused the request"
    assert len(requests) == 1


@pytest.mark.parametrize("phase", ["completion", "extraction", "embedding"])
def test_openrouter_sdk_retries_connection_failure_then_recovers(mock_openrouter, monkeypatch, phase):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    monkeypatch.setattr(OpenAI, "_calculate_retry_timeout", lambda *args, **kwargs: 0)
    attempts = 0

    def handler(path, body):
        nonlocal attempts
        attempts += 1
        if attempts <= 4:
            raise httpx.ConnectError("Controlled connection reset")
        if phase == "embedding":
            return embeddings(body, [1.0])
        if phase == "extraction":
            return completion('{"facts": [{"content": "Paris", "source_indices": [0]}]}')
        return completion("Paris")

    requests = mock_openrouter(handler)
    if phase == "embedding":
        result = providers.OpenAIEmbedder(config.embedding).embed(["Paris"])
        assert result[0][0] == 1.0
    else:
        llm = providers.OpenAILLM(config.llm)
        if phase == "extraction":
            assert llm.extract("Extract facts.", [], [{"id": "chunk-1", "content": "Paris"}]) == [
                {"content": "Paris", "source_chunk_ids": ["chunk-1"]},
            ]
        else:
            assert llm.complete("Answer briefly.", "Where?", json_response=False) == "Paris"
    assert len(requests) == attempts == 5


@pytest.mark.parametrize("phase", ["completion", "extraction", "embedding"])
def test_openrouter_sdk_retry_exhaustion_retains_connection_diagnostics(mock_openrouter, monkeypatch, phase):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    monkeypatch.setattr(OpenAI, "_calculate_retry_timeout", lambda *args, **kwargs: 0)

    def handler(path, body):
        raise httpx.ConnectError("Controlled connection reset")

    requests = mock_openrouter(handler)
    with pytest.raises(providers.ProviderError) as raised:
        if phase == "embedding":
            providers.OpenAIEmbedder(config.embedding).embed(["Paris"])
        else:
            llm = providers.OpenAILLM(config.llm)
            if phase == "extraction":
                llm.extract("Extract facts.", [], [{"id": "chunk-1", "content": "Paris"}])
            else:
                llm.complete("Answer briefly.", "Where?", json_response=False)
    error = raised.value
    assert isinstance(error.__cause__, APIConnectionError)
    assert isinstance(error.__cause__.__cause__, httpx.ConnectError)
    assert error.raw_response is None
    assert error.diagnostics["phase"] == phase
    assert error.diagnostics["model"] == getattr(config, "embedding" if phase == "embedding" else "llm").name
    assert error.diagnostics["exception_type"] == "APIConnectionError"
    assert error.diagnostics["cause_type"] == "ConnectError"
    assert len(requests) == 5


def test_openrouter_connection_diagnostics_retain_root_reason_and_redact_api_key(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)

    def handler(path, body):
        try:
            raise OSError("DNS lookup failed with openrouter-test-key")
        except OSError as error:
            raise httpx.ConnectError("TLS connection failed with openrouter-test-key") from error

    requests = mock_openrouter(handler)
    llm = providers.OpenAILLM(config.llm.model_copy(update={"retries": 0}))
    with pytest.raises(providers.ProviderError) as raised:
        llm.extract("Extract facts.", [], [{"id": "chunk-1", "content": "Paris"}])
    diagnostics = raised.value.diagnostics
    assert diagnostics["cause_chain"] == [
        {"type": "ConnectError", "message": "TLS connection failed with [REDACTED]"},
        {"type": "OSError", "message": "DNS lookup failed with [REDACTED]"},
    ]
    assert llm.client.api_key not in json.dumps(diagnostics)
    assert llm.client.api_key in str(raised.value.__cause__.__cause__)
    assert len(requests) == 1
