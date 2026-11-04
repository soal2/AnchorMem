import json
from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, OpenAI

from src import providers
from src.source_spans import build_source_units
from src.config import EvaluationConfig, ServiceConfig, load_config, resolved_path
from src.contracts import AddRequest, Message, SearchRequest
from src.topics import TOPICS
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


def fact_payload(text, ranges, **fields):
    return {"content": text, "source_unit_ranges": ranges, "topic": "profile", "scope": "default",
            "event_status": None, "conditions": None, "verification": None, "secondary_entries": [], **fields}


def secondary_payload(text, ranges, **fields):
    payload = fact_payload(text, ranges, **fields)
    payload.pop("secondary_entries")
    return payload


def mapped_fact(content, quotes, **fields):
    payload = fact_payload(content, [], **fields)
    payload.pop("source_unit_ranges")
    payload.pop("secondary_entries")
    payload.setdefault("parent_index", None)
    payload["source_quotes"] = [dict(chunk_id=chunk_id, quote=quote, occurrence=occurrence)
                                for chunk_id, quote, occurrence in quotes]
    payload["source_chunk_ids"] = list(dict.fromkeys(chunk_id for chunk_id, _, _ in quotes))
    return payload


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
    chunks = [{"message_id": "test-message", "id": "chunk-1", "content": "I live in Paris."},
              {"message_id": "test-message", "id": "chunk-2", "content": "I moved here last spring."}]
    prior = [{"role": "user", "content": "My name is Alice."}]
    facts = [mapped_fact("Lives in Paris.", [("chunk-1", chunks[0]["content"], 0)]),
             mapped_fact("Moved here last spring.", [("chunk-1", chunks[0]["content"], 0),
                                                     ("chunk-2", chunks[1]["content"], 0)])]

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
            assert set(fact_schema["required"]) == set(fact_payload("x", []))
            assert "parent_index" not in fact_schema["properties"]
            child_schema = fact_schema["properties"]["secondary_entries"]["items"]
            assert child_schema["additionalProperties"] is False
            assert set(child_schema["required"]) == set(secondary_payload("x", []))
            assert "parent_index" not in child_schema["properties"]
            assert "secondary_entries" not in child_schema["properties"]
            assert fact_schema["properties"]["topic"]["enum"] == list(TOPICS)
            source_schema = fact_schema["properties"]["source_unit_ranges"]
            assert source_schema["minItems"] == 1
            bound = len(build_source_units([chunk["content"] for chunk in chunks])[1]) - 1
            assert source_schema["items"] == {"type": "object", "properties": {
                "first": {"type": "integer", "minimum": 0, "maximum": bound},
                "last": {"type": "integer", "minimum": 0, "maximum": bound},
            }, "required": ["first", "last"], "additionalProperties": False}
            assert "source_unit_ids" not in fact_schema["properties"]
            assert "source_quotes" not in fact_schema["properties"]
            assert "source_spans" not in fact_schema["properties"]
            assert body["provider"]["require_parameters"] is True
            assert body["messages"][0] == {"role": "system", "content": "Extract grounded facts."}
            state = json.loads(body["messages"][1]["content"])
            assert state == {
                "project_id": "test-project",
                "new_chunks": [{"index": 0, "content": chunks[0]["content"],
                                "source_units": build_source_units([chunks[0]["content"]])[0][0]},
                               {"index": 1, "content": chunks[1]["content"],
                                "source_units": build_source_units([chunk["content"] for chunk in chunks])[0][1]}],
            }
            ids = [unit_id for chunk in state["new_chunks"] for unit_id, _ in chunk["source_units"]]
            assert ids == list(range(source_schema["items"]["properties"]["first"]["maximum"] + 1))
            return completion(json.dumps({"facts": [
                fact_payload("Lives in Paris.", [{"first": 0, "last": 4}]),
                fact_payload("Moved here last spring.", [{"first": 5, "last": 10}, {"first": 0, "last": 4}]),
            ]}))
        assert body["messages"] == [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Where?"}]
        return completion("Paris")

    requests = mock_openrouter(handler)
    llm = providers.OpenAILLM(config.llm)
    assert llm.extract("Extract grounded facts.", prior, chunks, project_id="test-project") == facts
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
            return completion(json.dumps({"facts": [fact_payload(fact, [{"first": 0, "last": 6}])]}))
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
    assert llm.extract("Extract facts.", [], [{"message_id": "test-message", "id": "chunk-1", "content": "Hello."}]) == []
    assert len(requests) == 1


@pytest.mark.parametrize("raw_response", [
    "not JSON", "null", "[]", "{}", '{"facts": null}', '{"facts": [], "extra": true}',
    '{"facts": [null]}', '{"facts": [{"content": "Paris", "source_indices": [0]}]}',
    *[json.dumps({"facts": [fact_payload("Paris", [{"first": 0, "last": 0}], **update)]}) for update in [
        {"content": ""}, {"content": " "}, {"content": 42}, {"extra": True},
        {"topic": "travel"}, {"topic": []}, {"scope": "unresolved"}, {"scope": []},
        {"parent_index": True}, {"parent_index": -1}, {"event_status": "invalid"}, {"event_status": []},
        {"conditions": " "}, {"verification": False},
        {"secondary_entries": None}, {"secondary_entries": {}}, {"secondary_entries": [None]},
        *[{"source_unit_ranges": value} for value in [
            None, {}, [], [None], [True], [False], [0.0], [1.0], ["0"], ["1"], [""], [" "],
            [-1], [2], [999999], ["p0"], ["w0"], [1, 1],
            [{"index": 0, "quote": "Paris", "start": None}],
        ]],
        *[{"source_unit_ranges": [source_range]} for source_range in [
            {"first": True, "last": 0}, {"first": 0, "last": False},
            {"first": "0", "last": 0}, {"first": 0, "last": 0.0},
            {"first": -1, "last": 0}, {"first": 0, "last": 1},
            {"first": 1}, {"last": 1}, {"first": 0, "last": 0, "extra": 0},
        ]],
        {"source_unit_ranges": [{"first": 0, "last": 0}] * 2},
    ]],
])
def test_openrouter_extraction_rejects_invalid_output_without_retry(mock_openrouter, raw_response):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    llm = providers.OpenAILLM(config.llm)
    with pytest.raises(providers.ProviderError) as raised:
        llm.extract("Extract facts.", [], [{"message_id": "test-message", "id": "chunk-1", "content": "Paris"}])
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
    raw_response = json.dumps({"facts": [fact_payload("Paris", [{"first": 0, "last": 0}])]})
    requests = mock_openrouter(lambda path, body: completion(raw_response, finish_reason=finish_reason))
    llm = providers.OpenAILLM(config.llm)
    with pytest.raises(providers.ProviderError) as raised:
        llm.extract("Extract facts.", [], [{"message_id": "test-message", "id": "chunk-1", "content": "Paris"}])
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
        llm.extract("Extract facts.", [], [{"message_id": "test-message", "id": "chunk-1", "content": "Paris"}])
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
            return completion(json.dumps({"facts": [fact_payload("Paris", [{"first": 0, "last": 0}])]}))
        return completion("Paris")

    requests = mock_openrouter(handler)
    if phase == "embedding":
        result = providers.OpenAIEmbedder(config.embedding).embed(["Paris"])
        assert result[0][0] == 1.0
    else:
        llm = providers.OpenAILLM(config.llm)
        if phase == "extraction":
            assert llm.extract("Extract facts.", [], [{"message_id": "test-message", "id": "chunk-1", "content": "Paris"}]) == [
                mapped_fact("Paris", [("chunk-1", "Paris", 0)]),
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
                llm.extract("Extract facts.", [], [{"message_id": "test-message", "id": "chunk-1", "content": "Paris"}])
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
        llm.extract("Extract facts.", [], [{"message_id": "test-message", "id": "chunk-1", "content": "Paris"}])
    diagnostics = raised.value.diagnostics
    assert diagnostics["cause_chain"] == [
        {"type": "ConnectError", "message": "TLS connection failed with [REDACTED]"},
        {"type": "OSError", "message": "DNS lookup failed with [REDACTED]"},
    ]
    assert llm.client.api_key not in json.dumps(diagnostics)
    assert llm.client.api_key in str(raised.value.__cause__.__cause__)
    assert len(requests) == 1


@pytest.mark.parametrize("project_id,expected", [(None, None), ("", None), (" \t", None), (" project-1 ", "project-1")])
def test_optional_project_context_normalizes_for_add_and_search(project_id, expected):
    add = AddRequest(request_id="r", user_id="u", session_id="s",
                     messages=[Message(role="user", content="A fact")], project_id=project_id)
    search = SearchRequest(query="A question", user_id="u", top_k=1, project_id=project_id)
    assert add.project_id == search.project_id == expected
    assert add.model_dump() == AddRequest.model_validate(add.model_dump()).model_dump()
    if expected is None:
        assert add.model_dump() == AddRequest(request_id="r", user_id="u", session_id="s",
                                             messages=[Message(role="user", content="A fact")]).model_dump()


@pytest.mark.parametrize("time_range,expected", [
    (None, None), ({}, None), ({"start": " ", "end": None}, None),
    ({"start": " 2024-02 "}, {"start": "2024-02", "end": None}),
    ({"end": "2024-03"}, {"start": None, "end": "2024-03"}),
    ({"start": "2024-02", "end": "2024-03"}, {"start": "2024-02", "end": "2024-03"}),
])
def test_optional_time_range_normalizes_missing_and_open_boundaries(time_range, expected):
    request = SearchRequest(query="A question", user_id="u", top_k=1, event_time_range=time_range)
    assert (request.event_time_range.model_dump() if request.event_time_range else None) == expected


@pytest.mark.parametrize("updates", [
    {"project_id": 1}, {"project_id": False}, {"project_id": []},
    {"event_time_range": "2024"}, {"event_time_range": {"start": 2024}},
    {"event_time_range": {"start": "yesterday"}},
    {"event_time_range": {"start": "2024-02-30"}},
    {"event_time_range": {"start": "2024", "end": "2024"}},
    {"event_time_range": {"start": "2025", "end": "2024"}},
    {"event_time_range": {"start": "2024-01-01", "end": "2024-01-02T00:00Z"}},
])
def test_invalid_optional_context_does_not_silently_broaden_search(updates):
    with pytest.raises(ValueError):
        SearchRequest(query="A question", user_id="u", top_k=1, **updates)


def ranges_for_quote(content, quote):
    _, lookup = build_source_units([content])
    start = content.index(quote)
    end = start + len(quote)
    first = next(unit_id for unit_id, (_, left, right) in lookup.items() if left == start and right <= end)
    last = next(unit_id for unit_id, (_, left, right) in lookup.items() if right == end and left >= start)
    return [{"first": first, "last": last}]


def test_extraction_separates_scopes_and_preserves_independent_secondary_quotes(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    content = "I prefer short replies; this project's replies must be in English. Short replies are preferred."
    payloads = [
        fact_payload("Prefers short replies", ranges_for_quote(content, "I prefer short replies"), topic="preferences", scope="global",
                     secondary_entries=[secondary_payload("Project replies must be in English",
                         ranges_for_quote(content, "this project's replies must be in English."),
                         topic="requirements", scope="project")]),
    ]
    calls = mock_openrouter(lambda path, body: completion(json.dumps({"facts": payloads})))
    facts = providers.OpenAILLM(config.llm).extract("Extract facts", [], [{"message_id": "test-message", "id": "c", "content": content}])
    assert facts[0]["source_quotes"] == [{"chunk_id": "c", "quote": "I prefer short replies", "occurrence": 0}]
    assert "English" not in facts[0]["source_quotes"][0]["quote"]
    assert facts[1]["source_quotes"] == [{"chunk_id": "c", "quote": "this project's replies must be in English.", "occurrence": 0}]
    assert facts[1]["topic"] == "requirements"
    assert facts[1]["scope"] == "project"
    assert facts[1]["parent_index"] == 0
    assert json.loads(calls[0][1]["messages"][1]["content"])["project_id"] is None
    assert len(calls) == 1


def test_extraction_selects_current_source_unit_ranges_instead_of_free_quotes(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)

    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        assert set(state) == {"project_id", "new_chunks"}
        assert "I donated my old car yesterday." not in json.dumps(body)
        assert "source_tokens" not in state["new_chunks"][0]
        assert [2, "Paris"] in state["new_chunks"][0]["source_units"]
        return completion(json.dumps({"facts": [fact_payload("Second Paris", [{"first": 2, "last": 2}])]}))

    requests = mock_openrouter(handler)
    assert providers.OpenAILLM(config.llm).extract("Extract", [
        {"role": "user", "content": "I donated my old car yesterday."},
    ], [{"message_id": "test-message", "id": "current", "content": "Paris then Paris"}]) == [
        mapped_fact("Second Paris", [("current", "Paris", 1)]),
    ]
    assert len(requests) == 1


def test_extraction_preserves_session_headers_and_metadata_while_selecting_a_clause(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    content = "[Session date: 2024-01-01]\nAlice: I live in Paris."
    source = {"message_id": "test-message", "id": "session", "content": content, "role": "user", "timestamp": 1704067200000,
              "source_order": 7}

    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        assert set(state) == {"project_id", "new_chunks"}
        chunk = state["new_chunks"][0]
        assert {key: value for key, value in chunk.items() if key != "source_units"} == {
            "index": 0, "content": content, "role": "user", "timestamp": 1704067200000,
            "source_order": 7, "timestamp_iso": "2024-01-01T00:00:00Z",
        }
        assert chunk["source_units"] == build_source_units([content])[0][0]
        assert [unit[0] for unit in chunk["source_units"]] == list(range(len(chunk["source_units"])))
        return completion(json.dumps({"facts": [fact_payload("Lives in Paris", ranges_for_quote(content, "I live in Paris."))]}))

    requests = mock_openrouter(handler)
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [source]) == [
        mapped_fact("Lives in Paris", [("session", "I live in Paris.", 0)]),
    ]
    assert len(requests) == 1


def test_extraction_large_sources_keep_schema_small_and_select_final_units(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    content = "word " * 1100 + "end."

    def handler(path, body):
        schema = body["response_format"]["json_schema"]["schema"]
        source_schema = schema["properties"]["facts"]["items"]["properties"]["source_unit_ranges"]
        assert source_schema["items"] == {"type": "object", "properties": {
            "first": {"type": "integer", "minimum": 0, "maximum": 1101},
            "last": {"type": "integer", "minimum": 0, "maximum": 1101},
        }, "required": ["first", "last"], "additionalProperties": False}
        assert len(json.dumps(schema)) < 3500
        chunk = json.loads(body["messages"][1]["content"])["new_chunks"][0]
        assert chunk["content"] == content
        assert len(chunk["source_units"]) == 1102
        assert chunk["source_units"][-2:] == [[1100, "end"], [1101, "."]]
        return completion(json.dumps({"facts": [fact_payload("The end", [{"first": 1100, "last": 1101}])]}))

    requests = mock_openrouter(handler)
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "long", "content": content}]) == [
        mapped_fact("The end", [("long", "end.", 0)]),
    ]
    assert len(requests) == 1


def test_extraction_skips_whitespace_only_sources_without_http(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: pytest.fail("Whitespace cannot support any fact"))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [
        {"message_id": "test-message", "id": "empty", "content": ""}, {"message_id": "test-message", "id": "blank", "content": " \n\t "},
    ]) == []
    assert requests == []


def test_extraction_mixed_blank_source_keeps_original_chunk_indices(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)

    def handler(path, body):
        chunks = json.loads(body["messages"][1]["content"])["new_chunks"]
        assert chunks[0] == {"index": 0, "content": " \n", "source_units": []}
        assert chunks[1] == {"index": 1, "content": "Paris", "source_units": [[0, "Paris"]]}
        return completion(json.dumps({"facts": [fact_payload("Paris", [{"first": 0, "last": 0}])]}))

    requests = mock_openrouter(handler)
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [
        {"message_id": "test-message", "id": "blank", "content": " \n"}, {"message_id": "test-message", "id": "valid", "content": "Paris"},
    ]) == [mapped_fact("Paris", [("valid", "Paris", 0)])]
    assert len(requests) == 1


@pytest.mark.parametrize("field,value", [
    ("source_unit_ids", [1]),
    ("source_quotes", [{"index": 0, "quote": "Paris", "start": None}]),
    ("source_quotes", [{"index": 0, "quote": "Paris", "occurrence": 0}]),
    ("source_spans", [{"index": 0, "start": 0, "end": 0, "quote": "Paris"}]),
])
def test_extraction_rejects_obsolete_wire_without_retry(mock_openrouter, field, value):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    payload = fact_payload("Paris", [])
    payload.pop("source_unit_ranges")
    payload[field] = value
    raw_response = json.dumps({"facts": [payload]})
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "current", "content": "Paris"}])
    assert isinstance(raised.value.__cause__, ValueError)
    assert raised.value.raw_response == raw_response
    assert len(requests) == 1


def test_extraction_captured_locomo_politics_clause_is_derived_from_selected_ids(mock_openrouter):
    # Static source from LoCoMo conv-41 D1:6, which failed v5 token arithmetic.
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    quote = "I'm really hoping to get into local politics. I love helping the community and making it a better place."
    content = "Yeah Maria, " + quote
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [
        fact_payload("John hopes to enter local politics", ranges_for_quote(content, quote)),
    ]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "john", "content": content}]) == [
        mapped_fact("John hopes to enter local politics", [("john", quote, 0)]),
    ]
    assert len(requests) == 1


def test_extraction_captured_longmemeval_handbag_evidence_keeps_literal_punctuation(mock_openrouter):
    # Static source from LongMemEval 95bcc1c8; no free quotation is accepted from v5 output.
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    content = "I'm looking for some advice on finding affordable handbags. I recently bought one from Forever 21 for $10 and it's been a great value, but I'm not sure where to find similar deals on better quality bags."
    quote = "I recently bought one from Forever 21 for $10 and it's been a great value"
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [
        fact_payload("A Forever 21 handbag cost $10", ranges_for_quote(content, quote)),
    ]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "bag", "content": content}]) == [
        mapped_fact("A Forever 21 handbag cost $10", [("bag", quote, 0)]),
    ]
    assert len(requests) == 1


def test_extraction_multiple_units_preserve_unicode_and_newlines(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    content = "Café\t上海，\nParis then Paris."
    quote = "Café\t上海，\nParis"
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [
        fact_payload("A literal multilingual source", ranges_for_quote(content, quote)),
    ]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "unicode", "content": content}]) == [
        mapped_fact("A literal multilingual source", [("unicode", quote, 0)]),
    ]
    assert len(requests) == 1


def test_extraction_primary_wire_uses_nested_secondary_entries(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    payload = fact_payload("Lives in Paris", [{"first": 0, "last": 0}])
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [payload]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "current", "content": "Paris"}]) == [
        mapped_fact("Lives in Paris", [("current", "Paris", 0)]),
    ]
    assert len(requests) == 1


def test_extraction_nested_entries_compute_flat_parent_indices_without_model_arithmetic(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    content = "Primary one. Child alpha. Child beta. Primary two. Child gamma."
    payloads = [
        fact_payload("Primary one", ranges_for_quote(content, "Primary one."), secondary_entries=[
            secondary_payload("Child alpha", ranges_for_quote(content, "Child alpha."), topic="requirements",
                              scope="project", conditions="If approved", verification="Confirm deployment"),
            secondary_payload("Child beta", ranges_for_quote(content, "Child beta.")),
        ]),
        fact_payload("Primary two", ranges_for_quote(content, "Primary two."), secondary_entries=[
            secondary_payload("Child gamma", ranges_for_quote(content, "Child gamma.")),
        ]),
    ]
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": payloads})))
    facts = providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "current", "content": content}])
    assert [fact["parent_index"] for fact in facts] == [None, 0, 0, None, 3]
    assert facts == [
        mapped_fact("Primary one", [("current", "Primary one.", 0)]),
        mapped_fact("Child alpha", [("current", "Child alpha.", 0)], parent_index=0,
                    topic="requirements", scope="project", conditions="If approved", verification="Confirm deployment"),
        mapped_fact("Child beta", [("current", "Child beta.", 0)], parent_index=0),
        mapped_fact("Primary two", [("current", "Primary two.", 0)]),
        mapped_fact("Child gamma", [("current", "Child gamma.", 0)], parent_index=3),
    ]
    assert len(requests) == 1


@pytest.mark.parametrize("update", [
    {"content": ""}, {"content": 42}, {"topic": "invalid"}, {"scope": "invalid"},
    {"event_status": "invalid"}, {"conditions": False}, {"verification": " "},
    {"source_unit_ranges": []}, {"source_unit_ranges": [True]}, {"source_unit_ranges": ["1"]},
    {"source_unit_ranges": [-1]}, {"source_unit_ranges": [999]},
    {"parent_index": 0}, {"parent_index": None}, {"secondary_entries": []},
])
def test_extraction_invalid_child_fails_entire_response_without_retry(mock_openrouter, update):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    child = secondary_payload("A child", [{"first": 0, "last": 0}])
    child.update(update)
    raw_response = json.dumps({"facts": [fact_payload("A primary", [{"first": 0, "last": 0}], secondary_entries=[child])]})
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "current", "content": "Paris"}])
    assert isinstance(raised.value.__cause__, ValueError)
    assert raised.value.raw_response == raw_response
    assert len(requests) == 1


@pytest.mark.parametrize("missing", ["content", "source_unit_ranges", "topic", "scope", "event_status", "conditions", "verification"])
def test_extraction_child_requires_its_own_complete_metadata(mock_openrouter, missing):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    child = secondary_payload("A child", [{"first": 0, "last": 0}])
    child.pop(missing)
    raw_response = json.dumps({"facts": [fact_payload("A primary", [{"first": 0, "last": 0}], secondary_entries=[child])]})
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "current", "content": "Paris"}])
    assert raised.value.raw_response == raw_response
    assert len(requests) == 1


@pytest.mark.parametrize("parent", [None, 0, 1])
def test_extraction_rejects_old_flat_parent_index_wire(mock_openrouter, parent):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    raw_response = json.dumps({"facts": [fact_payload("Old graph entry", [{"first": 0, "last": 0}], parent_index=parent)]})
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "current", "content": "Paris"}])
    assert raised.value.raw_response == raw_response
    assert len(requests) == 1


def test_extraction_source_ranges_select_exact_units_without_id_enumeration(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    payload = fact_payload("Second Paris", [])
    payload["source_unit_ranges"] = [{"first": 2, "last": 2}]
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [payload]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "current", "content": "Paris then Paris"}]) == [
        mapped_fact("Second Paris", [("current", "Paris", 1)]),
    ]
    assert len(requests) == 1


def test_extraction_long_literal_paragraph_uses_one_compact_range(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    content = " ".join(f"word{i}" for i in range(600)) + ".\nA final clause."
    selected = ranges_for_quote(content, content)
    assert len(selected) == 1
    payload = fact_payload("A long literal source", selected)
    raw_response = json.dumps({"facts": [payload]})
    assert len(raw_response) < 400
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    facts = providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "long", "content": content}])
    assert facts == [mapped_fact("A long literal source", [("long", content, 0)])]
    assert len(requests) == 1
    state = json.loads(requests[0][1]["messages"][1]["content"])
    assert state["new_chunks"][0]["content"] == content
    assert len(state["new_chunks"][0]["source_units"]) > 600


@pytest.mark.parametrize("source_range", [
    {"first": 2, "last": 1},
    {"first": 1, "last": 0},
    {"first": 3, "last": 1},
])
def test_extraction_invalid_physical_range_fails_entire_response_without_retry(mock_openrouter, source_range):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    raw_response = json.dumps({"facts": [fact_payload("Invalid physical range", [source_range])]})
    requests = mock_openrouter(lambda path, body: completion(raw_response))
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAILLM(config.llm).extract("Extract", [], [
            {"message_id": "test-message", "id": "first", "content": "Paris then Paris"}, {"message_id": "test-message", "id": "second", "content": "Berlin"},
        ])
    assert isinstance(raised.value.__cause__, ValueError)
    assert raised.value.raw_response == raw_response
    assert len(requests) == 1


def test_extraction_captured_v9_locomo_failure_has_one_monotonic_word_range(mock_openrouter):
    # conv-41 D1:3: v9 failed when clause and word candidate coordinates overlapped.
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    quote = "Been busy volunteering at the homeless shelter and keeping fit. Just started doing aerial yoga, it's great."
    content = quote + " Have you tried any other cool workout classes?"
    selected = ranges_for_quote(content, quote)
    assert len(selected) == 1
    assert selected[0]["first"] < selected[0]["last"]
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [
        fact_payload("Maria volunteers and started aerial yoga", selected),
    ]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "maria", "content": content}]) == [
        mapped_fact("Maria volunteers and started aerial yoga", [("maria", quote, 0)]),
    ]
    assert len(requests) == 1


def test_extraction_captured_v9_longmemeval_sports_source_keeps_newlines_and_endpoints(mock_openrouter):
    # 95bcc1c8 sports source appeared in the v9 failed first batch.
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    quote = "There are several popular sports at Queen's University Belfast, including:\n\n1. Rugby: Queen's Rugby Football Club is one of the oldest and most successful rugby clubs in Ireland."
    content = quote + " It has several teams that compete at various levels, including the all-important Ulster Bank All-Ireland League."
    selected = ranges_for_quote(content, quote)
    assert len(selected) == 1
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [
        fact_payload("Queen's has a successful rugby club", selected),
    ]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [{"message_id": "test-message", "id": "sports", "content": content}]) == [
        mapped_fact("Queen's has a successful rugby club", [("sports", quote, 0)]),
    ]
    assert len(requests) == 1


def test_global_source_range_cites_each_intersected_chunk_and_skips_empty_chunks(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    chunks = [
        {"message_id": "test-message", "id": "first", "content": "Paris then Paris"},
        {"message_id": "test-message", "id": "blank", "content": " \n"},
        {"message_id": "test-message", "id": "middle", "content": "Café\n上海"},
        {"message_id": "test-message", "id": "last", "content": "Berlin then Paris"},
    ]
    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        units = [unit_id for chunk in state["new_chunks"] for unit_id, _ in chunk["source_units"]]
        first = 2 if state["new_chunks"][0]["content"].startswith("Paris") else 0
        payload = fact_payload("A combined memory", [{"first": first, "last": units[-1]}])
        return completion(json.dumps({"facts": [payload]}))

    requests = mock_openrouter(handler)
    assert providers.OpenAILLM(config.llm).extract("Extract", [], chunks) == [
        mapped_fact("A combined memory", [("first", "Paris", 1)]),
        mapped_fact("A combined memory", [("middle", "Café\n上海", 0), ("last", "Berlin then Paris", 0)]),
    ]
    assert len(requests) == 2
    assert [chunk["content"] for _, body in requests
            for chunk in json.loads(body["messages"][1]["content"])["new_chunks"]] == [chunk["content"] for chunk in chunks]


def test_global_range_over_split_longmemeval_message_preserves_all_source_parts(mock_openrouter):
    # Static source excerpts from one 95bcc1c8 assistant message, split at list-item boundaries.
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    pieces = [
        "1. Player scouting: AI can analyze players' statistics, skill sets, and other factors to identify promising talent for recruitment.",
        "2. Game strategy: Analyzing opposing teams' tactics, strengths, and weaknesses can help create tailored game plans.",
        "3. Player performance analysis: AI can track individual player performance, offering personalized coaching and training recommendations.",
    ]
    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        last = state["new_chunks"][-1]["source_units"][-1][0]
        payload = fact_payload("AI supports sports analysis", [{"first": 0, "last": last}])
        return completion(json.dumps({"facts": [payload]}))

    requests = mock_openrouter(handler)
    chunks = [{"message_id": "test-message", "id": f"split-{index}", "content": content} for index, content in enumerate(pieces)]
    assert providers.OpenAILLM(config.llm).extract("Extract", [], chunks) == [
        mapped_fact("AI supports sports analysis", [(f"split-{index}", content, 0) for index, content in enumerate(pieces[:2])]),
        mapped_fact("AI supports sports analysis", [("split-2", pieces[2], 0)]),
    ]
    assert len(requests) == 2


def test_global_range_over_locomo_fundraiser_exchange_preserves_each_speaker_source(mock_openrouter):
    # Static conv-41 D16:6/D16:7 exchange; v11 could not cite a continuous multi-chunk interval.
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    chunks = [
        {"message_id": "test-message", "id": "maria", "role": "user", "content": "Thanks, John! Your help is really appreciated. If you know anyone who might be interested in volunteering for the event, let me know. We can do this!"},
        {"message_id": "test-message", "id": "john", "role": "user", "content": "Yep, Maria! I'll ask around to see if anyone I know wants to help. We'll find some awesome people for the cause. Let's make a change!"},
    ]
    _, lookup = build_source_units([chunk["content"] for chunk in chunks])
    payload = fact_payload("John will ask around for fundraiser volunteers", [{"first": 0, "last": len(lookup) - 1}])
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [payload]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], chunks) == [
        mapped_fact("John will ask around for fundraiser volunteers", [
            (chunk["id"], chunk["content"], 0) for chunk in chunks
        ]),
    ]
    assert len(requests) == 1


def test_extraction_separates_original_messages_before_building_their_source_schema(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: completion('{"facts": []}'))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [
        {"id": "first", "message_id": "message-1", "content": "Paris"},
        {"id": "second", "message_id": "message-2", "content": "Berlin"},
    ]) == []
    assert len(requests) == 2
    assert [[chunk["content"] for chunk in json.loads(body["messages"][1]["content"])["new_chunks"]]
            for _, body in requests] == [["Paris"], ["Berlin"]]
    for _, body in requests:
        endpoint_schema = body["response_format"]["json_schema"]["schema"]["properties"]["facts"]["items"]["properties"]["source_unit_ranges"]["items"]["properties"]
        assert endpoint_schema["first"]["maximum"] == endpoint_schema["last"]["maximum"] == 0


def test_extraction_keeps_all_fragments_of_one_message_in_one_request(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)

    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        assert [chunk["content"] for chunk in state["new_chunks"]] == ["Paris", "then Paris"]
        assert [unit_id for chunk in state["new_chunks"] for unit_id, _ in chunk["source_units"]] == [0, 1, 2]
        assert all("message_id" not in chunk for chunk in state["new_chunks"])
        return completion(json.dumps({"facts": [fact_payload("One complete message", [{"first": 0, "last": 2}])]}))

    requests = mock_openrouter(handler)
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [
        {"id": "fragment-1", "message_id": "one-message", "content": "Paris"},
        {"id": "fragment-2", "message_id": "one-message", "content": "then Paris"},
    ]) == [mapped_fact("One complete message", [("fragment-1", "Paris", 0), ("fragment-2", "then Paris", 0)])]
    assert len(requests) == 1


def test_extraction_offsets_nested_parent_indices_across_three_message_groups(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)

    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        assert len(state["new_chunks"]) == 1
        chunk = state["new_chunks"][0]
        word = chunk["content"]
        assert chunk["source_units"] == [[0, word]]
        children = 2 if word == "Paris" else 1
        payload = fact_payload(f"{word} primary", [{"first": 0, "last": 0}], secondary_entries=[
            secondary_payload(f"{word} child {number}", [{"first": 0, "last": 0}]) for number in range(children)
        ])
        return completion(json.dumps({"facts": [payload]}))

    requests = mock_openrouter(handler)
    chunks = [{"id": word, "message_id": f"message-{word}", "content": word} for word in ("Paris", "Berlin", "Tokyo")]
    facts = providers.OpenAILLM(config.llm).extract("Extract", [], chunks)
    assert [fact["parent_index"] for fact in facts] == [None, 0, 0, None, 3, None, 5]
    assert [fact["source_chunk_ids"] for fact in facts] == [["Paris"]] * 3 + [["Berlin"]] * 2 + [["Tokyo"]] * 2
    assert len(requests) == 3


def test_extraction_later_message_failure_raises_without_returning_partial_facts_or_retry(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    raw = '{"facts": null}'
    responses = iter([completion(json.dumps({"facts": [fact_payload("First fact", [{"first": 0, "last": 0}])]})), completion(raw)])
    requests = mock_openrouter(lambda path, body: next(responses))
    chunks = [{"id": word, "message_id": f"message-{word}", "content": word} for word in ("Paris", "Berlin", "Tokyo")]
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAILLM(config.llm).extract("Extract", [], chunks)
    assert raised.value.raw_response == raw
    assert isinstance(raised.value.__cause__, ValueError)
    assert len(requests) == 2
    assert [json.loads(body["messages"][1]["content"])["new_chunks"][0]["content"] for _, body in requests] == ["Paris", "Berlin"]


@pytest.mark.parametrize("invalid", [{}, {"message_id": None}, {"message_id": False}, {"message_id": 42},
                                      {"message_id": []}, {"message_id": ""}, {"message_id": " \t"}])
def test_all_message_ids_are_validated_before_any_api_request(mock_openrouter, invalid):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: pytest.fail("Invalid message_id must fail before every API request"))
    with pytest.raises(ValueError):
        providers.OpenAILLM(config.llm).extract("Extract", [], [
            {"id": "valid-first", "message_id": "valid", "content": "Paris"},
            {"id": "invalid-second", "content": "Berlin", **invalid},
        ])
    assert requests == []


def test_whitespace_message_group_uses_no_request_or_source_id_before_the_next_group(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: completion(json.dumps({"facts": [
        fact_payload("Paris", [{"first": 0, "last": 0}]),
    ]})))
    assert providers.OpenAILLM(config.llm).extract("Extract", [], [
        {"id": "blank", "message_id": "blank-message", "content": " \n\t"},
        {"id": "real", "message_id": "real-message", "content": "Paris"},
    ]) == [mapped_fact("Paris", [("real", "Paris", 0)])]
    assert len(requests) == 1
    state = json.loads(requests[0][1]["messages"][1]["content"])
    assert state["new_chunks"] == [{"index": 0, "content": "Paris", "source_units": [[0, "Paris"]]}]


def test_bounded_extraction_preserves_all_long_message_chunks_and_only_own_reference_context(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    chunks = [{"id": f"piece-{index}", "message_id": "long-message", "content": f"piece{index}",
               "role": "assistant", "source_order": index, "timestamp": 1704067200000} for index in range(7)]
    chunks.append({"id": "other", "message_id": "next-message", "content": "Next message", "source_order": 7})
    requests = mock_openrouter(lambda path, body: completion('{"facts": []}'))
    assert providers.OpenAILLM(config.llm).extract("Extract", [{"content": "FORBIDDEN_GOLD_ANSWER"}], chunks) == []
    states = [json.loads(body["messages"][1]["content"]) for _, body in requests]
    assert [len(state["new_chunks"]) for state in states] == [2, 2, 2, 1, 1]
    assert [chunk["content"] for state in states for chunk in state["new_chunks"]] == [chunk["content"] for chunk in chunks]
    assert all("FORBIDDEN_GOLD_ANSWER" not in json.dumps(body) for _, body in requests)
    assert "reference_context" not in states[0] and "reference_context" not in states[-1]
    for state in states[1:4]:
        reference = state["reference_context"]
        assert reference == {"role": "assistant", "content": "piece0", "source_order": 0,
                             "timestamp": 1704067200000, "timestamp_iso": "2024-01-01T00:00:00Z"}
        assert not {"id", "message_id", "index", "source_units"} & reference.keys()
    assert all(body["max_tokens"] == config.llm.max_output_tokens for _, body in requests)


def test_bounded_extraction_nested_parent_indices_offset_for_every_batch_and_message(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)

    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        text = state["new_chunks"][0]["content"]
        source = [{"first": 0, "last": 0}]
        payload = fact_payload(text, source, secondary_entries=[secondary_payload(text, source)])
        return completion(json.dumps({"facts": [payload]}))

    requests = mock_openrouter(handler)
    chunks = [{"id": f"chunk-{index}", "message_id": "first" if index < 5 else "second", "content": f"word{index}"}
              for index in range(6)]
    facts = providers.OpenAILLM(config.llm).extract("Extract", [], chunks)
    assert len(requests) == 4
    assert [fact["parent_index"] for fact in facts] == [None, 0, None, 2, None, 4, None, 6]
    assert [fact["source_chunk_ids"] for fact in facts] == [[f"chunk-{index}"] for index in (0, 0, 2, 2, 4, 4, 5, 5)]


@pytest.mark.parametrize("failure", ["length", "context_range"])
def test_bounded_extraction_rejects_later_batch_without_partial_return_or_content_retry(mock_openrouter, failure):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    first = completion(json.dumps({"facts": [fact_payload("Paris", [{"first": 0, "last": 0}])]}))
    bad = json.dumps({"facts": [fact_payload("Context cannot be cited", [{"first": 999, "last": 999}])]})
    responses = iter([first, completion(bad, finish_reason="length" if failure == "length" else "stop")])
    requests = mock_openrouter(lambda path, body: next(responses))
    chunks = [{"id": str(index), "message_id": "same", "content": "Paris"} for index in range(5)]
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAILLM(config.llm).extract("Extract", [], chunks)
    assert raised.value.raw_response == bad
    assert len(requests) == 2
    assert "reference_context" in json.loads(requests[-1][1]["messages"][1]["content"])


@pytest.mark.parametrize("orders", [[0, 2, 1], [0, 0, 2], [0, None, 2], [0, True, 2]])
def test_bounded_extraction_rejects_source_order_corruption_before_any_request(mock_openrouter, orders):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: pytest.fail("All ordering must be validated first"))
    chunks = [{"id": str(index), "message_id": "same", "content": "Paris", "source_order": order}
              for index, order in enumerate(orders)]
    with pytest.raises(ValueError, match="source_order"):
        providers.OpenAILLM(config.llm).extract("Extract", [], chunks)
    assert requests == []


def test_bounded_extraction_rejects_a_message_reappearing_after_another_message_before_calls(mock_openrouter):
    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    requests = mock_openrouter(lambda path, body: pytest.fail("Grouping must be validated before calls"))
    with pytest.raises(ValueError, match="contiguous"):
        providers.OpenAILLM(config.llm).extract("Extract", [], [
            {"id": str(index), "message_id": message, "content": "Paris"}
            for index, message in enumerate(("first", "second", "first"))
        ])
    assert requests == []


def test_bounded_extraction_plan_covers_all_engine_characters_with_original_chunk_overlap():
    from types import SimpleNamespace

    import tiktoken

    config, _ = load_config("configs/service/v1-b3.yaml", ServiceConfig)
    assert (config.chunk.max_tokens, config.chunk.overlap_tokens) == (512, 64)
    engine = SimpleNamespace(config=config, tokenizer=tiktoken.get_encoding(config.chunk.tokenizer))
    content = "[Session date: 2024-01-01]\nMaria: " + "完整原文 Café\n" * 1800 + "FINAL_SOURCE_MARKER"
    request = AddRequest(user_id="coverage", session_id="complete", request_id="coverage", messages=[Message(role="user", content=content)])
    _, chunks = MemoryEngine._chunks(engine, request, 0)
    batches = providers.extraction_batches(chunks)
    planned = [chunk for batch, _ in batches for chunk in batch]
    assert len(chunks) > 2 and planned == chunks
    assert all(1 <= len(batch) <= providers.EXTRACTION_CHUNKS_PER_CALL for batch, _ in batches)
    previous_end = 0
    restored = ""
    for chunk in planned:
        assert chunk["char_start"] <= previous_end
        assert chunk["content"] == content[chunk["char_start"]:chunk["char_end"]]
        restored += content[previous_end:chunk["char_end"]]
        previous_end = chunk["char_end"]
    assert previous_end == len(content) and restored == content
    assert "FINAL_SOURCE_MARKER" in planned[-1]["content"]
