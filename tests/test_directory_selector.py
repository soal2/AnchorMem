import copy
import json

import httpx
import pytest
from openai import OpenAI

from src import directory_selector
from src.config import ServiceConfig, load_config
from src.providers import ProviderError


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    clients = []

    def install(handler):
        requests = []

        def respond(request):
            assert request.headers["Authorization"] == "Bearer private-key"
            body = json.loads(request.content)
            requests.append((str(request.url), body))
            return handler(body)

        def factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            clients.append(client)
            return client

        monkeypatch.setattr(directory_selector, "OpenAI", factory)
        return requests

    yield install
    for client in clients:
        client.close()


def config():
    return load_config("configs/service/jevsight-d.yaml", ServiceConfig)[0].consolidation.decision_model.model_copy(
        update={"retries": 0})


def candidates():
    return [{"id": "memory-alpha", "path": "sessions/a", "abstract": "Travel plans", "overview": "Paris in May."},
            {"id": "memory-beta", "path": "sessions/b", "abstract": "Food preferences", "overview": "Enjoys sushi."}]


def answer(choice="relevant"):
    return {"type": "choice", "choice": choice,
            "probabilities": {key: .8 if key == choice else .1 for key in directory_selector.CRITERIA}}


def envelope():
    return {"model": "typesafe/jev-1.13-revision", "id": "jev-directories",
            "answers": {"d0": answer(), "d1": answer("irrelevant")},
            "usage": {"input_tokens": 100, "output_tokens": 4, "cost": .0000084}}


def test_select_batches_questions_and_binds_ids_in_code(transport):
    value = envelope()
    value["answers"] = {"d1": value["answers"]["d1"], "d0": value["answers"]["d0"]}
    calls = transport(lambda body: httpx.Response(200, json=value))
    events = []
    source = candidates()
    result = directory_selector.JevDirectorySelector(config(), events.append).select("Where am I going?", ["Paris", "Tokyo"], source)
    assert result == [{"id": item["id"], "choice": value["answers"][f"d{index}"]["choice"],
                       "probabilities": value["answers"][f"d{index}"]["probabilities"], "model": value["model"]}
                      for index, item in enumerate(source)]
    assert len(calls) == 1
    url, body = calls[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert body["model"] == "typesafe/jev-1.13"
    assert json.loads(body["state"]) == {
        "query": "Where am I going?", "options": ["Paris", "Tokyo"],
        "directories": {f"d{index}": {key: item[key] for key in ("path", "abstract", "overview")}
                        for index, item in enumerate(source)},
    }
    assert set(body["questions"]) == {"d0", "d1"}
    for key, question in body["questions"].items():
        assert question["type"] == "choice"
        assert question["criteria"] == directory_selector.CRITERIA
        assert directory_selector.INSTRUCTIONS in question["instructions"]
        assert f"Directory: {key}." in question["instructions"]
    assert events[0]["stage"] == "directory_selection_jev"
    assert events[0]["usage"] == value["usage"]
    assert events[0]["resolved_model"] == value["model"]
    assert events[0]["request_id"] == value["id"]
    assert events[0]["latency_seconds"] >= 0
    assert events[0]["success"] is True
    assert "private-key" not in json.dumps(events)


@pytest.mark.parametrize("mutation", [
    lambda value: value["answers"].pop("d1"),
    lambda value: value["answers"].update({"invented": answer()}),
    lambda value: value.update({"model": ""}),
    lambda value: value.update({"answers": []}),
    lambda value: value["answers"]["d0"].update({"type": "score"}),
    lambda value: value["answers"]["d0"].update({"id": "invented"}),
    lambda value: value["answers"]["d0"].update({"path": "other-user/private"}),
    lambda value: value["answers"]["d0"].update({"choice": "invented"}),
    lambda value: value["answers"]["d0"].update({"choice": "irrelevant"}),
    lambda value: value["answers"]["d0"].update({"probabilities": {"relevant": 1}}),
    lambda value: value["answers"]["d0"].update({"probabilities": {"relevant": .8, "irrelevant": .8, "uncertain": .8}}),
    lambda value: value["answers"]["d0"].update({"probabilities": {"relevant": True, "irrelevant": 0, "uncertain": 0}}),
])
def test_invalid_response_fails_without_fallback(transport, mutation):
    value = envelope()
    mutation(value)
    calls = transport(lambda body: httpx.Response(200, json=value))
    events = []
    with pytest.raises(ProviderError):
        directory_selector.JevDirectorySelector(config(), events.append).select("Travel?", None, candidates())
    assert len(calls) == 1
    assert len(events) == 1
    assert events[0]["success"] is False


def test_uncertain_is_preserved(transport):
    value = envelope()
    value["answers"]["d0"] = answer("uncertain")
    transport(lambda body: httpx.Response(200, json=value))
    result = directory_selector.JevDirectorySelector(config()).select("Unclear?", None, candidates())
    assert result[0]["choice"] == "uncertain"


def test_provider_failure_records_redacted_telemetry(transport):
    calls = transport(lambda body: httpx.Response(401, json={"error": {"message": "private-key Unauthorized", "type": "auth"}}))
    events = []
    with pytest.raises(ProviderError) as error:
        directory_selector.JevDirectorySelector(config(), events.append).select("Travel?", None, candidates())
    assert len(calls) == 1
    assert events[0]["success"] is False
    assert events[0]["usage"] is None
    assert "private-key" not in json.dumps(events)
    assert "private-key" not in json.dumps(error.value.diagnostics)


def test_empty_candidates_skip_provider(transport):
    calls = transport(lambda body: pytest.fail("empty candidates must not call Jev"))
    events = []
    assert directory_selector.JevDirectorySelector(config(), events.append).select("Query", None, []) == []
    assert calls == events == []


@pytest.mark.parametrize("invalid", [
    [{"id": "only-id"}],
    candidates() + [copy.deepcopy(candidates()[0])],
    [{"id": str(index), "path": str(index), "abstract": "x", "overview": "y"} for index in range(11)],
])
def test_invalid_candidates_fail_before_call(transport, invalid):
    calls = transport(lambda body: pytest.fail("invalid candidates must not call Jev"))
    with pytest.raises(ValueError):
        directory_selector.JevDirectorySelector(config()).select("Query", None, invalid)
    assert calls == []


def test_pins_decision_model():
    with pytest.raises(ValueError, match="pins"):
        directory_selector.JevDirectorySelector(config().model_copy(update={"name": "openai/gpt-4o-mini"}))
