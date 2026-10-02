import json

import httpx
import pytest
from openai import OpenAI

from src import providers
from src.config import ServiceConfig, load_config
from src.temporal import time_bounds, validate_temporal


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
            return handler(request.url.path, body)

        def factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            clients.append(client)
            return client

        monkeypatch.setattr(providers, "OpenAI", factory)
        return requests

    yield install
    for client in clients:
        client.close()


def config():
    return load_config("configs/service/v1-b3.yaml", ServiceConfig)[0]


def completion(value, finish_reason="stop", refusal=None, usage=None):
    return httpx.Response(200, json={
        "id": "chat-request", "object": "chat.completion", "created": 0,
        "model": "gpt-4o-mini-revision", "usage": usage,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": json.dumps(value), "refusal": refusal}}],
    })


def jev_response(answer=None):
    return httpx.Response(200, json={
        "model": "typesafe/jev-1.13-20260917", "id": "jev-request",
        "answers": {"relation": answer or {"type": "choice", "choice": "new", "probabilities": {"new": .8, "uncertain": .2}, "confidence": .3}},
        "usage": {"input_tokens": 100, "output_tokens": 2, "cost": .0000042},
    })


def test_jev_calls_absolute_decisions_url_and_records_real_usage(transport):
    calls = transport(lambda path, body: jev_response())
    events = []
    model = config().llm.model_copy(update={"name": "typesafe/jev-1.13"})
    result = providers.OpenAIDecision(model, "jev", events.append).decide({"fact": "new fact"}, {"new": "New fact", "uncertain": "Cannot determine"})
    assert result == {"choice": "new", "probabilities": {"new": .8, "uncertain": .2}, "model": "typesafe/jev-1.13-20260917"}
    url, body = calls[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert body["model"] == "typesafe/jev-1.13"
    assert json.loads(body["state"]) == {"fact": "new fact"}
    assert body["questions"]["relation"]["criteria"] == {"new": "New fact", "uncertain": "Cannot determine"}
    assert events[0]["usage"] == {"input_tokens": 100, "output_tokens": 2, "cost": .0000042}
    assert events[0]["resolved_model"] == result["model"]
    assert events[0]["request_id"] == "jev-request"
    assert events[0]["success"] is True
    assert events[0]["latency_seconds"] >= 0
    assert "private-key" not in json.dumps(events)


@pytest.mark.parametrize("answer", [
    {"type": "noul", "choice": "new", "probabilities": {"new": .8, "uncertain": .2}},
    {"type": "choice", "choice": "other", "probabilities": {"new": .8, "uncertain": .2}},
    {"type": "choice", "choice": "uncertain", "probabilities": {"new": .8, "uncertain": .2}},
    {"type": "choice", "choice": "new", "probabilities": {"new": 1}},
    {"type": "choice", "choice": "new", "probabilities": {"new": .8, "uncertain": .8}},
    {"type": "choice", "choice": "new", "probabilities": {"new": True, "uncertain": 0}},
    {"type": "choice", "choice": "new", "probabilities": {"new": 1.1, "uncertain": -.1}},
])
def test_jev_rejects_invalid_typed_output_without_fallback(transport, answer):
    calls = transport(lambda path, body: jev_response(answer))
    events = []
    model = config().llm.model_copy(update={"name": "typesafe/jev-1.13"})
    with pytest.raises(providers.ProviderError):
        providers.OpenAIDecision(model, "jev", events.append).decide({}, {"new": "new", "uncertain": "unknown"})
    assert len(calls) == 1
    assert events[0]["success"] is False


def test_llm_choice_uses_same_criteria_and_enum(transport):
    criteria = {"new": "An independent fact", "uncertain": "Unclear relation"}

    def handler(path, body):
        assert path == "/api/v1/chat/completions"
        assert json.loads(body["messages"][1]["content"]) == {"state": {"fact": "Paris"}, "criteria": criteria}
        assert body["messages"][0]["content"] == providers.OpenAIDecision.INSTRUCTIONS
        schema = body["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["properties"]["choice"]["enum"] == list(criteria)
        return completion({"choice": "new"})

    transport(handler)
    events = []
    result = providers.OpenAIDecision(config().llm, "llm", events.append).decide({"fact": "Paris"}, criteria)
    assert result == {"choice": "new", "probabilities": None, "model": "gpt-4o-mini-revision"}
    assert events[0]["usage"] is None


def test_temporal_extraction_preserves_metadata_and_date_precision(transport):
    source = {"id": "c1", "content": "I moved last year", "role": "user", "timestamp": 1704067200000, "source_order": 7}
    fields = {"occurred_start": "2023", "occurred_end": "2023", "occurred_precision": "year", "mentioned_at": "2024-01-01T00:00:00Z"}

    def handler(path, body):
        state = json.loads(body["messages"][1]["content"])
        chunk = state["new_chunks"][0]
        assert chunk == {"index": 0, **{key: value for key, value in source.items() if key != "id"}, "timestamp_iso": "2024-01-01T00:00:00Z"}
        assert state["prior_messages"][0]["timestamp_iso"] == "2024-01-01T00:00:00Z"
        schema = body["response_format"]["json_schema"]["schema"]["properties"]["facts"]["items"]
        assert set(schema["required"]) == {"content", "source_indices", *fields}
        return completion({"facts": [{"content": "Moved in 2023", "source_indices": [0], **fields}]})

    transport(handler)
    result = providers.OpenAILLM(config().llm).extract("Temporal facts", [source], [source], temporal=True)
    assert result == [{"content": "Moved in 2023", "source_chunk_ids": ["c1"], **fields}]


def test_temporal_extraction_accepts_literal_null_as_unknown(transport):
    source = {"id": "c1", "content": "I have used this since March 2, 2023."}
    transport(lambda path, body: completion({"facts": [{"content": "Uses this since March 2, 2023.",
        "source_indices": [0], "occurred_start": "2023-03-02", "occurred_end": "null",
        "occurred_precision": "day", "mentioned_at": None}]}))
    facts = providers.OpenAILLM(config().llm).extract("Temporal facts", [], [source], temporal=True)
    assert facts[0]["occurred_start"] == "2023-03-02"
    assert facts[0]["occurred_end"] is None
    assert validate_temporal({"occurred_end": "null"})["occurred_end"] is None


def test_temporal_extraction_never_uses_source_order_as_mention_date(transport):
    source = {"id": "c1", "content": "I use Bootstrap 5.3.0.", "role": "user",
              "timestamp": None, "source_order": 2}
    transport(lambda path, body: completion({"facts": [{"content": "Uses Bootstrap 5.3.0.",
        "source_indices": [0], "occurred_start": None, "occurred_end": None,
        "occurred_precision": None, "mentioned_at": "2"}]}))
    facts = providers.OpenAILLM(config().llm).extract("Temporal facts", [], [source], temporal=True)
    assert facts[0]["mentioned_at"] is None


@pytest.mark.parametrize("fields", [
    {"occurred_start": "2024-02-30", "occurred_precision": "day"},
    {"occurred_start": "yesterday", "occurred_precision": "day"},
    {"occurred_start": "2024-05", "occurred_end": "2024-04", "occurred_precision": "month"},
    {"occurred_start": "2024", "occurred_precision": None},
    {"occurred_precision": "day"},
    {"mentioned_at": 1704067200000},
    {"occurred_start": "2024", "occurred_precision": "unknown"},
])
def test_temporal_rejects_invalid_or_reversed_ranges(fields):
    with pytest.raises(ValueError):
        validate_temporal(fields)


def test_temporal_preserves_unknown_and_mixed_zones_without_comparing_them():
    assert validate_temporal({}) == dict.fromkeys(("occurred_start", "occurred_end", "occurred_precision", "mentioned_at"))
    fact = {"occurred_start": "2024-01-01T15:00+08:00", "occurred_end": "2024-01-01T10:00", "occurred_precision": "minute", "mentioned_at": "2024-01-02"}
    assert validate_temporal(fact) == fact


def test_rewrite_validates_exact_ids_and_nonempty_values(transport):
    events = []
    calls = transport(lambda path, body: completion({"o1": "New observation", "o2": "Another observation"}))
    result = providers.OpenAILLM(config().llm, events.append).rewrite("Write observations", [{"id": "o1"}, {"id": "o2"}])
    assert result == {"o1": "New observation", "o2": "Another observation"}
    schema = calls[0][1]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["o1", "o2"]
    assert schema["additionalProperties"] is False
    assert events[0]["stage"] == "consolidation_rewrite"


@pytest.mark.parametrize("value,finish_reason,refusal", [
    ({"other": "text"}, "stop", None),
    ({"o1": " "}, "stop", None),
    ({"o1": "text", "other": "text"}, "stop", None),
    ({"o1": "text"}, "length", None),
    ({"o1": "text"}, "stop", "Refused"),
])
def test_rewrite_rejects_bad_or_truncated_output(transport, value, finish_reason, refusal):
    calls = transport(lambda path, body: completion(value, finish_reason, refusal))
    events = []
    with pytest.raises(providers.ProviderError):
        providers.OpenAILLM(config().llm, events.append).rewrite("Write", [{"id": "o1"}])
    assert len(calls) == 1
    assert events[0]["success"] is False


def test_failed_request_telemetry_contains_no_secret(transport):
    transport(lambda path, body: httpx.Response(401, json={"error": {"message": "Unauthorized", "type": "auth"}}))
    events = []
    with pytest.raises(providers.ProviderError):
        providers.OpenAILLM(config().llm, events.append).complete("system", "user")
    assert events[0]["success"] is False
    assert events[0]["usage"] is None
    assert "private-key" not in json.dumps(events)


@pytest.mark.parametrize("start,end,precision", [
    ("2024-03", "2024", "month"),
    ("2024-02-29", "2024-02", "day"),
    ("2024-02-29T23:59", "2024-02-29", "minute"),
    ("2024-01-01T12:30:30.5Z", "2024-01-01T12:30Z", "second"),
])
def test_temporal_overlapping_partial_bounds_are_not_reversed(start, end, precision):
    fact = {"occurred_start": start, "occurred_end": end, "occurred_precision": precision, "mentioned_at": None}
    assert validate_temporal(fact) == fact


@pytest.mark.parametrize("start,end", [
    ("2025", "2024-12-31"), ("2024-03", "2024-02"),
    ("2024-01-01T12:31Z", "2024-01-01T12:30:59Z"),
    ("2024-01-01T12:00+00:00", "2024-01-01T12:00+01:00"),
])
def test_temporal_definitely_reversed_partial_bounds_are_rejected(start, end):
    with pytest.raises(ValueError, match="precedes"):
        validate_temporal({"occurred_start": start, "occurred_end": end, "occurred_precision": "minute"})


@pytest.mark.parametrize("value,lower,upper", [
    ("2024", "2024-01-01T00:00:00", "2024-12-31T23:59:59.999999"),
    ("2024-02", "2024-02-01T00:00:00", "2024-02-29T23:59:59.999999"),
    ("2024-02-29", "2024-02-29T00:00:00", "2024-02-29T23:59:59.999999"),
    ("2024-02-29T13+08:00", "2024-02-29T13:00:00+08:00", "2024-02-29T13:59:59.999999+08:00"),
    ("2024-02-29T13:14Z", "2024-02-29T13:14:00+00:00", "2024-02-29T13:14:59.999999+00:00"),
    ("2024-02-29T13:14:15", "2024-02-29T13:14:15", "2024-02-29T13:14:15.999999"),
    ("2024-02-29T13:14:15.12", "2024-02-29T13:14:15.120000", "2024-02-29T13:14:15.129999"),
])
def test_time_bounds_respect_source_precision_and_zone(value, lower, upper):
    bounds = time_bounds(value)
    assert tuple(item.isoformat() for item in bounds) == (lower, upper)


@pytest.mark.parametrize("probabilities", [
    {"new": .54, "a": .27, "b": .09, "c": .05, "d": .02, "e": .01, "f": .01},
    {"new": .51, "a": .5},
])
def test_jev_accepts_observed_cent_grid_rounding_without_normalization(transport, probabilities):
    answer = {"type": "choice", "choice": "new", "probabilities": probabilities, "confidence": .3}
    transport(lambda path, body: jev_response(answer))
    model = config().llm.model_copy(update={"name": "typesafe/jev-1.13"})
    result = providers.OpenAIDecision(model, "jev").decide({}, {key: key for key in probabilities})
    assert result["probabilities"] == probabilities
    assert sum(result["probabilities"].values()) == sum(probabilities.values())


def test_jev_keeps_selected_choice_when_rounded_probabilities_form_near_tie(transport):
    probabilities = {"new": .44, "target_supplement": .45, "uncertain": .11}
    answer = {"type": "choice", "choice": "new", "probabilities": probabilities, "confidence": .41}
    transport(lambda path, body: jev_response(answer))
    model = config().llm.model_copy(update={"name": "typesafe/jev-1.13"})
    result = providers.OpenAIDecision(model, "jev").decide({}, {key: key for key in probabilities})
    assert result["choice"] == "new"
    assert result["probabilities"] == probabilities
    with pytest.raises(ValueError, match="maximum probability"):
        providers.validate_choice_probabilities("new", {"new": .44, "target_supplement": .46, "uncertain": .10},
                                                {key: key for key in probabilities})


@pytest.mark.parametrize("probabilities", [
    {"new": .53, "a": .44},  # No normalized distribution in rounding intervals.
    {"new": .56, "a": .55, **{f"zero{i}": 0 for i in range(25)}},  # Clipping zeros matters.
    {"new": .543, "a": .446},  # Non-grid values retain the strict tolerance.
])
def test_jev_rejects_distribution_outside_rounding_bounds_and_keeps_raw_response(transport, probabilities):
    answer = {"type": "choice", "choice": "new", "probabilities": probabilities, "confidence": .3}
    transport(lambda path, body: jev_response(answer))
    model = config().llm.model_copy(update={"name": "typesafe/jev-1.13"})
    with pytest.raises(providers.ProviderError) as raised:
        providers.OpenAIDecision(model, "jev").decide({}, {key: key for key in probabilities})
    assert json.loads(raised.value.raw_response)["answers"]["relation"]["probabilities"] == probabilities


def test_shared_choice_validator_rejects_nonfinite_and_omitted_options():
    with pytest.raises(ValueError):
        providers.validate_choice_probabilities("new", {"new": float("nan"), "other": 0}, {"new": "new", "other": "other"})
    with pytest.raises(ValueError):
        providers.validate_choice_probabilities("new", {"new": 1}, {"new": "new", "other": "other"})
