import copy
import json

import httpx
import numpy as np
import pytest
from openai import OpenAI

from src import providers
from src.config import ModelConfig
from src.consolidation import consolidate


class Decisions:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.calls = []

    def decide(self, state, criteria):
        self.calls.append(copy.deepcopy((state, criteria)))
        choice = next(self.choices)
        if isinstance(choice, Exception):
            raise choice
        if callable(choice):
            choice = choice(state, criteria)
        if choice in {"duplicate", "supplement", "state_change", "correction", "conflict"}:
            choice = f"target0_{choice}"
        return {"choice": choice, "probabilities": {key: 1 / len(criteria) for key in criteria}, "model": "fake-jev"}


class Writer:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def rewrite(self, prompt, jobs):
        self.calls.append(copy.deepcopy(jobs))
        if self.result is not None:
            return self.result(jobs)
        return {job["id"]: job["previous_text"] + "; " + "; ".join(f["content"] for f in job["pending_facts"])
                for job in jobs}


def fact(identifier="f1", content="Alice works at Google", **times):
    return {"id": identifier, "content": content, "sources": [f"chunk-{identifier}"],
            "occurred_start": None, "occurred_end": None, "occurred_precision": None,
            "mentioned_at": "2026-09-01T12:00:00+00:00", **times}


def observation(identifier="o1", content="Alice works at Google", vector=None, **times):
    return {"id": identifier, "content": content, "revision": 4, "sources": [f"old-{identifier}"],
            "vector": np.array([1.0, 0.0]) if vector is None else np.array(vector),
            "source_facts": [fact(f"old-{identifier}", content, **times)], "status": "active"}


def run(facts, choices, existing=None, vectors=None, writer=None):
    model = Decisions(choices)
    writer = writer or Writer()
    changed, records = consolidate(facts, vectors or [np.array([1.0, 0.0]) for _ in facts],
                                   existing or [], model, writer, "preserve evidence", "r1")
    return changed, records, model, writer


def test_new_observation_is_deterministic_and_does_not_call_writer():
    changed, records, _, writer = run([fact()], ["new"])
    repeated, _, _, _ = run([fact()], ["new"])
    assert changed == repeated
    assert changed[0]["content"] == fact()["content"]
    assert changed[0]["revision"] == 1
    assert changed[0]["sources"] == ["f1"]
    assert records[0]["target_id"] == changed[0]["id"]
    assert records[0]["model"] == "fake-jev"
    assert writer.calls == []


def test_duplicate_adds_provenance_without_regenerating_old_text():
    old = observation()
    changed, records, _, writer = run([fact()], ["duplicate"], [old])
    assert changed[0]["id"] == "o1"
    assert changed[0]["content"] == old["content"]
    assert changed[0]["revision"] == 5
    assert changed[0]["sources"] == ["old-o1", "f1"]
    assert records[0]["relation"] == "duplicate"
    assert writer.calls == []
    assert old["sources"] == ["old-o1"]


def test_multiple_updates_see_pending_evidence_and_generate_one_revision():
    facts = [fact("f1", "Alice leads the team"), fact("f2", "Alice works remotely")]
    changed, _, model, writer = run(facts, ["supplement", "supplement"], [observation()])
    pending = model.calls[1][0]["candidate_observations"][0]["pending_facts"]
    assert [item["id"] for item in pending] == ["f1"]
    assert len(writer.calls) == 1 and len(writer.calls[0]) == 1
    assert [item["id"] for item in writer.calls[0][0]["pending_facts"]] == ["f1", "f2"]
    assert changed[0]["revision"] == 5
    assert changed[0]["sources"] == ["old-o1", "f1", "f2"]


def test_state_change_preserves_old_and_new_dates_for_writer():
    old = observation(occurred_start="2026-01-01", occurred_end="2026-01-01", occurred_precision="day",
                      mentioned_at="2026-01-02T12:00:00+00:00")
    new = fact(content="Alice moved to Meta on September 1", occurred_start="2026-09-01",
               occurred_end="2026-09-01", occurred_precision="day")
    changed, _, _, writer = run([new], ["state_change"], [old])
    job = writer.calls[0][0]
    assert job["source_facts"][0]["occurred_start"] == "2026-01-01"
    assert job["pending_facts"][0]["occurred_start"] == "2026-09-01"
    assert changed[0]["time_metadata"]["occurred_start"] == "2026-01-01"
    assert changed[0]["time_metadata"]["occurred_end"] == "2026-09-01"


def test_conflict_retains_both_claims_and_source_links():
    new = fact(content="Alice does not work at Google")
    changed, _, _, writer = run([new], ["conflict"], [observation()])
    job = writer.calls[0][0]
    assert job["previous_text"] == "Alice works at Google"
    assert job["pending_facts"][0]["content"] == "Alice does not work at Google"
    assert job["relations"] == [{"fact_id": "f1", "relation": "conflict"}]
    assert changed[0]["status"] == "conflicted"
    assert changed[0]["sources"] == ["old-o1", "f1"]


def test_uncertain_creates_separate_observation_without_editing_candidate():
    old = observation()
    changed, records, _, writer = run([fact(content="Another Alice works at Google")], ["uncertain"], [old])
    assert len(changed) == 1 and changed[0]["id"] != old["id"]
    assert changed[0]["status"] == "uncertain"
    assert changed[0]["sources"] == ["f1"]
    assert records[0]["target_id"] == changed[0]["id"]
    assert old["content"] == "Alice works at Google"
    assert writer.calls == []


def test_new_in_batch_observation_is_available_for_duplicate_and_update():
    facts = [fact("f1"), fact("f2"), fact("f3", "Alice works at Google remotely")]
    changed, records, model, writer = run(facts, ["new", "duplicate", "supplement"])
    assert len(changed) == 1
    assert changed[0]["revision"] == 1
    assert changed[0]["sources"] == ["f1", "f2", "f3"]
    assert {record["target_id"] for record in records} == {changed[0]["id"]}
    assert len(model.calls[2][0]["candidate_observations"][0]["pending_facts"]) == 2
    assert len(writer.calls[0]) == 1


def test_candidates_are_bounded_and_rank_by_cosine_not_vector_magnitude():
    old = [observation("exact", vector=[1, 1]), observation("large", vector=[1000, 0])]
    old += [observation(f"other-{i}", vector=[-1, -i - 1]) for i in range(5)]
    _, _, model, _ = run([fact()], ["new"], old, [np.array([1, 1])])
    candidates = model.calls[0][0]["candidate_observations"]
    assert len(candidates) == 5
    assert candidates[0]["id"] == "exact"


def test_pending_fact_vector_can_find_updated_observation_for_later_fact():
    facts = [fact("f1", "Alice moved"), fact("f2", "Alice's new work")]
    old = [observation("changed", vector=[1, 0]), observation("other", vector=[0.1, 0.9])]
    def select_changed(state, criteria):
        index = next(i for i, candidate in enumerate(state["candidate_observations"]) if candidate["id"] == "changed")
        return f"target{index}_state_change"
    _, records, model, _ = run(facts, [select_changed, "duplicate"], old,
                              [np.array([0, 1]), np.array([0, 1])])
    assert model.calls[1][0]["candidate_observations"][0]["id"] == "changed"
    assert records[1]["target_id"] == "changed"


def test_late_old_fact_keeps_source_recency_separate_from_arrival_order():
    old = observation(mentioned_at="2026-09-20T12:00:00+00:00")
    new = fact(mentioned_at="2025-01-01T12:00:00+00:00")
    changed, _, model, _ = run([new], ["supplement"], [old])
    assert model.calls[0][0]["new_fact"]["mentioned_at"] == "2025-01-01T12:00:00+00:00"
    assert "Arrival order does not establish current truth" in model.calls[0][0]["time_policy"]
    assert changed[0]["time_metadata"]["mentioned_at"] == "2026-09-20T12:00:00+00:00"


def test_unknown_and_mixed_timezone_times_are_not_filled_or_force_compared():
    unknown, _, _, _ = run([fact(mentioned_at=None)], ["new"])
    assert unknown[0]["time_metadata"]["mentioned_at"] is None
    old = observation(mentioned_at="2026-01-01T12:00:00")
    changed, _, _, writer = run([fact()], ["supplement"], [old])
    assert changed[0]["time_metadata"]["mentioned_at"] is None
    assert changed[0]["time_metadata"]["incomparable_fields"] == ["mentioned_at"]
    assert writer.calls[0][0]["source_facts"][0]["mentioned_at"] == "2026-01-01T12:00:00"


def test_coarse_iso_dates_keep_the_original_precision_and_values():
    old = observation(occurred_start="2025", occurred_end="2025", occurred_precision="year",
                      mentioned_at="2025")
    new = fact(occurred_start="2026-09", occurred_end="2026-09", occurred_precision="month",
               mentioned_at="2026-09")
    changed, _, _, writer = run([new], ["state_change"], [old])
    assert changed[0]["time_metadata"]["occurred_start"] == "2025"
    assert changed[0]["time_metadata"]["occurred_end"] == "2026-09"
    assert changed[0]["time_metadata"]["occurred_precision"] is None
    assert writer.calls[0][0]["source_facts"][0]["occurred_precision"] == "year"
    assert writer.calls[0][0]["pending_facts"][0]["occurred_precision"] == "month"


@pytest.mark.parametrize("old_end,new_end", [("2024", "2024-03"), ("2024-03", "2024")])
def test_end_aggregation_keeps_enclosing_coarse_year(old_end, new_end):
    old = observation(occurred_start="2024", occurred_end=old_end, occurred_precision="year")
    new = fact(occurred_start="2024-03", occurred_end=new_end, occurred_precision="month")
    changed, _, _, _ = run([new], ["supplement"], [old])
    assert changed[0]["time_metadata"]["occurred_start"] == "2024"
    assert changed[0]["time_metadata"]["occurred_end"] == "2024"


def test_overlapping_coarse_mention_dates_do_not_invent_recency_order():
    old = observation(mentioned_at="2024")
    changed, _, _, writer = run([fact(mentioned_at="2024-03")], ["supplement"], [old])
    assert changed[0]["time_metadata"]["mentioned_at"] is None
    assert changed[0]["time_metadata"]["incomparable_fields"] == ["mentioned_at"]
    assert writer.calls[0][0]["source_facts"][0]["mentioned_at"] == "2024"
    assert writer.calls[0][0]["pending_facts"][0]["mentioned_at"] == "2024-03"


@pytest.mark.parametrize("source_times,new_times,expected", [
    (("2026-06", "2026-06", "2026-06"), ("2024-01", "2024-01", "2026-09"), ("after", "before")),
    (("2024", "2024", "2024"), ("2024-03", "2024-03", "2024-03"), ("overlap", "overlap")),
    (("2024", "2024", "2026"), ("2025", "2025", "2025"), ("before", "after")),
    (("2026-01-01T10:00:00+08:00", "2026-01-01T10:00:00+08:00", "2026-01-01T10:00:00+08:00"),
     ("2026-01-01T03:00:00Z", "2026-01-01T03:00:00Z", "2026-01-01T02:00:00Z"), ("before", "overlap")),
    (("2024", None, None), ("2025", "2025", "2025"), ("unknown", "unknown")),
    (("2024", "2024", "2024"),
     ("2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"), ("unknown", "unknown")),
])
def test_candidate_time_hints_compare_source_intervals_to_incoming_without_inferring_truth(source_times, new_times, expected):
    def times(values):
        return dict(zip(("occurred_start", "occurred_end", "mentioned_at"), values))
    _, _, model, _ = run([fact(**times(new_times))], ["new"], [observation(**times(source_times))])
    candidate = model.calls[0][0]["candidate_observations"][0]
    assert candidate["temporal_comparisons"] == [{
        "fact_id": "old-o1", "event_to_new": expected[0], "mentioned_to_new": expected[1],
    }]
    assert tuple(candidate["source_facts"][0][key] for key in times(source_times)) == source_times


def test_time_hints_include_pending_facts_from_the_same_add():
    facts = [fact("f1", occurred_start="2024", occurred_end="2024", occurred_precision="year", mentioned_at="2026"),
             fact("f2", occurred_start="2025", occurred_end="2025", occurred_precision="year", mentioned_at="2025")]
    _, _, model, _ = run(facts, ["new", "supplement"])
    candidate = model.calls[1][0]["candidate_observations"][0]
    assert candidate["temporal_comparisons"] == [{
        "fact_id": "f1", "event_to_new": "before", "mentioned_to_new": "after",
    }]


def test_generator_decision_does_not_fabricate_probabilities():
    class GeneratorDecision:
        def decide(self, state, criteria):
            return {"choice": "new", "probabilities": None, "model": "fake-llm"}
    _, records = consolidate([fact()], [np.array([1, 0])], [], GeneratorDecision(), Writer(), "prompt", "r1")
    assert records[0]["probabilities"] is None
    assert records[0]["model"] == "fake-llm"


def test_invalid_probability_mass_is_rejected():
    class InvalidDecision:
        def decide(self, state, criteria):
            return {"choice": "new", "probabilities": {key: 0.1 for key in criteria}, "model": "fake"}
    with pytest.raises(ValueError, match="probability"):
        consolidate([fact()], [np.array([1, 0])], [], InvalidDecision(), Writer(), "prompt", "r1")


@pytest.mark.parametrize("existing_count", [0, 5])
def test_valid_rounded_jev_distribution_survives_provider_and_consolidation(monkeypatch, existing_count):
    monkeypatch.setenv("CONSOLIDATION_TEST_KEY", "test-key")
    probabilities = {}
    def respond(request):
        body = json.loads(request.content)
        criteria = body["questions"]["relation"]["criteria"]
        if existing_count:
            probabilities.update({key: .03 for key in criteria})
            probabilities["new"] = .21
            assert len(probabilities) == 27
            assert sum(probabilities.values()) == pytest.approx(.99)
        else:
            probabilities.update({"new": .8, "uncertain": .19995})
        assert set(criteria) == set(probabilities)
        return httpx.Response(200, json={
            "model": "typesafe/jev-1.13", "answers": {"relation": {
                "type": "choice", "choice": "new", "probabilities": probabilities,
            }},
        })
    def factory(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(providers, "OpenAI", factory)
    config = ModelConfig(name="typesafe/jev-1.13", base_url="https://openrouter.ai/api/v1",
                         api_key_env="CONSOLIDATION_TEST_KEY", temperature=0.0,
                         max_output_tokens=100, timeout_seconds=1.0, retries=0)
    model = providers.OpenAIDecision(config, "jev")
    try:
        existing = [observation(f"o{i}") for i in range(existing_count)]
        changed, records = consolidate([fact()], [np.array([1, 0])], existing, model, Writer(), "prompt", "r1")
    finally:
        model.client.close()
    assert changed[0]["sources"] == ["f1"]
    assert records[0]["probabilities"] == probabilities


@pytest.mark.parametrize("choice", ["missing-target", "target0_delete"])
def test_invalid_decision_is_rejected_without_writer(choice):
    writer = Writer()
    with pytest.raises(ValueError, match="choice"):
        run([fact()], [choice], [observation()], writer=writer)
    assert writer.calls == []


def test_failed_decision_propagates_without_mutating_inputs():
    old = observation()
    with pytest.raises(RuntimeError, match="decision unavailable"):
        run([fact()], [RuntimeError("decision unavailable")], [old])
    assert old["revision"] == 4
    assert old["sources"] == ["old-o1"]


@pytest.mark.parametrize("result", [lambda jobs: {}, lambda jobs: {"unexpected": "text"},
                                   lambda jobs: {jobs[0]["id"]: " "}, lambda jobs: {jobs[0]["id"]: 1}])
def test_invalid_writer_output_is_rejected(result):
    old = observation()
    with pytest.raises(ValueError, match="rewrite"):
        run([fact()], ["supplement"], [old], writer=Writer(result))
    assert old["revision"] == 4 and old["content"] == "Alice works at Google"


def test_writer_batches_at_most_eight_independent_targets():
    old = [observation(f"o{i}", vector=np.eye(9)[i]) for i in range(9)]
    facts = [fact(f"f{i}", f"Detail for o{i}") for i in range(9)]
    def select_target(state, criteria):
        wanted = "o" + state["new_fact"]["id"][1:]
        index = next(i for i, candidate in enumerate(state["candidate_observations"]) if candidate["id"] == wanted)
        return f"target{index}_supplement"
    changed, _, _, writer = run(facts, [select_target] * 9, old, list(np.eye(9)))
    assert len(changed) == 9
    assert [len(batch) for batch in writer.calls] == [8, 1]


def test_empty_batch_has_no_model_calls():
    changed, records, model, writer = run([], [])
    assert changed == [] and records == []
    assert model.calls == [] and writer.calls == []
