import numpy as np
import pytest

from src.consolidation import consolidate
from test_consolidation import Decisions, Writer, fact, observation, run


def event_fact(identifier="f1", **context):
    return fact(identifier, "Product release enters review", topic="events", matter_id=f"matter-{identifier}",
                **context)


def event_observation(identifier="o1", **context):
    return observation(identifier, "Product release is planned", topic="events", matter_id=f"matter-{identifier}",
                       **context)


def test_candidates_are_partitioned_before_the_top_five_limit():
    foreign = [observation(f"foreign-{i}", scope="project", project_id="other") for i in range(6)]
    foreign += [observation("global"), observation("different-topic", topic="requirements",
                                                scope="project", project_id="current")]
    matching = observation("matching", vector=[0, 1], scope="project", project_id="current")
    incoming = fact(scope="project", project_id="current")
    changed, _, model, _ = run([incoming], ["duplicate"], [*foreign, matching])
    assert [candidate["id"] for candidate in model.calls[0][0]["candidate_observations"]] == ["matching"]
    assert changed[0]["id"] == "matching"
    assert changed[0]["scope"] == "project" and changed[0]["project_id"] == "current"
    assert changed[0]["topic"] == "entities" and changed[0]["matter_id"] is None


def test_same_batch_stages_stay_in_their_own_topic_and_scope():
    incoming = [fact("global"), fact("project", scope="project", project_id="p"),
                fact("requirements", topic="requirements"), fact("global-duplicate")]
    changed, records, model, _ = run(incoming, ["new", "new", "new", "duplicate"])
    assert model.calls[1][0]["candidate_observations"] == []
    assert model.calls[2][0]["candidate_observations"] == []
    assert len(model.calls[3][0]["candidate_observations"]) == 1
    assert len(changed) == 3
    assert records[3]["target_id"] == records[0]["target_id"]


def test_unresolved_facts_and_observations_do_not_enter_consolidation():
    unresolved = fact("pending", scope="unresolved")
    old = observation("pending-old", scope="unresolved")
    changed, records, model, writer = run([unresolved, fact()], ["new"], [old])
    assert len(changed) == 1 and len(records) == 1 and len(model.calls) == 1
    assert model.calls[0][0]["candidate_observations"] == []
    assert changed[0]["sources"] == ["f1"]
    assert writer.calls == []


def test_same_matter_adds_independent_facet_without_rewriting_target():
    incoming = event_fact(scope="project", project_id="p")
    old = event_observation(scope="project", project_id="p")
    changed, records, model, writer = run([incoming], ["target0_same_matter"], [old])
    assert len(changed) == 1 and changed[0]["id"] != old["id"]
    assert changed[0]["revision"] == 1 and changed[0]["content"] == incoming["content"]
    assert changed[0]["sources"] == ["f1"]
    assert changed[0]["matter_id"] == incoming["matter_id"] == "matter-o1"
    assert records[0]["relation"] == "same_matter" and records[0]["related_target_id"] == "o1"
    assert model.calls[0][0]["candidate_observations"][0]["matter_id"] == "matter-o1"
    assert old["revision"] == 4 and old["sources"] == ["old-o1"]
    assert writer.calls == []


@pytest.mark.parametrize("relation", ["duplicate", "supplement", "state_change", "correction", "conflict"])
def test_existing_event_relations_inherit_target_matter(relation):
    incoming = event_fact()
    changed, _, _, writer = run([incoming], [relation], [event_observation()])
    assert changed[0]["id"] == "o1"
    assert changed[0]["matter_id"] == incoming["matter_id"] == "matter-o1"
    if writer.calls:
        assert writer.calls[0][0]["pending_facts"][0]["matter_id"] == "matter-o1"


@pytest.mark.parametrize("choice", ["new", "uncertain"])
def test_independent_event_keeps_initial_matter(choice):
    incoming = event_fact()
    changed, _, _, _ = run([incoming], [choice], [event_observation()])
    assert changed[0]["matter_id"] == incoming["matter_id"] == "matter-f1"


def test_same_matter_linkage_is_available_to_later_facts_in_the_batch():
    incoming = [event_fact("f1"), event_fact("f2")]
    def select_new_facet(state, criteria):
        index = next(i for i, candidate in enumerate(state["candidate_observations"])
                     if candidate["pending_facts"])
        assert state["candidate_observations"][index]["matter_id"] == "matter-o1"
        return f"target{index}_duplicate"
    changed, records, _, writer = run(incoming, ["target0_same_matter", select_new_facet],
                                      [event_observation()])
    assert len(changed) == 1 and changed[0]["sources"] == ["f1", "f2"]
    assert records[0]["target_id"] == records[1]["target_id"]
    assert {item["matter_id"] for item in incoming} == {"matter-o1"}
    assert writer.calls == []


def test_non_event_criteria_have_no_same_matter_choice():
    _, _, model, _ = run([fact()], ["new"], [observation()])
    assert not any(choice.endswith("_same_matter") for choice in model.calls[0][1])


def test_source_fact_metadata_is_preserved_for_decision_and_rewrite():
    incoming = fact(topic="methods", conditions="Linux host", verification="source_claimed")
    old = observation(topic="methods", conditions="Local development", verification="unverified")
    _, _, model, writer = run([incoming], ["supplement"], [old])
    new_state = model.calls[0][0]["new_fact"]
    pending = writer.calls[0][0]["pending_facts"][0]
    source = writer.calls[0][0]["source_facts"][0]
    for item in (new_state, pending):
        assert item["topic"] == "methods" and item["scope"] == "global"
        assert item["project_id"] is None and item["matter_id"] is None
        assert item["conditions"] == "Linux host" and item["verification"] == "source_claimed"
    assert source["conditions"] == "Local development" and source["verification"] == "unverified"


def test_event_status_is_preserved_in_pending_source_facts():
    incoming = event_fact(event_status="completed")
    _, _, _, writer = run([incoming], ["state_change"], [event_observation()])
    assert writer.calls[0][0]["pending_facts"][0]["event_status"] == "completed"


@pytest.mark.parametrize("field", ["topic", "scope", "project_id", "matter_id"])
def test_old_schema_metadata_is_rejected_without_model_calls(field):
    incoming = fact()
    del incoming[field]
    model, writer = Decisions([]), Writer()
    with pytest.raises(ValueError, match="context metadata"):
        consolidate([incoming], [np.array([1, 0])], [], model, writer, "prompt", "r")
    assert model.calls == [] and writer.calls == []
