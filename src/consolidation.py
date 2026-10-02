"""Stage source-grounded observation revisions without writing to the store."""

import hashlib
import json

import numpy as np

from .providers import validate_choice_probabilities
from .temporal import time_bounds


RELATIONS = ("duplicate", "supplement", "state_change", "correction", "conflict")
TIME_FIELDS = ("occurred_start", "occurred_end", "occurred_precision", "mentioned_at")


def _new_id(request_id: str, fact_id: str, choice: str) -> str:
    value = json.dumps(("observation", request_id, fact_id, choice), separators=(",", ":"))
    return "mem_" + hashlib.sha256(value.encode()).hexdigest()[:32]


def _unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)) or np.linalg.norm(value) == 0:
        raise ValueError("invalid consolidation vector")
    return value / np.linalg.norm(value)


def _fact(fact: dict) -> dict:
    return {key: fact[key] for key in ("id", "content", *TIME_FIELDS) if key in fact}


def _time_metadata(stage: dict) -> dict:
    sources = stage["source_facts"] + stage["pending_facts"]
    # A caller without source facts may still supply the existing observation's bounds.
    if not stage["source_facts"] and stage["base_revision"]:
        sources = [stage["existing_times"], *sources]
    result = {}
    incomparable = []
    for field in ("occurred_start", "occurred_end", "mentioned_at"):
        values = [(*time_bounds(f[field]), f[field]) for f in sources if f.get(field) is not None]
        awareness = {lower.utcoffset() is not None for lower, _, _ in values}
        if len(awareness) > 1:
            result[field] = None
            incomparable.append(field)
        elif not values:
            result[field] = None
        elif field == "occurred_start":
            earliest = min(lower for lower, _, _ in values)
            result[field] = max((item for item in values if item[0] == earliest), key=lambda item: item[1])[2]
        elif field == "occurred_end":
            latest = max(upper for _, upper, _ in values)
            result[field] = min((item for item in values if item[1] == latest), key=lambda item: item[0])[2]
        else:
            latest = max(values, key=lambda item: item[1])
            if any(item[:2] != latest[:2] and item[1] >= latest[0] and item[0] <= latest[1] for item in values):
                result[field] = None
                incomparable.append(field)
            else:
                result[field] = latest[2]
    precisions = {f["occurred_precision"] for f in sources if f.get("occurred_precision") is not None}
    result["occurred_precision"] = next(iter(precisions)) if len(precisions) == 1 else None
    result["incomparable_fields"] = incomparable
    return result


def _temporal_comparison(source: dict, incoming: dict) -> dict:
    result = {"fact_id": source["id"]}
    for kind in ("event", "mentioned"):
        intervals = []
        for fact in (source, incoming):
            if kind == "event":
                start, end = fact.get("occurred_start"), fact.get("occurred_end")
                intervals.append((time_bounds(start)[0], time_bounds(end)[1]) if start and end else None)
            else:
                value = fact.get("mentioned_at")
                intervals.append(time_bounds(value) if value else None)
        if (None in intervals
                or len({date.utcoffset() is not None for interval in intervals if interval for date in interval}) > 1):
            relation = "unknown"
        elif intervals[0][1] < intervals[1][0]:
            relation = "before"
        elif intervals[0][0] > intervals[1][1]:
            relation = "after"
        else:
            relation = "overlap"
        result[f"{kind}_to_new"] = relation
    return result


def _state(fact: dict, candidates: list[dict]) -> dict:
    return {
        "new_fact": _fact(fact),
        "candidate_observations": [
            {"id": candidate["id"], "content": candidate["content"],
             "source_facts": candidate["source_facts"], "pending_facts": candidate["pending_facts"],
             "pending_relations": candidate["relations"], "time_metadata": _time_metadata(candidate),
             "temporal_comparisons": [_temporal_comparison(source, fact)
                                      for source in candidate["source_facts"] + candidate["pending_facts"]]}
            for candidate in candidates
        ],
        "time_policy": "Use source mentioned_at and occurred intervals. Arrival order does not establish current truth. "
                       "Late old information may add history without replacing a newer state. Unknown times stay unknown. "
                       "temporal_comparisons are code-computed source-to-new interval relations, not truth judgments.",
    }


def _criteria(candidates: list[dict]) -> tuple[dict, dict]:
    criteria = {
        "new": "A distinct durable facet with no matching candidate. Shared topic alone is not a match.",
        "uncertain": "The entity, facet, or relation cannot be determined confidently; keep a separate observation.",
    }
    choices = {}
    descriptions = {
        "duplicate": "The same entity and facet, with no additional detail, condition, negation, or changed value.",
        "supplement": "The same entity and facet, adding compatible detail without changing its state.",
        "state_change": "The same entity and facet, explicitly describing a historical transition; preserve both states and dates.",
        "correction": "The same entity and facet, explicitly correcting an earlier claim rather than merely arriving later.",
        "conflict": "The same entity and facet, incompatible claims whose contradiction or temporal order remains unresolved.",
    }
    for index, candidate in enumerate(candidates):
        for relation in RELATIONS:
            choice = f"target{index}_{relation}"
            criteria[choice] = f"Target {candidate['id']}: {descriptions[relation]}"
            choices[choice] = (candidate["id"], relation)
    return criteria, choices


def consolidate(facts: list[dict], fact_vectors: list[np.ndarray], existing: list[dict],
                decision_model, llm, prompt: str, request_id: str) -> tuple[list[dict], list[dict]]:
    """Decide relations in source order, then generate at most one revision per target.

    Pending facts participate in subsequent decisions before text is rewritten. All
    returned revisions are ready for embedding and a single caller-owned transaction.
    """
    if len(facts) != len(fact_vectors):
        raise ValueError("fact vector count does not match facts")
    stages = {}
    for observation in existing:
        sources = list(dict.fromkeys(observation["sources"]))
        if not sources:
            raise ValueError("observation must have sources")
        stages[observation["id"]] = {
            "id": observation["id"], "content": observation["content"],
            "base_revision": observation["revision"], "sources": sources,
            "source_facts": [_fact(f) for f in observation.get("source_facts", [])],
            "existing_times": observation.get("time_metadata", observation),
            "pending_facts": [], "relations": [], "vectors": [_unit(observation["vector"])],
            "status": observation.get("status", "active"), "rewrite": False,
        }
    changed_ids = []
    decisions = []
    for fact, vector in zip(facts, fact_vectors):
        if not fact.get("content", "").strip() or not fact.get("sources"):
            raise ValueError("fact must have content and sources")
        vector = _unit(vector)
        ranked = sorted(stages.values(), key=lambda stage: (
            -max(float(vector @ candidate) for candidate in stage["vectors"]), stage["id"]))
        candidates = ranked[:5]
        criteria, choices = _criteria(candidates)
        result = decision_model.decide(_state(fact, candidates), criteria)
        if (not isinstance(result, dict) or not isinstance(result.get("choice"), str)
                or result["choice"] not in criteria):
            raise ValueError("invalid consolidation choice")
        probabilities = result.get("probabilities")
        if probabilities is not None:
            validate_choice_probabilities(result["choice"], probabilities, criteria)
        choice = result["choice"]
        if choice in ("new", "uncertain"):
            target_id = _new_id(request_id, fact["id"], choice)
            relation = choice
            stages[target_id] = {
                "id": target_id, "content": fact["content"], "base_revision": 0,
                "sources": [], "source_facts": [], "existing_times": {},
                "pending_facts": [], "relations": [], "vectors": [],
                "status": "uncertain" if choice == "uncertain" else "active", "rewrite": False,
            }
        else:
            target_id, relation = choices[choice]
        stage = stages[target_id]
        stage["sources"] = list(dict.fromkeys([*stage["sources"], fact["id"]]))
        stage["pending_facts"].append(_fact(fact))
        stage["relations"].append({"fact_id": fact["id"], "relation": relation})
        stage["vectors"].append(vector)
        stage["rewrite"] |= relation in ("supplement", "state_change", "correction", "conflict")
        if relation == "conflict":
            stage["status"] = "conflicted"
        if target_id not in changed_ids:
            changed_ids.append(target_id)
        decisions.append({"fact_id": fact["id"], "target_id": target_id, "relation": relation,
                          "choice": choice, "probabilities": dict(probabilities) if probabilities is not None else None,
                          "model": result.get("model")})

    jobs = [
        {"id": target_id, "previous_text": stages[target_id]["content"],
         "source_facts": stages[target_id]["source_facts"],
         "pending_facts": stages[target_id]["pending_facts"], "relations": stages[target_id]["relations"]}
        for target_id in changed_ids if stages[target_id]["rewrite"]
    ]
    for start in range(0, len(jobs), 8):
        batch = jobs[start:start + 8]
        rewritten = llm.rewrite(prompt, batch)
        if (not isinstance(rewritten, dict) or set(rewritten) != {job["id"] for job in batch}
                or any(not isinstance(text, str) or not text.strip() for text in rewritten.values())):
            raise ValueError("invalid observation rewrite")
        for target_id, content in rewritten.items():
            stages[target_id]["content"] = content.strip()
    changed = [
        {"id": target_id, "content": stages[target_id]["content"],
         "revision": stages[target_id]["base_revision"] + 1, "sources": stages[target_id]["sources"],
         "status": stages[target_id]["status"], "time_metadata": _time_metadata(stages[target_id])}
        for target_id in changed_ids
    ]
    return changed, decisions
