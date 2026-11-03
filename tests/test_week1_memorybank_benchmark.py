"""Offline checks for the MemoryBank paper evaluation benchmark."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src import week1_memorybank_benchmark as mb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    # Use the real frozen dataset under the project; prepare writes only under datasets/memorybank.
    manifest = mb.prepare(ROOT)
    return manifest


def test_source_files_match_frozen_sha256():
    raw = mb.load_raw(ROOT)
    assert set(raw["banks"]) == {"en", "cn"}
    assert len(raw["banks"]["en"]) == 15
    assert len(raw["banks"]["cn"]) == 15


def test_english_whitespace_user_names_join_to_probing_keys():
    raw = mb.load_raw(ROOT)
    # Raw JSON keys include padded names; indexed map must expose stripped keys.
    assert "Jason" in raw["banks"]["en"]
    assert "Linda" in raw["banks"]["en"]
    assert "Ivy" in raw["banks"]["en"]
    assert "Roland" in raw["banks"]["en"]
    selections = mb.select_full(raw["banks"], raw["probes"])
    jason = [item for item in selections if item["user"] == "Jason"]
    assert jason and all(item["group"] == "en:Jason" for item in jason)


def test_full_selection_is_deterministic_and_answer_independent(prepared):
    raw = mb.load_raw(ROOT)
    selected = mb.select_full(raw["banks"], raw["probes"])
    assert selected == mb.select_full(raw["banks"], raw["probes"])
    assert len(selected) == 200
    assert Counter_languages(selected) == {"en": 100, "cn": 100}
    assert len({item["group"] for item in selected}) == 30
    # Mutating unpublished answers cannot exist; mutate questions' surrounding bank meta instead.
    changed_banks = copy.deepcopy(raw["banks"])
    for language in changed_banks:
        for user in changed_banks[language]:
            changed_banks[language][user]["summary"] = {"leaked": "DO_NOT_USE"}
            changed_banks[language][user]["personality"] = {"leaked": "DO_NOT_USE"}
    assert mb.select_full(changed_banks, raw["probes"]) == selected


def Counter_languages(selected):
    from collections import Counter
    return dict(Counter(item["language"] for item in selected))




def test_empty_author_day_is_skipped_but_recorded(prepared):
    example = next(item for item in mb.load_full(ROOT) if item["user"] == "John Zhang")
    assert example["empty_dates"] == ["2023-05-05"]
    assert len(example["sessions"]) == 9
    assert all(session["messages"] for session in example["sessions"])

def test_normalize_preserves_ten_days_roles_and_strips_non_dialogue(prepared):
    examples = mb.load_full(ROOT)
    assert len(examples) == 200
    example = next(item for item in examples if item["id"] == "en:Emily#q000")
    history = mb.builder_history(example)
    assert len(history) == 10
    assert [session["id"] for session in history] == [f"session_{i}" for i in range(1, 11)]
    roles = [message["role"] for message in history[0]["messages"]]
    assert roles[0] == "user" and roles[1] == "assistant"
    assert example["source_messages"] == sum(len(s["messages"]) for s in history)
    assert example["source_tokens"] > 0
    assert example["answer"] is None
    assert "Emily" in example["evidence"] or "painting" in example["evidence"].lower()
    assert "DO_NOT_USE" not in str(history)
    assert "summary" not in str(history).lower() or "[Session date:" in history[0]["messages"][0]["content"]
    # Personality / overall_history must not enter builder history.
    blob = json.dumps(history, ensure_ascii=False)
    assert "gentle, kind-hearted" not in blob
    assert example["question_date"] == example["sessions"][-1]["date"] == "2023-05-06"


def test_builder_history_allowlist_excludes_answer_and_evidence():
    example = {
        "answer": "GOLD",
        "evidence": "EVIDENCE_ONLY",
        "meta_information": {"personality": "SECRET"},
        "sessions": [{
            "id": "session_1",
            "date": "2023-01-01",
            "messages": [{"role": "user", "content": "hi", "has_answer": True}],
        }],
    }
    assert mb.builder_history(example) == [{"id": "session_1", "messages": [{"role": "user", "content": "hi"}]}]


def test_judge_prompt_uses_evidence_when_answer_missing_and_parses_like_lme(prepared):
    example = next(item for item in mb.load_full(ROOT) if item["language"] == "en")
    prompt = mb.get_memorybank_judge_prompt("I recommended The Little Prince.", example)
    assert "Dialogue Evidence:" in prompt
    assert example["question"] in prompt
    assert "Answer yes or no only" in prompt
    assert mb.parse_memorybank_judge("Yes") is True
    assert mb.parse_memorybank_judge("no") is False
    assert mb.parse_memorybank_judge("  YES, correct ") is True
    with_answer = dict(example, answer="The Little Prince")
    gold_prompt = mb.get_memorybank_judge_prompt("The Little Prince", with_answer)
    assert "Correct Answer: The Little Prince" in gold_prompt
    assert "Dialogue Evidence:" not in gold_prompt


def test_prepare_manifest_counts_and_cost_estimate(prepared):
    assert prepared["counts"]["questions"] == 200
    assert prepared["counts"]["histories"] == 30
    assert prepared["counts"]["users_per_language"] == 15
    assert prepared["counts"]["dialogue_days_per_user"] == 10
    assert prepared["counts"]["source_tokens_all_histories"] > 0
    estimate = mb.estimate_cost(root=ROOT)
    assert estimate["questions"] == 200
    assert estimate["histories"] == 30
    assert estimate["source_tokens"] == prepared["counts"]["source_tokens_all_histories"]
    assert estimate["methods"]["vikingjev"]["total_usd_locomo_rate_plus_qa_judge"] > 0
    assert estimate["methods"]["mem0"]["total_usd_plus_judge"] > 0
    assert estimate["methods"]["memorybank_method"]["total_usd_plus_judge"] > 0
    # MemoryBank should be far cheaper than LongMemEval-S full (~$480 VikingJev).
    assert estimate["methods"]["vikingjev"]["total_usd_locomo_rate_plus_qa_judge"] < 5


def test_missing_user_is_rejected():
    raw = mb.load_raw(ROOT)
    probes = copy.deepcopy(raw["probes"])
    probes["en"][0] = {"Nobody": ["question?"]}
    with pytest.raises(ValueError, match="missing from en memory bank"):
        mb.select_full(raw["banks"], probes)
