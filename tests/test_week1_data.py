import ast
import copy
import hashlib
from collections import Counter

import pytest

from src.week1_data import (
    LME_CATEGORIES, LOCOMO_CATEGORIES, ROOT, SOURCES, builder_history, content_sha256,
    get_lme_judge_prompt, normalize_pilot, parse_lme_judge, score_locomo, select_pilot,
)


@pytest.fixture
def locomo():
    return [{"sample_id": group,
             "conversation": {"speaker_a": "Alice", "speaker_b": "Bob",
                              "session_2_date_time": "2023-01-02", "session_2": [{"speaker": "Bob", "text": "Last turn"}],
                              "session_1_date_time": "2023-01-01", "session_1": [
                                  {"speaker": "Alice", "text": "First turn", "blip_caption": "a red boat",
                                   "query": "DO_NOT_EXPOSE_IMAGE_QUERY", "dia_id": "D1:1"},
                                  {"speaker": "Bob", "text": "Second turn"}]},
             "qa": [{"category": category, "question": f"question-{category}-{index}",
                     "answer": "GOLD_ONLY", "evidence": ["EVIDENCE_ONLY"]}
                    for category in (1, 2, 3, 4, 5) for index in range(4)],
             "event_summary": "EVENT_SUMMARY_ONLY", "session_summary": "SESSION_SUMMARY_ONLY"}
            for group in ("conv-41", "conv-26")]


@pytest.fixture
def lme():
    return [{"question_id": f"{category}-{index}" + ("_abs" if index == 3 else ""),
             "question_type": category, "question": "Question", "answer": "GOLD_ONLY",
             "question_date": "2024-01-01", "answer_session_ids": ["GOLD_SESSION_ONLY"],
             "haystack_session_ids": [f"answer_GOLD_ID-{category}-{index}"], "haystack_dates": ["2023-06-01"],
             "haystack_sessions": [[{"role": "user", "content": "user evidence", "has_answer": True},
                                    {"role": "assistant", "content": "assistant evidence", "has_answer": False}]]}
            for category in LME_CATEGORIES for index in range(4)]


def test_selection_is_stratified_deterministic_and_answer_independent(locomo, lme):
    selected = select_pilot(locomo, lme)
    assert selected == select_pilot(locomo, lme)
    assert len(selected) == 32
    assert [item["dataset"] for item in selected[:2]] == ["locomo", "longmemeval"]
    for group in ("conv-41", "conv-26"):
        questions = [item for item in selected if item.get("group") == group]
        assert len(questions) == 10
        counts = Counter(item["category"] for item in questions)
        assert all(counts[category] >= 2 for category in LOCOMO_CATEGORIES)
        assert 5 not in counts
    long_questions = [item for item in selected if item["dataset"] == "longmemeval"]
    assert Counter(item["category"] for item in long_questions) == {category: 2 for category in LME_CATEGORIES}
    for category in ("knowledge-update", "multi-session"):
        assert sum("_abs" in item["id"] for item in long_questions if item["category"] == category) == 1
    changed_locomo, changed_lme = copy.deepcopy(locomo), copy.deepcopy(lme)
    for conversation in changed_locomo:
        for qa in conversation["qa"]:
            qa["answer"] = "Changed gold answer"
    for item in changed_lme:
        item["answer"] = "Changed gold answer"
    assert select_pilot(changed_locomo, changed_lme) == selected
    assert all(item["exposure"] == "development" for item in selected)
    assert all(item["prior_exposure"] == "known-history" for item in selected if item.get("group") == "conv-41")
    assert all(item["prior_exposure"] == "unknown" for item in selected if item.get("group") == "conv-26")


def test_normalized_full_history_preserves_image_roles_dates_and_strips_annotations(locomo, lme):
    examples = normalize_pilot(locomo, lme, select_pilot(locomo, lme))
    conversation = next(item for item in examples if item["dataset"] == "locomo")
    history = builder_history(conversation)
    assert [session["id"] for session in history] == ["session_1", "session_2"]
    assert conversation["source_messages"] == 3
    assert conversation["source_tokens"] > 0
    assert all(message["role"] == "user" for session in history for message in session["messages"])
    assert history[0]["messages"][0]["content"] == "[Session date: 2023-01-01]\nAlice: First turn\n[Image description: a red boat]"
    assert "Bob: Last turn" in history[-1]["messages"][-1]["content"]
    assert all(marker not in str(history) for marker in (
        "GOLD_ONLY", "EVIDENCE_ONLY", "EVENT_SUMMARY_ONLY", "SESSION_SUMMARY_ONLY", "DO_NOT_EXPOSE_IMAGE_QUERY"))
    long_example = next(item for item in examples if item["dataset"] == "longmemeval")
    long_history = builder_history(long_example)
    assert [message["role"] for message in long_history[0]["messages"]] == ["user", "assistant"]
    assert long_example["question_date"] == "2024-01-01"
    assert long_example["source_messages"] == 2
    assert long_history[0]["id"] == "session_1"
    assert "GOLD_ID" not in str(long_history)
    assert all("has_answer" not in message for message in long_history[0]["messages"])
    assert "GOLD_ONLY" not in str(long_history) and "GOLD_SESSION_ONLY" not in str(long_history)
    assert "[Session date: 2023-06-01]" in long_history[0]["messages"][0]["content"]


def test_builder_history_is_an_allowlist_and_not_the_gold_example():
    example = {"answer": "GOLD", "question": "QUESTION", "sessions": [
        {"id": "history", "answer_session_ids": ["leak"], "messages": [
            {"role": "user", "content": "legitimate", "has_answer": True, "answer": "GOLD"}]}]}
    assert builder_history(example) == [{"id": "history", "messages": [{"role": "user", "content": "legitimate"}]}]


def test_inconsistent_longmemeval_metadata_is_rejected(locomo, lme):
    lme[0]["haystack_dates"] = []
    selection = {"dataset": "longmemeval", "id": lme[0]["question_id"], "exposure": "development",
                 "prior_exposure": "unknown", "exposure_note": "test"}
    with pytest.raises(ValueError, match="inconsistent history"):
        normalize_pilot(locomo, lme, [selection])


def test_standard_content_hash_has_no_filename_prefix(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"benchmark content")
    second.write_bytes(first.read_bytes())
    assert content_sha256(first) == content_sha256(second) == hashlib.sha256(b"benchmark content").hexdigest()


@pytest.mark.parametrize("prediction,example,expected", [
    ("the cats are running", {"category": 4, "answer": "cats run"}, 0.8),
    ("dogs, cats", {"category": 1, "answer": "cats, dogs"}, 1.0),
    ("cats", {"category": 1, "answer": "cats, dogs"}, 0.5),
    ("Psychology", {"category": 3, "answer": "Psychology; counseling"}, 1.0),
    ("2022", {"category": 2, "answer": 2022}, 1.0),
    ("", {"category": 4, "answer": "cats"}, 0.0),
])
def test_official_locomo_f1_stemming_subanswers_open_domain(prediction, example, expected):
    assert score_locomo(prediction, example) == pytest.approx(expected)


def test_locomo_adversarial_category_not_in_pilot():
    with pytest.raises(ValueError, match="categories 1"):
        score_locomo("no information available", {"category": 5, "answer": "unknown"})


def test_official_judge_prompt_refusal_and_question_types():
    example = {"id": "question_abs", "category": "knowledge-update", "question": "How old?", "answer": "Not provided"}
    prompt = get_lme_judge_prompt("I do not know", example)
    assert "unanswerable question" in prompt and "Explanation: Not provided" in prompt
    assert "Model Response: I do not know" in prompt
    example["id"] = "question"
    assert "previous information along with an updated answer" in get_lme_judge_prompt("42", example)
    example["category"] = "temporal-reasoning"
    assert "off-by-one errors" in get_lme_judge_prompt("42", example)
    example["category"] = "single-session-preference"
    assert "Rubric: Not provided" in get_lme_judge_prompt("42", example)


@pytest.mark.parametrize("response,result", [("yes", True), (" NO\n", False), ("Yes", True)])
def test_judge_valid_yes_no(response, result):
    assert parse_lme_judge(response) is result


@pytest.mark.parametrize("response,expected", [
    ("yes, but", True), ("no, yesterday", True), ("yesterday", True),
    ("yes. no.", True), ("Yes.", True), ("No.", False),
    ("No, the response is incorrect.", False), ("", False), (" \n", False),
])
def test_judge_matches_frozen_official_label_statement(response, expected):
    path = ROOT / "data" / "week1" / SOURCES["longmemeval_scorer"]["path"]
    assert content_sha256(path) == SOURCES["longmemeval_scorer"]["sha256"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "label" for target in node.targets)]
    assert len(assignments) == 1
    # Execute the author's actual assignment rather than another local implementation.
    namespace = {"eval_response": response.strip()}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["label"] is expected
    assert parse_lme_judge(response) is namespace["label"]
