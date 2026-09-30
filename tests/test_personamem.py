import csv
import json

import pytest

from src import evaluation
from src.config import EvaluationConfig, load_config
from src.contracts import Evidence, SearchResponse


@pytest.fixture
def persona_dataset(tmp_path):
    history_path = tmp_path / "data/chat_history_32k/persona1.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(json.dumps({"metadata": {"persona_id": 1}, "chat_history": [
        {"role": "user", "content": "I live in Paris."},
        {"role": "assistant", "content": "I will remember that."},
    ]}))
    rows = [{"persona_id": "1", "chat_history_32k_link": "data/chat_history_32k/persona1.json",
             "user_query": repr({"role": "user", "content": f"Which city? {index}"}),
             "short_persona": "This metadata must not enter memory.", "topic_preference": "city",
             "correct_answer": "Paris", "incorrect_answers": json.dumps(["London", "Berlin", "Rome"])}
            for index in range(2)]
    with (tmp_path / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path, history_path


def test_personamem_loads_official_history_and_four_shuffled_choices(persona_dataset):
    root, history_path = persona_dataset
    examples = evaluation.load_examples("personamem", root, 42)
    repeated = evaluation.load_examples("personamem", root, 42)
    assert examples == repeated
    assert examples[0]["sessions"] == examples[1]["sessions"]
    assert examples[0]["sessions"][0]["id"] != examples[0]["id"]
    assert examples[0]["sessions"][0]["messages"] == json.loads(history_path.read_text())["chat_history"]
    for example in examples:
        assert sorted(example["options"]) == ["Berlin", "London", "Paris", "Rome"]
        assert example["options"]["ABCD".index(example["gold_choice"])] == "Paris"
        assert evaluation.score_personamem(json.dumps({"choice": example["gold_choice"]}), example["gold_choice"])
    orders = {tuple(evaluation.load_examples("personamem", root, seed)[0]["options"]) for seed in range(8)}
    assert len(orders) > 1


def test_personamem_history_content_changes_session_identity(persona_dataset):
    root, history_path = persona_dataset
    before = evaluation.load_examples("personamem", root, 42)[0]["sessions"][0]["id"]
    history = json.loads(history_path.read_text())
    history["chat_history"][0]["content"] = "I moved to Berlin."
    history_path.write_text(json.dumps(history))
    after = evaluation.load_examples("personamem", root, 42)[0]["sessions"][0]["id"]
    assert before != after


def test_personamem_history_extra_fields_do_not_enter_memory(persona_dataset):
    root, history_path = persona_dataset
    history = json.loads(history_path.read_text())
    history["chat_history"][0]["additional_field"] = "This is not message text."
    history_path.write_text(json.dumps(history))
    messages = evaluation.load_examples("personamem", root, 42)[0]["sessions"][0]["messages"]
    assert messages == [{"role": "user", "content": "I live in Paris."},
                        {"role": "assistant", "content": "I will remember that."}]


def test_personamem_skips_empty_assistant_without_replacement(persona_dataset):
    root, history_path = persona_dataset
    history = json.loads(history_path.read_text())
    original = history["chat_history"].copy()
    history["chat_history"].append({"role": "assistant", "content": ""})
    content = json.dumps(history)
    history_path.write_text(content)
    messages = evaluation.load_examples("personamem", root, 42)[0]["sessions"][0]["messages"]
    assert messages == original
    assert history_path.read_text() == content


def test_personamem_missing_history_is_not_replaced(persona_dataset):
    root, history_path = persona_dataset
    history_path.unlink()
    with pytest.raises(ValueError, match="history"):
        evaluation.load_examples("personamem", root, 42)


@pytest.mark.parametrize("history", [
    {"metadata": {"persona_id": 2}, "chat_history": [{"role": "user", "content": "Paris"}]},
    {"metadata": {"persona_id": 1}, "chat_history": []},
    {"metadata": {"persona_id": 1}, "chat_history": [{"role": "assistant", "content": ""}]},
    {"metadata": {"persona_id": 1}, "messages": [{"role": "user", "content": "Paris"}]},
    {"metadata": {"persona_id": 1}, "chat_history": [{"role": "invalid", "content": "Paris"}]},
    {"metadata": {"persona_id": 1}, "chat_history": [{"role": "user", "content": " "}]},
    {"metadata": {"persona_id": 1}, "chat_history": [{"role": "user", "content": ["Paris"]}]},
])
def test_personamem_rejects_invalid_history(persona_dataset, history):
    root, history_path = persona_dataset
    history_path.write_text(json.dumps(history))
    with pytest.raises(ValueError):
        evaluation.load_examples("personamem", root, 42)


@pytest.mark.parametrize("incorrect", ["not JSON", "[]", '["London"]', '["Paris", "Berlin", "Rome"]',
                                         '["London", "London", "Rome"]', '["London", "Berlin", 1]'])
def test_personamem_rejects_invalid_options(persona_dataset, incorrect):
    root, _ = persona_dataset
    csv_path = root / "benchmark.csv"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["incorrect_answers"] = incorrect
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        evaluation.load_examples("personamem", root, 42)


@pytest.mark.parametrize("answer, correct", [
    ('{"choice":"B"}', True), ('{"choice":"A"}', False),
    ('{"choice":"UNKNOWN"}', False), ('{"choice":["A","B"]}', False),
    ('{"choice":"A or B"}', False), ('{"choice":"E"}', False),
    ('{"choice":"b"}', False), ('{"choice":1}', False),
    ('{"choice":"B","other":"A"}', False), ('{"choice":"A","choice":"B"}', False),
    ('{"choice":"B"} because Paris', False), ('[]', False), ('null', False),
    ('Paris', False), ('', False),
])
def test_personamem_only_correct_single_choice_passes(answer, correct):
    assert evaluation.score_personamem(answer, "B") is correct


def test_personamem_wrong_city_regression():
    # 即使标准答案本来就在四个选项中，选错仍必须失败。
    assert not evaluation.score_personamem('{"choice":"B"}', "A")


@pytest.mark.parametrize("selected_text, expected_label", [("Paris", "CORRECT"), ("London", "WRONG")])
def test_personamem_evaluation_choices_history_reuse_and_isolated_runs(persona_dataset, monkeypatch, selected_text, expected_label):
    root, _ = persona_dataset
    (root / "source.json").write_text(json.dumps({"repository": "test-personamem", "revision": "fixed"}))
    config, config_path = load_config("configs/evaluation/personamem-dev.yaml", EvaluationConfig)
    config = config.model_copy(update={"data_path": str(root), "data_sha256": evaluation.sha256(root),
                                       "output_dir": str(root.parent / (root.name + "-runs")), "sample_limit": None,
                                       "answer": config.answer.model_copy(update={"name": "test-answer"}),
                                       "judge": config.judge.model_copy(update={"name": "must-not-create-judge"})})
    monkeypatch.setattr(evaluation, "_commit", lambda path: config.contract_commit)
    requests, searches, model_prompts, databases = [], [], [], []

    class FakeLLM:
        def __init__(self, model_config):
            assert model_config.name != "must-not-create-judge"

        def complete(self, system, prompt, json_response=True):
            assert json_response is True
            body = json.loads(prompt)
            assert set(body) == {"memories", "question", "options"}
            assert list(body["options"]) == list("ABCD")
            model_prompts.append(body)
            choice = next(key for key, text in body["options"].items() if text == selected_text)
            return " \n" + json.dumps({"choice": choice}) + "\n "

    class FakeEngine:
        def __init__(self, service, *args):
            databases.append(service.database_dir)

        def add(self, request):
            requests.append(request)

        def search(self, request):
            searches.append(request)
            return SearchResponse(data=[Evidence(id="e1", content="I live in Paris.", score=1.0)])

    monkeypatch.setattr(evaluation, "OpenAILLM", FakeLLM)
    monkeypatch.setattr(evaluation, "OpenAIEmbedder", lambda config: object())
    monkeypatch.setattr(evaluation, "MemoryEngine", FakeEngine)
    first_run = evaluation.evaluate(config, config_path)
    second_run = evaluation.evaluate(config, config_path)
    assert first_run != second_run and databases[0] != databases[1]
    assert all("memory-index" in path for path in databases)
    assert requests[0].request_id == requests[1].request_id
    assert requests[0].messages == requests[1].messages
    assert "metadata" not in requests[0].model_dump_json()
    predictions = [json.loads(line) for line in (first_run / "predictions.jsonl").read_text().splitlines()]
    for prediction, search, prompt in zip(predictions, searches, model_prompts):
        assert search.options == prediction["options"] == list(prompt["options"].values())
        assert prediction["label"] == expected_label
        assert prediction["selected_choice"] in "ABCD"
        assert prediction["answer"] == " \n" + json.dumps({"choice": prediction["selected_choice"]}) + "\n "
        assert prediction["history_path"] and prediction["history_sha256"]
    manifest = json.loads((first_run / "manifest.json").read_text())
    assert manifest["scoring_protocol"] == "personamem-choice-v1"
    assert manifest["data_source"]["provenance"]["revision"] == "fixed"
    metrics = json.loads((first_run / "metrics.json").read_text())
    assert metrics["requested"] == metrics["evaluated"] == 2
    assert metrics["completion_rate"] == 1
    assert metrics["correct"] == (2 if expected_label == "CORRECT" else 0)
