import ast
import hashlib
import json
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import EvaluationConfig, ServiceConfig, config_hash, load_config, resolved_path, write_snapshot
from .contracts import AddRequest, Message, SearchRequest
from .engine import MemoryEngine
from .providers import OpenAIEmbedder, OpenAILLM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
    else:
        files = [path]
    for file in files:
        digest.update(str(file.relative_to(path) if path.is_dir() else file.name).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_examples(dataset: str, path: Path) -> list[dict]:
    """Return explicit histories and questions; no benchmark answers enter retrieval."""
    examples = []
    if dataset == "locomo":
        conversations = {item["sample_id"]: item for item in _jsonl(path / "conversations.jsonl")}
        for question in _jsonl(path / "questions.jsonl"):
            conversation = conversations[question["sample_id"]]
            sessions = []
            for session in conversation["sessions"]:
                messages = []
                for turn in session["messages"]:
                    content = turn.get("text") or ""
                    if not content.strip():
                        continue
                    speaker = turn.get("speaker", "")
                    messages.append({"role": turn.get("role", "user"), "content": f"{speaker}: {content}"})
                sessions.append({"id": str(session["session_index"]), "messages": messages})
            examples.append({"id": question["qa_id"], "group": question["sample_id"], "sessions": sessions,
                             "question": question["question"], "answers": question["answer"], "category": question.get("category", "unknown"),
                             "speaker_a": question.get("speaker_a", "speaker 1"), "speaker_b": question.get("speaker_b", "speaker 2")})
    elif dataset == "longmemeval":
        records = json.loads(path.read_text(encoding="utf-8"))
        for item in records:
            session_ids = item.get("haystack_session_ids", list(range(len(item["haystack_sessions"]))))
            sessions = []
            for index, turns in enumerate(item["haystack_sessions"]):
                messages = [{"role": turn["role"], "content": turn["content"]} for turn in turns if isinstance(turn.get("content"), str) and turn["content"].strip()]
                sessions.append({"id": str(session_ids[index]), "messages": messages})
            history_hash = hashlib.sha256(json.dumps(sessions, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
            examples.append({"id": item["question_id"], "group": history_hash, "sessions": sessions,
                             "question": item["question"], "answers": [item["answer"]], "category": item.get("question_type", "unknown")})
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return examples


def split_examples(examples: list[dict], split: str, seed: int, fraction: float, limit: int | None) -> list[dict]:
    groups = sorted({item["group"] for item in examples})
    random.Random(seed).shuffle(groups)
    border = 8 if len(groups) == 10 else max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    included = set(groups[:border] if split == "dev" else groups[border:])
    selected = [item for item in examples if item["group"] in included]
    return selected[:limit] if limit is not None else selected


def _commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _contract_prompts(path: Path) -> tuple[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"OPEN_ENDED_ANSWER_TEMPLATE", "ACCURACY_PROMPT"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values["OPEN_ENDED_ANSWER_TEMPLATE"], values["ACCURACY_PROMPT"]


def _answer_prompt(template: str, example: dict, evidence: str) -> str:
    values = {"speaker_1_name": example.get("speaker_a", "speaker 1"), "speaker_2_name": example.get("speaker_b", "speaker 2"),
              "speaker_1_memories": evidence, "speaker_2_memories": "", "question": example["question"]}
    return re.sub(r"\{\{(speaker_1_name|speaker_2_name|speaker_1_memories|speaker_2_memories|question)\}\}",
                  lambda match: values[match.group(1)], template)


def _judge_prompt(template: str, example: dict, gold: str, answer: str) -> str:
    values = {"question": example["question"], "gold_answer": gold, "generated_answer": answer}
    return re.sub(r"\{(question|gold_answer|generated_answer)\}", lambda match: values[match.group(1)], template)


def _append(path: Path, item: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluate(config: EvaluationConfig, config_path: Path):
    service_path = resolved_path(config_path, config.service_config_path)
    service, service_path = load_config(service_path, ServiceConfig)
    dataset_path = resolved_path(config_path, config.data_path)
    actual_data_hash = sha256(dataset_path)
    if actual_data_hash != config.data_sha256:
        raise ValueError(f"dataset SHA-256 mismatch: found {actual_data_hash}; set data_sha256 to the frozen dataset hash")
    contract_path = resolved_path(config_path, config.contract_repository_path)
    if _commit(contract_path) != config.contract_commit:
        raise ValueError("AML contract repository commit mismatch")
    examples = split_examples(load_examples(config.dataset, dataset_path), config.split, config.seed, config.dev_fraction, config.sample_limit)
    if not examples:
        raise ValueError("selected evaluation split is empty")
    run_dir = resolved_path(config_path, config.output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + config.run_label)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_snapshot(service, run_dir / "resolved-service-config.yaml")
    write_snapshot(config, run_dir / "resolved-eval-config.yaml")
    pipeline_name = "locomo-refined" if config.dataset == "locomo" else "longmemeval-s"
    answer_prompt, judge_prompt = _contract_prompts(contract_path / "data" / pipeline_name / "pipeline.py")
    manifest = {"label": "local-development", "run_label": config.run_label, "created_at": datetime.now(timezone.utc).isoformat(),
                "code_commit": _commit(Path(__file__).resolve().parents[2]), "contract_commit": config.contract_commit,
                "service_config_hash": config_hash(service), "eval_config_hash": config_hash(config), "data_sha256": actual_data_hash,
                "answer_prompt_sha256": hashlib.sha256(answer_prompt.encode()).hexdigest(),
                "judge_prompt_sha256": hashlib.sha256(judge_prompt.encode()).hexdigest(),
                "models": {"extract": service.llm.name, "embedding": service.embedding.name, "answer": config.answer.name, "judge": config.judge.name},
                "dataset": config.dataset, "split": config.split, "question_count": len(examples)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for filename in ("predictions.jsonl", "retrieval.jsonl", "failures.jsonl"):
        (run_dir / filename).touch()
    engine = MemoryEngine(service, service_path, OpenAILLM(service.llm), OpenAIEmbedder(service.embedding))
    answer_llm, judge_llm = OpenAILLM(config.answer), OpenAILLM(config.judge)
    counts = defaultdict(lambda: [0, 0])
    failures = defaultdict(int)
    for example in examples:
        user_id = f"local:{config.dataset}:{example['group']}"
        stage = "request"
        try:
            for session in example["sessions"]:
                for offset in range(0, len(session["messages"]), 20):
                    messages = [Message.model_validate(turn) for turn in session["messages"][offset:offset + 20]]
                    if messages:
                        engine.add(AddRequest(request_id=f"{user_id}:{session['id']}:{offset}", user_id=user_id,
                                              session_id=f"{user_id}:{session['id']}", messages=messages))
            retrieved = engine.search(SearchRequest(query=example["question"], user_id=user_id, top_k=100))
            _append(run_dir / "retrieval.jsonl", {"id": example["id"], "data": [item.model_dump() for item in retrieved.data]})
            evidence = "\n".join(item.content for item in retrieved.data)
            stage = "answer"
            answer = answer_llm.complete("You answer questions from retrieved memories.", _answer_prompt(answer_prompt, example, evidence), json_response=False).strip()
            stage = "judge"
            labels = []
            for gold in example["answers"]:
                verdict_raw = judge_llm.complete("You evaluate answers.", _judge_prompt(judge_prompt, example, str(gold), answer))
                verdict = json.loads(verdict_raw)
                if verdict.get("label") not in {"CORRECT", "WRONG"}:
                    raise ValueError("judge returned no valid label")
                labels.append(verdict["label"])
                if verdict["label"] == "CORRECT":
                    break
            group = example["category"]
            _append(run_dir / "predictions.jsonl", {"id": example["id"], "answer": answer, "label": "CORRECT" if "CORRECT" in labels else "WRONG", "category": group})
            counts[group][0] += 1
            counts[group][1] += int("CORRECT" in labels)
        except Exception as error:
            category = f"{stage}_failure"
            failures[category] += 1
            _append(run_dir / "failures.jsonl", {"id": example["id"], "category": category, "error": str(error)})
    total = sum(value[0] for value in counts.values())
    correct = sum(value[1] for value in counts.values())
    metrics = {"label": "local-development", "dataset": config.dataset, "evaluated": total, "correct": correct,
               "accuracy": correct / total if total else None, "categories": {key: {"count": n, "correct": c, "accuracy": c / n} for key, (n, c) in counts.items()}, "failures": dict(failures)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return run_dir
