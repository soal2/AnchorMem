"""Frozen native benchmark inputs and official metrics for the week-one pilot."""

import ast
import hashlib
import json
import random
import re
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
LOCOMO_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LME_DATA_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LME_EVALUATOR_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
LOCOMO_CATEGORIES = {4: "single-hop", 1: "multi-hop", 2: "temporal", 3: "open-domain"}
LME_CATEGORIES = (
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "temporal-reasoning", "knowledge-update",
)
SOURCES = {
    "locomo": {
        "path": "datasets/locomo10.json", "revision": LOCOMO_REVISION,
        "url": f"https://raw.githubusercontent.com/snap-research/LoCoMo/{LOCOMO_REVISION}/data/locomo10.json",
        "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    },
    "longmemeval": {
        "path": "datasets/longmemeval_s_cleaned.json", "revision": LME_DATA_REVISION,
        "url": f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{LME_DATA_REVISION}/longmemeval_s_cleaned.json",
        "download_url": f"https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/{LME_DATA_REVISION}/longmemeval_s_cleaned.json",
        "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    },
    "locomo_scorer": {
        "path": "scorers/locomo_evaluation.py", "revision": LOCOMO_REVISION,
        "url": f"https://raw.githubusercontent.com/snap-research/LoCoMo/{LOCOMO_REVISION}/task_eval/evaluation.py",
        "sha256": "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd",
    },
    "longmemeval_scorer": {
        "path": "scorers/longmemeval_evaluate_qa.py", "revision": LME_EVALUATOR_REVISION,
        "url": f"https://raw.githubusercontent.com/xiaowu0162/LongMemEval/{LME_EVALUATOR_REVISION}/src/evaluation/evaluate_qa.py",
        "sha256": "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
    },
}


def content_sha256(path: Path) -> str:
    """Standard SHA256 of file bytes, without a filename or directory prefix."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _download(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=120) as source, temporary.open("wb") as destination:
        while block := source.read(1024 * 1024):
            destination.write(block)
    temporary.replace(path)


def select_pilot(locomo: list[dict], longmemeval: list[dict], seed: int = 42) -> list[dict]:
    """Select by type and source order, never by gold answer or method results."""
    rng = random.Random(seed)
    selections = []
    conversations = {item["sample_id"]: item for item in locomo}
    for group in ("conv-41", "conv-26"):
        qas = conversations[group]["qa"]
        indices = []
        for category in (1, 2, 3, 4):
            candidates = [i for i, qa in enumerate(qas) if qa["category"] == category]
            if len(candidates) < 2:
                raise ValueError(f"LoCoMo {group} category {category} has fewer than two questions")
            indices.extend(rng.sample(candidates, 2))
        remaining = [i for i, qa in enumerate(qas) if qa["category"] in LOCOMO_CATEGORIES and i not in indices]
        indices.extend(rng.sample(remaining, 2))
        for index in indices:
            selections.append({"dataset": "locomo", "id": f"{group}#q{index:04d}",
                               "group": group, "qa_index": index, "category": qas[index]["category"]})
    for category in LME_CATEGORIES:
        candidates = [item for item in longmemeval if item["question_type"] == category]
        answerable = [item for item in candidates if "_abs" not in item["question_id"]]
        if category in ("knowledge-update", "multi-session"):
            abstentions = [item for item in candidates if "_abs" in item["question_id"]]
            selected = rng.sample(answerable, 1) + rng.sample(abstentions, 1)
        else:
            selected = rng.sample(answerable, 2)
        for item in selected:
            selections.append({"dataset": "longmemeval", "id": item["question_id"], "category": category})
    # The runner can verify both end-to-end chains before extending the pilot.
    first_lme = next(i for i, selection in enumerate(selections) if selection["dataset"] == "longmemeval")
    ordered = [selections[0], selections[first_lme]] + selections[1:first_lme] + selections[first_lme + 1:]
    for item in ordered:
        item["exposure"] = "development"
        item["prior_exposure"] = "known-history" if item.get("group") == "conv-41" else "unknown"
        item["exposure_note"] = (
            "conv-41 history was used in the existing LoCoMo-Refined smoke; exact original QA exposure is unknown."
            if item.get("group") == "conv-41" else
            "No complete prior exposure registry is available; selection and all pilot results are development-exposed."
        )
    return ordered


def _locomo_sessions(conversation: dict) -> list[dict]:
    sessions = []
    keys = sorted((key for key, value in conversation.items()
                   if re.fullmatch(r"session_\d+", key) and isinstance(value, list)),
                  key=lambda key: int(key.split("_")[1]))
    for key in keys:
        date = conversation.get(f"{key}_date_time")
        messages = []
        for turn in conversation[key]:
            # Explicit allowlist: QA, event/session summaries, image query terms
            # and evidence annotations never enter the memory input.
            content = f"{turn['speaker']}: {turn.get('text', '')}"
            if turn.get("blip_caption"):
                content += f"\n[Image description: {turn['blip_caption']}]"
            if date:
                content = f"[Session date: {date}]\n{content}"
            messages.append({"role": "user", "content": content})
        sessions.append({"id": key, "messages": messages})
    return sessions


def _lme_sessions(item: dict) -> list[dict]:
    histories = item["haystack_sessions"]
    ids, dates = item["haystack_session_ids"], item["haystack_dates"]
    if not (len(histories) == len(ids) == len(dates)):
        raise ValueError(f"LongMemEval {item['question_id']} has inconsistent history metadata")
    sessions = []
    for index, (date, turns) in enumerate(zip(dates, histories, strict=True), start=1):
        messages = []
        for turn in turns:
            if turn["role"] not in ("user", "assistant") or not isinstance(turn["content"], str):
                raise ValueError(f"Invalid history message in LongMemEval {item['question_id']}")
            content = turn["content"]
            if date:
                content = f"[Session date: {date}]\n{content}"
            messages.append({"role": turn["role"], "content": content})
        # Source IDs include gold-derived "answer_" prefixes. Use ordinal
        # boundaries, so builder metadata cannot reveal an answer session.
        sessions.append({"id": f"session_{index}", "messages": messages})
    return sessions


def normalize_pilot(locomo: list[dict], longmemeval: list[dict], selections: list[dict]) -> list[dict]:
    locomo_by_id = {item["sample_id"]: item for item in locomo}
    lme_by_id = {item["question_id"]: item for item in longmemeval}
    histories = {}
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    examples = []
    for selection in selections:
        dataset = selection["dataset"]
        if dataset == "locomo":
            item = locomo_by_id[selection["group"]]
            qa = item["qa"][selection["qa_index"]]
            group = item["sample_id"]
            if group not in histories:
                histories[group] = _locomo_sessions(item["conversation"])
            sessions = histories[group]
            example = {"category": qa["category"], "category_name": LOCOMO_CATEGORIES[qa["category"]],
                       "question": qa["question"], "answer": str(qa["answer"])}
        else:
            item = lme_by_id[selection["id"]]
            sessions = _lme_sessions(item)
            serialized = json.dumps(sessions, ensure_ascii=False, sort_keys=True).encode()
            group = hashlib.sha256(serialized).hexdigest()[:20]
            example = {"category": item["question_type"], "question": item["question"],
                       "question_date": item["question_date"], "answer": str(item["answer"]),
                       "abstention": "_abs" in item["question_id"]}
        example.update({"dataset": dataset, "id": selection["id"], "group": group,
                        "sessions": sessions, "exposure": selection["exposure"],
                        "prior_exposure": selection["prior_exposure"], "exposure_note": selection["exposure_note"]})
        example["source_messages"] = sum(len(session["messages"]) for session in sessions)
        example["source_tokens"] = sum(len(encoding.encode(message["content"], disallowed_special=()))
                                       for session in sessions for message in session["messages"])
        examples.append(example)
    return examples


def builder_history(example: dict) -> list[dict]:
    """The only field crossing into a memory builder is allowlisted history."""
    return [{"id": session["id"], "messages": [{"role": message["role"], "content": message["content"]}
             for message in session["messages"]]} for session in example["sessions"]]


def prepare(root: Path) -> dict:
    directory = Path(root) / "data" / "week1"
    directory.mkdir(parents=True, exist_ok=True)
    previous = _read_json(directory / "prepared.json") if (directory / "prepared.json").is_file() else {}
    sources = {}
    for name, spec in SOURCES.items():
        path = directory / spec["path"]
        if not path.is_file():
            _download(path, spec.get("download_url", spec["url"]))
        checksum = content_sha256(path)
        expected = spec.get("sha256") or previous.get("sources", {}).get(name, {}).get("sha256")
        if expected and checksum != expected:
            raise ValueError(f"Frozen {name} SHA256 mismatch")
        sources[name] = {**spec, "sha256": checksum, "bytes": path.stat().st_size}
    locomo = _read_json(directory / SOURCES["locomo"]["path"])
    lme = _read_json(directory / SOURCES["longmemeval"]["path"])
    locomo_counts = Counter(str(qa["category"]) for item in locomo for qa in item["qa"])
    lme_counts = Counter(item["question_type"] for item in lme)
    if len(locomo) != 10 or sum(locomo_counts[str(k)] for k in LOCOMO_CATEGORIES) != 1540:
        raise ValueError("Unexpected native LoCoMo population")
    if len(lme) != 500 or sum("_abs" in item["question_id"] for item in lme) != 30 or set(lme_counts) != set(LME_CATEGORIES):
        raise ValueError("Unexpected LongMemEval-S population")
    selections = select_pilot(locomo, lme)
    selection_path = directory / "selection.json"
    selection = {"seed": 42, "selection_policy": "type stratification and seeded source-index sampling; no answer/performance selection",
                 "development_exposure": True, "examples": selections}
    if selection_path.is_file() and _read_json(selection_path) != selection:
        raise ValueError("Existing frozen pilot selection differs; use a separate experiment directory")
    _save_json(selection_path, selection)
    pilot = normalize_pilot(locomo, lme, selections)
    manifest = {"sources": sources, "data_hash_algorithm": "SHA256(file bytes)",
                "counts": {"locomo_conversations": len(locomo), "locomo_categories": dict(sorted(locomo_counts.items())),
                           "locomo_answerable": 1540, "longmemeval_questions": len(lme),
                           "longmemeval_categories": dict(sorted(lme_counts.items())), "longmemeval_abstentions": 30},
                "pilot": {"examples": len(pilot), "dataset_counts": dict(Counter(item["dataset"] for item in pilot)),
                          "histories": len({(item["dataset"], item["group"]) for item in pilot}),
                          "source_messages_by_history": {f"{item['dataset']}:{item['group']}": item["source_messages"] for item in pilot},
                          "source_tokens_by_history": {f"{item['dataset']}:{item['group']}": item["source_tokens"] for item in pilot}},
                "selection_sha256": content_sha256(selection_path),
                "tokenizer": "tiktoken.encoding_for_model('gpt-4o-mini')",
                "scoring": {"locomo": "Official AST functions normalize_answer, f1_score, f1; PorterStemmer; category 3 semicolon prefix",
                            "longmemeval": "Official get_anscheck_prompt and label expression; yes substring in stripped lowercase judge response"}}
    _save_json(directory / "prepared.json", manifest)
    return manifest


def load_pilot(root: Path) -> list[dict]:
    directory = Path(root) / "data" / "week1"
    manifest = _read_json(directory / "prepared.json")
    for name, spec in manifest["sources"].items():
        if content_sha256(directory / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Frozen {name} SHA256 mismatch")
    selection_path = directory / "selection.json"
    if content_sha256(selection_path) != manifest["selection_sha256"]:
        raise ValueError("Frozen pilot selection SHA256 mismatch")
    return normalize_pilot(_read_json(directory / SOURCES["locomo"]["path"]),
                           _read_json(directory / SOURCES["longmemeval"]["path"]),
                           _read_json(selection_path)["examples"])


@lru_cache(maxsize=2)
def _official_functions(filename: str, names: tuple[str, ...]) -> dict:
    """Compile only named pure official functions; never import heavy scorer modules."""
    source = (ROOT / "data" / "week1" / "scorers" / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Required official functions missing in {filename}")
    import regex
    from nltk.stem import PorterStemmer
    namespace = {"regex": regex, "string": string, "Counter": Counter, "np": np, "ps": PorterStemmer()}
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def score_locomo(answer: str, example: dict) -> float:
    category = int(example["category"])
    if category not in LOCOMO_CATEGORIES:
        raise ValueError("Week-one LoCoMo F1 includes only categories 1–4")
    gold = str(example["answer"])
    if category == 3:
        gold = gold.split(";")[0].strip()
    scorer = _official_functions("locomo_evaluation.py", ("normalize_answer", "f1_score", "f1"))
    return float(scorer["f1" if category == 1 else "f1_score"](answer, gold))


def get_lme_judge_prompt(answer: str, example: dict) -> str:
    if example["category"] not in LME_CATEGORIES:
        raise ValueError("Unknown LongMemEval question type")
    scorer = _official_functions("longmemeval_evaluate_qa.py", ("get_anscheck_prompt",))
    return scorer["get_anscheck_prompt"](example["category"], example["question"], str(example["answer"]), answer,
                                          abstention="_abs" in example["id"])


def parse_lme_judge(response: str) -> bool:
    """Execute the frozen author's label expression (evaluate_qa.py lines 112–113)."""
    path = ROOT / "data" / "week1" / SOURCES["longmemeval_scorer"]["path"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "label" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Required official LongMemEval label assignment missing or ambiguous")
    expression = ast.Expression(body=assignments[0].value)
    return eval(compile(expression, str(path), "eval"), {"eval_response": response.strip()})
