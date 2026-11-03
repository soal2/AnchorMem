"""MemoryBank paper evaluation set as a week-one-style benchmark.

Loads the SiliconFriend eval histories and probing questions (zh+en) into the
same normalized example shape used by LoCoMo / LongMemEval pilots. Official
reference answers are not published; judge prompts fall back to dialogue
evidence while remaining binary-yes/no compatible with the existing LME judge
parser.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
DATASET = "memorybank"
COMMIT = "cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9"
REPO = "zhongwanjun/MemoryBank-SiliconFriend"
ARCHIVE_URL = f"https://api.github.com/repos/{REPO}/tarball/{COMMIT}"
LANGUAGES = ("en", "cn")
CATEGORY = "memory-probe"
TOKENIZER_NAME = "gpt-4o-mini"

SOURCES = {
    "en_memory_bank": {
        "path": "datasets/memorybank/en/memory_bank_en.json",
        "sha256": "27c302202f9675d87dcaa73e3a42dc0fca8999dcff3c6e7882a787c8a08d9bc7",
        "url": f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/eval_data/en/memory_bank_en.json",
    },
    "en_probing": {
        "path": "datasets/memorybank/en/probing_questions_en.jsonl",
        "sha256": "84538a41e5a9c8000850c19008a85445b01bf52c9960c9d735d21a4b73e5c648",
        "url": f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/eval_data/en/probing_questions_en.jsonl",
    },
    "cn_memory_bank": {
        "path": "datasets/memorybank/cn/memory_bank_cn.json",
        "sha256": "6098382314c7e5cf1d4564a3b587e9863b1e083f2ba56ce3fbbea68a1c91d287",
        "url": f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/eval_data/cn/memory_bank_cn.json",
    },
    "cn_probing": {
        "path": "datasets/memorybank/cn/probing_questions_cn.jsonl",
        "sha256": "10f26828cc14a98b6dca39223d7536b7b9af38f1fe68d7851070a64ec0339aa7",
        "url": f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/eval_data/cn/probing_questions_cn.jsonl",
    },
    "memorybank_scorer": {
        "path": "scorers/memorybank_evaluate_qa.py",
        "sha256": None,  # filled after first prepare of authored scorer
        "url": "local:data/week1/scorers/memorybank_evaluate_qa.py",
    },
}


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _download(path: Path, url: str) -> None:
    from urllib.request import urlopen
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=120) as source, temporary.open("wb") as destination:
        while block := source.read(1024 * 1024):
            destination.write(block)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _normalize_user_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def _index_users(bank: dict) -> dict[str, dict]:
    indexed = {}
    for raw_name, payload in bank.items():
        name = _normalize_user_name(raw_name)
        if not name:
            raise ValueError("MemoryBank user name is empty after normalization")
        if name in indexed:
            raise ValueError(f"Duplicate MemoryBank user after normalization: {name}")
        indexed[name] = payload
    return indexed


def _sessions_from_history(history: dict) -> tuple[list[dict], list[str]]:
    if not isinstance(history, dict) or not history:
        raise ValueError("MemoryBank history must be a non-empty date→turns mapping")
    sessions = []
    skipped_empty_dates = []
    for date in sorted(history):
        turns = history[date]
        if not isinstance(turns, list):
            raise ValueError(f"MemoryBank day {date} turns must be a list")
        if not turns:
            # Author file has a rare empty day (en/John Zhang/2023-05-05); keep calendar
            # fidelity in metadata but do not emit an empty builder session.
            skipped_empty_dates.append(date)
            continue
        messages = []
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError(f"Invalid turn on {date}")
            query = turn.get("query")
            response = turn.get("response")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Missing user query on {date}")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"Missing assistant response on {date}")
            # Keep only dialogue text; never inject summary/personality/meta fields.
            user_content = f"[Session date: {date}]\n{query.strip()}"
            assistant_content = f"[Session date: {date}]\n{response.strip()}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})
        sessions.append({"id": f"session_{len(sessions) + 1}", "date": date, "messages": messages})
    if not sessions:
        raise ValueError("MemoryBank history has no non-empty dialogue days")
    return sessions, skipped_empty_dates


def _evidence_text(sessions: list[dict]) -> str:
    blocks = []
    for session in sessions:
        date = session.get("date") or session["id"]
        lines = [f"### {date}"]
        for message in session["messages"]:
            role = "User" if message["role"] == "user" else "AI"
            # Strip the duplicated session-date prefix inside evidence blocks.
            content = message["content"]
            content = re.sub(r"^\[Session date: [^\]]+\]\n", "", content)
            lines.append(f"{role}: {content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def load_raw(root: Path | None = None) -> dict:
    """Load and verify the frozen MemoryBank eval files."""
    root = Path(root or ROOT)
    directory = root / "data" / "week1"
    verified = {}
    for name, spec in SOURCES.items():
        if name == "memorybank_scorer":
            continue
        path = directory / spec["path"]
        if not path.is_file():
            # Prefer the already-frozen MemoryBank author tree when present (no network).
            relative = Path(spec["path"]).relative_to("datasets/memorybank")
            mirror = directory / "memorybank" / "source" / "eval_data" / relative
            if mirror.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(mirror.read_bytes())
            else:
                _download(path, spec["url"])
        checksum = content_sha256(path)
        if checksum != spec["sha256"]:
            raise ValueError(f"Frozen MemoryBank file changed: {spec['path']}")
        verified[name] = {**spec, "bytes": path.stat().st_size}
    banks = {
        "en": _index_users(_read_json(directory / SOURCES["en_memory_bank"]["path"])),
        "cn": _index_users(_read_json(directory / SOURCES["cn_memory_bank"]["path"])),
    }
    probes = {
        "en": _read_jsonl(directory / SOURCES["en_probing"]["path"]),
        "cn": _read_jsonl(directory / SOURCES["cn_probing"]["path"]),
    }
    return {"sources": verified, "banks": banks, "probes": probes}


def select_full(banks: dict, probes: dict) -> list[dict]:
    """Deterministic full-set selector: all probing questions in language then file order."""
    selections = []
    for language in LANGUAGES:
        bank = banks[language]
        for row in probes[language]:
            if not isinstance(row, dict) or len(row) != 1:
                raise ValueError("Each probing JSONL row must map exactly one user to questions")
            (raw_name, questions), = row.items()
            user = _normalize_user_name(raw_name)
            if user not in bank:
                raise ValueError(f"Probing user {raw_name!r} missing from {language} memory bank")
            if not isinstance(questions, list) or not questions:
                raise ValueError(f"No probing questions for {language}:{user}")
            for index, question in enumerate(questions):
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Empty probing question for {language}:{user}#{index}")
                selections.append({
                    "dataset": DATASET,
                    "id": f"{language}:{user}#q{index:03d}",
                    "group": f"{language}:{user}",
                    "language": language,
                    "user": user,
                    "qa_index": index,
                    "category": CATEGORY,
                    "exposure": "development",
                    "prior_exposure": "unknown",
                    "exposure_note": (
                        "MemoryBank SiliconFriend eval set; no official answer file; "
                        "all items treated as development-exposed until a held-out policy is frozen."
                    ),
                })
    return selections


def normalize(banks: dict, probes: dict, selections: list[dict]) -> list[dict]:
    encoding = tiktoken.encoding_for_model(TOKENIZER_NAME)
    histories: dict[str, list[dict]] = {}
    evidence_cache: dict[str, str] = {}
    empty_days_by_group: dict[str, list[str]] = {}
    # Rebuild question lists keyed by normalized user.
    questions_by_user = {language: {} for language in LANGUAGES}
    for language in LANGUAGES:
        for row in probes[language]:
            (raw_name, questions), = row.items()
            questions_by_user[language][_normalize_user_name(raw_name)] = questions
    examples = []
    for selection in selections:
        language = selection["language"]
        user = selection["user"]
        group = selection["group"]
        if group not in histories:
            payload = banks[language][user]
            history = payload.get("history")
            if not isinstance(history, dict) or len(history) != 10:
                raise ValueError(f"{group} expected 10 calendar days in history, found {0 if not isinstance(history, dict) else len(history)}")
            sessions, skipped = _sessions_from_history(history)
            histories[group] = sessions
            evidence_cache[group] = _evidence_text(sessions)
            # Stash skipped empty dates on a side map for example metadata.
            empty_days_by_group[group] = skipped
        sessions = histories[group]
        question = questions_by_user[language][user][selection["qa_index"]]
        meta = banks[language][user].get("meta_information") or {}
        example = {
            "dataset": DATASET,
            "id": selection["id"],
            "group": group,
            "language": language,
            "user": user,
            "category": CATEGORY,
            "category_name": CATEGORY,
            "question": question.strip(),
            # Official answers are unpublished; keep the field for future gold files.
            "answer": None,
            "evidence": evidence_cache[group],
            "question_date": sessions[-1]["date"],
            "empty_dates": list(empty_days_by_group.get(group, [])),
            "sessions": sessions,
            "meta_information": {
                "name": _normalize_user_name(meta.get("name", user)),
                "personality": meta.get("personality"),
                "hobbies": meta.get("hobbies"),
                "speaking_style": meta.get("speaking_style"),
            },
            "exposure": selection["exposure"],
            "prior_exposure": selection["prior_exposure"],
            "exposure_note": selection["exposure_note"],
        }
        example["source_messages"] = sum(len(session["messages"]) for session in sessions)
        example["source_tokens"] = sum(
            len(encoding.encode(message["content"], disallowed_special=()))
            for session in sessions for message in session["messages"]
        )
        examples.append(example)
    return examples


def builder_history(example: dict) -> list[dict]:
    """Allowlisted history only; never pass answers, evidence, or personality into builders."""
    return [{
        "id": session["id"],
        "messages": [{"role": message["role"], "content": message["content"]}
                     for message in session["messages"]],
    } for session in example["sessions"]]


def prepare(root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    directory = root / "data" / "week1"
    raw = load_raw(root)
    banks, probes = raw["banks"], raw["probes"]
    for language in LANGUAGES:
        if len(banks[language]) != 15:
            raise ValueError(f"Expected 15 {language} users, found {len(banks[language])}")
    selections = select_full(banks, probes)
    selection_path = directory / "datasets/memorybank/selection-full.json"
    selection = {
        "seed": None,
        "selection_policy": "full published probing set; language then file order; no answer/performance selection",
        "development_exposure": True,
        "dataset": DATASET,
        "languages": list(LANGUAGES),
        "examples": selections,
    }
    if selection_path.is_file() and _read_json(selection_path) != selection:
        raise ValueError("Existing MemoryBank full selection differs; use a separate experiment directory")
    _save_json(selection_path, selection)
    examples = normalize(banks, probes, selections)
    scorer_path = directory / SOURCES["memorybank_scorer"]["path"]
    scorer_sha = content_sha256(scorer_path)
    sources = {**raw["sources"], "memorybank_scorer": {**SOURCES["memorybank_scorer"], "sha256": scorer_sha,
                                                       "bytes": scorer_path.stat().st_size}}
    history_groups = sorted({item["group"] for item in examples})
    tokens_by_history = {}
    messages_by_history = {}
    for item in examples:
        tokens_by_history[item["group"]] = item["source_tokens"]
        messages_by_history[item["group"]] = item["source_messages"]
    unique_history_tokens = sum(tokens_by_history.values())
    unique_history_messages = sum(messages_by_history.values())
    manifest = {
        "dataset": DATASET,
        "repo": REPO,
        "commit": COMMIT,
        "archive_url": ARCHIVE_URL,
        "paper": "arXiv:2305.10250",
        "data_hash_algorithm": "SHA256(file bytes)",
        "sources": sources,
        "counts": {
            "languages": list(LANGUAGES),
            "users_per_language": 15,
            "dialogue_days_per_user": 10,
            "histories": len(history_groups),
            "questions": len(examples),
            "questions_per_language": dict(Counter(item["language"] for item in examples)),
            "source_messages_all_histories": unique_history_messages,
            "source_tokens_all_histories": unique_history_tokens,
            "source_messages_by_history": messages_by_history,
            "source_tokens_by_history": tokens_by_history,
        },
        "selection_path": str(selection_path.relative_to(directory)),
        "selection_sha256": content_sha256(selection_path),
        "tokenizer": f"tiktoken.encoding_for_model({TOKENIZER_NAME!r})",
        "scoring": {
            "protocol": "memorybank-binary-v1",
            "compatible_with": "LongMemEval yes/no judge parse",
            "gold_answers": "unpublished; judge uses dialogue evidence unless answer is supplied",
            "scorer": SOURCES["memorybank_scorer"]["path"],
            "paper_human_metrics_not_replicated": [
                "Memory Retrieval Accuracy (0/1)",
                "Response Correctness (0/0.5/1)",
                "Contextual Coherence (0/0.5/1)",
                "Model Ranking Score",
            ],
        },
        "open_issues": [
            "No official reference answers (GitHub issue #22).",
            "Paper cites ~194 questions; frozen files contain 100 en + 100 cn = 200.",
            "English memory bank keys may include surrounding whitespace; loader strips names.",
            "zh/en are parallel corpora, not translations guaranteed to share answer keys.",
            "Binary LLM judge is a proxy for the paper's human 0/0.5/1 rubrics.",
        ],
    }
    _save_json(directory / "datasets/memorybank/prepared.json", manifest)
    return manifest


def load_full(root: Path | None = None) -> list[dict]:
    root = Path(root or ROOT)
    directory = root / "data" / "week1"
    manifest = _read_json(directory / "datasets/memorybank/prepared.json")
    for name, spec in manifest["sources"].items():
        if name == "memorybank_scorer":
            path = directory / spec["path"]
        else:
            path = directory / spec["path"]
        if content_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen MemoryBank {name} SHA256 mismatch")
    selection_path = directory / manifest["selection_path"]
    if content_sha256(selection_path) != manifest["selection_sha256"]:
        raise ValueError("Frozen MemoryBank selection SHA256 mismatch")
    raw = load_raw(root)
    return normalize(raw["banks"], raw["probes"], _read_json(selection_path)["examples"])


@lru_cache(maxsize=1)
def _scorer_functions() -> dict:
    path = ROOT / "data" / "week1" / SOURCES["memorybank_scorer"]["path"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_anscheck_prompt"]
    if len(selected) != 1:
        raise ValueError("memorybank_evaluate_qa.py must define get_anscheck_prompt")
    namespace: dict = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), path.name, "exec"), namespace)
    return namespace


def get_memorybank_judge_prompt(model_answer: str, example: dict) -> str:
    if example.get("category") not in {CATEGORY, "memorybank", "memory-retrieval"}:
        raise ValueError("Unknown MemoryBank category")
    scorer = _scorer_functions()
    return scorer["get_anscheck_prompt"](
        example.get("category", CATEGORY),
        example["question"],
        example.get("answer"),
        model_answer,
        abstention=False,
        evidence=example.get("evidence"),
    )


def parse_memorybank_judge(response: str) -> bool:
    """Same binary parse as LongMemEval's frozen label expression."""
    return "yes" in response.strip().lower()


def estimate_cost(examples: list[dict] | None = None, root: Path | None = None) -> dict:
    """Offline cost estimate from tiktoken source tokens and week-one cost-basis rates."""
    if examples is None:
        prepared = ROOT / "data/week1/datasets/memorybank/prepared.json"
        if not prepared.is_file():
            prepare(root)
        examples = load_full(root)
    # Unique histories: cost is paid once per group, then per-question QA.
    by_group = {}
    for item in examples:
        by_group[item["group"]] = item
    source_tokens = sum(item["source_tokens"] for item in by_group.values())
    source_messages = sum(item["source_messages"] for item in by_group.values())
    n_histories = len(by_group)
    n_questions = len(examples)
    # Observed week-one rates from cost-basis-brief (LoCoMo short-dialogue anchor preferred
    # for MemoryBank's short daily chats; LME long-history rates listed for sensitivity).
    rates = {
        "vikingjev_build_per_source_token_locomo": 1.780167154689487e-05,
        "vikingjev_build_per_source_token_lme": 8.21159677861141e-06,
        "vikingjev_build_per_source_token_fallback": 0.0000181,
        "vikingjev_qa_per_question_locomo": 0.000210421,
        "vikingjev_qa_per_question_fallback": 0.0002,
        "mem0_build_per_source_token_locomo": 1.2125100525827405e-06,
        "mem0_qa_per_question_locomo": 0.000141324,
        "memorybank_method_build_per_source_token_locomo": 7.743042238031412e-07,
        "memorybank_method_qa_per_question_locomo": 0.000116859,
        # Evidence-backed judge is larger than LME's short prompt; use LME QA judge
        # sample (~$0.00043) as a lower bound and a conservative $0.001 for full evidence.
        "judge_per_question_lme_sample": 0.00043,
        "judge_per_question_evidence_conservative": 0.001,
    }
    vj_build = source_tokens * rates["vikingjev_build_per_source_token_locomo"]
    vj_build_fallback = source_tokens * rates["vikingjev_build_per_source_token_fallback"]
    vj_qa = n_questions * rates["vikingjev_qa_per_question_locomo"]
    vj_judge = n_questions * rates["judge_per_question_evidence_conservative"]
    mem0_build = source_tokens * rates["mem0_build_per_source_token_locomo"]
    mem0_qa = n_questions * rates["mem0_qa_per_question_locomo"]
    mb_build = source_tokens * rates["memorybank_method_build_per_source_token_locomo"]
    mb_qa = n_questions * rates["memorybank_method_qa_per_question_locomo"]
    estimate = {
        "currency": "USD",
        "scope": "offline estimate for full MemoryBank probing set; no experiment authorization",
        "histories": n_histories,
        "questions": n_questions,
        "source_messages": source_messages,
        "source_tokens": source_tokens,
        "tokenizer": TOKENIZER_NAME,
        "rates": rates,
        "methods": {
            "vikingjev": {
                "build_usd_locomo_rate": round(vj_build, 6),
                "build_usd_fallback_0.0000181": round(vj_build_fallback, 6),
                "qa_usd": round(vj_qa, 6),
                "judge_usd_conservative": round(vj_judge, 6),
                "total_usd_locomo_rate_plus_qa_judge": round(vj_build + vj_qa + vj_judge, 6),
            },
            "mem0": {
                "build_usd_locomo_rate": round(mem0_build, 6),
                "qa_usd": round(mem0_qa, 6),
                "judge_usd_conservative": round(vj_judge, 6),
                "total_usd_plus_judge": round(mem0_build + mem0_qa + vj_judge, 6),
            },
            "memorybank_method": {
                "build_usd_locomo_rate": round(mb_build, 6),
                "qa_usd": round(mb_qa, 6),
                "judge_usd_conservative": round(vj_judge, 6),
                "total_usd_plus_judge": round(mb_build + mb_qa + vj_judge, 6),
            },
            "rag": {
                "note": "RAG excluded from primary week-one cost basis comparison",
            },
        },
        "notes": [
            "Build cost scales with unique histories (30 = 15 users × 2 languages), not per question.",
            "LoCoMo short-dialogue rates are the primary anchor; MemoryBank days are short chats.",
            "Judge cost assumes evidence-backed prompts (full user dialogue); gold-answer prompts would be cheaper.",
            "Does not change the existing $20 budget cap; estimate only.",
        ],
    }
    return estimate
