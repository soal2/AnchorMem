"""Validate explicitly selected baseline artifacts before importing a fresh pilot."""

import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import tiktoken

from .week1_data import parse_lme_judge, score_locomo


BASELINES = ("rag", "mem0", "memorybank")


def history_sha256(example):
    from .week1_tasks import history_digest

    return history_digest(example)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_hashes(path):
    """Hash durable index files; logs and OS lock files have no scientific state."""
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("inherited index cannot contain symlinks")
        if item.is_file() and item.name not in {"worker.log", ".DS_Store", "LOCK", ".lock"} and not item.name.endswith(".lock"):
            result[str(item.relative_to(path))] = _digest(item)
    if not result:
        raise ValueError("inherited index has no durable files")
    return result


def _symbol(path, *names):
    body = ast.parse(path.read_text()).body
    for name in names:
        node = next((node for node in body if getattr(node, "name", None) == name), None)
        if node is None:
            raise ValueError(f"inherited source has no symbol: {name}")
        body = getattr(node, "body", [])
    return ast.dump(node)


def _verify_protocol(root, parent, manifest):
    old = json.loads((parent / "manifest.json").read_text())
    for name in ("selection_sha256", "answer_system", "answer_model", "judge_model",
                 "evidence_budget_tokens", "rag_source", "baseline_sources", "budget_usd"):
        if old.get(name) != manifest.get(name):
            raise ValueError(f"baseline inheritance protocol changed: {name}")
    for selection in (root / "data/week1/selection.json", parent / "selection.json"):
        if _digest(selection) != manifest["selection_sha256"]:
            raise ValueError("baseline inheritance selection changed")
    for current, saved in ((root / "uv.lock", parent / "uv.lock"),
                           (root / "pyproject.toml", parent / "pyproject.toml"),
                           (root / "data/week1/prepared.json", parent / "data-manifest.json")):
        if current.read_bytes() != saved.read_bytes():
            raise ValueError(f"baseline inheritance dependencies or data changed: {current.name}")
    if (root / "src/week1_data.py").read_bytes() != (parent / "source/week1_data.py").read_bytes():
        raise ValueError("baseline inheritance data adapter or scoring changed")
    eligible = set(BASELINES)
    for filename, affected in (("week1_baseline_worker.py", {"mem0", "memorybank"}),
                               ("week1_memorybank.py", {"memorybank"})):
        if (root / "src" / filename).read_bytes() != (parent / "source" / filename).read_bytes():
            eligible.difference_update(affected)
    symbols = {"week1_runner.py": [("render_evidence",)],
               "week1_gateway.py": [("Gateway", "_prepare")]}
    for filename, names in symbols.items():
        for name in names:
            if _symbol(root / "src" / filename, *name) != _symbol(parent / "source" / filename, *name):
                raise ValueError(f"baseline inheritance scientific implementation changed: {filename}:{'.'.join(name)}")
    method_symbols = {"rag": {"week1_runner.py": [("AuthorRAG", "__init__"), ("AuthorRAG", "ingest"),
                                                  ("AuthorRAG", "search")],
                              "providers.py": [("OpenAIEmbedder", "__init__"), ("OpenAIEmbedder", "embed")]},
                      "mem0": {"week1_runner.py": [("BaselineProcess", "ingest")]}}
    for method, files in method_symbols.items():
        if any(_symbol(root / "src" / filename, *name) != _symbol(parent / "source" / filename, *name)
               for filename, names in files.items() for name in names):
            eligible.discard(method)
    source_files = [(root / "data/week1/rag-source/evaluation/src/rag.py", parent / "author-source/rag.py")]
    for method in ("mem0", "memorybank"):
        archive = root / ("data/week1/memorybank/source.tar.gz" if method == "memorybank" else "data/week1/baselines/mem0.tar.gz")
        source_files.extend(((archive, parent / f"author-source/{method}.tar.gz"),
                             (root / f"data/week1/envs/{method}-requirements.lock", parent / f"{method}-requirements.lock")))
    for current in (root / "data/week1/scorers").glob("*"):
        if current.is_file():
            source_files.append((current, parent / "author-source/scorers" / current.name))
    if any(not saved.is_file() or current.read_bytes() != saved.read_bytes() for current, saved in source_files):
        raise ValueError("baseline inheritance pinned author sources changed")
    return eligible


def _verify_index(root, parent, method, example):
    path = parent / "indexes" / method / hashlib.sha256(example["group"].encode()).hexdigest()[:20]
    hashes = index_hashes(path)
    if method == "rag":
        source = root / "data/week1/rag-source/evaluation/src/rag.py"
        spec = importlib.util.spec_from_file_location("week1_inherited_author_rag", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manager = module.RAGManager.__new__(module.RAGManager)
        manager.calculate_embedding = lambda text: [0.0]
        history = [{"timestamp": session["id"], "speaker": message["role"], "text": message["content"]}
                   for session in example["sessions"] for message in session["messages"]]
        previous = os.environ.get("EMBEDDING_MODEL")
        try:
            os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
            expected, _ = manager.create_chunks(history, chunk_size=500)
        finally:
            if previous is None:
                os.environ.pop("EMBEDDING_MODEL", None)
            else:
                os.environ["EMBEDDING_MODEL"] = previous
        chunks = json.loads((path / "rag.json").read_text())
        vectors = np.load(path / "rag.npy", allow_pickle=False)
        if chunks != expected or vectors.shape != (len(expected), 1024) or not np.isfinite(vectors).all():
            raise ValueError("inherited RAG index does not cover the full frozen history")
    else:
        state = json.loads((path / "worker-state.json").read_text())
        if method == "mem0":
            expected = {f"{session['id']}:{offset}" for session in example["sessions"]
                        for offset in range(0, len(session["messages"]), 20)}
            if (state.get("method") != "mem0" or len(state.get("completed_adds", [])) != len(expected)
                    or set(state["completed_adds"]) != expected):
                raise ValueError("inherited Mem0 index lacks complete Add receipts")
        elif (state.get("history_complete") is not True or state.get("group") != example["group"]
              or state.get("source_messages") != sum(len(s["messages"]) for s in example["sessions"])):
            raise ValueError("inherited MemoryBank index lacks a full-history receipt")
        pinned = json.loads((parent / "manifest.json").read_text())["baseline_sources"][method]["commit"]
        if state.get("commit") != pinned:
            raise ValueError("inherited index uses a different author commit")
    return path, hashes


def validate_inheritance(root, parent_dir, manifest, examples):
    """Return only complete baseline histories and matching scored QA stages; never ours."""
    root, parent = Path(root).resolve(), Path(parent_dir).resolve()
    if not parent.is_relative_to(root / "artifacts/week1"):
        raise ValueError("inherited run must belong to this experiment")
    eligible = _verify_protocol(root, parent, manifest)
    rows = {name: _records(parent / f"{name}.jsonl")
            for name in ("ingestion", "retrieval", "answers", "judges", "predictions")}
    groups = {}
    for example in examples:
        key = (example["dataset"], example["group"])
        groups.setdefault(key, []).append(example)
        if history_sha256(example) != history_sha256(groups[key][0]):
            raise ValueError("questions sharing a history have different full source content")
    result = []
    tokenizer = tiktoken.get_encoding("o200k_base")
    for (dataset, group), questions in groups.items():
        example = questions[0]
        count = sum(len(session["messages"]) for session in example["sessions"])
        for method in BASELINES:
            if method not in eligible:
                continue
            ingested = [r for r in rows["ingestion"] if r.get("method") == method and r.get("dataset") == dataset
                        and r.get("group") == group and r.get("success") is True]
            if not ingested:
                continue
            ingestion = ingested[-1]
            if ingestion.get("source_messages") != count:
                raise ValueError("inherited ingestion message count differs from full history")
            path, hashes = _verify_index(root, parent, method, example)
            stages = []
            for question in questions:
                def matches(row):
                    return (row.get("method"), row.get("dataset"), row.get("id"), row.get("group")) == (method, dataset, question["id"], group)
                predicted = [r for r in rows["predictions"] if matches(r)]
                if not predicted:
                    continue
                prediction = predicted[-1]
                answers = [r for r in rows["answers"] if matches(r) and r.get("answer") == prediction.get("answer")
                           and r.get("evidence_tokens") == prediction.get("evidence_tokens")]
                if not answers or answers[-1].get("finish_reason") != "stop":
                    raise ValueError("inherited answer lacks a matching complete response")
                answer = answers[-1]
                retrieved = [r for r in rows["retrieval"] if matches(r) and r.get("evidence_tokens") == answer["evidence_tokens"]
                             and r.get("search_seconds") == answer.get("search_seconds")]
                if not retrieved:
                    raise ValueError("inherited prediction lacks matching retrieval")
                retrieval = retrieved[-1]
                context = "\n\n".join(f"[Memory {r['id']}]\n{r['content']}" for r in retrieval["data"])
                if len(tokenizer.encode(context, disallowed_special=())) != answer["evidence_tokens"] or answer["evidence_tokens"] > 16000:
                    raise ValueError("inherited evidence token receipt differs from actual text")
                judge = None
                if dataset == "longmemeval":
                    judges = [r for r in rows["judges"] if matches(r) and r.get("judge_response") == prediction.get("judge_response")]
                    if not judges or judges[-1].get("finish_reason") != "stop":
                        raise ValueError("inherited judge lacks a matching complete response")
                    judge = judges[-1]
                    score = float(parse_lme_judge(judge["judge_response"]))
                else:
                    score = float(score_locomo(answer["answer"], question))
                if not math.isfinite(prediction["score"]) or not math.isclose(score, prediction["score"], abs_tol=1e-12):
                    raise ValueError("inherited score differs from official scoring")
                stages.append({"example": question, "id": question["id"], "retrieval": retrieval,
                               "answer": answer, "judge": judge, "prediction": prediction})
            if method == "memorybank":
                completed = [stage["id"] for stage in stages]
                if completed != [question["id"] for question in questions[:len(completed)]]:
                    raise ValueError("inherited MemoryBank results are not a frozen-order prefix")
                if any(r.get("method") == method and r.get("dataset") == dataset and r.get("group") == group
                       and r.get("id") not in completed for r in rows["retrieval"]):
                    raise ValueError("inherited MemoryBank has unaccounted recall reinforcement")
            result.append({"method": method, "dataset": dataset, "group": group,
                           "history_sha256": history_sha256(example), "source_messages": count,
                           "source_tokens": example["source_tokens"], "index_path": path,
                           "index_sha256": hashes, "ingestion": ingestion, "questions": stages,
                           "parent_run_id": parent.name})
    return result
