import hashlib
import json

import pytest
import tiktoken

from src import week1_lineage as lineage


@pytest.fixture
def inherited(tmp_path, monkeypatch):
    parent = tmp_path / "artifacts/week1/parent"
    parent.mkdir(parents=True)
    example = {"dataset": "locomo", "group": "history", "id": "q0", "category": 1,
               "question": "What does the user like?", "answer": "tea", "source_tokens": 5,
               "sessions": [{"id": "s1", "messages": [{"role": "user", "content": "I like tea."}]}]}
    content = "[Memory m]\nI like tea."
    tokens = len(tiktoken.get_encoding("o200k_base").encode(content))
    identity = {key: example[key] for key in ("dataset", "group", "id", "category")}
    identity["method"] = "rag"
    rows = {"ingestion": [{**identity, "success": True, "source_messages": 1}],
            "retrieval": [{**identity, "evidence_tokens": tokens, "search_seconds": .1,
                           "data": [{"id": "m", "content": "I like tea."}]}],
            "answers": [{**identity, "answer": "tea", "finish_reason": "stop",
                         "evidence_tokens": tokens, "search_seconds": .1}],
            "predictions": [{**identity, "answer": "tea", "score": 1., "evidence_tokens": tokens}]}

    def save():
        for name, values in rows.items():
            (parent / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))

    save()
    monkeypatch.setattr(lineage, "_verify_protocol", lambda *args: set(lineage.BASELINES))
    monkeypatch.setattr(lineage, "_verify_index", lambda *args: (parent / "index", {"state": "digest"}))
    return tmp_path, parent, example, rows, save


def test_inheritance_imports_only_full_baseline_stages_and_matches_retrieval(inherited):
    root, parent, example, rows, save = inherited
    rows["retrieval"].insert(0, {**rows["retrieval"][0], "search_seconds": 100,
                                  "data": [{"id": "bad", "content": "unmatched old search"}]})
    rows["predictions"].append({**rows["predictions"][0], "method": "ours"})
    rows["ingestion"].append({**rows["ingestion"][0], "method": "ours"})
    save()
    result = lineage.validate_inheritance(root, parent, {}, [example])
    assert len(result) == 1 and result[0]["method"] == "rag"
    assert result[0]["history_sha256"] == lineage.history_sha256(example)
    assert result[0]["questions"][0]["retrieval"]["data"][0]["id"] == "m"
    assert result[0]["questions"][0]["judge"] is None


@pytest.mark.parametrize("change,match", [("partial_history", "message count"),
    ("truncated_answer", "complete response"), ("changed_score", "official scoring"),
    ("missing_retrieval", "matching retrieval"), ("wrong_tokens", "token receipt")])
def test_inheritance_refuses_incomplete_or_mismatched_evidence(inherited, change, match):
    root, parent, example, rows, save = inherited
    if change == "partial_history":
        rows["ingestion"][0]["source_messages"] = 0
    elif change == "truncated_answer":
        rows["answers"][0]["finish_reason"] = "length"
    elif change == "changed_score":
        rows["predictions"][0]["score"] = 0
    elif change == "missing_retrieval":
        rows["retrieval"].clear()
    else:
        rows["retrieval"][0]["data"][0]["content"] += " More unrecorded evidence."
    save()
    with pytest.raises(ValueError, match=match):
        lineage.validate_inheritance(root, parent, {}, [example])


def test_memorybank_inheritance_rejects_unaccounted_reinforcement(inherited):
    root, parent, example, rows, save = inherited
    for values in rows.values():
        for row in values:
            row["method"] = "memorybank"
    rows["retrieval"].append({**rows["retrieval"][0], "id": "unscored-search"})
    save()
    with pytest.raises(ValueError, match="unaccounted recall"):
        lineage.validate_inheritance(root, parent, {}, [example])


def test_memorybank_inheritance_requires_frozen_question_order_prefix(inherited):
    root, parent, example, rows, save = inherited
    for values in rows.values():
        for row in values:
            row["method"] = "memorybank"
    save()
    earlier = {**example, "id": "earlier-unanswered"}
    with pytest.raises(ValueError, match="frozen-order prefix"):
        lineage.validate_inheritance(root, parent, {}, [earlier, example])


def test_mem0_incomplete_add_receipts_are_not_a_full_history(tmp_path):
    parent = tmp_path / "parent"
    example = {"group": "history", "sessions": [{"id": "s1", "messages": [
        {"role": "user", "content": str(i)} for i in range(41)]}]}
    path = parent / "indexes/mem0" / hashlib.sha256(b"history").hexdigest()[:20]
    path.mkdir(parents=True)
    (path / "worker-state.json").write_text(json.dumps({"method": "mem0", "completed_adds": ["s1:0", "s1:20"]}))
    with pytest.raises(ValueError, match="complete Add receipts"):
        lineage._verify_index(tmp_path, parent, "mem0", example)


def test_index_hashes_reject_symlinks_and_exclude_only_nonstate_files(tmp_path):
    (tmp_path / "state.db").write_text("durable")
    (tmp_path / "state.db-wal").write_text("durable journal")
    (tmp_path / "worker.log").write_text("runtime log")
    hashes = lineage.index_hashes(tmp_path)
    assert set(hashes) == {"state.db", "state.db-wal"}
    (tmp_path / "link").symlink_to(tmp_path / "state.db")
    with pytest.raises(ValueError, match="symlinks"):
        lineage.index_hashes(tmp_path)


def test_changed_method_adapter_rebuilds_only_affected_baseline(tmp_path):
    parent = tmp_path / "artifacts/week1/parent"
    texts = {
        "src/week1_runner.py": "class AuthorRAG:\n def __init__(self): pass\n def ingest(self): pass\n def search(self): pass\nclass BaselineProcess:\n def ingest(self): pass\ndef render_evidence(): return 'unchanged'\n",
        "src/providers.py": "class OpenAIEmbedder:\n def __init__(self): pass\n def embed(self): pass\n",
        "src/week1_gateway.py": "class Gateway:\n def _prepare(self): pass\n",
        "src/week1_baseline_worker.py": "original worker",
        "src/week1_memorybank.py": "original MemoryBank adapter",
        "src/week1_data.py": "original data and scoring adapter",
        "uv.lock": "locked", "pyproject.toml": "fixed", "data/week1/prepared.json": "{}",
        "data/week1/selection.json": '{"examples": []}',
        "data/week1/rag-source/evaluation/src/rag.py": "original rag",
        "data/week1/baselines/mem0.tar.gz": "original mem0 archive",
        "data/week1/memorybank/source.tar.gz": "original MemoryBank archive",
        "data/week1/envs/mem0-requirements.lock": "fixed mem0",
        "data/week1/envs/memorybank-requirements.lock": "fixed MemoryBank",
    }
    destinations = {"data/week1/prepared.json": "data-manifest.json", "data/week1/selection.json": "selection.json",
        "data/week1/rag-source/evaluation/src/rag.py": "author-source/rag.py",
        "data/week1/baselines/mem0.tar.gz": "author-source/mem0.tar.gz",
        "data/week1/memorybank/source.tar.gz": "author-source/memorybank.tar.gz",
        "data/week1/envs/mem0-requirements.lock": "mem0-requirements.lock",
        "data/week1/envs/memorybank-requirements.lock": "memorybank-requirements.lock"}
    for name, content in texts.items():
        current = tmp_path / name
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text(content)
        saved = parent / destinations.get(name, "source/" + name[4:] if name.startswith("src/") else name)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(content)
    manifest = {"selection_sha256": hashlib.sha256(texts["data/week1/selection.json"].encode()).hexdigest()}
    (parent / "manifest.json").write_text(json.dumps(manifest))
    assert lineage._verify_protocol(tmp_path, parent, manifest) == set(lineage.BASELINES)
    (tmp_path / "src/week1_memorybank.py").write_text("save successful author FAISS state")
    assert lineage._verify_protocol(tmp_path, parent, manifest) == {"rag", "mem0"}
    source = tmp_path / "src/week1_runner.py"
    source.write_text(source.read_text().replace("return 'unchanged'", "return 'changed reader'"))
    with pytest.raises(ValueError, match="scientific implementation changed"):
        lineage._verify_protocol(tmp_path, parent, manifest)
