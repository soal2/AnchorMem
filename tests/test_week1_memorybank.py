from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading
from time import perf_counter
from types import SimpleNamespace

import pytest

from src import week1_memorybank as memorybank


ROOT = Path(__file__).resolve().parents[1]


def session(date="1:56 pm on 8 May, 2023", texts=("A: first", "B: second", "A: third")):
    return {"id": "session_1", "messages": [{"role": "user", "content": f"[Session date: {date}]\n{text}"}
                                            for text in texts]}


def test_complete_sessions_and_same_date_merge_preserve_every_message():
    sessions = [session(texts=tuple(f"speaker: original {i:03d};" for i in range(23))),
                session(texts=("tail message",))]
    bank, clock, count = memorybank.memory_input(sessions)
    dialogs = bank["pilot"]["history"]["2023-05-08"]
    assert clock == "2023-05-08"
    assert count == 24
    assert len(dialogs) == 13
    text = "\n".join(d["query"] + d["response"] for d in dialogs)
    for item in sessions:
        for message in item["messages"]:
            assert text.count(message["content"]) == 1
    assert dialogs[11]["response"] == ""
    assert dialogs[-1]["query"].endswith("tail message")


def test_benchmark_clock_uses_question_date_without_wall_clock():
    bank, clock, count = memorybank.memory_input([session("2023/05/20 (Sat) 13:27")],
                                                "2023/05/30 (Tue) 23:40")
    assert clock == "2023-05-30"
    assert list(bank["pilot"]["history"]) == ["2023-05-20"]
    assert count == 3


@pytest.mark.parametrize("sessions,date", [([], None), ([session("bad")], None),
    ([session()], "2023-05-07"), ([{"id": "x", "messages": [{"role": "user", "content": "undated"}]}], None)])
def test_incomplete_or_future_history_dates_rejected(sessions, date):
    with pytest.raises(ValueError):
        memorybank.memory_input(sessions, date)


def test_parent_ingestion_allowlists_history_and_clock():
    process = memorybank.MemoryBankProcess.__new__(memorybank.MemoryBankProcess)
    sent = []
    process.call = lambda request: sent.append(request)
    example = {"group": "g", "sessions": [session()], "question_date": "2023-05-08",
               "question": "secret QA", "answer": "gold", "evidence": ["gold source"]}
    example["sessions"][0]["messages"][0]["answer"] = "annotation"
    process.ingest(example)
    assert set(sent[0]) == {"action", "sessions", "group", "question_date"}
    assert set(sent[0]["sessions"][0]["messages"][0]) == {"role", "content"}
    assert "secret QA" not in json.dumps(sent)
    assert "gold" not in json.dumps(sent)
    assert "annotation" not in json.dumps(sent)


def test_summary_transport_uses_author_scaffolding_and_full_prompt_once():
    calls = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or response)))
    module = SimpleNamespace(LLMClientSimple=type("AuthorClient", (), {}))
    memorybank.install_summary_transport(module, client)
    prompt = "original source dialogue\n" * 10000
    assert module.LLMClientSimple().generate_text_simple(prompt, 1) == "summary"
    assert len(calls) == 1
    assert calls[0]["messages"][-1]["content"] == prompt
    assert calls[0]["model"] == "openai/gpt-4o-mini"
    assert calls[0]["temperature"] == 0
    assert calls[0]["max_tokens"] == 400
    assert calls[0]["messages"][2]["role"] == "system"


def test_summary_transport_propagates_error_without_retry_or_cut():
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("This model's maximum context")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    module = SimpleNamespace(LLMClientSimple=type("AuthorClient", (), {}))
    memorybank.install_summary_transport(module, client)
    prompt = "uncut source" * 10000
    with pytest.raises(RuntimeError, match="maximum context"):
        module.LLMClientSimple().generate_text_simple(prompt, 1)
    assert len(calls) == 1
    assert calls[0]["messages"][-1]["content"] == prompt


def test_remote_embedding_batches_serially_and_validates_dimensions():
    calls = []
    def embed(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(i)] * 1024)
                                     for i in reversed(range(len(kwargs["input"])))])
    encoder = memorybank.RemoteEmbeddings(SimpleNamespace(embeddings=SimpleNamespace(create=embed)))
    result = encoder.embed_documents([str(i) for i in range(23)])
    assert [len(c["input"]) for c in calls] == [10, 10, 3]
    assert len(result) == 23
    assert result[1] == [1.0] * 1024
    assert all(c["dimensions"] == 1024 and c["model"] == memorybank.EMBEDDING_MODEL for c in calls)
    assert encoder.embed_query("query") == [0.0] * 1024


def test_embedding_bad_dimension_rejected():
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.0] * 4)])))
    with pytest.raises(ValueError, match="dimensions"):
        memorybank.RemoteEmbeddings(client).embed_query("query")


@pytest.mark.parametrize("network_failure", [False, True])
def test_retrieval_persists_native_docstore_only_after_successful_network_check(tmp_path, network_failure):
    worker = memorybank.MemoryBankWorker.__new__(memorybank.MemoryBankWorker)
    worker.path = tmp_path
    worker.clock = "2023-05-08"
    worker.memory_file = tmp_path / "memory.json"
    worker.memory_file.write_text(json.dumps({"pilot": {"overall_history": "history", "overall_personality": "personality"}}))
    calls = []
    def check():
        calls.append("check")
        if network_failure and len(calls) == 3:
            raise RuntimeError("failed native network request")
    def search(*args, **kwargs):
        calls.append("search")
        return ["remembered"], "2023-05-08"
    worker.network = SimpleNamespace(check=check)
    worker.retrieval = SimpleNamespace(search_memory=search)
    worker.vector_store = SimpleNamespace(save_local=lambda path: calls.append(("save", path)))
    if network_failure:
        with pytest.raises(RuntimeError, match="failed native network"):
            worker.search("query")
        assert calls == ["check", "search", "check"]
    else:
        assert worker.search("query")[0]["content"].endswith("remembered")
        assert calls == ["check", "search", "check", ("save", str(tmp_path / "faiss"))]


@pytest.mark.skipif(not (ROOT / "data/week1/memorybank/manifest.json").exists(), reason="Author environment not prepared")
@pytest.mark.parametrize("dataset", ["synthetic", "locomo", "longmemeval"])
def test_real_author_worker_summary_faiss_forgetting_and_reinforcement(tmp_path, dataset):
    """Only HTTP model responses are mocked; author memory algorithms and FAISS run."""
    memorybank.verify(ROOT)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append((self.path, body, self.headers.get("Authorization")))
            if self.path.endswith("/embeddings"):
                result = {"object": "list", "data": [{"object": "embedding", "index": i,
                    "embedding": [0.01 * (i + 1)] * 1024} for i, _ in enumerate(body["input"])],
                    "model": body["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1}}
            else:
                result = {"id": "offline", "object": "chat.completion", "created": 0,
                    "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "The original speakers discussed a garden."}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = f"http://127.0.0.1:{server.server_port}"
    path = tmp_path / "native-index"
    if dataset == "synthetic":
        example = {"group": "offline", "sessions": [session(texts=(
            "Alice: I planted a garden.", "Bob: I remember your flowers."))]}
    else:
        from src.week1_data import load_pilot
        example = next(e for e in load_pilot(ROOT) if e["dataset"] == dataset)
    converted, expected_clock, expected_count = memorybank.memory_input(example["sessions"], example.get("question_date"))
    started = perf_counter()
    try:
        process = memorybank.MemoryBankProcess(ROOT, path, gateway)
        try:
            state = process.ingest(example)
            assert state["source_messages"] == expected_count
            assert state["clock"] == expected_clock
            before = json.loads((path / "memory.json").read_text())["pilot"]
            if dataset == "synthetic":
                assert before["history"]["2023-05-08"][0]["memory_strength"] == 1
            initial_docstore = (path / "faiss/index.pkl").read_bytes()
            evidence = process.search("What was planted?")
            assert "garden" in evidence[0]["content"]
            assert "Overall personality" in evidence[0]["content"]
            after = json.loads((path / "memory.json").read_text())["pilot"]
            if dataset == "synthetic":
                assert after["history"]["2023-05-08"][0]["memory_strength"] > 1
                assert after["summary"]["2023-05-08"]["memory_strength"] == 1
            assert "What was planted?" not in json.dumps(after)
            assert (path / "faiss/index.faiss").exists()
            searched_docstore = (path / "faiss/index.pkl").read_bytes()
            assert searched_docstore != initial_docstore
        finally:
            process.close()
        before_reopen = len(requests)
        reopened = memorybank.MemoryBankProcess(ROOT, path, gateway)
        try:
            assert len(requests) == before_reopen
            assert (path / "faiss/index.pkl").read_bytes() == searched_docstore
            assert reopened.search("Flowers?")
            if dataset == "synthetic":
                continued = json.loads((path / "memory.json").read_text())["pilot"]
                assert continued["history"]["2023-05-08"][0]["memory_strength"] > after["history"]["2023-05-08"][0]["memory_strength"]
        finally:
            reopened.close()
        generations = [body for route, body, _ in requests if route.endswith("/chat/completions")]
        assert len(generations) == 2 * len(converted["pilot"]["history"]) + 2
        assert all(body["temperature"] == 0 and body["model"] == memorybank.MODEL for body in generations)
        assert all(key == "Bearer week1-local-gateway" for _, _, key in requests)
        print(f"{dataset}: {expected_count} complete source messages, {len(generations)} native summaries, "
              f"{len(requests)} local mock requests, {perf_counter() - started:.3f}s")
    finally:
        server.shutdown()
        server.server_close()
