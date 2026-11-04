from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from src import week1_baseline_worker as workers


def test_gateway_rewrites_absolute_api_url_and_uses_dummy_key():
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    network = workers.GatewayNetwork("http://127.0.0.1:8123", httpx.MockTransport(handle))
    network.client().post("https://api.openai.com/v1/chat/completions",
                          headers={"Authorization": "Bearer outside-key"}, json={})
    assert str(seen[0].url) == "http://127.0.0.1:8123/api/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer week1-local-gateway"


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://example.com", "http://localhost/api/v1"])
def test_gateway_requires_loopback_base_host(url):
    with pytest.raises(ValueError):
        workers.GatewayNetwork(url)


def test_gateway_error_blocks_subsequent_api_requests():
    requests = []
    network = workers.GatewayNetwork("http://127.0.0.1:8123", httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(402, json={"error": "budget"})))
    client = network.client()
    assert client.post("https://api.openai.com/v1/embeddings", json={}).status_code == 402
    with pytest.raises(RuntimeError, match="requests stopped"):
        client.post("https://api.openai.com/v1/embeddings", json={})
    assert len(requests) == 1


def test_remote_encoder_returns_ordered_float32_shared_dimensions():
    calls = []

    def embed(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[2.0] * 1024),
                                     SimpleNamespace(index=0, embedding=[1.0] * 1024)])

    encoder = workers.RemoteEncoder(SimpleNamespace(embeddings=SimpleNamespace(create=embed)))
    result = encoder.encode(["first", "second"], convert_to_tensor=True)
    assert result.shape == (2, 1024)
    assert result.dtype == np.float32
    assert result[0, 0] == 1.0
    assert result[1, 0] == 2.0
    assert calls[0]["model"] == "openai/text-embedding-3-small"
    assert calls[0]["dimensions"] == 1024
    assert encoder.get_config_dict()["model_name"] == "openai/text-embedding-3-small"
    assert encoder.encode([]).shape == (0, 1024)


def test_amem_factory_patch_covers_new_retrievers_and_deterministic_llm(monkeypatch):
    temperatures = []

    class NativeController:
        def get_completion(self, prompt, response_format, temperature=0.7):
            temperatures.append(temperature)
            return "{}"

    module = SimpleNamespace(SentenceTransformer=None, OpenAIController=NativeController)
    import openai
    monkeypatch.setattr(workers.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(openai, "OpenAI", lambda: object())
    patched = workers.load_amem_module(SimpleNamespace())
    first = patched.SentenceTransformer("all-MiniLM-L6-v2")
    rebuilt = patched.SentenceTransformer(first.get_config_dict()["model_name"])
    assert isinstance(first, workers.RemoteEncoder)
    assert isinstance(rebuilt, workers.RemoteEncoder)
    assert rebuilt.get_config_dict()["model_name"] == workers.EMBEDDING_MODEL
    NativeController().get_completion("prompt", {}, temperature=0.7)
    assert temperatures == [0]


def test_mem0_batching_preserves_every_role_message_and_idempotency():
    worker = workers.BaselineWorker.__new__(workers.BaselineWorker)
    worker.method = "mem0"
    worker.completed_adds = set()
    worker.network = SimpleNamespace(check=lambda: None)
    calls = []
    worker.system = SimpleNamespace(add=lambda messages, **kwargs:
                                    calls.append((messages, kwargs)) or {"results": []})
    worker.persist = lambda: None
    messages = [{"role": "assistant" if number % 2 else "user", "content": str(number)}
                for number in range(45)]
    request = {"messages": messages, "request_id": "session-one", "session_id": "s1"}
    assert worker.add(request)["messages_processed"] == 45
    assert [len(call[0]) for call in calls] == [20, 20, 5]
    assert [message for call in calls for message in call[0]] == messages
    assert all(call[1]["infer"] is True for call in calls)
    assert worker.add(request) == {"already_completed": True}
    assert len(calls) == 3


def test_amem_calls_original_author_retrieval_and_preserves_all_context():
    native_context = "author formatted evidence including linked neighbors\n" * 50
    calls = []
    worker = workers.BaselineWorker.__new__(workers.BaselineWorker)
    worker.method = "amem"
    worker.network = SimpleNamespace(check=lambda: None)
    worker.system = SimpleNamespace(find_related_memories_raw=lambda query, k:
                                    calls.append((query, k)) or native_context)
    rows = worker.search({"query": "question", "top_k": 2})
    assert calls == [("question", 2)]
    assert rows == [{"id": "amem-native-context", "content": native_context, "score": None}]


def test_protocol_handles_malformed_json_and_redirects_native_prints(monkeypatch, tmp_path):
    class FakeWorker:
        def __init__(self, *args):
            print("author debug output")
        def metadata(self):
            return {"method": "amem"}
        def close(self):
            return None

    monkeypatch.setattr(workers, "BaselineWorker", FakeWorker)
    output = io.StringIO()
    workers.serve("amem", tmp_path, "http://127.0.0.1:9",
                  io.StringIO('bad json\n[]\n{"action":"init"}\n{"action":"close"}\n'), output)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 4
    assert rows[0]["ok"] is False
    assert rows[1]["ok"] is False
    assert rows[2]["result"] == {"method": "amem"}
    assert rows[3]["ok"] is True


@pytest.fixture
def fake_gateway():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if self.path.endswith("/embeddings"):
                texts = payload["input"]
                result = {"object": "list", "model": payload["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1},
                          "data": [{"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * 1023}
                                   for i, text in enumerate(texts)]}
            else:
                text = str(payload["messages"])
                if "should_evolve" in text:
                    content = {"should_evolve": False, "actions": [], "suggested_connections": [],
                               "tags_to_update": [], "new_context_neighborhood": [], "new_tags_neighborhood": []}
                elif "structured analysis" in text:
                    content = {"keywords": ["tea", "preference", "drink"], "context": "Alice likes tea", "tags": ["food"]}
                elif "You are a smart memory manager" in text:
                    content = {"memory": [{"id": "0", "text": "Alice likes tea", "event": "ADD"}]}
                else:
                    content = {"facts": ["Alice likes tea"]}
                result = {"id": "offline", "object": "chat.completion", "created": 1, "model": payload["model"],
                          "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()
    thread.join()


@pytest.mark.parametrize("method", ["mem0", "amem"])
def test_installed_native_worker_add_search_restore_and_history_isolation(method, tmp_path, fake_gateway):
    python = workers.ENVS / method / "bin/python"
    if not python.exists() or not (workers.ENVS / f"{method}-requirements.lock").exists():
        pytest.skip("Pinned optional environments have not been prepared")
    gateway_url, requests = fake_gateway

    def run(root, actions):
        if method == "mem0":
            actions = [{**action, "manifest": {"mem0_prompt_date": "2026-11-03"}}
                       if action.get("action") == "init" else action for action in actions]
        result = subprocess.run([str(python), str(Path(workers.__file__)), "--method", method,
                                 "--root", str(root), "--gateway-url", gateway_url],
                                input="".join(json.dumps(action) + "\n" for action in actions),
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        assert all(row["ok"] for row in rows), (rows, result.stderr)
        return rows

    root = tmp_path / "history-one"
    add = {"action": "add", "session_id": "s1", "request_id": "session-one", "date": "2024-01-01",
           "messages": [{"role": "user", "content": "2024-01-01 Alice: I like tea."},
                        {"role": "assistant", "content": "2024-01-01 Assistant: Your preference is tea."}]}
    search = {"action": "search", "query": "What does Alice like?", "top_k": 20}
    rows = run(root, [{"action": "init"}, add, search, {"action": "close"}])
    assert rows[2]["result"]
    before = len(requests)
    restored = run(root, [{"action": "init"}, add, search, {"action": "close"}])
    assert restored[1]["result"] == {"already_completed": True}
    assert len(requests) - before == 1  # Only the search embedding; restore is offline.
    assert restored[2]["result"] == rows[2]["result"]
    empty = run(tmp_path / "history-two", [{"action": "init"}, search, {"action": "close"}])
    assert empty[1]["result"] == []
    assert all(payload["model"] in {workers.MODEL, workers.EMBEDDING_MODEL} for path, payload in requests)
    assert all(payload["temperature"] == 0 for path, payload in requests if path.endswith("/chat/completions"))


def test_installed_amem_native_consolidation_keeps_remote_encoder(fake_gateway):
    python = workers.ENVS / "amem" / "bin/python"
    if not (workers.ENVS / "amem-requirements.lock").exists():
        pytest.skip("Pinned optional A-Mem environment has not been prepared")
    gateway_url, requests = fake_gateway
    code = """
import contextlib, sys
from src.week1_baseline_worker import GatewayNetwork, patch_openai, load_amem_module, RemoteEncoder, EMBEDDING_MODEL, MODEL
with contextlib.redirect_stdout(sys.stderr):
    network = GatewayNetwork(sys.argv[1])
    patch_openai(network)
    module = load_amem_module(network)
    system = module.AgenticMemorySystem(model_name=EMBEDDING_MODEL, llm_model=MODEL, api_key='week1-local-gateway')
    system.add_note('Alice likes tea.', time='2024-01-01')
    old_retriever = system.retriever
    system.consolidate_memories()
    assert old_retriever is not system.retriever
    assert isinstance(system.retriever.model, RemoteEncoder)
    assert system.retriever.model.get_config_dict()['model_name'] == EMBEDDING_MODEL
    assert system.retriever.embeddings.shape == (1, 1024)
print('native consolidation passed')
"""
    result = subprocess.run([str(python), "-c", code, gateway_url], cwd=workers.PROJECT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "native consolidation passed"
    embedding_calls = [payload for path, payload in requests if path.endswith("/embeddings")]
    assert len(embedding_calls) == 2
    assert all(payload["model"] == workers.EMBEDDING_MODEL for payload in embedding_calls)
