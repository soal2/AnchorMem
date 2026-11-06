"""Isolated JSONL workers for the frozen 2025 Mem0 and A-Mem baselines.

Run with the corresponding data/week1/envs/<method>/bin/python. Model
traffic is restricted to the local, budgeted experiment gateway.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import re
from datetime import date
import subprocess
import sys
import tarfile
from urllib.parse import urlsplit
from urllib.request import urlopen


PROJECT = Path(__file__).resolve().parents[1]
BASELINES = PROJECT / "data/week1/baselines"
ENVS = PROJECT / "data/week1/envs"
COMMITS = {
    "mem0": "07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4",
    "amem": "1df003993661073a173f9bf6e7106a642ff02700",
}
REPOS = {"mem0": "mem0ai/mem0", "amem": "WujiangXu/A-mem"}
MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
DIMENSIONS = 1024
DUMMY_KEY = "week1-local-gateway"
DEPENDENCIES = {
    "mem0": ["mem0ai==0.1.94", "openai==1.61.1", "httpx==0.28.1",
             "qdrant-client==1.13.3"],
    "amem": ["numpy==1.24.3", "openai==1.61.1", "torch==2.4.0",
             "sentence-transformers==3.4.1", "scikit-learn==1.6.1",
             "transformers==4.46.3", "nltk==3.9.1", "rank-bm25==0.2.2",
             "litellm==1.59.1", "httpx==0.27.2"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> dict:
    """Fetch immutable archives and install isolated, reproducibly frozen deps."""
    BASELINES.mkdir(parents=True, exist_ok=True)
    ENVS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for method in ("mem0",):
        archive = BASELINES / f"{method}.tar.gz"
        url = f"https://api.github.com/repos/{REPOS[method]}/tarball/{COMMITS[method]}"
        if not archive.exists():
            with urlopen(url, timeout=120) as response:
                archive.write_bytes(response.read())
        source = BASELINES / method
        if not source.exists():
            source.mkdir()
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    relative = Path(*Path(member.name).parts[1:])
                    if not relative.parts:
                        continue
                    if member.issym() or member.islnk() or ".." in relative.parts:
                        raise ValueError("Archive contains unsafe path or link")
                    member.name = str(relative)
                    bundle.extract(member, source, filter="data")
        python = ENVS / method / "bin/python"
        if not python.exists():
            subprocess.run(["uv", "venv", "--python", "3.12" if method == "mem0" else "3.11",
                            str(ENVS / method)], check=True, stdout=sys.stderr)
        lock_path = ENVS / f"{method}-requirements.lock"
        requirements = (["--requirement", str(lock_path)] if lock_path.exists()
                        else DEPENDENCIES[method])
        subprocess.run(["uv", "pip", "install", "--python", str(python),
                        *requirements], check=True, stdout=sys.stderr)
        lock = subprocess.check_output(["uv", "pip", "freeze", "--python", str(python)], text=True)
        lock_path.write_text(lock)
        source_files = sorted(source.rglob("*.py"))
        manifest[method] = {
            "repo": REPOS[method], "commit": COMMITS[method], "archive_url": url,
            "archive_sha256": sha256(archive), "python": str(python),
            "python_version": subprocess.check_output([str(python), "--version"], text=True).strip(),
            "dependency_lock": str(ENVS / f"{method}-requirements.lock"),
            "dependency_lock_sha256": sha256(ENVS / f"{method}-requirements.lock"),
            "source_sha256": {str(path.relative_to(source)): sha256(path) for path in source_files},
            "adapter": "src/week1_baseline_worker.py", "telemetry": False,
            "embedding_model": EMBEDDING_MODEL, "embedding_dimensions": DIMENSIONS,
            "adaptations": [
                "Original pinned source imported directly; author source files are unmodified",
                "All OpenAI clients use the local budget gateway, dummy key, zero SDK retries",
                "Generation model and temperature fixed for comparison",
                "History role/date input and process JSONL transport adaptation",
                ("Local persistent Qdrant; graph disabled; native Memory.add(infer=True)/search"
                 if method == "mem0" else
                 "Remote SentenceTransformer factory, including native consolidation; native add_note/find_related_memories_raw"),
                "Index persistence for offline reopen; answers are never added",
            ],
        }
    (BASELINES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class GatewayNetwork:
    """Rewrite even absolute SDK URLs and stop after any failed model request."""
    def __init__(self, url: str, transport=None):
        import httpx
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("A baseline requires a loopback HTTP budget gateway")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("gateway-url must be the base host without a path")
        self.timeout = float(os.environ.get("WEEK1_LOCAL_REQUEST_TIMEOUT", "240"))
        self.url = url.rstrip("/")
        self.failed = None
        self.transport = transport or httpx.HTTPTransport(retries=0)
        network = self

        class LocalTransport(httpx.BaseTransport):
            def handle_request(self, request):
                if network.failed:
                    raise RuntimeError(network.failed)
                route = request.url.path.rstrip("/")
                if route.endswith("/chat/completions"):
                    route = "/api/v1/chat/completions"
                elif route.endswith("/embeddings"):
                    route = "/api/v1/embeddings"
                else:
                    raise ValueError(f"Unsupported model route: {route}")
                headers = dict(request.headers)
                headers.pop("host", None)
                headers["authorization"] = f"Bearer {DUMMY_KEY}"
                local = httpx.Request(request.method, network.url + route,
                                      headers=headers, content=request.read(),
                                      extensions=request.extensions)
                try:
                    response = network.transport.handle_request(local)
                except Exception:
                    network.failed = "Gateway transport failed; subsequent model requests stopped"
                    raise
                if response.status_code >= 400:
                    region = response.status_code == 403 and b"region" in response.read().lower()
                    network.failed = (f"Gateway returned HTTP {response.status_code}"
                                      + (" region refusal" if region else "") + "; model requests stopped")
                return response

            def close(self):
                # The shared transport is closed by the worker, not individual clients.
                pass

        self.local_transport_class = LocalTransport

    def client(self):
        import httpx
        return httpx.Client(transport=self.local_transport_class(), timeout=self.timeout, trust_env=False)

    def check(self):
        if self.failed:
            raise RuntimeError(self.failed)

    def close(self):
        self.transport.close()


def patch_openai(network: GatewayNetwork):
    """Enforce gateway + dummy key + no retries for all upstream constructors."""
    import openai
    original = openai.OpenAI

    class LocalOpenAI(original):
        def __init__(self, *args, **kwargs):
            if args:
                raise ValueError("OpenAI client construction must use named parameters")
            kwargs.update(api_key=DUMMY_KEY, base_url=network.url + "/api/v1",
                          max_retries=0, timeout=network.timeout, http_client=network.client())
            super().__init__(**kwargs)

    openai.OpenAI = LocalOpenAI
    return LocalOpenAI


class RemoteEncoder:
    """SentenceTransformer interface using the common remote embedding model."""
    def __init__(self, client):
        self.client = client

    def get_config_dict(self):
        return {"model_name": EMBEDDING_MODEL}

    def encode(self, sentences, **kwargs):
        import numpy as np
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        response = self.client.embeddings.create(model=EMBEDDING_MODEL,
                                                 input=texts, dimensions=DIMENSIONS)
        rows = sorted(response.data, key=lambda item: item.index)
        result = np.asarray([item.embedding for item in rows], dtype=np.float32)
        if result.shape != (len(texts), DIMENSIONS):
            raise ValueError("Remote embedding response has incorrect dimensions/count")
        return result[0] if single else result


def load_amem_module(network: GatewayNetwork):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    sys.path.insert(0, str(BASELINES / "amem"))
    module = importlib.import_module("memory_layer")
    import openai
    # Patch the factory itself before system creation. Consolidation and every
    # retriever construction therefore also use the common remote encoder.
    module.SentenceTransformer = lambda *args, **kwargs: RemoteEncoder(openai.OpenAI())
    native_completion = module.OpenAIController.get_completion

    def deterministic_completion(self, prompt, response_format, temperature=0):
        return native_completion(self, prompt, response_format, temperature=0)

    module.OpenAIController.get_completion = deterministic_completion
    return module


def patch_mem0_prompt(manifest):
    """Pin only the adapter's imported objects; never mutate frozen source files."""
    prompt_date = manifest["mem0_prompt_date"]
    date.fromisoformat(prompt_date)
    prompts = importlib.import_module("mem0.configs.prompts")
    utils = importlib.import_module("mem0.memory.utils")
    prompt, count = re.subn(r"Today's date is \d{4}-\d{2}-\d{2}\.",
                            f"Today's date is {prompt_date}.", prompts.FACT_RETRIEVAL_PROMPT)
    if count != 1:
        raise ValueError("Frozen Mem0 date marker changed")
    prompts.FACT_RETRIEVAL_PROMPT = utils.FACT_RETRIEVAL_PROMPT = prompt


class BaselineWorker:
    def __init__(self, method: str, root: Path, gateway_url: str, manifest=None):
        self.method = method
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.network = GatewayNetwork(gateway_url)
        os.environ["OPENAI_API_KEY"] = DUMMY_KEY
        os.environ["OPENAI_BASE_URL"] = self.network.url + "/api/v1"
        os.environ["MEM0_TELEMETRY"] = "False"
        os.environ["MEM0_DIR"] = str(self.root / "mem0-state")
        os.environ["LITELLM_LOG"] = "ERROR"
        os.environ["LITELLM_TELEMETRY"] = "False"
        for name in ("OPENROUTER_API_KEY", "OPENAI_API_BASE"):
            os.environ.pop(name, None)
        patch_openai(self.network)
        self.state_file = self.root / "worker-state.json"
        self.completed_adds = set()
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
            if state["method"] != method or state["commit"] != COMMITS[method]:
                raise ValueError("Persistent baseline index does not match the pinned method")
            self.completed_adds.update(state["completed_adds"])
        if method == "mem0":
            sys.path.insert(0, str(BASELINES / "mem0"))
            from mem0 import Memory
            patch_mem0_prompt(manifest or {})
            self.system = Memory.from_config({
                "llm": {"provider": "openai", "config": {
                    "model": MODEL, "temperature": 0, "max_tokens": 4096,
                    "openai_base_url": self.network.url + "/api/v1", "api_key": DUMMY_KEY}},
                "embedder": {"provider": "openai", "config": {
                    "model": EMBEDDING_MODEL, "embedding_dims": DIMENSIONS,
                    "openai_base_url": self.network.url + "/api/v1", "api_key": DUMMY_KEY}},
                "vector_store": {"provider": "qdrant", "config": {
                    "collection_name": "week1", "embedding_model_dims": DIMENSIONS,
                    "path": str(self.root / "qdrant"), "on_disk": True}},
                "history_db_path": str(self.root / "history.db"), "version": "v1.1",
            })
        elif method == "amem":
            module = load_amem_module(self.network)
            self.system = module.AgenticMemorySystem(model_name=EMBEDDING_MODEL,
                                                     llm_backend="openai", llm_model=MODEL,
                                                     api_key=DUMMY_KEY)
            saved = self.root / "amem-state.pkl"
            if saved.exists():
                with saved.open("rb") as handle:
                    state = pickle.load(handle)
                self.system.memories = state["memories"]
                self.system.evo_cnt = state["evo_cnt"]
                self.system.evo_threshold = state["evo_threshold"]
                self.system.retriever.load(str(self.root / "retriever.pkl"),
                                           str(self.root / "embeddings.npy"))
        else:
            raise ValueError("Unknown baseline")

    def metadata(self):
        native_source = Path(sys.modules[type(self.system).__module__].__file__).resolve()
        return {"method": self.method, "commit": COMMITS[self.method],
                "model": MODEL, "temperature": 0, "embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": DIMENSIONS, "graph": False,
                "completed_adds": sorted(self.completed_adds),
                "mem0_batch_size": 20 if self.method == "mem0" else None,
                "max_tokens": 4096 if self.method == "mem0" else 1000,
                "native_source": str(native_source), "native_source_sha256": sha256(native_source)}

    def add(self, request):
        request_id = request.get("request_id")
        if request_id and request_id in self.completed_adds:
            return {"already_completed": True}
        messages = request["messages"]
        for message in messages:
            if message["role"] not in {"user", "assistant"} or not isinstance(message["content"], str):
                raise ValueError("History messages must have user/assistant role and string content")
        results = []
        if self.method == "mem0":
            for offset in range(0, len(messages), 20):
                self.network.check()
                result = self.system.add(messages[offset:offset + 20], user_id="history", infer=True,
                                         metadata={"session_id": request.get("session_id", "")})
                self.network.check()
                results.extend(result["results"])
        else:
            for message in messages:
                self.network.check()
                note_id = self.system.add_note(
                    content=f"{message['role']}: {message['content']}",
                    time=str(request.get("date") or request.get("timestamp") or request.get("session_id") or "unknown"))
                self.network.check()
                results.append({"id": note_id})
        if request_id:
            self.completed_adds.add(request_id)
        self.persist()
        return {"messages_processed": len(messages), "results": results}

    def search(self, request):
        self.network.check()
        top_k = int(request.get("top_k", 20))
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.method == "mem0":
            found = self.system.search(request["query"], user_id="history", limit=top_k)["results"]
            results = [{"id": item["id"], "content": item["memory"], "score": item.get("score")}
                       for item in found]
        else:
            # Invoke the original author method, including its linked expansion
            # and formatting. The outer reader applies the common token budget.
            content = self.system.find_related_memories_raw(request["query"], k=top_k)
            results = ([{"id": "amem-native-context", "content": content, "score": None}]
                       if content else [])
        self.network.check()
        return results

    def persist(self):
        if self.method == "amem":
            self.system.retriever.save(str(self.root / "retriever.pkl"), str(self.root / "embeddings.npy"))
            temp = self.root / "amem-state.tmp"
            with temp.open("wb") as handle:
                pickle.dump({"memories": self.system.memories, "evo_cnt": self.system.evo_cnt,
                             "evo_threshold": self.system.evo_threshold}, handle)
            temp.replace(self.root / "amem-state.pkl")
        temp = self.root / "worker-state.tmp"
        temp.write_text(json.dumps({"method": self.method, "commit": COMMITS[self.method],
                                    "completed_adds": sorted(self.completed_adds)}) + "\n")
        temp.replace(self.state_file)

    def close(self):
        self.persist()
        if self.method == "mem0":
            self.system.vector_store.client.close()
            self.system._telemetry_vector_store.client.close()
            self.system.db.connection.close()
        self.network.close()


def serve(method: str, root: Path, gateway_url: str, input_stream=sys.stdin, output_stream=sys.stdout):
    worker = None
    for line in input_stream:
        request = {}
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("Worker request must be a JSON object")
            request = parsed
            action = request["action"]
            with contextlib.redirect_stdout(sys.stderr):
                if action == "init":
                    if worker is not None:
                        raise ValueError("Worker is already initialized")
                    worker = BaselineWorker(method, root, gateway_url, request.get("manifest"))
                    result = worker.metadata()
                elif worker is None:
                    raise ValueError("init must be called first")
                elif action == "add":
                    result = worker.add(request)
                elif action == "search":
                    result = worker.search(request)
                elif action == "close":
                    worker.close()
                    result = {"closed": True}
                else:
                    raise ValueError("Unknown worker action")
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {"ok": False, "error": str(error), "error_type": type(error).__name__}
        output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        output_stream.flush()
        if request.get("action") == "close":
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=list(COMMITS))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--gateway-url")
    args = parser.parse_args()
    if args.prepare:
        with contextlib.redirect_stdout(sys.stderr):
            manifest = prepare()
        print(json.dumps({"ok": True, "result": manifest}))
    else:
        if not all((args.method, args.root, args.gateway_url)):
            parser.error("--method, --root and --gateway-url are required")
        serve(args.method, args.root, args.gateway_url)


if __name__ == "__main__":
    main()
