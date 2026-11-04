"""Pinned MemoryBank author code, isolated behind the pilot JSONL boundary."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import select
import subprocess
import sys
import tarfile
from urllib.request import urlopen

COMMIT = "cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9"
REPO = "zhongwanjun/MemoryBank-SiliconFriend"
MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEPENDENCIES = ["langchain==0.0.146", "openai==1.61.1", "httpx==0.27.2",
                "numpy==1.26.4", "faiss-cpu==1.8.0.post1", "unstructured==0.10.30",
                "torch==2.4.0"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(root):
    return json.loads((Path(root) / "data/week1/memorybank/manifest.json").read_text())


def verify(root):
    root = Path(root)
    folder = root / "data/week1/memorybank"
    info = metadata(root)
    if info["commit"] != COMMIT or info["repo"] != REPO:
        raise ValueError("MemoryBank source identity differs from the frozen author commit")
    for relative, expected in info["files_sha256"].items():
        if digest(folder / relative) != expected:
            raise ValueError(f"MemoryBank frozen file changed: {relative}")
    lock = root / "data/week1/envs/memorybank-requirements.lock"
    if digest(lock) != info["dependency_lock_sha256"]:
        raise ValueError("MemoryBank dependency lock changed")
    return info


def prepare(root):
    root = Path(root).resolve()
    folder = root / "data/week1/memorybank"
    if (folder / "manifest.json").exists():
        return verify(root)
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "source.tar.gz"
    url = f"https://api.github.com/repos/{REPO}/tarball/{COMMIT}"
    if not archive.exists():
        with urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
    source = folder / "source"
    source.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            relative = Path(*Path(member.name).parts[1:])
            if not relative.parts:
                continue
            if member.issym() or member.islnk() or ".." in relative.parts or relative.is_absolute():
                raise ValueError("Unsafe MemoryBank source archive entry")
            member.name = str(relative)
            bundle.extract(member, source, filter="data")
    environment = root / "data/week1/envs/memorybank"
    python = environment / "bin/python"
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", "3.11", str(environment)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(python), *DEPENDENCIES], check=True)
    lock = environment.with_name("memorybank-requirements.lock")
    lock.write_text(subprocess.check_output(["uv", "pip", "freeze", "--python", str(python)], text=True))
    info = {"repo": REPO, "commit": COMMIT, "archive_url": url,
            "python": str(python), "python_version": subprocess.check_output([str(python), "--version"], text=True).strip(),
            "dependency_lock": str(lock), "dependency_lock_sha256": digest(lock),
            "files_sha256": {str(p.relative_to(folder)): digest(p) for p in
                              [archive, *sorted(source.rglob("*.py")), source / "requirement.txt", source / "LICENSE"]},
            "native_entrypoints": ["summarize_memory.summarize_memory",
                "forget_memory.MemoryForgetterLoader.initial_load_forget_and_save",
                "forget_memory.LocalMemoryRetrieval.init_memory_vector_store",
                "forget_memory.LocalMemoryRetrieval.load_memory_index",
                "forget_memory.LocalMemoryRetrieval.search_memory",
                "forget_memory.MemoryForgetterLoader.update_memory_when_searched"],
            "adaptations": ["Author source files remain byte-identical to the immutable archive",
                "Only LLMClientSimple.generate_text_simple transport replaced; author summary/personality prompts unchanged",
                "HuggingFaceEmbeddings factory replaced with common remote 1024-dimensional embeddings",
                "Generation gpt-4o-mini temperature 0; author max_tokens 400; gateway only; retries 0; no truncation",
                "Full sessions paired in order with original role/date/speaker text retained; same dates merged in original order",
                "Clock is benchmark question date, or latest history session date when absent; seed 42",
                "Author FAISS retrieval, ChineseTextSplitter, random forgetting, and reinforcement run directly",
                "Original exp(-t / 5 * S) formula retained; original summary memories are not reinforced",
                "Original author top_k 6 and chunk_size 200 retained; shared answer reader receives native recall and overall summaries",
                "macOS loader uses the torch library directory so torch and FAISS share one OpenMP runtime",
                "No answer writeback; only author recall reinforcement writes memory"],
            "embedding_model": EMBEDDING_MODEL, "embedding_dimensions": 1024,
            "generation": {"model": MODEL, "temperature": 0, "max_tokens": 400, "retries": 0},
            "retrieval": {"top_k": 6, "chunk_size": 200, "backend": "author FAISS"},
            "forgetting": {"seed": 42, "formula": "math.exp(-t / 5*S)",
                "clock": "official question date or latest full-history session date",
                "summary_reinforcement": False},
            "maintained_benchmark_backend": False}
    (folder / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    return verify(root)


def benchmark_date(value):
    for pattern in ("%Y-%m-%d", "%Y/%m/%d (%a) %H:%M", "%I:%M %p on %d %B, %Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported benchmark date: {value}")


def memory_input(sessions, question_date=None):
    """Convert the complete allowlisted history, retaining every original byte."""
    history = {}
    count = 0
    for session in sessions:
        messages = session["messages"]
        if not messages:
            continue
        dates = []
        for message in messages:
            match = re.match(r"\[Session date: ([^\n]+)\]\n", message["content"])
            if not match:
                raise ValueError("MemoryBank requires the frozen session date on every history message")
            dates.append(benchmark_date(match.group(1)))
        if len(set(dates)) != 1:
            raise ValueError("History session contains inconsistent dates")
        dialogs = history.setdefault(dates[0], [])
        for offset in range(0, len(messages), 2):
            pair = messages[offset:offset + 2]
            texts = [f"[History role: {m['role']}]\n{m['content']}" for m in pair]
            dialogs.append({"query": texts[0], "response": texts[1] if len(texts) == 2 else ""})
        count += len(messages)
    if not history:
        raise ValueError("MemoryBank cannot build an empty history")
    clock = benchmark_date(question_date) if question_date else max(history)
    if max(history) > clock:
        raise ValueError("History contains a date after the benchmark clock")
    return {"pilot": {"history": history}}, clock, count


def install_summary_transport(module, client):
    def generate(self, prompt, prompt_num, language="en"):
        if prompt_num != 1 or language != "en":
            raise ValueError("The pilot fixes one English author summary per request")
        response = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=400,
            top_p=1.0, frequency_penalty=0.4, presence_penalty=0.2,
            stop=["<|im_end|>", "¬人类¬"], n=1, messages=[
                {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                {"role": "system", "content": "Sure, I will do my best to assist you."},
                {"role": "user", "content": prompt}])
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("MemoryBank returned an empty summary")
        return content
    module.LLMClientSimple.generate_text_simple = generate


class RemoteEmbeddings:
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        result = []
        for offset in range(0, len(texts), 10):
            batch = texts[offset:offset + 10]
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=1024)
            vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            if len(vectors) != len(batch) or any(len(row) != 1024 for row in vectors):
                raise ValueError("MemoryBank embedding dimensions or count differ from the fixed encoder")
            result.extend(vectors)
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class MemoryBankWorker:
    def __init__(self, project, path, gateway_url):
        from src.week1_baseline_worker import DUMMY_KEY, GatewayNetwork, patch_openai
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        verify(project)
        self.network = GatewayNetwork(gateway_url)
        os.environ["OPENAI_API_KEY"] = DUMMY_KEY
        client_type = patch_openai(self.network)
        self.client = client_type()
        source = Path(project) / "data/week1/memorybank/source/memory_bank"
        sys.path.insert(0, str(source))
        import importlib
        self.summary = importlib.import_module("summarize_memory")
        self.native = importlib.import_module("memory_retrieval.forget_memory")
        install_summary_transport(self.summary, self.client)
        self.native.HuggingFaceEmbeddings = lambda **kwargs: RemoteEmbeddings(self.client)
        self.retrieval = self.native.LocalMemoryRetrieval()
        self.retrieval.init_cfg(language="en")
        self.memory_file = self.path / "memory.json"
        self.state_file = self.path / "worker-state.json"
        self.clock = None
        self.vector_store = None
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
            if state["commit"] != COMMIT or not state["history_complete"]:
                raise ValueError("MemoryBank index is incomplete or uses different author source")
            self.clock = state["clock"]
            self._open_index()

    def _open_index(self):
        self.retrieval.user = "pilot"
        self.retrieval.memory_path = str(self.memory_file)
        self.retrieval.memory_loader = self.native.MemoryForgetterLoader(str(self.memory_file), "en")
        self.retrieval.memory_loader.load_memories(str(self.memory_file))
        index = self.path / "faiss"
        self.vector_store = self.retrieval.load_memory_index(str(index)) if index.exists() else None

    def ingest(self, request):
        if self.clock is not None:
            raise ValueError("MemoryBank history is already complete")
        memory, clock, count = memory_input(request["sessions"], request.get("question_date"))
        self.memory_file.write_text(json.dumps(memory, ensure_ascii=False))
        self.summary.summarize_memory(str(self.memory_file), name="pilot", language="en")
        self.network.check()
        random.seed(42)
        self.retrieval.init_memory_vector_store(str(self.memory_file), str(self.path / "faiss"),
                                                user_name="pilot", cur_date=clock)
        self.network.check()
        self.clock = clock
        self._open_index()
        state = {"commit": COMMIT, "group": request["group"], "history_complete": True,
                 "source_messages": count, "clock": clock, "seed": 42}
        self.state_file.write_text(json.dumps(state) + "\n")
        return state

    def search(self, query):
        if self.clock is None:
            raise ValueError("MemoryBank history is incomplete")
        self.network.check()
        recalled, dates = (self.retrieval.search_memory(query, self.vector_store, cur_date=self.clock)
                           if self.vector_store is not None else ([], ""))
        self.network.check()
        if self.vector_store is not None:
            self.vector_store.save_local(str(self.path / "faiss"))
        user = json.loads(self.memory_file.read_text())["pilot"]
        context = [f"Overall history: {user['overall_history']}",
                   f"Overall personality: {user['overall_personality']}"]
        if recalled:
            context.append(f"Recalled memories ({dates}):\n" + "\n".join(recalled))
        return [{"id": "memorybank-author-context", "content": "\n\n".join(context), "score": None}]

    def close(self):
        self.client.close()
        self.network.close()


class MemoryBankProcess:
    def __init__(self, root, path, gateway_url):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.log = (self.path / "worker.log").open("a")
        env = {key: value for key, value in os.environ.items() if not
               (key.endswith("_KEY") or key.endswith("_TOKEN") or "SECRET" in key)}
        env.update(OPENAI_API_KEY="week1-local-gateway", PYTHONUNBUFFERED="1",
                   NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        if sys.platform == "darwin":
            env["DYLD_LIBRARY_PATH"] = str(Path(root) / "data/week1/envs/memorybank/lib/python3.11/site-packages/torch/lib")
        self.process = subprocess.Popen([str(Path(root) / "data/week1/envs/memorybank/bin/python"),
            "-m", "src.week1_memorybank", "worker", "--project", str(root), "--path", str(path),
            "--gateway-url", gateway_url], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, env=env)
        try:
            self.call({"action": "init"}, timeout=180)
        except Exception:
            self.process.kill()
            self.process.wait()
            self.log.close()
            raise

    def call(self, request, timeout=14400):
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError("MemoryBank worker timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("MemoryBank worker exited; see worker.log")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "MemoryBank worker failed"))
        return response.get("result")

    def ingest(self, example):
        sessions = [{"id": s["id"], "messages": [{"role": m["role"], "content": m["content"]}
                     for m in s["messages"]]} for s in example["sessions"]]
        return self.call({"action": "ingest", "sessions": sessions, "group": example["group"],
                          "question_date": example.get("question_date")})

    def search(self, question):
        return self.call({"action": "search", "query": question})

    def close(self):
        try:
            if self.process.poll() is None:
                self.call({"action": "close"}, timeout=15)
                self.process.wait(timeout=30)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


def serve(project, path, gateway_url):
    worker = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            action = request["action"]
            with contextlib.redirect_stdout(sys.stderr):
                if action == "init":
                    if worker is not None:
                        raise ValueError("MemoryBank worker already initialized")
                    worker = MemoryBankWorker(project, path, gateway_url)
                    result = {"commit": COMMIT}
                elif worker is None:
                    raise ValueError("Initialize MemoryBank before use")
                elif action == "ingest":
                    result = worker.ingest(request)
                elif action == "search":
                    result = worker.search(request["query"])
                elif action == "close":
                    worker.close()
                    result = None
                else:
                    raise ValueError("Unknown MemoryBank action")
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)
        if request.get("action") == "close":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify", "worker"])
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--path", type=Path)
    parser.add_argument("--gateway-url")
    args = parser.parse_args()
    if args.command == "worker":
        serve(args.project, args.path, args.gateway_url)
    else:
        print(json.dumps(prepare(args.project) if args.command == "prepare" else verify(args.project), indent=2))
