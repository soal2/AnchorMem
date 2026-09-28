import hashlib
import json
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken

from .config import ServiceConfig, index_fingerprint, resolved_path
from .contracts import AddRequest, AddResponse, Evidence, SearchRequest, SearchResponse
from .providers import Embedder, LLM
from .storage import Store


def stable_id(*parts: object) -> str:
    data = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return "mem_" + hashlib.sha256(data.encode()).hexdigest()[:32]


def normalized(vector: list[float], dimension: int) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid embedding dimension or value")
    length = np.linalg.norm(result)
    if length == 0:
        raise ValueError("zero embedding")
    return result / length


class MemoryEngine:
    def __init__(self, config: ServiceConfig, config_path: Path, llm: LLM, embedder: Embedder):
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.tokenizer = tiktoken.get_encoding(config.chunk.tokenizer)
        self.prompt = resolved_path(config_path, config.facts.prompt_path).read_text(encoding="utf-8")
        self.store = Store(resolved_path(config_path, config.database_dir), index_fingerprint(config, self.prompt))
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock(self, user_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(user_id, threading.Lock())

    def _chunks(self, request: AddRequest, first_seq: int) -> tuple[list[dict], list[dict]]:
        messages, chunks = [], []
        step = self.config.chunk.max_tokens - self.config.chunk.overlap_tokens
        for offset, message in enumerate(request.messages):
            seq = first_seq + offset
            message_id = stable_id(request.user_id, request.session_id, seq, message.role, message.content)
            messages.append({"id": message_id, "session_id": request.session_id, "seq": seq, "role": message.role, "content": message.content, "timestamp": message.timestamp})
            tokens = self.tokenizer.encode(message.content)
            for chunk_seq, start in enumerate(range(0, len(tokens), step)):
                end = min(start + self.config.chunk.max_tokens, len(tokens))
                content = self.tokenizer.decode(tokens[start:end])
                chunks.append({"id": stable_id(request.user_id, request.session_id, seq, chunk_seq, content), "message_id": message_id,
                               "session_id": request.session_id, "message_seq": seq, "chunk_seq": chunk_seq,
                               "token_start": start, "token_end": end, "content": content, "timestamp": message.timestamp})
                if end == len(tokens):
                    break
        return messages, chunks

    def add(self, request: AddRequest) -> AddResponse:
        payload_hash = hashlib.sha256(request.model_dump_json(exclude_none=False).encode()).hexdigest()
        with self._lock(request.user_id), self.store.connect(request.user_id) as db:
            previous = self.store.existing_request(db, request.request_id, payload_hash)
            if previous is not None:
                return AddResponse.model_validate(previous)
            first_seq = self.store.next_seq(db, request.session_id)
            prior = self.store.prior_messages(db, request.session_id)
            used = 0
            context = []
            for item in reversed(prior):
                size = len(self.tokenizer.encode(item["content"]))
                if used + size > self.config.facts.prior_context_tokens:
                    break
                used += size
                context.append(item)
            context.reverse()
            messages, chunks = self._chunks(request, first_seq)
            raw_facts = self.llm.extract(self.prompt, context, [{"id": c["id"], "content": c["content"]} for c in chunks])
            chunk_ids = {c["id"] for c in chunks}
            facts = []
            fact_ids = set()
            for fact in raw_facts:
                if not isinstance(fact, dict) or not isinstance(fact.get("content"), str) or not fact["content"].strip():
                    raise ValueError("invalid extracted fact")
                sources = fact.get("source_chunk_ids")
                if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or s not in chunk_ids for s in sources):
                    raise ValueError("fact must cite at least one new chunk")
                fact_id = stable_id(request.request_id, fact["content"], sorted(set(sources)))
                if fact_id not in fact_ids:
                    facts.append({"id": fact_id, "content": fact["content"], "sources": sorted(set(sources))})
                    fact_ids.add(fact_id)
            texts = [c["content"] for c in chunks] + [f["content"] for f in facts]
            vectors = [normalized(v, self.config.embedding.dimensions) for v in self.embedder.embed(texts)]
            if len(vectors) != len(texts):
                raise ValueError("embedding count does not match text count")
            response = AddResponse(request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)
            try:
                db.execute("BEGIN IMMEDIATE")
                db.executemany("INSERT INTO messages VALUES (:id,:session_id,:seq,:role,:content,:timestamp)", messages)
                db.executemany("INSERT INTO chunks VALUES (:id,:message_id,:session_id,:message_seq,:chunk_seq,:token_start,:token_end,:content,:timestamp)", chunks)
                db.executemany("INSERT INTO chunks_fts(content,chunk_id) VALUES (?,?)", [(c["content"], c["id"]) for c in chunks])
                db.executemany("INSERT INTO facts VALUES (?,?,?)", [(f["id"], f["content"], self.config.llm.name) for f in facts])
                db.executemany("INSERT INTO fact_sources VALUES (?,?)", [(f["id"], s) for f in facts for s in f["sources"]])
                db.executemany("INSERT INTO embeddings VALUES (?,?,?,?,?)", [
                    ("chunk" if i < len(chunks) else "fact", (chunks + facts)[i]["id"], self.config.embedding.name,
                     self.config.embedding.dimensions, vectors[i].tobytes()) for i in range(len(vectors))
                ])
                db.execute("INSERT INTO requests VALUES (?,?,?)", (request.request_id, payload_hash, response.model_dump_json()))
                db.commit()
            except Exception:
                db.rollback()
                raise
            return response

    def _vector_rank(self, db, query_vector: np.ndarray, owner_type: str, limit: int) -> list[str]:
        rows = db.execute("SELECT owner_id,vector FROM embeddings WHERE owner_type=?", (owner_type,)).fetchall()
        if not rows:
            return []
        matrix = np.frombuffer(b"".join(row["vector"] for row in rows), dtype=np.float32).reshape(len(rows), self.config.embedding.dimensions)
        scores = matrix @ query_vector
        positions = np.argsort(-scores, kind="stable")[:limit]
        return [rows[int(i)]["owner_id"] for i in positions]

    @staticmethod
    def _bm25_rank(db, query: str, limit: int) -> list[str]:
        words = re.findall(r"\w+", query, flags=re.UNICODE)
        if not words:
            return []
        expression = " OR ".join('"' + word.replace('"', '') + '"' for word in words)
        rows = db.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (expression, limit)).fetchall()
        return [row[0] for row in rows]

    def search(self, request: SearchRequest) -> SearchResponse:
        if request.top_k < 1:
            return SearchResponse(data=[])
        with self.store.connect(request.user_id, create=False) as db:
            if db is None:
                return SearchResponse(data=[])
            db.execute("BEGIN")
            try:
                if db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is None:
                    return SearchResponse(data=[])
                retrieval = self.config.retrieval
                query_specs = [(request.query, retrieval.question_weight)]
                if request.options and retrieval.options_weight > 0:
                    query_specs.append((request.query + "\n" + "\n".join(request.options), retrieval.options_weight))
                query_vectors = [normalized(v, self.config.embedding.dimensions) for v in self.embedder.embed([q for q, _ in query_specs])]
                if len(query_vectors) != len(query_specs):
                    raise ValueError("query embedding count mismatch")
                scores: dict[str, float] = defaultdict(float)
                for (query, query_weight), vector in zip(query_specs, query_vectors):
                    routes = []
                    if retrieval.bm25_weight:
                        routes.append((list(enumerate(self._bm25_rank(db, query, retrieval.bm25_candidates), 1)), retrieval.bm25_weight))
                    if retrieval.chunk_vector_weight:
                        routes.append((list(enumerate(self._vector_rank(db, vector, "chunk", retrieval.chunk_vector_candidates), 1)), retrieval.chunk_vector_weight))
                    if retrieval.fact_vector_weight:
                        fact_ids = self._vector_rank(db, vector, "fact", retrieval.fact_vector_candidates)
                        mapped = []
                        for rank, fact_id in enumerate(fact_ids, 1):
                            mapped.extend((rank, row[0]) for row in db.execute("SELECT chunk_id FROM fact_sources WHERE fact_id=? ORDER BY chunk_id", (fact_id,)))
                        routes.append((mapped, retrieval.fact_vector_weight))
                    for ranked_ids, route_weight in routes:
                        seen = set()
                        for rank, chunk_id in ranked_ids:
                            if chunk_id not in seen:
                                scores[chunk_id] += query_weight * route_weight / (retrieval.rrf_k + rank)
                                seen.add(chunk_id)
                if not scores:
                    return SearchResponse(data=[])
                seeds = sorted(scores, key=lambda key: (-scores[key], key))[:retrieval.seed_count]
                all_rows = db.execute("SELECT * FROM chunks ORDER BY session_id,message_seq,chunk_seq").fetchall()
                by_id = {row["id"]: row for row in all_rows}
                sessions: dict[str, list] = defaultdict(list)
                for row in all_rows:
                    sessions[row["session_id"]].append(row)
                selected: dict[str, set[int]] = defaultdict(set)
                for seed in seeds:
                    row = by_id[seed]
                    session = sessions[row["session_id"]]
                    position = next(i for i, item in enumerate(session) if item["id"] == seed)
                    selected[row["session_id"]].update(range(max(0, position - retrieval.neighbor_count), min(len(session), position + retrieval.neighbor_count + 1)))
                windows = []
                for session_id, positions in selected.items():
                    ordered = sorted(positions)
                    groups = []
                    for position in ordered:
                        if not groups or position > groups[-1][-1] + 1:
                            groups.append([])
                        groups[-1].append(position)
                    for group in groups:
                        rows = [sessions[session_id][i] for i in group]
                        content = "\n".join(row["content"] for row in rows)
                        score = max(scores.get(row["id"], 0) for row in rows)
                        timestamp = next((row["timestamp"] for row in rows if row["timestamp"] is not None), None)
                        created_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat().replace("+00:00", "Z") if timestamp is not None else None
                        windows.append(Evidence(id=stable_id(*[row["id"] for row in rows]), content=content, score=score, created_at=created_at))
                windows.sort(key=lambda item: (-item.score, item.id))
                result, used = [], 0
                for item in windows:
                    size = len(self.tokenizer.encode(item.content))
                    if used + size > retrieval.evidence_budget_tokens:
                        continue
                    result.append(item)
                    used += size
                    if len(result) >= min(request.top_k, retrieval.max_results):
                        break
                return SearchResponse(data=result)
            finally:
                db.rollback()
