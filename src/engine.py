import hashlib
import json
import re
import threading
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken

from .config import ServiceConfig, index_fingerprint, resolved_path
from .contracts import AddRequest, AddResponse, Evidence, SearchRequest, SearchResponse
from .directory_memory import DirectoryMemory
from .providers import Embedder, LLM, OpenAIDecision
from .storage import Store


def stable_id(*parts: object) -> str:
    # 所有消息、chunk、事实和证据窗口都使用确定性 ID，保证同一输入重启后仍能定位到同一条记录。
    data = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return "mem_" + hashlib.sha256(data.encode()).hexdigest()[:32]


def normalized(vector: list[float], dimension: int) -> np.ndarray:
    # 归一化后，两个向量的点积就是 cosine similarity；后面的向量检索只需要做矩阵乘法。
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid embedding dimension or value")
    length = np.linalg.norm(result)
    if length == 0:
        raise ValueError("zero embedding")
    return result / length


class MemoryEngine:
    """记忆系统的核心：负责写入记忆、建立索引和返回可追溯的原文证据。"""

    def __init__(self, config: ServiceConfig, config_path: Path, llm: LLM, embedder: Embedder, decision_model=None, directory_selector=None):
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.decision_model = decision_model
        self.directory_selector = directory_selector
        self.tokenizer = tiktoken.get_encoding(config.chunk.tokenizer)
        # 事实提炼的规则来自外部 prompt；prompt、分块配置和 embedding 配置会参与索引指纹。
        self.prompt = resolved_path(config_path, config.facts.prompt_path).read_text(encoding="utf-8")
        self.consolidation_prompt = ""
        decision_rules = ""
        temporal_rules = hashlib.sha256(Path(__file__).with_name("temporal.py").read_bytes()).hexdigest() if config.facts.temporal else ""
        if config.consolidation.mode != "off":
            if decision_model is None:
                raise ValueError("enabled consolidation requires a decision model")
            self.consolidation_prompt = resolved_path(config_path, config.consolidation.prompt_path).read_text(encoding="utf-8")
            decision_rules = hashlib.sha256(Path(__file__).with_name("consolidation.py").read_bytes()
                                            + Path(__file__).with_name("providers.py").read_bytes()
                                            + OpenAIDecision.INSTRUCTIONS.encode()).hexdigest()
        directory_rules = ""
        if config.directory_selector is not None:
            if directory_selector is None:
                raise ValueError("directory memory requires a directory selector")
            directory_rules = hashlib.sha256(Path(__file__).with_name("directory_memory.py").read_bytes()
                                             + Path(__file__).with_name("directory_selector.py").read_bytes()).hexdigest()
        self.store = Store(resolved_path(config_path, config.database_dir),
                           index_fingerprint(config, self.prompt, self.consolidation_prompt, decision_rules,
                                             temporal_rules, directory_rules))
        self.directories = DirectoryMemory(self.store.root, self.tokenizer) if config.directory_selector else None
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock(self, user_id: str) -> threading.Lock:
        # 每个 user 串行写入，避免同一用户的 seq 分配和 SQLite 事务互相覆盖。
        with self._locks_guard:
            return self._locks.setdefault(user_id, threading.Lock())

    def _chunks(self, request: AddRequest, first_seq: int) -> tuple[list[dict], list[dict]]:
        """把消息切成带来源位置的重叠 token chunk，并保留 session 顺序。"""
        messages, chunks = [], []
        step = self.config.chunk.max_tokens - self.config.chunk.overlap_tokens
        for offset, message in enumerate(request.messages):
            seq = first_seq + offset
            message_id = stable_id(request.user_id, request.session_id, seq, message.role, message.content)
            messages.append({"id": message_id, "session_id": request.session_id, "seq": seq, "role": message.role, "content": message.content, "timestamp": message.timestamp})
            tokens = self.tokenizer.encode(message.content)
            for chunk_seq, start in enumerate(range(0, len(tokens), step)):
                # 例如 max_tokens=512、overlap_tokens=64 时，窗口每次前进 448 token。
                end = min(start + self.config.chunk.max_tokens, len(tokens))
                content = self.tokenizer.decode(tokens[start:end])
                chunks.append({"id": stable_id(request.user_id, request.session_id, seq, chunk_seq, content), "message_id": message_id,
                               "session_id": request.session_id, "message_seq": seq, "chunk_seq": chunk_seq,
                               "token_start": start, "token_end": end, "content": content, "timestamp": message.timestamp})
                if end == len(tokens):
                    break
        return messages, chunks

    def add(self, request: AddRequest) -> AddResponse:
        """写入一批新消息，并同时建立原文索引、事实索引和向量索引。

        写入顺序是：历史上下文 -> 新消息分块 -> LLM 提炼事实 -> embedding -> SQLite 原子事务。
        LLM 或 embedding 任一步失败时，事务尚未提交，因此不会留下半成品记忆。
        """
        payload_hash = hashlib.sha256(request.model_dump_json(exclude_none=False).encode()).hexdigest()
        with self._lock(request.user_id), self.store.connect(request.user_id) as db:
            # request_id + payload_hash 实现幂等：同请求直接返回旧响应；同 ID 不同内容则报错。
            previous = self.store.existing_request(db, request.request_id, payload_hash)
            if previous is not None:
                return AddResponse.model_validate(previous)
            first_seq = self.store.next_seq(db, request.session_id)

            # 只把当前 session 最近的历史消息交给事实模型，用于解析“他/那里/之前”等指代。
            # 历史消息不会被当成本次新增事实的来源，事实仍必须引用本次新 chunk。
            prior = self.store.prior_messages(db, request.session_id)
            if self.directories:
                identities = [row[0] for row in db.execute(
                    "SELECT id FROM messages WHERE session_id=? ORDER BY seq", (request.session_id,))]
                sources = self.directories.messages(db, request.user_id, identities)
                prior = [{key: sources[identity][key] for key in ("role", "content", "timestamp")}
                         for identity in identities]
            used = 0
            context = []
            for item in reversed(prior):
                size = len(self.tokenizer.encode(item["content"]))
                if used + size > self.config.facts.prior_context_tokens:
                    break
                used += size
                context.append(item if self.config.facts.temporal else {key: item[key] for key in ("role", "content")})
            context.reverse()
            messages, chunks = self._chunks(request, first_seq)

            # Provider 将模型引用的整数索引映射为 source_chunk_ids，交给这里校验。
            if self.config.facts.temporal:
                from .temporal import validate_temporal

                roles = {message["id"]: message["role"] for message in messages}
                inputs = [{"id": c["id"], "content": c["content"], "role": roles[c["message_id"]],
                           "timestamp": c["timestamp"], "source_order": i} for i, c in enumerate(chunks)]
                raw_facts = self.llm.extract(self.prompt, context, inputs, temporal=True)
            else:
                raw_facts = self.llm.extract(self.prompt, context, [{"id": c["id"], "content": c["content"]} for c in chunks])
            chunk_ids = {c["id"] for c in chunks}
            chunk_order = {c["id"]: i for i, c in enumerate(chunks)}
            chunk_timestamps = {c["id"]: c["timestamp"] for c in chunks}
            facts = []
            fact_ids = set()
            for fact in raw_facts:
                # 拒绝无内容、无来源或引用旧 chunk 的事实，避免事实索引脱离原文证据。
                if not isinstance(fact, dict) or not isinstance(fact.get("content"), str) or not fact["content"].strip():
                    raise ValueError("invalid extracted fact")
                sources = fact.get("source_chunk_ids")
                if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or s not in chunk_ids for s in sources):
                    raise ValueError("fact must cite at least one new chunk")
                temporal = validate_temporal(fact) if self.config.facts.temporal else {
                    "occurred_start": None, "occurred_end": None, "occurred_precision": None, "mentioned_at": None}
                if self.config.facts.temporal and all(chunk_timestamps[s] is not None for s in sources):
                    temporal["mentioned_at"] = datetime.fromtimestamp(max(chunk_timestamps[s] for s in sources) / 1000,
                                                                     timezone.utc).isoformat().replace("+00:00", "Z")
                identity_parts = [request.request_id, fact["content"], sorted(set(sources))]
                if self.config.facts.temporal:
                    identity_parts.append(temporal)
                fact_id = stable_id(*identity_parts)
                if fact_id not in fact_ids:
                    facts.append({"id": fact_id, "content": fact["content"], "sources": sorted(set(sources)), **temporal})
                    fact_ids.add(fact_id)
            if self.config.facts.temporal:
                facts.sort(key=lambda fact: min(chunk_order[source] for source in fact["sources"]))

            # 原文 chunk 和提炼 fact 共用一次 embedding 调用，但在数据库中用 owner_type 区分。
            texts = [c["content"] for c in chunks] + [f["content"] for f in facts]
            directory = self.directories.build(request, messages, facts) if self.directories else None
            if directory:
                texts.append(directory["overview"])
            vectors = [normalized(v, self.config.embedding.dimensions) for v in self.embedder.embed(texts)]
            if len(vectors) != len(texts):
                raise ValueError("embedding count does not match text count")
            observations, decisions, observation_vectors = [], [], []
            if self.config.consolidation.mode != "off" and facts:
                from .consolidation import consolidate

                existing = self.store.current_observations(db)
                for observation in existing:
                    observation["vector"] = np.frombuffer(observation["vector"], dtype=np.float32)
                observations, decisions = consolidate(facts, vectors[len(chunks):len(chunks) + len(facts)], existing, self.decision_model,
                                                     self.llm, self.consolidation_prompt, request.request_id)
                observation_vectors = [normalized(v, self.config.embedding.dimensions)
                                       for v in self.embedder.embed([item["content"] for item in observations])]
                if len(observation_vectors) != len(observations):
                    raise ValueError("observation embedding count mismatch")
            recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            response = AddResponse(request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)
            with self.directories.publish(directory, db) if directory else nullcontext():
                self._commit_add(db, request, response, payload_hash, messages, chunks, facts, vectors,
                                 observations, decisions, observation_vectors, recorded_at, directory)
            return response

    def _commit_add(self, db, request, response, payload_hash, messages, chunks, facts, vectors,
                    observations, decisions, observation_vectors, recorded_at, directory):
        try:
            # 只有所有外部模型调用成功后才开始事务；下面的表必须整体提交。
            db.execute("BEGIN IMMEDIATE")
            db.executemany("INSERT INTO messages VALUES (:id,:session_id,:seq,:role,:content,:timestamp)",
                           [{**message, "content": ""} for message in messages] if directory else messages)
            db.executemany("INSERT INTO chunks VALUES (:id,:message_id,:session_id,:message_seq,:chunk_seq,:token_start,:token_end,:content,:timestamp)",
                           [{**chunk, "content": ""} for chunk in chunks] if directory else chunks)
            db.executemany("INSERT INTO chunks_fts(content,chunk_id) VALUES (?,?)", [(c["content"], c["id"]) for c in chunks])
            db.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?)",
                [(f["id"], f["content"], self.config.llm.name, f["occurred_start"], f["occurred_end"],
                  f["occurred_precision"], f["mentioned_at"], recorded_at) for f in facts])
            db.executemany("INSERT INTO fact_sources VALUES (?,?)", [(f["id"], s) for f in facts for s in f["sources"]])
            db.executemany("INSERT INTO embeddings VALUES (?,?,?,?,?)", [
                # vectors 的前 len(chunks) 个属于原文，剩余部分属于事实。
                ("chunk" if i < len(chunks) else "fact", (chunks + facts)[i]["id"], self.config.embedding.name,
                 self.config.embedding.dimensions, vectors[i].tobytes()) for i in range(len(chunks) + len(facts))
            ])
            self.store.save_observations(db, observations, decisions, request.request_id, recorded_at)
            db.executemany("""INSERT INTO embeddings VALUES ('observation',?,?,?,?)
                ON CONFLICT(owner_type,owner_id) DO UPDATE SET
                model=excluded.model,dimension=excluded.dimension,vector=excluded.vector""",
                [(observation["id"], self.config.embedding.name, self.config.embedding.dimensions, vector.tobytes())
                 for observation, vector in zip(observations, observation_vectors)])
            if directory:
                db.execute("""INSERT INTO memory_directories VALUES
                    (:id,:path,:parent_path,:abstract,:overview,:source_sha256)""", directory)
                db.executemany("INSERT INTO message_sources VALUES (?,?)",
                               [(message["id"], directory["id"]) for message in messages])
                db.execute("INSERT INTO embeddings VALUES ('directory',?,?,?,?)",
                           (directory["id"], self.config.embedding.name,
                            self.config.embedding.dimensions, vectors[-1].tobytes()))
            db.execute("INSERT INTO requests VALUES (?,?,?)", (request.request_id, payload_hash, response.model_dump_json()))
            db.commit()
        except BaseException:
            db.rollback()
            raise

    def _select_directories(self, db, request, ranked_chunks):
        mapping = {row["id"]: row["directory_id"] for row in db.execute("""
            SELECT c.id, s.directory_id FROM chunks c JOIN message_sources s ON s.message_id=c.message_id""")}
        candidates = []
        seen = set()
        for chunk_id in ranked_chunks:
            directory_id = mapping[chunk_id]
            if directory_id not in seen:
                candidates.append(dict(db.execute("SELECT id,path,abstract,overview FROM memory_directories WHERE id=?",
                                                  (directory_id,)).fetchone()))
                seen.add(directory_id)
            if len(candidates) == 10:
                break
        decisions = self.directory_selector.select(request.query, request.options, candidates)
        if (len(decisions) != len(candidates) or {item["id"] for item in decisions} != seen
                or any(item["choice"] not in {"relevant", "irrelevant", "uncertain"} for item in decisions)):
            raise ValueError("directory selection returned invalid candidate decisions")
        accepted = {item["id"] for item in decisions if item["choice"] != "irrelevant"}
        return {chunk_id for chunk_id, directory_id in mapping.items() if directory_id in accepted}

    def _vector_rank(self, db, query_vector: np.ndarray, owner_type: str, limit: int) -> list[str]:
        """按 cosine similarity 对 chunk 或 fact 做向量召回，只返回排名后的 ID。"""
        rows = db.execute("SELECT owner_id,vector FROM embeddings WHERE owner_type=?", (owner_type,)).fetchall()
        if not rows:
            return []
        matrix = np.frombuffer(b"".join(row["vector"] for row in rows), dtype=np.float32).reshape(len(rows), self.config.embedding.dimensions)
        scores = matrix @ query_vector
        positions = np.argsort(-scores, kind="stable")[:limit]
        return [rows[int(i)]["owner_id"] for i in positions]

    @staticmethod
    def _bm25_rank(db, query: str, limit: int) -> list[str]:
        """使用 SQLite FTS5 对原始 chunk 做 lexical/BM25 召回。"""
        words = re.findall(r"\w+", query, flags=re.UNICODE)
        if not words:
            return []
        expression = " OR ".join('"' + word.replace('"', '') + '"' for word in words)
        rows = db.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (expression, limit)).fetchall()
        return [row[0] for row in rows]

    def search(self, request: SearchRequest) -> SearchResponse:
        """混合检索并返回原文证据，而不是直接返回 LLM 提炼的事实。

        具体流程：多路召回 -> 加权 RRF 融合 -> 选 seed -> 恢复 session 邻居 ->
        合并连续窗口 -> 按 token 预算截断。
        """
        if request.top_k < 1:
            return SearchResponse(data=[])
        with self.store.connect(request.user_id, create=False) as db:
            if db is None:
                return SearchResponse(data=[])
            # 搜索使用 SQLite 读事务，保证一次 search 看到一致的索引快照。
            db.execute("BEGIN")
            try:
                if db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is None:
                    return SearchResponse(data=[])
                retrieval = self.config.retrieval

                # 基础 query 必须参与；如果请求带 options，则把“问题+选项”作为第二个检索视角。
                query_specs = [(request.query, retrieval.question_weight)]
                if request.options and retrieval.options_weight > 0:
                    query_specs.append((request.query + "\n" + "\n".join(request.options), retrieval.options_weight))
                query_vectors = [normalized(v, self.config.embedding.dimensions) for v in self.embedder.embed([q for q, _ in query_specs])]
                if len(query_vectors) != len(query_specs):
                    raise ValueError("query embedding count mismatch")
                scores: dict[str, float] = defaultdict(float)
                for (query, query_weight), vector in zip(query_specs, query_vectors):
                    routes = []
                    # 每条 route 各自返回一个排名列表；route 的原始分数不会跨算法直接相加。
                    if retrieval.bm25_weight:
                        routes.append((list(enumerate(self._bm25_rank(db, query, retrieval.bm25_candidates), 1)), retrieval.bm25_weight))
                    if retrieval.chunk_vector_weight:
                        routes.append((list(enumerate(self._vector_rank(db, vector, "chunk", retrieval.chunk_vector_candidates), 1)), retrieval.chunk_vector_weight))
                    if retrieval.fact_vector_weight:
                        # fact 只用于找到相关内容，随后通过 fact_sources 映射回可追溯的原文 chunk。
                        fact_ids = self._vector_rank(db, vector, "fact", retrieval.fact_vector_candidates)
                        mapped = []
                        for rank, fact_id in enumerate(fact_ids, 1):
                            mapped.extend((rank, row[0]) for row in db.execute("SELECT chunk_id FROM fact_sources WHERE fact_id=? ORDER BY chunk_id", (fact_id,)))
                        routes.append((mapped, retrieval.fact_vector_weight))
                    if self.config.consolidation.mode != "off":
                        observation_ids = self._vector_rank(db, vector, "observation", 50)
                        mapped = []
                        for rank, observation_id in enumerate(observation_ids, 1):
                            rows = db.execute("""SELECT DISTINCT fs.chunk_id FROM observations o
                                JOIN observation_sources os ON os.observation_id=o.id AND os.revision=o.current_revision
                                JOIN fact_sources fs ON fs.fact_id=os.fact_id
                                WHERE o.id=? ORDER BY fs.chunk_id""", (observation_id,))
                            mapped.extend((rank, row[0]) for row in rows)
                        routes.append((mapped, 1.0))
                    if self.directories:
                        directory_ids = self._vector_rank(db, vector, "directory", 50)
                        mapped = []
                        for rank, directory_id in enumerate(directory_ids, 1):
                            rows = db.execute("""SELECT c.id FROM chunks c
                                JOIN message_sources s ON s.message_id=c.message_id
                                WHERE s.directory_id=? ORDER BY c.message_seq,c.chunk_seq""", (directory_id,))
                            mapped.extend((rank, row[0]) for row in rows)
                        routes.append((mapped, 1.0))
                    for ranked_ids, route_weight in routes:
                        seen = set()
                        for rank, chunk_id in ranked_ids:
                            if chunk_id not in seen:
                                # 加权 Reciprocal Rank Fusion：排名越靠前，贡献越大；seen 防止同一路重复计分。
                                scores[chunk_id] += query_weight * route_weight / (retrieval.rrf_k + rank)
                                seen.add(chunk_id)
                if not scores:
                    return SearchResponse(data=[])

                # 先选少量高分 seed，再以 seed 为中心恢复上下文，避免每个召回 chunk 都单独返回。
                ranked = sorted(scores, key=lambda key: (-scores[key], key))
                allowed = self._select_directories(db, request, ranked) if self.directories else None
                seeds = [key for key in ranked if allowed is None or key in allowed][:retrieval.seed_count]
                if not seeds:
                    return SearchResponse(data=[])
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
                    # neighbor_count=1 时，同时取 seed 前后各一个 chunk；B0/B1 为 0，B2/B3 为 1。
                    selected[row["session_id"]].update(
                        i for i in range(max(0, position - retrieval.neighbor_count), min(len(session), position + retrieval.neighbor_count + 1))
                        if allowed is None or session[i]["id"] in allowed)
                hydrated = {}
                if self.directories:
                    rows = [sessions[session_id][i] for session_id, positions in selected.items() for i in positions]
                    hydrated = {row["id"]: row for row in self.directories.hydrate_chunks(db, request.user_id, rows)}
                windows = []
                for session_id, positions in selected.items():
                    ordered = sorted(positions)
                    groups = []
                    for position in ordered:
                        # 相邻位置合并为一个窗口；不连续的位置拆成多个证据窗口。
                        if not groups or position > groups[-1][-1] + 1:
                            groups.append([])
                        groups[-1].append(position)
                    for group in groups:
                        rows = [hydrated.get(sessions[session_id][i]["id"], sessions[session_id][i]) for i in group]
                        content = "\n".join(row["content"] for row in rows)
                        score = max(scores.get(row["id"], 0) for row in rows)
                        timestamp = next((row["timestamp"] for row in rows if row["timestamp"] is not None), None)
                        created_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat().replace("+00:00", "Z") if timestamp is not None else None
                        windows.append(Evidence(id=stable_id(*[row["id"] for row in rows]), content=content, score=score, created_at=created_at))
                windows.sort(key=lambda item: (-item.score, item.id))

                # 最终返回原始 chunk 拼成的 evidence，并同时受 token 预算和 top_k/max_results 限制。
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
