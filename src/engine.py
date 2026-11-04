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
from .topics import TOPICS, candidates as topic_candidates, selected_topics
from .providers import Embedder, LLM, OpenAIDecision
from .storage import Store
from .temporal import event_overlaps, validate_temporal


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
    """Synchronously store scoped entries and return their original source excerpts."""

    def __init__(self, config: ServiceConfig, config_path: Path, llm: LLM, embedder: Embedder,
                 decision_model=None, directory_selector=None):
        self.config, self.llm, self.embedder = config, llm, embedder
        self.decision_model, self.directory_selector = decision_model, directory_selector
        self.tokenizer = tiktoken.get_encoding(config.chunk.tokenizer)
        self.prompt = resolved_path(config_path, config.facts.prompt_path).read_text(encoding="utf-8")
        self.consolidation_prompt = ""
        decision_rules = ""
        if config.consolidation.mode != "off":
            if decision_model is None:
                raise ValueError("enabled consolidation requires a decision model")
            self.consolidation_prompt = resolved_path(config_path, config.consolidation.prompt_path).read_text(encoding="utf-8")
            decision_rules = hashlib.sha256(Path(__file__).with_name("consolidation.py").read_bytes()
                                            + Path(__file__).with_name("providers.py").read_bytes()
                                            + OpenAIDecision.INSTRUCTIONS.encode()).hexdigest()
        if config.directory_selector is not None and directory_selector is None:
            raise ValueError("directory memory requires a directory selector")
        temporal_rules = hashlib.sha256(Path(__file__).with_name("temporal.py").read_bytes()).hexdigest()
        directory_rules = hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in
            ("engine.py", "contracts.py", "topics.py", "directory_memory.py", "directory_selector.py", "source_spans.py"))).hexdigest()
        self.store = Store(resolved_path(config_path, config.database_dir), index_fingerprint(
            config, self.prompt, self.consolidation_prompt, decision_rules, temporal_rules, directory_rules))
        self.directories = DirectoryMemory(self.store.root, self.tokenizer) if config.directory_selector else None
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock(self, user_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(user_id, threading.Lock())

    def _chunks(self, request: AddRequest, first_seq: int) -> tuple[list[dict], list[dict]]:
        """Token-sized extraction windows retain exact Unicode character boundaries."""
        messages, chunks = [], []
        step = self.config.chunk.max_tokens - self.config.chunk.overlap_tokens
        for offset, message in enumerate(request.messages):
            seq = first_seq + offset
            message_id = stable_id(request.user_id, request.session_id, seq, message.role, message.content)
            messages.append({"id": message_id, "session_id": request.session_id, "seq": seq,
                             "role": message.role, "content": message.content, "timestamp": message.timestamp})
            tokens = self.tokenizer.encode(message.content)
            _, offsets = self.tokenizer.decode_with_offsets(tokens)
            for chunk_seq, start in enumerate(range(0, len(tokens), step)):
                end = min(start + self.config.chunk.max_tokens, len(tokens))
                char_start, char_end = offsets[start], offsets[end] if end < len(tokens) else len(message.content)
                if char_end > char_start:
                    content = message.content[char_start:char_end]
                    chunks.append({"id": stable_id(message_id, chunk_seq, content), "message_id": message_id,
                                   "session_id": request.session_id, "message_seq": seq, "chunk_seq": chunk_seq,
                                   "char_start": char_start, "char_end": char_end, "content": content,
                                   "timestamp": message.timestamp, "role": message.role, "source_order": len(chunks)})
                if end == len(tokens):
                    break
        return messages, chunks

    def _entries(self, request, raw_facts, extraction_chunks):
        """Resolve exact quotes to excerpts; model-produced project IDs are never used."""
        if not isinstance(raw_facts, list):
            raise ValueError("extraction must return a facts list")
        by_id = {chunk["id"]: chunk for chunk in extraction_chunks}
        facts, chunks, identities = [], {}, []
        metadata_fields = ("topic", "scope", "parent_index", "event_status", "conditions", "verification")
        for index, raw in enumerate(raw_facts):
            if (not isinstance(raw, dict) or not isinstance(raw.get("content"), str) or not raw["content"].strip()
                    or any(key not in raw for key in metadata_fields) or not isinstance(raw["topic"], str)
                    or raw["topic"] not in TOPICS
                    or raw["scope"] not in ("default", "global", "project")):
                raise ValueError("invalid extracted entry metadata")
            parent = raw["parent_index"]
            if parent is not None and (type(parent) is not int or parent < 0 or parent >= len(raw_facts) or parent == index):
                raise ValueError("invalid parent entry reference")
            if raw["event_status"] not in (None, "planned", "in_progress", "completed", "cancelled", "unknown"):
                raise ValueError("invalid event status")
            if any(raw[key] is not None and (not isinstance(raw[key], str) or not raw[key].strip())
                   for key in ("conditions", "verification")):
                raise ValueError("invalid method conditions or verification")
            topic = raw["topic"]
            scope = "global" if topic in ("profile", "preferences") or raw["scope"] == "global" else (
                "project" if request.project_id is not None else "unresolved" if raw["scope"] == "project" else "global")
            project_id = request.project_id if scope == "project" else None
            quotes = raw.get("source_quotes")
            sources = raw.get("source_chunk_ids")
            if (not isinstance(quotes, list) or not quotes or not isinstance(sources, list) or not sources
                    or any(not isinstance(source, str) or source not in by_id for source in sources)):
                raise ValueError("entry must cite at least one new chunk with exact quotes")
            excerpt_ids, original_ids = [], set()
            for quote in quotes:
                if (not isinstance(quote, dict) or set(quote) != {"chunk_id", "quote", "occurrence"}
                        or not isinstance(quote["chunk_id"], str) or quote["chunk_id"] not in by_id
                        or not isinstance(quote["quote"], str)
                        or not quote["quote"].strip() or type(quote["occurrence"]) is not int or quote["occurrence"] < 0):
                    raise ValueError("invalid source quote")
                original = by_id[quote["chunk_id"]]
                position = -1
                for _ in range(quote["occurrence"] + 1):
                    position = original["content"].find(quote["quote"], 0 if position < 0 else position + len(quote["quote"]))
                    if position < 0:
                        raise ValueError("source quote is not present in cited chunk")
                begin = original["char_start"] + position
                finish = begin + len(quote["quote"])
                excerpt_id = stable_id(original["message_id"], begin, finish, topic, scope, project_id)
                chunks[excerpt_id] = {**original, "id": excerpt_id, "char_start": begin, "char_end": finish,
                                      "content": quote["quote"], "topic": topic, "scope": scope, "project_id": project_id}
                excerpt_ids.append(excerpt_id)
                original_ids.add(original["id"])
            if original_ids != set(sources):
                raise ValueError("source quote references do not match cited chunks")
            temporal = validate_temporal(raw) if self.config.facts.temporal else {
                "occurred_start": None, "occurred_end": None, "occurred_precision": None, "mentioned_at": None}
            if self.config.facts.temporal and all(by_id[source]["timestamp"] is not None for source in sources):
                temporal["mentioned_at"] = datetime.fromtimestamp(max(by_id[s]["timestamp"] for s in sources) / 1000,
                                                                  timezone.utc).isoformat().replace("+00:00", "Z")
            metadata = {"topic": topic, "scope": scope, "project_id": project_id,
                        "event_status": (raw["event_status"] or "unknown") if topic == "events" else None,
                        "conditions": raw["conditions"] if topic == "methods" else None,
                        "verification": raw["verification"] if topic == "methods" else None}
            identity = stable_id(request.request_id, raw["content"], sorted(set(excerpt_ids)), temporal, metadata, parent)
            identities.append(identity)
            facts.append({"id": identity, "content": raw["content"], "sources": sorted(set(excerpt_ids)),
                          "parent_index": parent, "matter_id": stable_id("matter", identity) if topic == "events" else None,
                          **metadata, **temporal})
        for fact in facts:
            parent = fact.pop("parent_index")
            if parent is not None and raw_facts[parent]["parent_index"] is not None:
                raise ValueError("auxiliary entries must reference a primary entry; cycles are forbidden")
            fact["parent_fact_id"] = identities[parent] if parent is not None else None
        unique_facts = list({fact["id"]: fact for fact in facts}.values())
        unique_facts.sort(key=lambda fact: (min((chunks[source]["message_seq"], chunks[source]["char_start"])
                                               for source in fact["sources"]), fact["id"]))
        ordered_chunks = sorted(chunks.values(), key=lambda c: (c["message_seq"], c["char_start"], c["char_end"], c["id"]))
        counts = defaultdict(int)
        for chunk in ordered_chunks:
            chunk["chunk_seq"] = counts[chunk["message_id"]]
            counts[chunk["message_id"]] += 1
        return unique_facts, ordered_chunks

    def add(self, request: AddRequest) -> AddResponse:
        payload_hash = hashlib.sha256(request.model_dump_json(exclude_none=False).encode()).hexdigest()
        with self._lock(request.user_id), self.store.connect(request.user_id) as db:
            previous = self.store.existing_request(db, request.request_id, payload_hash)
            if previous is not None:
                return AddResponse.model_validate(previous)
            prior = self.store.prior_messages(db, request.session_id, request.project_id)
            if self.directories:
                prior = self.directories.hydrate_chunks(db, request.user_id, prior)
            context, used = [], 0
            for item in reversed(prior):
                size = len(self.tokenizer.encode(item["content"]))
                if used + size > self.config.facts.prior_context_tokens:
                    break
                context.append({key: item[key] for key in ("role", "content", "timestamp")})
                used += size
            context.reverse()
            messages, extraction_chunks = self._chunks(request, self.store.next_seq(db, request.session_id))
            raw_facts = self.llm.extract(self.prompt, context, extraction_chunks,
                                         temporal=self.config.facts.temporal, project_id=request.project_id)
            facts, chunks = self._entries(request, raw_facts, extraction_chunks)
            active_chunks = [c for c in chunks if c["scope"] != "unresolved"]
            active_facts = [f for f in facts if f["scope"] != "unresolved"]
            owners = [("chunk", c) for c in active_chunks] + [("fact", f) for f in active_facts]
            vectors = [normalized(v, self.config.embedding.dimensions) for v in
                       self.embedder.embed([item["content"] for _, item in owners])] if owners else []
            if len(vectors) != len(owners):
                raise ValueError("embedding count does not match text count")
            observations, decisions, observation_vectors = [], [], []
            if self.config.consolidation.mode != "off" and active_facts:
                from .consolidation import consolidate
                partitions = {(f["topic"], f["scope"], f["project_id"]) for f in active_facts}
                existing = self.store.current_observations(db, partitions)
                for observation in existing:
                    observation["vector"] = np.frombuffer(observation["vector"], dtype=np.float32)
                observations, decisions = consolidate(active_facts, vectors[len(active_chunks):], existing,
                    self.decision_model, self.llm, self.consolidation_prompt, request.request_id)
                observation_vectors = [normalized(v, self.config.embedding.dimensions) for v in
                    self.embedder.embed([item["content"] for item in observations])] if observations else []
                if len(observation_vectors) != len(observations):
                    raise ValueError("observation embedding count mismatch")
            for fact in facts:
                scope_path = "global" if fact["scope"] == "global" else "unresolved" if fact["scope"] == "unresolved" else (
                    "projects/" + hashlib.sha256(fact["project_id"].encode()).hexdigest())
                fact["logical_path"] = scope_path + "/" + fact["topic"] + ("/" + fact["matter_id"] if fact["matter_id"] else "")
            unresolved = sum(f["scope"] == "unresolved" for f in facts)
            response = AddResponse(request_id=request.request_id, user_id=request.user_id, session_id=request.session_id,
                unresolved_entry_count=unresolved, warnings=["Project-specific entries were saved with unresolved scope and are excluded from retrieval."] if unresolved else [])
            directory = self.directories.build(request, messages, facts) if self.directories else None
            recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with self.directories.publish(directory, db) if directory else nullcontext():
                self._commit_add(db, request, response, payload_hash, messages, chunks, facts, owners, vectors,
                                 observations, decisions, observation_vectors, recorded_at, directory)
            return response

    def _commit_add(self, db, request, response, payload_hash, messages, chunks, facts, owners, vectors,
                    observations, decisions, observation_vectors, recorded_at, directory):
        try:
            db.execute("BEGIN IMMEDIATE")
            db.executemany("INSERT INTO messages VALUES (:id,:session_id,:seq,:role,:content,:timestamp)",
                           [{**m, "content": ""} for m in messages] if directory else messages)
            db.executemany("""INSERT INTO chunks VALUES (:id,:message_id,:session_id,:message_seq,:chunk_seq,
                :char_start,:char_end,:content,:timestamp,:topic,:scope,:project_id)""",
                [{**c, "content": ""} for c in chunks] if directory else chunks)
            db.executemany("INSERT INTO chunks_fts(content,chunk_id) VALUES (?,?)",
                           [(c["content"], c["id"]) for c in chunks if c["scope"] != "unresolved"])
            columns = ("id", "content", "extraction_version", "occurred_start", "occurred_end", "occurred_precision",
                       "mentioned_at", "recorded_at", "topic", "scope", "project_id", "parent_fact_id", "matter_id",
                       "event_status", "conditions", "verification", "logical_path")
            db.executemany("INSERT INTO facts VALUES (" + ",".join(":" + key for key in columns) + ")",
                           [{**f, "extraction_version": self.config.llm.name, "recorded_at": recorded_at} for f in facts])
            db.executemany("INSERT INTO fact_sources VALUES (?,?)", [(f["id"], s) for f in facts for s in f["sources"]])
            db.executemany("INSERT INTO embeddings VALUES (?,?,?,?,?)", [
                (kind, item["id"], self.config.embedding.name, self.config.embedding.dimensions, vector.tobytes())
                for (kind, item), vector in zip(owners, vectors)])
            self.store.save_observations(db, observations, decisions, request.request_id, recorded_at)
            db.executemany("""INSERT INTO embeddings VALUES ('observation',?,?,?,?)
                ON CONFLICT(owner_type,owner_id) DO UPDATE SET
                model=excluded.model,dimension=excluded.dimension,vector=excluded.vector""",
                [(o["id"], self.config.embedding.name, self.config.embedding.dimensions, v.tobytes())
                 for o, v in zip(observations, observation_vectors)])
            if directory:
                db.execute("INSERT INTO memory_directories VALUES (:id,:path,:parent_path,:abstract,:overview,:source_sha256)", directory)
                db.executemany("INSERT INTO message_sources VALUES (?,?)", [(m["id"], directory["id"]) for m in messages])
            db.execute("INSERT INTO requests VALUES (?,?,?)", (request.request_id, payload_hash, response.model_dump_json()))
            db.commit()
        except BaseException:
            db.rollback()
            raise

    def _prepare_scope(self, db, request):
        # A temporary relation avoids SQLite parameter limits for large corpora.
        db.execute("CREATE TEMP TABLE IF NOT EXISTS search_facts (id TEXT PRIMARY KEY)")
        db.execute("DELETE FROM search_facts")
        rows = db.execute("SELECT * FROM facts WHERE scope='global' OR (scope='project' AND project_id=?)",
                          (request.project_id,)).fetchall()
        identities = [(f["id"],) for f in rows if event_overlaps(f, request.event_time_range)]
        db.executemany("INSERT INTO search_facts VALUES (?)", identities)

    @staticmethod
    def _scoped_owners(owner_type):
        if owner_type == "chunk":
            return "SELECT fs.chunk_id FROM fact_sources fs JOIN search_facts s ON s.id=fs.fact_id"
        if owner_type == "fact":
            return "SELECT id FROM search_facts"
        if owner_type == "observation":
            return """SELECT os.observation_id FROM observation_sources os JOIN observations o
                ON o.id=os.observation_id AND o.current_revision=os.revision
                JOIN search_facts s ON s.id=os.fact_id"""
        raise ValueError("unsupported vector owner")

    def _vector_rank(self, db, query_vector, owner_type, limit):
        rows = db.execute("SELECT owner_id,vector FROM embeddings WHERE owner_type=? AND owner_id IN (" +
                          self._scoped_owners(owner_type) + ") ORDER BY owner_id", (owner_type,)).fetchall()
        if not rows:
            return []
        matrix = np.frombuffer(b"".join(row["vector"] for row in rows), dtype=np.float32).reshape(
            len(rows), self.config.embedding.dimensions)
        positions = np.argsort(-(matrix @ query_vector), kind="stable")[:limit]
        return [rows[int(i)]["owner_id"] for i in positions]

    def _bm25_rank(self, db, query, limit):
        words = re.findall(r"\w+", query, flags=re.UNICODE)
        if not words:
            return []
        expression = " OR ".join('"' + word.replace('"', '') + '"' for word in words)
        return [row[0] for row in db.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "AND chunk_id IN (" + self._scoped_owners("chunk") + ") ORDER BY bm25(chunks_fts) LIMIT ?", (expression, limit))]

    def _retrieval_vector_count(self, db):
        config = self.config.retrieval
        kinds = (["chunk"] if config.chunk_vector_weight else []) + (["fact"] if config.fact_vector_weight else [])
        if self.config.consolidation.mode != "off":
            kinds.append("observation")
        return sum(db.execute("SELECT COUNT(*) FROM embeddings WHERE owner_type=? AND owner_id IN (" +
            self._scoped_owners(kind) + ")", (kind,)).fetchone()[0] for kind in kinds)

    def _select_topics(self, db, request):
        available = {row[0] for row in db.execute("SELECT DISTINCT f.topic FROM facts f JOIN search_facts s ON s.id=f.id")}
        topics = [item for item in topic_candidates() if item["id"] in available]
        decisions = self.directory_selector.select(request.query, request.options, topics)
        selected = selected_topics(decisions, [t["id"] for t in topics])
        if not selected:
            db.execute("DELETE FROM search_facts")
        else:
            db.execute("DELETE FROM search_facts WHERE id NOT IN (SELECT id FROM facts WHERE topic IN (" +
                       ",".join("?" for _ in selected) + "))", selected)

    def search(self, request: SearchRequest) -> SearchResponse:
        with self.store.connect(request.user_id, create=False) as db:
            if db is None:
                return SearchResponse(data=[])
            db.execute("BEGIN")
            try:
                self._prepare_scope(db, request)
                if db.execute("SELECT 1 FROM search_facts LIMIT 1").fetchone() is None:
                    return SearchResponse(data=[])
                retrieval = self.config.retrieval
                if self.directory_selector and self._retrieval_vector_count(db) > retrieval.directory_routing_threshold:
                    self._select_topics(db, request)
                    if db.execute("SELECT 1 FROM search_facts LIMIT 1").fetchone() is None:
                        return SearchResponse(data=[])
                query_specs = [(request.query, retrieval.question_weight)]
                if request.options and retrieval.options_weight:
                    query_specs.append((request.query + "\n" + "\n".join(request.options), retrieval.options_weight))
                vectors = [normalized(v, self.config.embedding.dimensions) for v in
                           self.embedder.embed([q for q, _ in query_specs])]
                if len(vectors) != len(query_specs):
                    raise ValueError("query embedding count mismatch")
                allowed = {row[0] for row in db.execute(self._scoped_owners("chunk"))}
                scores = defaultdict(float)
                for (query, weight), vector in zip(query_specs, vectors):
                    routes = []
                    if retrieval.bm25_weight:
                        routes.append((list(enumerate(self._bm25_rank(db, query, retrieval.bm25_candidates), 1)), retrieval.bm25_weight))
                    if retrieval.chunk_vector_weight:
                        routes.append((list(enumerate(self._vector_rank(db, vector, "chunk", retrieval.chunk_vector_candidates), 1)), retrieval.chunk_vector_weight))
                    if retrieval.fact_vector_weight:
                        mapped = []
                        for rank, identity in enumerate(self._vector_rank(db, vector, "fact", retrieval.fact_vector_candidates), 1):
                            mapped.extend((rank, row[0]) for row in db.execute(
                                "SELECT chunk_id FROM fact_sources WHERE fact_id=? ORDER BY chunk_id", (identity,)))
                        routes.append((mapped, retrieval.fact_vector_weight))
                    if self.config.consolidation.mode != "off":
                        mapped = []
                        for rank, identity in enumerate(self._vector_rank(db, vector, "observation", 50), 1):
                            mapped.extend((rank, row[0]) for row in db.execute("""SELECT DISTINCT fs.chunk_id
                                FROM observations o JOIN observation_sources os ON os.observation_id=o.id
                                AND os.revision=o.current_revision JOIN fact_sources fs ON fs.fact_id=os.fact_id
                                JOIN search_facts s ON s.id=os.fact_id WHERE o.id=? ORDER BY fs.chunk_id""", (identity,)))
                        routes.append((mapped, 1.0))
                    for ranked, route_weight in routes:
                        seen = set()
                        for rank, identity in ranked:
                            if identity in allowed and identity not in seen:
                                scores[identity] += weight * route_weight / (retrieval.rrf_k + rank)
                                seen.add(identity)
                if not scores:
                    return SearchResponse(data=[])
                seeds = sorted(scores, key=lambda key: (-scores[key], key))[:retrieval.seed_count]
                sessions = defaultdict(list)
                locations = {}
                for row in db.execute("SELECT * FROM chunks ORDER BY session_id,message_seq,chunk_seq"):
                    locations[row["id"]] = (row["session_id"], len(sessions[row["session_id"]]))
                    sessions[row["session_id"]].append(dict(row))
                selected = defaultdict(set)
                for seed in seeds:
                    session_id, position = locations[seed]
                    session, origin = sessions[session_id], sessions[session_id][position]
                    for i in range(max(0, position - retrieval.neighbor_count), min(len(session), position + retrieval.neighbor_count + 1)):
                        row = session[i]
                        if (row["id"] in allowed and abs(row["message_seq"] - origin["message_seq"]) <= retrieval.neighbor_count
                                and (row["topic"], row["scope"], row["project_id"]) == (origin["topic"], origin["scope"], origin["project_id"])):
                            selected[session_id].add(i)
                entries = defaultdict(list)
                for row in db.execute("SELECT fs.chunk_id,fs.fact_id FROM fact_sources fs JOIN search_facts s ON s.id=fs.fact_id ORDER BY fs.fact_id"):
                    entries[row["chunk_id"]].append(row["fact_id"])
                windows = []
                for session_id, positions in selected.items():
                    groups = []
                    for position in sorted(positions):
                        current = sessions[session_id][position]
                        previous = sessions[session_id][groups[-1][-1]] if groups else None
                        adjacent = previous is not None and position == groups[-1][-1] + 1 and (
                            (current["message_id"] == previous["message_id"] and current["char_start"] <= previous["char_end"])
                            or current["message_seq"] == previous["message_seq"] + 1)
                        same_scope = previous is not None and all(current[k] == previous[k] for k in ("topic", "scope", "project_id"))
                        if not adjacent or not same_scope:
                            groups.append([])
                        groups[-1].append(position)
                    for group in groups:
                        rows = [sessions[session_id][i] for i in group]
                        if self.directories:
                            rows = self.directories.hydrate_chunks(db, request.user_id, rows)
                        contents = []
                        previous = None
                        for row in rows:
                            start = max(0, previous["char_end"] - row["char_start"]) if previous and previous["message_id"] == row["message_id"] else 0
                            contents.append(row["content"][start:])
                            previous = row
                        timestamp = next((r["timestamp"] for r in rows if r["timestamp"] is not None), None)
                        origin = rows[0]
                        windows.append(Evidence(id=stable_id(*[r["id"] for r in rows]), content="\n".join(c for c in contents if c),
                            score=max(scores.get(r["id"], 0) for r in rows),
                            created_at=datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat().replace("+00:00", "Z") if timestamp is not None else None,
                            entry_ids=sorted({identity for r in rows for identity in entries[r["id"]]}), topic=origin["topic"],
                            scope=origin["scope"], project_id=origin["project_id"],
                            requirement_priority=(2 if origin["scope"] == "project" else 1) if origin["topic"] == "requirements" else None))
                result, used = [], 0
                for item in sorted(windows, key=lambda item: (-item.score, item.id)):
                    size = len(self.tokenizer.encode(item.content))
                    if used + size <= retrieval.evidence_budget_tokens:
                        result.append(item)
                        used += size
                    if len(result) >= min(request.top_k, retrieval.max_results):
                        break
                return SearchResponse(data=result)
            finally:
                db.rollback()
