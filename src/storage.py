import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, response_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER NOT NULL,
  role TEXT NOT NULL, content TEXT NOT NULL, timestamp INTEGER,
  UNIQUE(session_id, seq)
);
CREATE TABLE IF NOT EXISTS memory_directories (
  id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, parent_path TEXT NOT NULL,
  abstract TEXT NOT NULL, overview TEXT NOT NULL, source_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message_sources (
  message_id TEXT PRIMARY KEY REFERENCES messages(id),
  directory_id TEXT NOT NULL REFERENCES memory_directories(id)
);
CREATE INDEX IF NOT EXISTS message_sources_directory ON message_sources(directory_id);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, message_id TEXT NOT NULL REFERENCES messages(id),
  session_id TEXT NOT NULL, message_seq INTEGER NOT NULL, chunk_seq INTEGER NOT NULL,
  token_start INTEGER NOT NULL, token_end INTEGER NOT NULL, content TEXT NOT NULL,
  timestamp INTEGER, UNIQUE(message_id, chunk_seq)
);
CREATE INDEX IF NOT EXISTS chunks_order ON chunks(session_id, message_seq, chunk_seq);
CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY, content TEXT NOT NULL, extraction_version TEXT NOT NULL,
  occurred_start TEXT, occurred_end TEXT, occurred_precision TEXT, mentioned_at TEXT,
  recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_sources (
  fact_id TEXT NOT NULL REFERENCES facts(id), chunk_id TEXT NOT NULL REFERENCES chunks(id),
  PRIMARY KEY(fact_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS fact_sources_chunk ON fact_sources(chunk_id);
CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY, current_revision INTEGER NOT NULL CHECK(current_revision >= 1)
);
CREATE TABLE IF NOT EXISTS observation_revisions (
  observation_id TEXT NOT NULL REFERENCES observations(id), revision INTEGER NOT NULL,
  content TEXT NOT NULL, temporal_json TEXT NOT NULL, status TEXT NOT NULL,
  request_id TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(observation_id, revision)
);
CREATE TABLE IF NOT EXISTS observation_sources (
  observation_id TEXT NOT NULL, revision INTEGER NOT NULL,
  fact_id TEXT NOT NULL REFERENCES facts(id),
  PRIMARY KEY(observation_id, revision, fact_id),
  FOREIGN KEY(observation_id, revision) REFERENCES observation_revisions(observation_id, revision)
);
CREATE INDEX IF NOT EXISTS observation_sources_fact ON observation_sources(fact_id);
CREATE TABLE IF NOT EXISTS consolidation_decisions (
  request_id TEXT NOT NULL, fact_id TEXT NOT NULL REFERENCES facts(id),
  observation_id TEXT NOT NULL REFERENCES observations(id), relation TEXT NOT NULL,
  decision_json TEXT NOT NULL, PRIMARY KEY(request_id, fact_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
  owner_type TEXT NOT NULL CHECK(owner_type IN ('chunk','fact','observation','directory')),
  owner_id TEXT NOT NULL, model TEXT NOT NULL, dimension INTEGER NOT NULL,
  vector BLOB NOT NULL, PRIMARY KEY(owner_type, owner_id)
);
CREATE TRIGGER IF NOT EXISTS embeddings_source_check BEFORE INSERT ON embeddings BEGIN
  SELECT CASE
    WHEN NEW.owner_type='chunk' AND NOT EXISTS(SELECT 1 FROM chunks WHERE id=NEW.owner_id) THEN RAISE(ABORT, 'embedding chunk source missing')
    WHEN NEW.owner_type='fact' AND NOT EXISTS(SELECT 1 FROM facts WHERE id=NEW.owner_id) THEN RAISE(ABORT, 'embedding fact source missing')
    WHEN NEW.owner_type='observation' AND NOT EXISTS(SELECT 1 FROM observations WHERE id=NEW.owner_id) THEN RAISE(ABORT, 'embedding observation source missing')
    WHEN NEW.owner_type='directory' AND NOT EXISTS(SELECT 1 FROM memory_directories WHERE id=NEW.owner_id) THEN RAISE(ABORT, 'embedding directory source missing')
  END;
END;
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, chunk_id UNINDEXED, tokenize='unicode61');
"""


def user_db_path(root: Path, user_id: str) -> Path:
    return root / f"{hashlib.sha256(user_id.encode()).hexdigest()}.sqlite3"


class Store:
    def __init__(self, root: Path, fingerprint: str):
        self.root = root
        self.fingerprint = fingerprint
        root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self, user_id: str, create: bool = True):
        path = user_db_path(self.root, user_id)
        if not create and not path.exists():
            yield None
            return
        new_database = not path.exists()
        connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            if new_database:
                connection.executescript(SCHEMA)
                connection.execute("INSERT OR IGNORE INTO metadata VALUES ('schema_version','3')")
                connection.execute("INSERT OR IGNORE INTO metadata VALUES ('index_fingerprint',?)", (self.fingerprint,))
            stored = connection.execute("SELECT value FROM metadata WHERE key='index_fingerprint'").fetchone()
            if stored is None or stored[0] != self.fingerprint:
                raise ValueError("index configuration changed; use a new database directory")
            schema = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if schema is None or schema[0] != "3":
                raise ValueError("database schema version is unsupported; use a new database directory")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def existing_request(connection: sqlite3.Connection, request_id: str, payload_hash: str):
        row = connection.execute("SELECT payload_hash,response_json FROM requests WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            return None
        if row[0] != payload_hash:
            raise FileExistsError("request_id already exists with different payload")
        return json.loads(row[1])

    @staticmethod
    def next_seq(connection: sqlite3.Connection, session_id: str) -> int:
        return connection.execute("SELECT COALESCE(MAX(seq),0)+1 FROM messages WHERE session_id=?", (session_id,)).fetchone()[0]

    @staticmethod
    def prior_messages(connection: sqlite3.Connection, session_id: str) -> list[dict]:
        rows = connection.execute("SELECT role,content,timestamp FROM messages WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def current_observations(connection: sqlite3.Connection) -> list[dict]:
        rows = connection.execute("""
            SELECT o.id, o.current_revision AS revision, r.content, r.temporal_json, r.status, e.vector
            FROM observations o JOIN observation_revisions r
              ON r.observation_id=o.id AND r.revision=o.current_revision
            JOIN embeddings e ON e.owner_type='observation' AND e.owner_id=o.id
            ORDER BY o.id
        """).fetchall()
        observations = {row["id"]: {**dict(row), "time_metadata": json.loads(row["temporal_json"]),
                                    "sources": [], "source_facts": []} for row in rows}
        for row in connection.execute("""
            SELECT s.observation_id, f.* FROM observation_sources s
            JOIN observations o ON o.id=s.observation_id AND o.current_revision=s.revision
            JOIN facts f ON f.id=s.fact_id ORDER BY s.observation_id,f.id
        """):
            observation = observations[row["observation_id"]]
            observation["sources"].append(row["id"])
            observation["source_facts"].append({key: row[key] for key in
                ("id", "content", "occurred_start", "occurred_end", "occurred_precision", "mentioned_at")})
        return list(observations.values())

    @staticmethod
    def save_observations(connection: sqlite3.Connection, observations: list[dict],
                          decisions: list[dict], request_id: str, created_at: str):
        for observation in observations:
            connection.execute("""INSERT INTO observations VALUES (?,?)
                ON CONFLICT(id) DO UPDATE SET current_revision=excluded.current_revision""",
                (observation["id"], observation["revision"]))
            temporal = observation["time_metadata"]
            connection.execute("INSERT INTO observation_revisions VALUES (?,?,?,?,?,?,?)",
                (observation["id"], observation["revision"], observation["content"],
                 json.dumps(temporal, ensure_ascii=False), observation.get("status", "active"), request_id, created_at))
            connection.executemany("INSERT INTO observation_sources VALUES (?,?,?)",
                [(observation["id"], observation["revision"], source) for source in observation["sources"]])
        connection.executemany("INSERT INTO consolidation_decisions VALUES (?,?,?,?,?)",
            [(request_id, item["fact_id"], item["target_id"], item["relation"],
              json.dumps(item, ensure_ascii=False)) for item in decisions])
