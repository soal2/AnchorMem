"""Immutable source directories with separately indexed, bounded directory summaries."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class DirectoryMemory:
    def __init__(self, root: Path, tokenizer):
        self.root, self.tokenizer = root, tokenizer

    def build(self, request, messages: list[dict], facts: list[dict]) -> dict:
        # User-controlled identifiers never become filesystem path components.
        parent = f"user/{_hash(request.user_id)}/memories/sessions/{_hash(request.session_id)}"
        source = json.dumps({"user_id": request.user_id, "session_id": request.session_id,
                             "request_id": request.request_id, "messages": messages},
                            ensure_ascii=False, sort_keys=True, indent=2).encode()
        lines = [f"Session: {request.session_id}",
                 f"Source order: {messages[0]['seq']}–{messages[-1]['seq']}",
                 "Source timestamps (milliseconds; null means unknown): "
                 + json.dumps([message["timestamp"] for message in messages])]
        if facts:
            for fact in facts:
                timing = {key: fact.get(key) for key in
                          ("occurred_start", "occurred_end", "occurred_precision", "mentioned_at")}
                lines.append(f"- {fact['content']} | {json.dumps(timing, ensure_ascii=False)}")
        else:
            lines.extend(f"- {message['role']}: {message['content']}" for message in messages)
        overview_tokens = self.tokenizer.encode("\n".join(lines))
        overview = self.tokenizer.decode(overview_tokens[:1024])
        if len(overview_tokens) > 1024:
            overview += "\n[Overview truncated; absence here does not prove irrelevance.]"
        abstract = self.tokenizer.decode(self.tokenizer.encode("\n".join(lines[3:]))[:256])
        source_sha256 = hashlib.sha256(source).hexdigest()
        # Content-addressed versions also make retries safe if a model produces a
        # different summary after a crash left an unpublished directory behind.
        identity = _hash(json.dumps([request.request_id, source_sha256, abstract, overview]))
        path = f"{parent}/{identity}"
        return {"id": identity, "path": path, "parent_path": parent,
                "abstract": abstract, "overview": overview,
                "source_sha256": source_sha256, "source": source}

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()) or path == self.root.resolve():
            raise ValueError("source directory escapes memory root")
        return path

    @contextmanager
    def publish(self, entry: dict, db):
        """Publish immutable files before the DB commit; unregistered files are never read.

        A crash between rename and commit may leave an unreferenced directory. Retrying
        the identical content reuses it after byte verification. New summaries receive
        their own version path; orphaned versions remain invisible to retrieval.
        """
        target = self._path(entry["path"])
        files = {"messages.json": entry["source"],
                 ".abstract.md": entry["abstract"].encode(), ".overview.md": entry["overview"].encode()}
        created = False
        if target.exists():
            if any(not (target / name).is_file() or (target / name).read_bytes() != data
                   for name, data in files.items()):
                raise ValueError("immutable source directory already contains different data")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".pending-", dir=target.parent) as staging:
                for name, data in files.items():
                    with (Path(staging) / name).open("wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                with_dir = os.open(staging, os.O_RDONLY)
                try:
                    os.fsync(with_dir)
                finally:
                    os.close(with_dir)
                os.rename(staging, target)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            created = True
        try:
            yield
        except BaseException:
            # A caller can be interrupted just after SQLite commits. Never remove
            # a source that is already registered by that completed transaction.
            registered = db.execute("SELECT 1 FROM memory_directories WHERE id=?", (entry["id"],)).fetchone()
            if created and registered is None:
                shutil.rmtree(target)
            raise

    def messages(self, db, user_id: str, message_ids: list[str]) -> dict[str, dict]:
        if not message_ids:
            return {}
        result = {}
        # Resolve only registered sources inside this user's current SQLite snapshot.
        rows = db.execute("""SELECT d.*, s.message_id FROM memory_directories d
            JOIN message_sources s ON s.directory_id=d.id
            WHERE s.message_id IN (""" + ",".join("?" for _ in message_ids) + ")", message_ids).fetchall()
        entries = {row["id"]: row for row in rows}
        for row in entries.values():
            source = (self._path(row["path"]) / "messages.json").read_bytes()
            if hashlib.sha256(source).hexdigest() != row["source_sha256"]:
                raise ValueError("original memory integrity check failed")
            payload = json.loads(source)
            if payload["user_id"] != user_id:
                raise ValueError("original memory belongs to another user")
            result.update({message["id"]: message for message in payload["messages"]})
        if not set(message_ids).issubset(result):
            raise ValueError("original memory source is missing")
        return {identity: result[identity] for identity in message_ids}

    def hydrate_chunks(self, db, user_id: str, rows: list) -> list[dict]:
        messages = self.messages(db, user_id, list(dict.fromkeys(row["message_id"] for row in rows)))
        tokens = {identity: self.tokenizer.encode(message["content"]) for identity, message in messages.items()}
        return [{**dict(row), "content": self.tokenizer.decode(
            tokens[row["message_id"]][row["token_start"]:row["token_end"]])} for row in rows]
