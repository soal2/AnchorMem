"""Check every frozen history and actual bounded extraction envelope without API calls."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tiktoken

from src.config import ServiceConfig, load_config
from src.contracts import AddRequest, Message
from src.engine import MemoryEngine
from src.providers import EXTRACTION_CHUNKS_PER_CALL, OpenAILLM, extraction_batches
from src.week1_data import load_pilot


def audit(root):
    config, _ = load_config(root / "configs/service/vikingjev.yaml", ServiceConfig)
    engine = MemoryEngine.__new__(MemoryEngine)
    engine.config, engine.tokenizer = config, tiktoken.get_encoding(config.chunk.tokenizer)
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.config, llm.telemetry = config.llm, None
    captured = []

    def create(**request):
        captured.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"facts":[]}', refusal=None), finish_reason="stop")])

    llm.client = SimpleNamespace(api_key="offline", chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    prompt = (root / "prompts/extract-temporal-facts.txt").read_text()
    histories = {}
    for example in load_pilot(root):
        histories.setdefault((example["dataset"], example["group"]), example)
    checks, history_counts, add_batches = [], [], 0
    for example in histories.values():
        source_messages = source_characters = source_chunks = requests = 0
        for session in example["sessions"]:
            for offset in range(0, len(session["messages"]), 20):
                request = AddRequest(request_id="offline", user_id="offline", session_id=str(session["id"]),
                    messages=[Message.model_validate(m) for m in session["messages"][offset:offset + 20]])
                messages, chunks = engine._chunks(request, offset + 1)
                grouped = defaultdict(list)
                for chunk in chunks:
                    grouped[chunk["message_id"]].append(chunk)
                for message in messages:
                    covered = 0
                    for chunk in grouped[message["id"]]:
                        assert chunk["char_start"] <= covered < chunk["char_end"]
                        assert chunk["content"] == message["content"][chunk["char_start"]:chunk["char_end"]]
                        covered = chunk["char_end"]
                    assert covered == len(message["content"]), "history has an uncovered character interval"
                    source_messages += 1
                    source_characters += covered
                batches = extraction_batches(chunks)
                assert [c for batch, _ in batches for c in batch] == chunks
                llm.extract(prompt, [], chunks, temporal=True, project_id="paper:" + example["group"])
                bodies = [json.loads(call["messages"][1]["content"]) for call in captured]
                assert len(captured) == len(batches)
                for (batch, reference), payload, body in zip(batches, captured, bodies, strict=True):
                    assert 1 <= len(body["new_chunks"]) <= EXTRACTION_CHUNKS_PER_CALL
                    assert [c["content"] for c in body["new_chunks"]] == [c["content"] for c in batch]
                    assert len({c["message_id"] for c in batch}) == 1
                    if reference:
                        assert reference["message_id"] == batch[0]["message_id"]
                        assert body["reference_context"]["content"] == reference["content"]
                        assert not {"id", "index", "message_id", "source_units"}.intersection(body["reference_context"])
                    token_count = len(engine.tokenizer.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=()))
                    checks.append({"group": example["group"], "session": str(session["id"]), "offset": offset,
                        "chunks": len(batch), "reference_chunks": int(reference is not None),
                        "serialized_request_tokens": token_count,
                        "input_plus_output_plus_1024_margin": token_count + config.llm.max_output_tokens + 1024})
                requests += len(captured)
                captured.clear()
                source_chunks += len(chunks)
                add_batches += 1
        history_counts.append({"dataset": example["dataset"], "group": example["group"],
            "messages": source_messages, "characters": source_characters,
            "chunks": source_chunks, "extraction_requests": requests})
    largest = max(checks, key=lambda row: row["input_plus_output_plus_1024_margin"])
    assert largest["input_plus_output_plus_1024_margin"] <= 128000
    return {"scope": "fixed-two-chunk-full-history-envelope-audit", "paid_calls": 0,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
            ("src/providers.py", "src/source_spans.py", "src/engine.py", "src/temporal.py", "prompts/extract-temporal-facts.txt")},
        "selection_sha256": hashlib.sha256((root / "data/week1/selection.json").read_bytes()).hexdigest(),
        "histories": len(histories), "add_batches": add_batches, "extraction_requests": len(checks),
        "max_chunks_per_request": EXTRACTION_CHUNKS_PER_CALL, "history_truncated": False,
        "all_message_character_intervals_covered": True, "all_source_chunks_processed_once": True,
        "reference_context_is_same_original_message": True, "qa_answers_or_annotations_sent": False,
        "all_within_context": True, "context_limit": 128000, "output_limit": config.llm.max_output_tokens,
        "largest_request": largest, "history_counts": history_counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history_counts"}, ensure_ascii=False, indent=2))
