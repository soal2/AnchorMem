"""Freeze, execute, and report the four Jevsight experiments without changing scorers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from src.config import EvaluationConfig, load_config, resolved_path
from src.evaluation import _call_summary, load_examples, sha256, split_examples


PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts" / "jevsight"
CONFIGS = PROJECT / "configs" / "evaluation" / "jevsight"
DATASETS = ("locomo", "longmemeval", "personamem", "beam", "clbench")
VARIANTS = ("a", "b", "c", "d")
SEED = 42
VALIDATION_QUOTAS = {
    "locomo": {"2": 6, "3": 4, "1": 5, "4": 5},
    "longmemeval": {"temporal-reasoning": 6, "knowledge-update": 6, "multi-session": 6,
                   "single-session-user": 4, "single-session-assistant": 4, "single-session-preference": 4},
    "personamem": 20,
    "beam": {category: 1 for category in ("abstention", "contradiction_resolution", "event_ordering",
             "information_extraction", "instruction_following", "knowledge_update", "multi_session_reasoning",
             "preference_following", "summarization", "temporal_reasoning")},
    "clbench": 10,
}
TIME_UPDATE_CATEGORIES = {
    "locomo": {"2"},
    "longmemeval": {"temporal-reasoning", "knowledge-update"},
    "beam": {"temporal_reasoning", "knowledge_update"},
}
USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "provider_cost_credits", "cost_usd")


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def historical_examples(root=PROJECT / "artifacts" / "runs"):
    used = {dataset: {"ids": set(), "groups": set()} for dataset in DATASETS}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        dataset = manifest.get("dataset")
        if dataset not in used:
            continue
        for filename in ("predictions.jsonl", "failures.jsonl"):
            for record in _read_jsonl(manifest_path.parent / filename):
                if "id" in record:
                    used[dataset]["ids"].add(str(record["id"]))
                if "group" in record:
                    used[dataset]["groups"].add(str(record["group"]))
    return used


def choose_validation(dataset, examples, used):
    def order(example):
        return hashlib.sha256(f"{SEED}:{dataset}:{example['id']}".encode()).digest()

    available = sorted((item for item in examples if str(item["id"]) not in used["ids"]), key=order)
    quota = VALIDATION_QUOTAS[dataset]
    if isinstance(quota, dict):
        selected = []
        for category, count in quota.items():
            candidates = [item for item in available if str(item["category"]) == category]
            if len(candidates) < count:
                raise ValueError(f"{dataset}: insufficient unused validation questions for {category}: {len(candidates)} < {count}")
            selected.extend(candidates[:count])
        return selected
    selected = []
    groups = set(used["groups"])
    for item in available:
        group = str(item["group"])
        if group not in groups:
            selected.append(item)
            groups.add(group)
        if len(selected) == quota:
            return selected
    raise ValueError(f"{dataset}: insufficient unused validation groups: {len(selected)} < {quota}")


def _frozen_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to change a frozen experiment file: {path}")
        return
    path.write_text(content, encoding="utf-8")


def _relative(path, config_path):
    return os.path.relpath(path, config_path.parent)


def prepare():
    used = historical_examples()
    jobs = []
    for dataset in DATASETS:
        base_path = PROJECT / "configs" / "evaluation" / f"{dataset}-dev.yaml"
        base, base_path = load_config(base_path, EvaluationConfig)
        data_path = resolved_path(base_path, base.data_path)
        if sha256(data_path) != base.data_sha256:
            raise ValueError(f"{dataset}: dataset SHA-256 mismatch")
        examples = load_examples(dataset, data_path, SEED)
        pools = {split: split_examples(examples, split, SEED, base.dev_fraction, None) for split in ("dev", "validation")}
        shared_sessions = None
        if dataset == "longmemeval":
            dev_sessions = {session["id"] for item in pools["dev"] for session in item["sessions"]}
            validation_sessions = {session["id"] for item in pools["validation"] for session in item["sessions"]}
            shared_sessions = len(dev_sessions & validation_sessions)
        for split in ("dev", "validation"):
            selected = (pools[split][:base.sample_limit] if split == "dev"
                        else choose_validation(dataset, pools[split], used[dataset]))
            selection = {"dataset": dataset, "split": split, "seed": SEED, "data_sha256": base.data_sha256,
                         "selection_method": "existing configured dev prefix" if split == "dev" else "SHA-256 ordering within fixed quotas; exclude previously executed question IDs",
                         "quotas": VALIDATION_QUOTAS[dataset] if split == "validation" else len(selected),
                         "isolation": {"unit": "complete-history hash" if dataset == "longmemeval" else "conversation/persona/task group",
                                       "shared_dev_validation_source_sessions": shared_sessions,
                                       "note": "LongMemEval question and complete-history split does not guarantee source-session disjointness" if dataset == "longmemeval" else "group-disjoint dev and validation; PersonaMem/CLBench use one unseen group per selected question"},
                         "questions": [{key: str(item[key]) for key in ("id", "group", "category")} for item in selected]}
            selection_path = ARTIFACTS / "selections" / f"{split}-{dataset}.json"
            _frozen_write(selection_path, json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
            for variant in VARIANTS:
                path = CONFIGS / split / f"{variant}-{dataset}.yaml"
                raw = base.model_dump(mode="json")
                raw.update(service_config_path=_relative(PROJECT / "configs" / "service" / f"jevsight-{variant}.yaml", path),
                           data_path=_relative(data_path, path),
                           contract_repository_path=_relative(resolved_path(base_path, base.contract_repository_path), path),
                           split=split, seed=SEED, sample_limit=len(selected),
                           question_ids_path=_relative(selection_path, path),
                           run_label=f"jevsight-{split}-{variant}-{dataset}", output_dir=_relative(ARTIFACTS / "runs", path))
                _frozen_write(path, yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
                jobs.append({"stage": split, "variant": variant, "dataset": dataset,
                             "question_count": len(selected), "config_path": str(path.relative_to(PROJECT)),
                             "selection_path": str(selection_path.relative_to(PROJECT))})
    experiment = {"name": "Jevsight", "seed": SEED, "variants": {"a": "B3", "b": "B3 + write-time temporal semantics",
                  "c": "temporal + consolidation with LLM relationship decisions", "d": "same consolidation with Jev relationship decisions"},
                  "scope": "configured local development and fixed validation samples; not complete benchmark suites or official AML scores",
                  "question_counts_per_variant": {stage: sum(job["question_count"] for job in jobs if job["stage"] == stage and job["variant"] == "a") for stage in ("dev", "validation")},
                  "jobs": jobs}
    _frozen_write(ARTIFACTS / "experiment.json", json.dumps(experiment, ensure_ascii=False, indent=2) + "\n")
    return experiment


def _source_hash():
    root = PROJECT / "src"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_matches(run_dir, config_path, code_hash):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    config, config_path = load_config(config_path, EvaluationConfig)
    service_path = resolved_path(config_path, config.service_config_path)
    service = yaml.safe_load(service_path.read_text())
    expected = {"code_snapshot_sha256": code_hash, "data_sha256": config.data_sha256,
                "eval_config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "service_config_file_sha256": hashlib.sha256(service_path.read_bytes()).hexdigest()}
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    for name, value in (("extract", service["facts"]["prompt_path"]),
                        ("consolidation", service.get("consolidation", {}).get("prompt_path"))):
        if value is not None:
            current = hashlib.sha256(resolved_path(service_path, value).read_bytes()).hexdigest()
            if manifest.get("prompt_files", {}).get(name, {}).get("sha256") != current:
                return False
    chosen = json.loads((run_dir / "chosen-questions.json").read_text())
    selection = json.loads(resolved_path(config_path, config.question_ids_path).read_text())
    return chosen == selection["questions"]


def _complete(run_dir, question_count):
    path = run_dir / "metrics.json"
    if not path.exists():
        return False
    try:
        metrics = json.loads(path.read_text())
        predictions = _read_jsonl(run_dir / "predictions.jsonl")
        retrieval = _read_jsonl(run_dir / "retrieval.jsonl")
        questions = json.loads((run_dir / "chosen-questions.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    chosen = {item["id"] for item in questions}
    return (metrics.get("requested") == metrics.get("evaluated") == question_count
            and metrics.get("failed") == 0 and len(predictions) == len(retrieval) == question_count
            and len(chosen) == question_count
            and {item["id"] for item in predictions} == {item["id"] for item in retrieval} == chosen)


def matching_runs(job, code_hash):
    config_path = PROJECT / job["config_path"]
    label = f"jevsight-{job['stage']}-{job['variant']}-{job['dataset']}"
    return [path for path in sorted((ARTIFACTS / "runs").glob(f"*-{label}"), reverse=True)
            if _run_matches(path, config_path, code_hash)]


def execute_job(job, code_hash):
    previous = matching_runs(job, code_hash)
    for path in previous:
        if _complete(path, job["question_count"]):
            print(f"DONE {job['stage']}/{job['variant']}/{job['dataset']} {path}", flush=True)
            return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = ARTIFACTS / "jobs" / f"{timestamp}-{job['stage']}-{job['variant']}-{job['dataset']}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    record = {**job, "code_snapshot_sha256": code_hash, "started_at": datetime.now(timezone.utc).isoformat(),
              "returncode": None, "log_path": str(prefix.with_suffix(".log").relative_to(PROJECT))}
    prefix.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"START {job['stage']}/{job['variant']}/{job['dataset']} log={record['log_path']}", flush=True)
    with prefix.with_suffix(".log").open("x", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, "-m", "src.cli", "evaluate", "--config", str(PROJECT / job["config_path"])],
                                   cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=completed.returncode, completed_at=datetime.now(timezone.utc).isoformat())
    prefix.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    succeeded = any(_complete(path, job["question_count"]) for path in matching_runs(job, code_hash))
    print(f"{'DONE' if succeeded else 'FAILURE'} {job['stage']}/{job['variant']}/{job['dataset']} returncode={completed.returncode}", flush=True)


def development_complete(experiment, code_hash):
    jobs = [job for job in experiment["jobs"] if job["stage"] == "dev"]
    expected = {(variant, dataset) for variant in VARIANTS for dataset in DATASETS}
    if len(jobs) != len(expected) or {(job["variant"], job["dataset"]) for job in jobs} != expected:
        raise ValueError("experiment must contain exactly all 20 development jobs")
    return all(any(_complete(path, job["question_count"]) for path in matching_runs(job, code_hash)) for job in jobs)


def _resource_comparison(old, new):
    result = {}
    for scope in ("model_usage", "provider_stage_usage"):
        left, right = old.get(scope, {}), new.get(scope, {})
        stages = {}
        for stage in sorted(left.keys() | right.keys()):
            before, after = left.get(stage, {}), right.get(stage, {})
            delta = {key: after.get(key, 0) - before.get(key, 0)
                     for key in ("calls", "failed_calls", "latency_seconds")}
            for key in USAGE_FIELDS:
                a = before.get(key) if before.get("calls", 0) else 0
                b = after.get(key) if after.get("calls", 0) else 0
                missing = before.get(key + "_missing_calls", 0) + after.get(key + "_missing_calls", 0)
                delta[key] = b - a if a is not None and b is not None and missing == 0 else None
            stages[stage] = {"left": before, "right": after, "delta": delta}
        result[scope] = stages
    return result


def _observed_total_tokens(call):
    usage = call.get("usage") or {}
    total = usage.get("total_tokens")
    if type(total) in {int, float}:
        return total
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    return prompt + completion if type(prompt) in {int, float} and type(completion) in {int, float} else None


def _complete_sum(values):
    return sum(values) if all(type(value) in {int, float} for value in values) else None


def _decision_efficiency(row):
    if row["status"] != "executed" or row["variant"] not in {"c", "d"}:
        return None
    calls = _read_jsonl(PROJECT / row["run_path"] / "model-calls.jsonl")
    stage = "decision_llm" if row["variant"] == "c" else "decision_jev"
    decisions = [call for call in calls if call.get("stage") == stage]
    ingestion = [call for call in calls if call.get("evaluation_stage") == "ingestion"]
    count = len(decisions)
    decision_seconds = _complete_sum([call.get("latency_seconds") for call in decisions])
    decision_tokens = _complete_sum([_observed_total_tokens(call) for call in decisions])
    decision_credits = _complete_sum([(call.get("usage") or {}).get("cost") for call in decisions])
    return {"stage": row["stage"], "dataset": row["dataset"], "variant": row["variant"],
            "decision_calls": count, "failed_decision_calls": sum(call.get("success") is False for call in decisions),
            "decision_model_seconds": decision_seconds,
            "decision_seconds_per_call": decision_seconds / count if count and decision_seconds is not None else None,
            "decision_total_tokens": decision_tokens,
            "decision_tokens_per_call": decision_tokens / count if count and decision_tokens is not None else None,
            "decision_provider_credits": decision_credits,
            "ingestion_wall_seconds": row["metrics"]["actual_ingestion_seconds"],
            "ingestion_total_tokens": _complete_sum([_observed_total_tokens(call) for call in ingestion]),
            "ingestion_provider_credits": _complete_sum([(call.get("usage") or {}).get("cost") for call in ingestion])}


def _ratio(after, before):
    return after / before if after is not None and before not in {None, 0} else None


def _comparison_cases(before, after, dataset):
    cases = []
    for key in before:
        a, b = before[key], after[key]
        old_score, new_score = a.get("beam_score"), b.get("beam_score")
        if a["label"] == b["label"] == "CORRECT" and old_score == new_score:
            continue
        outcome = ("gained_correct" if a["label"] != "CORRECT" and b["label"] == "CORRECT" else
                   "lost_correct" if a["label"] == "CORRECT" and b["label"] != "CORRECT" else "both_incorrect")
        cases.append({"id": key, "group": str(a.get("group", "")), "category": str(a.get("category", "")),
                      "outcome": outcome, "left_label": a["label"], "right_label": b["label"],
                      "left_prediction": a.get("answer"), "right_prediction": b.get("answer"),
                      "left_primary_score": old_score if dataset == "beam" else int(a["label"] == "CORRECT"),
                      "right_primary_score": new_score if dataset == "beam" else int(b["label"] == "CORRECT")})
    return cases


def _validation_acceptance(by_job, comparisons):
    native = {}
    jev = {}
    for dataset in DATASETS:
        a, c, d = (by_job[("validation", variant, dataset)] for variant in ("a", "c", "d"))
        for left, target in ((a, native), (c, jev)):
            item = {"status": "pending", "criterion": "D primary metric >= comparison variant primary metric"}
            if left["status"] == d["status"] == "executed":
                old, new = left["metrics"], d["metrics"]
                if old["primary_metric"] != new["primary_metric"]:
                    raise ValueError(f"{dataset}: primary metrics differ between paired runs")
                item.update(status="pass" if new["primary_value"] >= old["primary_value"] else "fail",
                            primary_metric=new["primary_metric"], left_value=old["primary_value"], d_value=new["primary_value"])
            target[dataset] = item
    subsets = [item for item in comparisons if item["stage"] == "validation" and item["contrast"] == "D-A"
               and item["dataset"] in TIME_UPDATE_CATEGORIES]
    temporal = {"status": "pending", "criterion": "D-A time/update net correct > 0",
                "categories": {dataset: sorted(categories) for dataset, categories in TIME_UPDATE_CATEGORIES.items()}}
    if len(subsets) == len(TIME_UPDATE_CATEGORIES) and all(item["status"] == "executed" for item in subsets):
        gained = sum(item["time_update"]["gained_correct"] for item in subsets)
        lost = sum(item["time_update"]["lost_correct"] for item in subsets)
        temporal.update(status="pass" if gained > lost else "fail", gained_correct=gained, lost_correct=lost,
                        net_correct=gained - lost, paired_questions=sum(item["time_update"]["paired_questions"] for item in subsets))

    def status(checks):
        values = [item["status"] for item in checks]
        return "pending" if "pending" in values else ("pass" if all(value == "pass" for value in values) else "fail")

    return {"optimization_status": status([*native.values(), temporal]), "d_vs_a_native_metrics": native,
            "time_update": temporal, "jev_status": status([*native.values(), temporal, *jev.values()]),
            "jev_comparison_status": status(jev.values()), "d_vs_c_native_metrics": jev,
            "validation_policy": "Frozen validation is not used for tuning. No alternate prompts or models are tried in response to validation quality failures; only infrastructure/transport failures may be rerun."}


def _usage_text(summary, key):
    if not summary.get("calls", 0):
        return "0 (no calls)"
    value = summary.get(key)
    missing = summary.get(key + "_missing_calls", 0)
    known = "unavailable" if value is None else f"{value:g}"
    return f"{known}; missing {missing}/{summary['calls']}"


def _attempt_history(attempts, code_hash):
    runs = [(path.parent, json.loads(path.read_text())) for path in sorted((ARTIFACTS / "runs").glob("*/manifest.json"))]
    history = []
    for attempt in attempts:
        start = datetime.fromisoformat(attempt["started_at"])
        later = [datetime.fromisoformat(item["started_at"]) for item in attempts
                 if all(item.get(key) == attempt.get(key) for key in ("stage", "variant", "dataset"))
                 and datetime.fromisoformat(item["started_at"]) > start]
        end = datetime.fromisoformat(attempt["completed_at"]) if attempt.get("completed_at") else min(later, default=None)
        label = f"jevsight-{attempt['stage']}-{attempt['variant']}-{attempt['dataset']}"
        artifacts = []
        for path, manifest in runs:
            if manifest.get("run_label") != label or manifest.get("code_snapshot_sha256") != attempt["code_snapshot_sha256"]:
                continue
            created = datetime.fromisoformat(manifest["created_at"])
            if created < start or (end is not None and created > end):
                continue
            observed = {}
            malformed = {}
            for filename in ("model-calls.jsonl", "ingestion.jsonl", "predictions.jsonl", "failures.jsonl"):
                observed[filename] = []
                malformed[filename] = 0
                source = path / filename
                if not source.exists():
                    continue
                for line in source.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        observed[filename].append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed[filename] += 1
            calls = observed["model-calls.jsonl"]
            artifacts.append({"run_path": str(path.relative_to(PROJECT)),
                              "code_snapshot_sha256": manifest["code_snapshot_sha256"],
                              "complete_success": _complete(path, attempt["question_count"]),
                              "observed_predictions": len(observed["predictions.jsonl"]),
                              "observed_failures": len(observed["failures.jsonl"]),
                              "observed_ingestion_seconds": sum(item.get("seconds", 0) for item in observed["ingestion.jsonl"]),
                              "observed_model_usage": _call_summary(calls),
                              "observed_provider_stage_usage": _call_summary([{**item, "evaluation_stage": item["stage"]} for item in calls]),
                              "malformed_records": {key: value for key, value in malformed.items() if value}})
        history.append({**attempt, "matches_current_source": attempt["code_snapshot_sha256"] == code_hash,
                        "run_artifacts": artifacts})
    return history


def report(experiment):
    code_hash = _source_hash()
    attempts = [{**json.loads(path.read_text()), "attempt_path": str(path.relative_to(PROJECT))}
                for path in sorted((ARTIFACTS / "jobs").glob("*.json"))]
    rows = []
    by_job = {}
    for job in experiment["jobs"]:
        matches = matching_runs(job, code_hash)
        completed = next((path for path in matches if _complete(path, job["question_count"])), None)
        latest = completed or (matches[0] if matches else None)
        relevant = [attempt for attempt in attempts if all(attempt.get(key) == job[key] for key in ("stage", "variant", "dataset"))
                    and attempt.get("code_snapshot_sha256") == code_hash]
        status = ("executed" if completed else "running" if any(item.get("returncode") is None for item in relevant)
                  else "failure" if latest else "pending")
        metrics = json.loads((latest / "metrics.json").read_text()) if latest and (latest / "metrics.json").exists() else None
        row = {**job, "status": status, "run_path": str(latest.relative_to(PROJECT)) if latest else None,
               "metrics": metrics, "attempts": [{"returncode": item["returncode"], "log_path": item["log_path"]} for item in relevant]}
        rows.append(row)
        by_job[(job["stage"], job["variant"], job["dataset"])] = row
    comparisons = []
    for stage in ("dev", "validation"):
        for dataset in DATASETS:
            for left, right in (("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")):
                old, new = by_job[(stage, left, dataset)], by_job[(stage, right, dataset)]
                item = {"stage": stage, "dataset": dataset, "contrast": f"{right.upper()}-{left.upper()}", "status": "pending"}
                if old["status"] == new["status"] == "executed":
                    a = {row["id"]: row for row in _read_jsonl(PROJECT / old["run_path"] / "predictions.jsonl")}
                    b = {row["id"]: row for row in _read_jsonl(PROJECT / new["run_path"] / "predictions.jsonl")}
                    if set(a) != set(b):
                        raise ValueError("paired experiment question IDs differ")
                    item.update(status="executed", paired_questions=len(a),
                                gained_correct=sum(a[key]["label"] != "CORRECT" and b[key]["label"] == "CORRECT" for key in a),
                                lost_correct=sum(a[key]["label"] == "CORRECT" and b[key]["label"] != "CORRECT" for key in a),
                                primary_metric=new["metrics"]["primary_metric"],
                                primary_delta=new["metrics"]["primary_value"] - old["metrics"]["primary_value"])
                    categories = TIME_UPDATE_CATEGORIES.get(dataset, set())
                    subset = [key for key in a if str(a[key].get("category")) in categories]
                    gained = sum(a[key]["label"] != "CORRECT" and b[key]["label"] == "CORRECT" for key in subset)
                    lost = sum(a[key]["label"] == "CORRECT" and b[key]["label"] != "CORRECT" for key in subset)
                    item["time_update"] = {"categories": sorted(categories), "paired_questions": len(subset),
                                           "gained_correct": gained, "lost_correct": lost, "net_correct": gained - lost}
                    item["resources"] = _resource_comparison(old["metrics"], new["metrics"])
                    if (left, right) in (("a", "d"), ("c", "d")):
                        item["cases"] = _comparison_cases(a, b, dataset)
                        item["left_run_path"], item["right_run_path"] = old["run_path"], new["run_path"]
                comparisons.append(item)
    acceptance = _validation_acceptance(by_job, comparisons)
    decision_efficiency = [item for row in rows if (item := _decision_efficiency(row)) is not None]
    efficiency_by_job = {(item["stage"], item["dataset"], item["variant"]): item for item in decision_efficiency}
    decision_efficiency_pairs = []
    for stage in ("dev", "validation"):
        for dataset in DATASETS:
            c, d = (efficiency_by_job.get((stage, dataset, variant)) for variant in ("c", "d"))
            if c is None or d is None:
                continue
            decision_efficiency_pairs.append({"stage": stage, "dataset": dataset,
                "c_decision_calls": c["decision_calls"], "d_decision_calls": d["decision_calls"],
                **{f"d_over_c_{key}": _ratio(d[key], c[key]) for key in (
                    "decision_seconds_per_call", "decision_total_tokens", "decision_provider_credits",
                    "ingestion_wall_seconds", "ingestion_total_tokens", "ingestion_provider_credits")}})
    payload = {"experiment": "Jevsight", "scope": experiment["scope"], "current_code_snapshot_sha256": code_hash,
               "status_counts": {status: sum(row["status"] == status for row in rows) for status in ("executed", "running", "failure", "pending")},
               "runs": rows, "comparisons": comparisons, "validation_acceptance": acceptance,
               "decision_efficiency": decision_efficiency, "decision_efficiency_pairs": decision_efficiency_pairs,
               "attempt_history": _attempt_history(attempts, code_hash)}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Jevsight experiment report", "", experiment["scope"], "",
             "A: B3; B: time; C: time + LLM relationship consolidation; D: same consolidation + Jev.", "",
             "Each configuration is scored once on the frozen samples. Remote Answer/Judge and extraction outputs can vary across runs even at temperature zero, so small score differences are point estimates rather than proof of a robust gain.", "",
             "BEAM uses mean rubric score; CLBench uses all-rubrics accuracy; PersonaMem uses exact choice. Missing usage/cost is unavailable, never estimated as zero.", "",
             "| Split | Variant | Dataset | Status | Requested/evaluated/failed | Primary metric | Correct/requested | Actual ingestion s | Search P95 s |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        m = row["metrics"]
        if m is None:
            detail = f"{row['question_count']}/—/— | — | — | — | —"
        else:
            value = m.get("primary_value")
            primary = (f"{m['primary_metric']}={value:.4f}" if value is not None else f"{m['primary_metric']}=unavailable")
            if row["status"] != "executed":
                primary = "invalid (incomplete)"
            p95 = m.get("latency", {}).get("search", {}).get("p95_seconds")
            detail = f"{m['requested']}/{m['evaluated']}/{m['failed']} | {primary} | {m['correct']}/{m['requested']} | {m.get('actual_ingestion_seconds', 0):.2f} | {f'{p95:.3f}' if p95 is not None else '—'}"
        lines.append(f"| {row['stage']} | {row['variant'].upper()} | {row['dataset']} | {row['status']} | {detail} |")
    lines += ["", "## Paired comparisons", "", "| Split | Dataset | Contrast | Status | Gained/lost correct | Primary delta |", "| --- | --- | --- | --- | --- | --- |"]
    for item in comparisons:
        details = f"{item['gained_correct']}/{item['lost_correct']} | {item['primary_delta']:+.4f}" if item["status"] == "executed" else "— | —"
        lines.append(f"| {item['stage']} | {item['dataset']} | {item['contrast']} | {item['status']} | {details} |")
    lines += ["", "## Frozen validation acceptance", "", acceptance["validation_policy"], "",
              f"Optimization: **{acceptance['optimization_status']}**. Jev acceptance: **{acceptance['jev_status']}**; D-C native metric comparison alone: **{acceptance['jev_comparison_status']}**.", "",
              "The optimization requires D >= A on each of the five native primary metrics and positive D-A net correct on the fixed time/update subset. Jev additionally requires D >= C on every native primary metric. Missing paired results keep acceptance pending; no thresholds are relaxed.", "",
              "| Dataset | D >= A | D >= C |", "| --- | --- | --- |"]
    for dataset in DATASETS:
        lines.append(f"| {dataset} | {acceptance['d_vs_a_native_metrics'][dataset]['status']} | {acceptance['d_vs_c_native_metrics'][dataset]['status']} |")
    temporal = acceptance["time_update"]
    lines += ["", f"Time/update: **{temporal['status']}**; gained/lost/net correct: "
              + (f"{temporal['gained_correct']}/{temporal['lost_correct']}/{temporal['net_correct']:+d} over {temporal['paired_questions']} paired questions." if temporal['status'] != 'pending' else "unavailable until all required pairs complete."),
              "Included categories: LoCoMo 2; LongMemEval temporal-reasoning and knowledge-update; BEAM temporal_reasoning and knowledge_update. BEAM event_ordering and contradiction_resolution are outside this acceptance subset.",
              "", "## Relationship decision speed and cost", "",
              "C and D differ only in the relationship decision model. Model seconds sum observed decision-call latency; Add seconds are the measured first-ingestion wall time. Jev total tokens are input plus output when the provider omits a total. A missing call value makes the corresponding total unavailable. Provider credits are not USD.", "",
              "| Split | Dataset | Variant | Decision calls/failed | Decision model s | s/call | Decision tokens | Tokens/call | Decision credits | First Add wall s | All ingestion tokens | All ingestion credits |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    def display(value, digits=3):
        return f"{value:.{digits}f}" if value is not None else "—"
    for item in decision_efficiency:
        lines.append(f"| {item['stage']} | {item['dataset']} | {item['variant'].upper()} | {item['decision_calls']}/{item['failed_decision_calls']} | "
                     f"{display(item['decision_model_seconds'])} | {display(item['decision_seconds_per_call'])} | "
                     f"{display(item['decision_total_tokens'], 0)} | {display(item['decision_tokens_per_call'], 1)} | "
                     f"{display(item['decision_provider_credits'], 6)} | {display(item['ingestion_wall_seconds'])} | "
                     f"{display(item['ingestion_total_tokens'], 0)} | {display(item['ingestion_provider_credits'], 6)} |")
    lines += ["", "D/C ratios below 1 mean Jev used less time, tokens, or provider credits on that measure. Calls can differ because the models may choose different consolidation actions; compare both per-call speed and total workload cost.", "",
              "| Split | Dataset | Decision calls C/D | Decision s/call D/C | Decision tokens D/C | Decision credits D/C | First Add wall D/C | All ingestion tokens D/C | All ingestion credits D/C |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in decision_efficiency_pairs:
        lines.append(f"| {item['stage']} | {item['dataset']} | {item['c_decision_calls']}/{item['d_decision_calls']} | "
                     + " | ".join(display(item[f"d_over_c_{key}"]) for key in (
                         "decision_seconds_per_call", "decision_total_tokens", "decision_provider_credits",
                         "ingestion_wall_seconds", "ingestion_total_tokens", "ingestion_provider_credits")) + " |")
    lines += ["",
              "", "## Actual provider usage by stage", "", "Tokens and provider credits are sums of observed values; missing-call counts are shown explicitly. Provider credits are not converted to USD. A missing delta in report.json means at least one required usage value is unavailable.", "",
              "| Split | Variant | Dataset | Scope/stage | Calls/failed | Input tokens | Output tokens | Total tokens | Provider credits | USD | Model latency s | Resolved models (missing calls) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if row["metrics"] is None:
            continue
        for scope in ("model_usage", "provider_stage_usage"):
            for stage, usage in row["metrics"].get(scope, {}).items():
                models = ", ".join(usage.get("resolved_models", [])) or "unavailable"
                lines.append(f"| {row['stage']} | {row['variant'].upper()} | {row['dataset']} | {scope}/{stage} | {usage['calls']}/{usage['failed_calls']} | "
                             + " | ".join(_usage_text(usage, key) for key in USAGE_FIELDS)
                             + f" | {usage['latency_seconds']:.3f} | {models} ({usage.get('resolved_model_missing_calls', usage['calls'])}) |")
    lines += ["", "## D-A and D-C prediction cases", "", "Changed correctness and remaining incorrect predictions are recorded with IDs, categories, labels, exact predictions and per-question primary scores in report.json. Run paths identify original predictions.jsonl files.", "",
              "| Split | Dataset | Contrast | Case ID | Outcome | Left/right labels |", "| --- | --- | --- | --- | --- | --- |"]
    for item in comparisons:
        for case in item.get("cases", []):
            lines.append(f"| {item['stage']} | {item['dataset']} | {item['contrast']} | {case['id']} | {case['outcome']} | {case['left_label']}/{case['right_label']} |")
    lines += ["", "## All execution attempts, including interrupted development", "",
              "The main comparisons use only complete runs matching the current frozen inputs. This appendix retains previous source hashes and interruption reasons. Usage below contains observed provider records, including failed calls; an interrupted in-flight call can be unrecorded, so these sums do not claim total billed cost.", "",
              "| Split | Variant | Dataset | Current source | Return code | Attempt log | Reason | Observed run artifacts |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for attempt in payload["attempt_history"]:
        paths = ", ".join(item["run_path"] for item in attempt["run_artifacts"]) or "none"
        lines.append(f"| {attempt['stage']} | {attempt['variant'].upper()} | {attempt['dataset']} | {attempt['matches_current_source']} | {attempt.get('returncode')} | {attempt['log_path']} | {attempt.get('interruption_reason', '—')} | {paths} |")
    lines += ["", "| Run artifact | Observed stage | Calls/failed | Tokens | Provider credits | USD | Model latency s |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for attempt in payload["attempt_history"]:
        for artifact in attempt["run_artifacts"]:
            for stage, usage in artifact["observed_model_usage"].items():
                lines.append(f"| {artifact['run_path']} | {stage} | {usage['calls']}/{usage['failed_calls']} | {_usage_text(usage, 'total_tokens')} | {_usage_text(usage, 'provider_cost_credits')} | {_usage_text(usage, 'cost_usd')} | {usage['latency_seconds']:.3f} |")
    lines += ["", "Full per-stage paired resource deltas, missing-value counts, model latency, failed attempts and paths are in report.json and each run's model-calls.jsonl. Both stage scopes summarize the same calls from different perspectives and must not be added together.",
              "", "Validation freezes new question/complete-history samples; LongMemEval source sessions can overlap between splits. LoCoMo multi-hop validation comes from conv-26. CLBench remains the existing local semantic task evaluation with rubrics visible to Answer, not an official or pure retrieval score."]
    (ARTIFACTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="freeze IDs and create all 40 evaluation configs; no model calls")
    parser.add_argument("--run", choices=("dev", "validation", "all"), help="run the chosen stage, resuming complete successful runs only")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--report", action="store_true", help="write the current truthful report without model calls")
    args = parser.parse_args()
    if not (args.prepare or args.run or args.report):
        parser.error("choose --prepare, --run, or --report")
    experiment = prepare() if args.prepare else json.loads((ARTIFACTS / "experiment.json").read_text())
    variants, datasets = args.variants.split(","), args.datasets.split(",")
    if any(item not in VARIANTS for item in variants) or any(item not in DATASETS for item in datasets):
        parser.error("unknown variant or dataset")
    if args.run:
        code_hash = _source_hash()
        jobs = [job for job in experiment["jobs"] if job["variant"] in variants and job["dataset"] in datasets
                and (args.run == "all" or job["stage"] == args.run)]
        for stage in ("dev", "validation"):
            selected = [job for job in jobs if job["stage"] == stage]
            if stage == "validation" and selected and not development_complete(experiment, code_hash):
                print("Validation remains pending because all 20 development runs have not completed successfully under the current inputs.", flush=True)
                break
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda job: execute_job(job, code_hash), selected))
    result = report(experiment)
    print(json.dumps(result["status_counts"]), flush=True)
    print(f"Report: {ARTIFACTS / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
