"""Run one frozen benchmark question after directory-first retrieval, without rebuilding its index."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from src import evaluation
from src.config import EvaluationConfig, load_config
from src.engine import MemoryEngine

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "artifacts/vikingjev/directory-all-top4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["locomo", "longmemeval", "personamem", "beam", "clbench"])
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    baseline = json.loads((PROJECT / "docs/results/vikingjev-benchmark-smoke.json").read_text())["results"][args.dataset]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = []
    if args.retry_failed:
        prior = json.loads((OUTPUT / f"{args.dataset}.json").read_text())
        metrics = json.loads((PROJECT / prior["run"] / "metrics.json").read_text())
        if metrics["failed"] == 0 and metrics["evaluated"] == metrics["requested"]:
            raise ValueError("refusing to repeat a completed benchmark")
        previous = prior.get("attempts", [prior["run"]])
        (OUTPUT / f"{args.dataset}.retry-{len(previous)}.started").open("x").close()
    else:
        (OUTPUT / f"{args.dataset}.started").open("x").close()
    vector_reads, routing = [], []

    class MeasuredEngine(MemoryEngine):
        def _vector_rank(self, db, vector, kind, limit, directory_ids=None):
            started = perf_counter()
            record = {"kind": kind, "scope_directories": None if directory_ids is None else len(directory_ids),
                      "vectors_loaded": 0}

            class CountedConnection:
                def execute(self, sql, parameters):
                    cursor = db.execute(sql, parameters)

                    class CountedCursor:
                        def fetchall(self):
                            rows = cursor.fetchall()
                            record["vectors_loaded"] += len(rows)
                            return rows
                    return CountedCursor()
            result = super()._vector_rank(CountedConnection(), vector, kind, limit, directory_ids)
            record["seconds"] = perf_counter() - started
            vector_reads.append(record)
            return result

        def _select_directories(self, db, request):
            ranked = [r[0] for r in db.execute("SELECT DISTINCT topic_id FROM directory_topics WHERE selected=1 ORDER BY topic_id")]
            decisions = []
            original = self.directory_selector.select
            def measured_select(*args):
                batch = original(*args)
                decisions.extend(batch)
                return batch
            self.directory_selector.select = measured_select
            try:
                selected = super()._select_directories(db, request)
            finally:
                self.directory_selector.select = original
            routing.append({"candidates": ranked, "accepted": selected, "decisions": decisions})
            return selected

    evaluation.MemoryEngine = MeasuredEngine
    config, path = load_config(PROJECT / f"configs/evaluation/vikingjev-smoke/{args.dataset}.yaml", EvaluationConfig)
    config = config.model_copy(update={"run_label": f"directory-all-top4-{args.dataset}", "output_dir": str(OUTPUT / "runs")})
    run = evaluation.evaluate(config, path, reuse_index_from=PROJECT / baseline["run_path"])
    (run / "retrieval-work.json").write_text(json.dumps({"vector_reads": vector_reads, "directory_routing": routing}, indent=2))
    (run / "comparison-runner.py").write_bytes(Path(__file__).read_bytes())
    (OUTPUT / f"{args.dataset}.json").write_text(json.dumps({"baseline_run": baseline["run_path"],
                                                           "run": str(run.relative_to(PROJECT)),
                                                           "attempts": [*previous, str(run.relative_to(PROJECT))]}, indent=2))
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics["evaluated"] != metrics["requested"] or metrics["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
