# Full LoCoMo benchmark

From `/workspace/AnchorMem`, prepare the MemoryBank runtime (no model calls):

```bash
.venv/bin/python -m src.week1_runner prepare --benchmark locomo --methods memorybank
```

Mem0 is already prepared on this box. To prepare both baseline environments on
another box, omit `--methods memorybank`. The LoCoMo path validates the existing
`data/week1/datasets/locomo10.json` and LoCoMo scorer, without loading or fetching
LongMemEval. The legacy default remains the mixed 32-question pilot.

The existing MemoryBank author manifest is preserved. If its original platform
lock is unavailable, prepare installs the adapter's pinned direct dependencies
and freezes the resolved local environment into
`data/week1/envs/memorybank-requirements.lock`. The separate
`memorybank-environment.json` receipt binds that lock to the unchanged author
manifest. Subsequent prepare operations reuse and validate this local lock.

The operator supplies the key in their shell; preflight does not read `.env` or
call a model:

```bash
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
.venv/bin/python -m src.week1_runner preflight --benchmark locomo
```

Start the authorized full run only when ready:

```bash
.venv/bin/python -m src.week1_runner run --benchmark locomo --budget-usd 20
```

This selects `ours` (VikingJev), `mem0`, and `memorybank`, and all 1,540 questions
with category != 5 across 10 complete histories. Scoring uses the existing
LoCoMo Token F1 function; there are no judge calls. The gateway, shared
`artifacts/week1/api-ledger.jsonl`, $20 total cap, and retry logic are reused.
Prior spending counts toward that cap; this does not authorize another $20.

Optional `--methods ours mem0` selects an ordered subset. Optional
`--max-questions 10` takes the first 10 scorable questions in source order,
while retaining complete conversation histories. For a resume, supply
`--run-dir artifacts/week1/<run-id>` and the same benchmark, budget, methods,
and question cap used originally. Run selections and locks are frozen in the
new run directory; existing runs are not amended.
