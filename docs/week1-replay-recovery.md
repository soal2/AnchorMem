# Recovery after the replay fix

Run these commands from `/workspace/AnchorMem`, with the old runner stopped.
The fix does not edit the frozen baselines, budget, task database, or live manifest.
The operator must run amend-code with the current working tree before resuming.

1. Back up SQLite with its backup API (includes committed WAL contents):

```bash
.venv/bin/python - <<'PY'
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
run = Path('artifacts/week1/20261103T012529354847Z')
source = sqlite3.connect(f'file:{run / "tasks.sqlite3"}?mode=ro', uri=True)
backup = run / ('tasks.sqlite3.backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
with sqlite3.connect(backup) as target:
    source.backup(target)
source.close()
print(backup)
PY
```

2. Record the current code amendment:

```bash
.venv/bin/python -m src.week1_runner amend-code --run-dir artifacts/week1/20261103T012529354847Z --reason 'Re-evaluate legacy failures from unbilled checkpoint evidence; bound recovery requeues'
```

3. Preview the exact resume requeue selection without writing to either database
or starting the runner. This prints task identity only, never error payloads:

```bash
.venv/bin/python - <<'PYCODE'
import sqlite3
from pathlib import Path
from src.week1_tasks import TaskStore
run = Path('artifacts/week1/20261103T012529354847Z').resolve()
store = TaskStore.__new__(TaskStore)
store.path = run / 'tasks.sqlite3'
store.connection = sqlite3.connect(store.path.as_uri() + '?mode=ro', uri=True)
store.connection.row_factory = sqlite3.Row
try:
    selected = store.requeue(4, run.parent / 'api-call-checkpoints.sqlite3', dry_run=True)
    for task_id in selected:
        task = store.get(task_id)
        print(task['method'], task['kind'], task['group'], task['id'], task_id)
    print('Would requeue:', len(selected))
finally:
    store.close()
PYCODE
```

Resume now performs this reclassification itself. Do not run the former manual
SQL requeue snippet: it bypassed billing evidence and attempt bounds. Only
VikingJev (`ours`) and Mem0 are eligible. Legacy non-retryable failures get a
bounded recovery allowance when checkpoints prove the calls unbilled (region
refusals, replay mismatches, or unsent connections). Each recovery requeue is
persisted separately, so repeated resumes cannot reset that allowance. Genuine
400s without replay mismatch remain failed. Possibly billed checkpoints,
including uncertain cn:焦彦, remain blocked for manual review. Existing successful
stages, failure history, call checkpoints and the $20 budget are preserved.

4. Start exactly one runner in the foreground (do not run a second copy):

```bash
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
.venv/bin/python -m src.week1_runner run --benchmark memorybank --budget-usd 20 --run-dir artifacts/week1/20261103T012529354847Z
```

The exclusive executor lock rejects a second runner. The legacy run pins
`mem0_prompt_date` to `2026-11-03` before initializing a worker; new runs persist
their date at creation. Check the run's manifest/status and ingestion records for
successful Mem0 Chinese units. Never delete or reconcile call checkpoints merely
to bypass an uncertain/possibly billed call.
