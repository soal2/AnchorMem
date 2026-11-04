# Recovery after the replay fix

Run these commands from `/workspace/AnchorMem`, with the old runner stopped.
The fix does not edit the frozen baselines, budget, task database, or live manifest.
The operator must run amend-code revision 2 after the fix commit and before resuming.

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

2. The other operator runs the already planned revision 2 amendment:

```bash
.venv/bin/python -m src.week1_runner amend-code --run-dir artifacts/week1/20261103T012529354847Z --reason 'Pin Mem0 run date; classify region refusals; bound pauses; prohibit possibly billed resends'
```

3. Requeue recoverable failures and interrupted tasks without deleting failure
history, completed stages, ingestion units, or call checkpoints:

```bash
.venv/bin/python - <<'PY'
import json
from src.week1_tasks import TaskStore, retryable_error
store = TaskStore('artifacts/week1/20261103T012529354847Z/tasks.sqlite3')
selected = []
with store.connection:
    for row in store.rows():
        if row['method'] not in {'ours', 'mem0'}:
            continue
        error = row['error'] or {}
        text = json.dumps(error).lower()
        if 'uncertain' in text or 'will not be repeated' in text:
            continue
        recoverable = retryable_error(error) or 'week1_replay_mismatch' in text
        if row['state'] == 'running' or (row['state'] == 'failed' and recoverable):
            store.connection.execute("UPDATE tasks SET state='pending',error=NULL WHERE task_id=?", (row['task_id'],))
            selected.append(row['task_id'])
    # Release QA only when its build is runnable or already complete.
    for row in store.rows('qa'):
        if row['method'] not in {'ours', 'mem0'} or row['state'] != 'blocked':
            continue
        if store.build(row['method'], row['group'])['state'] in {'pending', 'running', 'success'}:
            store.connection.execute("UPDATE tasks SET state='pending',error=NULL WHERE task_id=?", (row['task_id'],))
            selected.append(row['task_id'])
print(f'Requeued {len(selected)} tasks')
store.close()
PY
```

This explicitly grants one new task pass even if historical failure counts reached
four. It preserves the audit history and does not reset the gateway attempt limit.
Real 400s, invalid extraction output, and uncertain calls stay failed. The gateway
still blocks any possibly billed resend and any unsafe changed checkpoint.
Budget-stopped tasks are handled by the runner's existing resume policy; full
reservations remain charged against the unchanged $20 cap.

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
