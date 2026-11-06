# Runner hang diagnostics and recovery

The runner enables all-thread faulthandler output after acquiring its executor
lock, before opening the task database or starting the gateway. Output appends to
`<run-dir>/thread-stacks.log`. `--stack-dump-seconds` defaults to 300 and produces
periodic dumps even during healthy long operations. On a runner started with this
code, request an immediate dump with `kill -USR1 <pid>`. This command is documented
only; no signal was sent to the existing live runner. That runner does not gain
these handlers from an on-disk code update.

`--progress-timeout-seconds` defaults to 900 (15 minutes). A separate daemon tracks
mtime/size changes to runtime-status.json, the task database/WAL, and the shared
API ledger/checkpoint database/WAL. It uses a monotonic last-progress timestamp and records
the UTC timestamp in the final dump. Stack dumps themselves do not count as
progress. Both intervals must be positive. Set the progress timeout above any
intentional pause or expected long local computation. The current cumulative
failure-pause allowance is at most 600 seconds.

On expiry the watchdog dumps every Python thread, flushes the stack log, and calls
`os._exit(124)`. It takes no engine, task, gateway, or transport locks and does not
run cleanup or join abandoned workers. A C extension holding the GIL can delay
the Python watchdog; the native faulthandler timer still provides periodic stacks.

SQLite WAL commits and the fsynced reservation ledger remain authoritative.
Resume resets running tasks to pending, preserves committed units/stages, and
replays successful API checkpoints. Resume creates a unique unpublished work directory for each attempt, so a
surviving baseline child cannot write into the resumed attempt. Old work is left
unpublished rather than reused or deleted; no child is signalled by the watchdog.
Unknown-cost and interrupted requests retain their full cost upper bounds even
when retried; the authorized hard cap remains $20. Transient deadline failures
(including wrapped or legacy incorrectly non-retryable failures) requeue builds
and QA, unblock dependencies, and retain the existing task/call attempt limits.
An exit can leave manifest/runtime status saying running; those are display
snapshots, not the transaction authority.

## Static audit findings

- Gateway state RLock covered upstream I/O and retry sleeps. State locking now
  covers reservation and completion transactions only; waits for it are bounded.
  A separate bounded request gate preserves serial checkpoint ordinals/context.
  Abandoned I/O workers never acquire either application lock.
- Upstream HTTP pools could be shared with abandoned attempts. On a production
  attempt deadline, a fresh client/pool is used for subsequent attempts; old pool
  cleanup runs only in an unjoined daemon. Injected offline test transports retain
  their existing behavior.
- Server shutdown, routed-client close, and native transport close could block
  cleanup. They now run with wall-clock bounds. Server thread joins and executor
  child reaping in the operational runner have explicit timeouts. Local HTTP handler socket I/O is bounded.
- Native ingestion holds its per-user lock/transaction during provider calls, but
  the abandoned worker owns only detached transport I/O, never the engine or DB.
  Existing socket-deadline regressions verify rollback, lock release, and retry on
  the same engine. Frozen method/baseline sources were left unchanged.
- Transport queues already use timed gets; there are no executor shutdown/join
  or Condition waits in the runner/gateway path. No abandoned worker is joined.
  Frozen baseline adapters can still contain their own waits; the independent
  process watchdog bounds those without changing the frozen sources.
  These are candidates and mitigations, not proof of the historical futex cause;
  stacks from a future hang are needed to identify that cause.

## Resume the affected LoCoMo run

Do this only after the current executor has ended; do not start another executor
for the active run. No amendment or resume was performed during this repair.
The operational amendment step is required by the existing frozen-code verifier.
From the repository root:

```bash
cd /workspace/AnchorMem
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
uv run python scripts/week1_experiments.py amend-code \
  --run-dir artifacts/week1/20261105T122803255750Z \
  --reason 'Bound deadlock waits, add stack/progress watchdog, preserve deadline retries'
uv run python scripts/week1_experiments.py run \
  --benchmark locomo --methods ours mem0 memorybank --budget-usd 20 \
  --run-dir artifacts/week1/20261105T122803255750Z \
  --attempt-deadline-seconds 180 \
  --progress-timeout-seconds 900 --stack-dump-seconds 300
```

For a new instrumented runner, `kill -USR1 <pid>` appends all-thread stacks to
`/workspace/AnchorMem/artifacts/week1/20261105T122803255750Z/thread-stacks.log`.
