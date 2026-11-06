LoCoMo conv-26 decision rejection, 2026-11-06

Run: artifacts/week1/20261105T122803255750Z. Pulled memorybank-benchmark;
origin was already current. No authenticated inference requests were sent.

Public GET https://openrouter.ai/api/v1/models omits typesafe/jev-1.13
(and other decision models); typesafe/jev-router is listed. The model page
https://openrouter.ai/typesafe/jev-1.13 still lists the original slug, one
provider and 262,144-token context. No evidence of rename/deprecation was found.
The run's stored successful responses resolve to typesafe/jev-1.13-20260917.
Keep the pinned slug; the public catalog discrepancy is not this failure's cause.

The original 400 body was not persisted in failures.jsonl. There is no upstream
400 for this task in the shared ledger. Offline reproduction on a temporary copy
of the committed index replayed all 48 session_5:0 responses, verifying each
prepared request's SHA256 against the shared read-only checkpoint database.
Request 49 failed locally in Gateway._prepare before reservation or network I/O:
HTTP 400, reconstructed handler body:
{"error":{"code":"week1_input_too_long","message":"decisions exceed conservative input bound; no truncation","type":"invalid_request_error"}}

The exact rejected payload has a nonempty JSON state, 32 relation criteria,
no invalid control characters and no chat messages or generation parameters
(the decisions API uses state/questions). Encoded state: 25,740 bytes;
question plus name/overhead: 6,181 bytes; fixed overhead: 256 bytes.
Total: 32,177, exceeding the local 32,000-byte per-question conservative bound.
This is not evidence that the model's 262,144-token context was exceeded.
Compacting JSON formatting reduces encoded state to 24,429 bytes and total to
30,866. Every evidence value is retained. Original HTTP payload: 31,913 bytes.

Gateway compacts only otherwise-rejected JSON states. Prior successful requests
remain byte-identical, so checkpoint replay remains valid. Pricing bounds and
budget are unchanged. Oversize evidence after compaction is still rejected.
The runner now retains nested HTTP status/body in redacted failure artifacts.
The precise legacy receipt gets a one-time requeue migration, retaining the total
four-attempt limit (two already used); other 400s retain permanent classification.
Read-only requeue dry-run confirmed eligibility against the real checkpoint DB.

The live process and manifest were left alone. Once that process exits, record
the operational amendment with amend-code before running the resume command.
No frozen archive, baseline source, or run source snapshot was modified.

Validation: full `uv run pytest` completed with 903 passed, 10 skipped and one
existing Starlette/httpx deprecation warning. `git diff --check` passed.
