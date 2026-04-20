# Golden end-to-end cases

Each case directory here drives a replay-only run of the tracer. No live
MCP, no live OpenAI — every tool + LLM result comes from the case's
`recording.jsonl`.

## Directory layout

```
tests/golden/case_<id>/
    input.json          # TracerConfig fields (dict passed to TracerConfig(**input))
    recording.jsonl     # captured via AGENT_RECORD=1 on a real run
    expected.json       # subset-based metrics (see tests/golden/metrics.py)
    metadata.yaml       # optional free-form notes (not parsed)
```

Directories starting with `_` or `.` are ignored (that's where
`_template_case/` lives as a shape example).

## Workflow to add a new case

1. Run the tracer live with `AGENT_RECORD=1`:

   ```bash
   AGENT_RECORD=1 AGENT_RECORDINGS_DIR=recordings \
     .venv/bin/python -m agent.cli "..." <victim> eth
   ```

2. Copy the resulting `recordings/<date>/<trace_id>.jsonl` into
   `tests/golden/case_<id>/recording.jsonl`.

3. Write `input.json` with the same `TracerConfig` you ran.

4. Write `expected.json` asserting only the outcomes you care about
   (terminal addresses, path count range, traced amount, chains,
   specific entity roles). Unspecified metrics short-circuit to ok.

5. `.venv/bin/pytest tests/golden/case_<id> -v` — it should pass. Any
   later code change that breaks the replay will show up as a metric
   failure.

## Metrics

See `tests/golden/metrics.py`. They're all opt-in per case:

* `terminal_addresses` — required terminal endpoints.
* `paths_min` / `paths_max` — bracketed path count.
* `traced_amount` + `traced_tolerance` — FIFO-traced volume.
* `chains` — required cross-chain coverage.
* `entity_roles` — `{address: role}` map.

Add a new metric by dropping a function in `metrics.py` with the same
`(ok, detail)` return shape and appending it to `ALL_METRICS`.
