# Fixed-set oracle diagnostics (E2E)

This describes the end-to-end behavior for the **optional** fixed-set oracle metric (`gating.fixed_oracle.*`).

## Setup

- Generate a fixed set once (deterministic):
  - `yz oracle-set-gen --id fixed_v1 --n 4096 --seed 123`
  - This writes `configs/oracle_sets/fixed_v1.json`
- Enable in a run config:
  - `gating.fixed_oracle.enabled: true`
  - `gating.fixed_oracle.set_id: "fixed_v1"`

## Expected runtime behavior

- During each iteration, the controller spawns `yz oracle-fixed-worker ...` **asynchronously**.
- Gating/promotion/finalize must **not** wait for fixed-set completion.
- When the worker finishes, it appends one NDJSON event:
  - `runs/<run_id>/logs/metrics.ndjson`:
    - `event: "oracle_fixed_summary_v1"`
    - `iter_idx`, `set_id`, `num_states`
    - overall + breakdown match rates

## TUI verification

- Open the run in `yz tui`
- From Dashboard:
  - press `o` to open the Oracle screen
- The fixed-set column shows:
  - `-` if no `oracle_fixed_summary_v1` has arrived for that iteration yet
  - the match rate once it arrives

