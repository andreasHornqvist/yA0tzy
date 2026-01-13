---
task: Implement TUI run extension + replay + server management + oracle diagnostics (spec)
test_command: "cargo test -p yz-tui"
---

# Task: TUI Feature Suite (Spec + Acceptance Criteria)

Define an implementation-ready spec for the following TUI features, with **clear acceptance criteria** and **end-to-end test procedures** where possible:

- Extend runs (start from existing)
- Game replay (gating replay per seed)
- Manage servers/workers (do not kill on TUI quit)
- New Oracle screen (incl. fixed oracle set)

## Requirements

1. Covers 4 features:
   - Extend runs
   - Gating replay viewer
   - Manage servers/workers
   - Oracle screen (incl. fixed oracle set)
2. Includes explicit, verifiable acceptance criteria using `[ ]` checkboxes.
3. Includes end-to-end test procedures (manual steps are OK).
4. Defines any required persisted data/contracts (file locations + minimal fields).
5. States backward-compat expectations for older runs (missing fields render as `-`).

---

## Success Criteria

### 1) Extend Runs (fork/continue from existing)

#### User flow (UX)
- From Dashboard `(d)`:
  - Press **`e`** = Extend run
  - Select run -> enter new name -> edit config (pre-filled from source) -> confirm
  - New run starts and **continues counting iterations** from the source run

#### Copy rules (deterministic + safe)
- **Copied**:
  - `config.yaml` (and optionally `config.draft.yaml`)
  - `models/best.pt` (+ any required artifacts)
  - optionally `replay/` (toggle)
  - `run.json` is re-created as a new manifest derived from source (see below)
- **Not copied**:
  - `logs/`, `logs_workers/`
  - `gate_workers/`, `replay_workers/`, `logs_gate_workers/`
  - stop/cancel files

#### Manifest rules (iteration continuation)
- New run must:
  - set `created_ts_ms` to “now”
  - set `run_id` to new name
  - set next iteration index to source run’s `controller_iteration_idx`
  - record provenance (“extended from <src>”) in the new run metadata

#### Acceptance criteria
- [ ] On Dashboard `(d)`, pressing **`e`** opens an “Extend Run” modal/screen.
- [ ] User can select a source run from `runs/`.
- [ ] User can enter a new run name (and it does not conflict with quit keys).
- [ ] Config editor defaults to the source run’s config snapshot.
- [ ] Confirm creates `runs/<new_id>/` with valid `config.yaml` and `run.json`.
- [ ] New controller continues at `iter_idx == src.controller_iteration_idx`.
- [ ] Toggle “include replay” works (on: replay copied; off: replay empty/absent).
- [ ] Destination `logs/metrics.ndjson` starts fresh (no source history mixed in).

#### End-to-end test
- [ ] **E2E-ER-1**: fork + continue
  - Preconditions: pick a completed run (e.g. `runs/p9`) with `controller_iteration_idx > 5`
  - Steps: Dashboard `(d)` -> `e` -> select `p9` -> name `p9_ext_1` -> confirm
  - Expected:
    - `runs/p9_ext_1/run.json` exists and parses
    - first new iteration index == old `controller_iteration_idx`
    - `runs/p9_ext_1/logs/metrics.ndjson` contains only new events

---

### 2) Game Replay (analysis) — gating replay per seed

#### Data contract (required)
Persist per-game gating traces (rolls/actions/turn progression + final scores) in a discoverable format:
- location (pick one):
  - `runs/<id>/gate_replays/seed_<seed>_<game_idx>.json`, or
  - `runs/<id>/logs/gate_replays.ndjson`
- must include:
  - seed, game index, side assignment (best/candidate), perspective info
  - per-step: turn index, roll(s), chosen action
  - final: totals, diff, winner
  - paired seed swap: link the two games for a seed as a pair

#### UX
- From Dashboard `(d)`: press **`r`** to open “Gating Replay”
- Left panel: seed list
- Right: if `paired_seed_swap=true`, show **pair side-by-side**, synced by turn index
- Controls:
  - Space: step forward one roll/action
  - Back-step (e.g. `b`)
  - Toggle “show whole game” vs step mode

#### Acceptance criteria
- [ ] Gate workers persist replay traces for gating games.
- [ ] Replay traces can be loaded after the run is done (no workers required).
- [ ] New TUI replay screen exists.
- [ ] User can select a seed and view games for that seed.
- [ ] If paired seed swap is enabled, the pair is shown side-by-side with clear labels.
- [ ] Space advances deterministically; back-step works.
- [ ] Endgame stats are always visible (totals/diff/winner).

#### End-to-end test
- [ ] **E2E-GR-1**: paired replay rendering
  - Preconditions: run with `paired_seed_swap=true` completes >= 20 gate games
  - Steps: open replay screen -> pick a seed with a full pair -> step 5 turns
  - Expected: both columns stay in sync by turn; final totals match gating report

---

### 3) Manage servers/workers (no shutdown on `q`)

#### UX
- On System/Inference screen (`I`): show a “Manage / Servers” panel
- Lists active processes (infer/controller/selfplay-worker/gate-worker) with PID, run association, endpoints, last-seen
- Kill action: select -> `k` -> confirm -> SIGTERM then SIGKILL (timeout)

#### Acceptance criteria
- [ ] System/Inference screen has a “Manage / Servers” panel.
- [ ] Panel lists discovered processes with PID + type + run association where possible.
- [ ] Key endpoints show when available (infer bind, `metrics_bind`).
- [ ] Kill flow works with confirmation and escalation.
- [ ] Quitting the TUI with `q` does **not** shut down any servers/workers.
- [ ] Restarting the TUI can reattach to already-running runs and show live metrics.

#### End-to-end test
- [ ] **E2E-MS-1**: restart TUI without stopping servers
  - Preconditions: start a run (controller + infer active)
  - Steps: quit TUI -> relaunch TUI
  - Expected: run continues; TUI shows it progressing

---

### 4) New Oracle screen (two-panel + fixed oracle set)

#### Goal
Make oracle diagnostics **comparable across iterations** by adding a **fixed oracle state set** evaluation.

#### UX (two panel)
- Left: iterations table with:
  - existing oracle accuracy (current)
  - regret vs oracle (gating games)
  - **fixed-set oracle accuracy**
- Right: breakdown for selected iteration (by move type, turn buckets, etc.)

#### Data contract (fixed oracle set)
- Store deterministic oracle sets under `configs/oracle_sets/<id>.json` (or `.ndjson`)
- Config references: `oracle.fixed_set_id: <id>` (name TBD)
- Per-iteration results persisted to `logs/metrics.ndjson` (preferred) with breakdowns

#### Acceptance criteria
- [ ] New TUI Oracle screen exists (keybinding TBD; e.g. `o`) with two-panel layout.
- [ ] Fixed oracle set is evaluated on the **same states** each iteration.
- [ ] Fixed oracle metrics are persisted per iteration.
- [ ] Right panel shows breakdown by move type and turn buckets.
- [ ] Backward compatible: older runs without these fields render with `-`.

#### End-to-end test
- [ ] **E2E-OR-1**: fixed oracle stability
  - Preconditions: create `oracle_set_dev_v1` with N states
  - Steps: run 3 iterations with minimal/no changes -> compare fixed-set oracle acc
  - Expected: values are comparable and don’t jump due to distribution shift alone

---

## Open Questions (resolve before implementation)

1. [ ] Keybindings:
   - Extend run: keep `e` on Dashboard?
   - Replay: `r` on Dashboard vs Performance screen?
   - Oracle: `o`?
2. [ ] Extend-run: support “copy logs” option, or always start fresh logs?
3. [ ] Replay trace format: per-game JSON files vs one NDJSON stream?
4. [ ] Fixed oracle set size: how many states (N) is “cheap enough” per iteration on CPU?
