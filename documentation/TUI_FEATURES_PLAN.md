### TUI Feature Plan (Spec + Acceptance Criteria + Test Criteria)

This document specifies a set of TUI features with **clear, validatable acceptance criteria** and **end-to-end test procedures** where feasible.

Scope in this iteration:
- **Extend runs** (fork/continue from an existing run from Dashboard)
- **Game replay (analysis)** (gating replay per seed, paired seed swap visualization)
- **Manage servers** (discover running workers/servers, view stats/config, kill)
- **New Oracle screen** (iteration-indexed oracle metrics + fixed oracle set)

Non-goals (explicitly out of scope for this spec unless added later):
- Changing core training/search algorithms (beyond computing new oracle metrics)
- GUI (non-TUI) tooling
- Remote cluster orchestration (beyond local process discovery/kill)

---

## Glossary / Terms

- **Run**: A folder under `runs/<run_id>/` with `run.json`, `config.yaml`, `logs/metrics.ndjson`, `models/`, `replay/`, etc.
- **Iteration**: `iter_idx` as used across `run.json` and NDJSON events.
- **Best**: current promoted model (`models/best.pt`).
- **Candidate**: training model (`models/candidate.pt`).
- **Gate**: evaluation games between best and candidate (seeded, optionally paired seed swap).
- **Paired seed swap**: For each seed, play two games with swapped player perspectives.
- **Oracle**: reference policy/value oracle (currently used for oracle match metrics during gating).

---

## Feature 1: Extend Runs (start from existing run)

### Goal
From Dashboard `(d)`, the user can fork an existing run into a new run directory, with config defaults taken from the source run, and continue iter indexing from the source run.

### User flow (UX)
- On Dashboard `(d)`:
  - Press **`e`** = Extend run.
  - Select a run (from the visible run list).
  - Enter:
    - **New run name** (destination `runs/<new_id>/`)
    - **Config edit** (defaults to the source run’s `config.yaml` / config snapshot)
  - Confirm -> copies run folder (see copy rules) -> starts controller using the new folder.

### Copy rules (deterministic + safe)
The new run should contain:
- **Copied**
  - `config.yaml` (and optionally `config.draft.yaml` if the TUI uses it)
  - `run.json` as a *new* manifest derived from source (see manifest rules)
  - `models/best.pt` and any other required model artifacts
  - `replay/` (optional, controlled by a toggle; see below)
  - `configs/` snapshot files if the run uses them (as applicable)
- **Not copied**
  - `logs/` and `logs_workers/` (to avoid mixing history unless we explicitly want it)
  - `gate_workers/`, `replay_workers/`, `logs_gate_workers/` (runtime ephemeral)
  - any `cancel.request`, stop files, etc.

Recommended toggles (TUI prompts):
- **Include replay buffer**: `Yes/No` (default: `Yes` if goal is “continue training”; `No` if goal is “restart training from same best”).
- **Include candidate + optimizer state**: `Yes/No` (default: `No` unless `continuous_candidate_training=true` and user explicitly wants to continue candidate).

### Manifest rules (iteration continuation)
The new run must:
- Set `created_ts_ms` to “now”
- Set `run_id` to the new name
- Set `controller_iteration_idx` (start) to the source run’s **current** iteration idx
- Preserve provenance by recording source run in the new run manifest (new fields or existing `init_from`/metadata).

Two modes:
- **Continue iteration numbers** (requested):
  - If source ended at iteration `N` (last completed = `N-1`), next iteration in new run should be `N`.
  - TUI should display something like “Extending from `<src>` at iter `<N>`”.

### Acceptance criteria (checkboxes)
- [ ] On Dashboard `(d)`, pressing **`e`** opens an “Extend Run” modal/screen.
- [ ] The modal lists runs from `runs/` and allows selection.
- [ ] The “new run name” input accepts typical characters and does not conflict with global quit keys.
- [ ] The config editor is pre-filled with the source run config snapshot.
- [ ] On confirm, a new directory `runs/<new_id>/` is created.
- [ ] The new run has a valid `config.yaml` and `run.json` and is startable.
- [ ] The controller for the new run begins at `iter_idx == src.controller_iteration_idx` (continues counting).
- [ ] If “Include replay buffer” is **off**, `runs/<new_id>/replay/` is empty (or absent) and training still works.
- [ ] If “Include replay buffer” is **on**, `runs/<new_id>/replay/` exists and matches the chosen copy behavior.
- [ ] No logs from the source run are mixed into the destination run’s `logs/metrics.ndjson` (unless explicitly enabled as a future option).

### End-to-end test procedure
- **E2E-ER-1: fork + continue**
  - Preconditions: pick a completed run `runs/p9` (or any) with `controller_iteration_idx > 5`.
  - Steps:
    - Open TUI, go Dashboard `(d)`.
    - Press `e`, select `p9`, name it `p9_ext_1`, keep config defaults, include replay on.
    - Confirm start.
  - Expected:
    - `runs/p9_ext_1/run.json` exists and is valid JSON.
    - TUI shows iteration numbers continuing (first new iteration is the old run’s next index).
    - New `logs/metrics.ndjson` starts fresh (no old events).

---

## Feature 2: Game Replay (analysis) — gating replay per seed

### Goal
Provide a “nice TUI” for inspecting gating games by seed, including paired seed swap side-by-side visualization so it’s easy to see differences between perspectives.

### Key requirement
We need **persisted per-game gating traces** (rolls/actions/turn progression + final score breakdown) in a discoverable format.

#### Proposed data contract (minimal viable)
For each gating game executed by gate workers, write a compact JSON (or NDJSON) trace:
- location: `runs/<id>/gate_replays/seed_<seed>_<game_idx>.json` (or `gate_replays.ndjson`)
- contains:
  - seed, game index, which side is best/candidate, perspective info
  - per-step:
    - turn index
    - dice roll(s) (if applicable)
    - chosen action
    - intermediate scoreboard deltas (optional)
  - final:
    - total scores for both players
    - score diff
    - winner
    - optional oracle stats for the game

Paired seed swap:
- Two games for the same seed should be linkable as a “pair”.

### User flow (UX)
- From Dashboard `(d)` during/after gating:
  - Press **`r`** = Replay (or another key, to be decided) -> opens “Gating Replay” screen.
- Replay screen:
  - Select **seed** from list (left panel)
  - If paired:
    - show **two games side-by-side** (best/candidate swapped perspectives) synchronized by turn index
  - Controls:
    - **Space**: step forward one “roll/action”
    - **Shift+Space** (or `b`): step backward
    - **g**: toggle “show whole game” vs “step-by-step”
    - **t**: toggle details (rolls only / rolls+actions / include derived stats)
  - Show final endgame stats prominently (score breakdown, diff, winner)

### Acceptance criteria (checkboxes)
- [ ] Gate workers persist replay traces for all gating games (or at least for `seed_set_id` runs).
- [ ] Replay traces are keyed by seed and can be loaded without the original workers running.
- [ ] A new TUI screen exists for gating replay browsing.
- [ ] User can select a seed and see all games for that seed.
- [ ] If `paired_seed_swap=true`, the UI shows the pair side-by-side with clear labels of perspective.
- [ ] Space advances a single step deterministically; reverse-step works.
- [ ] “Show whole game” mode renders full history (rolls/turns/actions) without requiring stepping.
- [ ] Endgame stats are always visible (score totals, diff, winner).

### End-to-end test procedure
- **E2E-GR-1: paired replay rendering**
  - Preconditions: run with `paired_seed_swap=true` completes at least 20 gate games.
  - Steps:
    - Open replay screen, choose a seed with a full pair.
    - Step through first 5 turns, compare both columns.
  - Expected:
    - Two perspectives stay in sync by turn index.
    - Final totals match those recorded in gating summary.

---

## Feature 3: Manage Servers / Workers (without breaking running runs)

### Goal
Add a screen (on **`I`** / “Inference”) that shows active workers/servers, their key stats + config, and provides an option to kill them.

Also: **do not shut off servers on `q`**. The TUI should be restartable after code changes and be able to reattach to already-running runs if backward-compatible.

### Design notes
We have multiple “process types”:
- Python inference server (socket + Prometheus metrics endpoint)
- `yz controller`
- `yz selfplay-worker` processes
- `yz gate-worker` processes

To manage them, we need:
- Discovery (PID + command line + run_dir association)
- Status (running/dead)
- Light stats (from progress files + `/metrics` where applicable)
- Kill action (SIGTERM first, then SIGKILL)

### Acceptance criteria (checkboxes)
- [ ] A new “Manage / Servers” panel exists on the System/Inference screen (`I`).
- [ ] The panel lists discovered processes with:
  - [ ] PID
  - [ ] type (infer/controller/selfplay-worker/gate-worker)
  - [ ] run_id/run_dir association when possible
  - [ ] key endpoints (infer bind, metrics_bind) when available
  - [ ] last-seen timestamp / alive indicator
- [ ] Selecting a process and pressing `k` prompts “Kill? (y/n)”.
- [ ] Kill uses SIGTERM; if still alive after a timeout, escalates to SIGKILL (documented).
- [ ] Pressing `q` to quit the TUI does **not** shut down running servers/workers.
- [ ] Restarting the TUI shows existing running runs and their live metrics.

### End-to-end test procedure
- **E2E-MS-1: restart TUI without stopping servers**
  - Preconditions: start a run (controller + infer server active).
  - Steps:
    - Quit TUI with `q`.
    - Relaunch TUI.
  - Expected:
    - Run still progressing; TUI reconnects and shows live stats.

---

## Feature 4: New Oracle Screen (two panel + fixed oracle set)

### Goal
Add a new screen that makes oracle-related learning diagnostics *comparable across iterations* by adding a **Fixed Oracle Set** evaluation.

### Screen layout
Two panels:
- **Left panel**: table of iterations with oracle metrics:
  - Current oracle accuracy (as today)
  - Regret vs oracle (for gating games)
  - **New: Fixed oracle set accuracy** (same states each iteration)
- **Right panel**: details for selected iteration:
  - breakdown by type and turn (mark vs reroll, turn index buckets, etc.)
  - optional histograms/sparklines

### Fixed oracle set (data + evaluation)
We need a deterministic, versioned set of game states:
- stored as `configs/oracle_sets/<id>.json` (or `.ndjson`)
- referenced by config: `oracle.fixed_set_id: <id>` (or reuse gating seed set infrastructure)
- evaluation runs each iteration (cheap, bounded) and logs:
  - overall accuracy
  - by turn bucket
  - by action type

### Acceptance criteria (checkboxes)
- [ ] New TUI screen exists (keybinding TBD; e.g. `o`) with two-panel layout.
- [ ] Left panel lists iterations and includes fixed-set oracle metrics.
- [ ] Fixed oracle set evaluation is computed on the **same** states each iteration.
- [ ] Fixed oracle metrics are persisted to `logs/metrics.ndjson` (or a stable JSON) per iteration.
- [ ] Right panel shows breakdowns by move type and turn buckets for the selected iteration.
- [ ] Metrics remain backward compatible: older runs without fixed-set data still render (show `-`).

### End-to-end test procedure
- **E2E-OR-1: fixed oracle is stable**
  - Preconditions: create `oracle_set_dev_v1` with N states.
  - Steps:
    - Run 3 iterations with no training changes (or very small).
    - Compare fixed oracle accuracy across iterations.
  - Expected:
    - Values are comparable and do not jump due to changing state distribution alone.

---

## Cross-cutting: Testing & CI expectations

### Unit/Integration tests (code-level)
- [ ] Rust unit tests for any new schema parsing / replay trace encoding.
- [ ] Golden test for replay trace rendering (at least deterministic formatting).
- [ ] Python tests for fixed oracle set evaluation (deterministic output given fixed model + states).

### Backward compatibility
- [ ] TUI can open and render runs created before these changes (missing fields show `-`).
- [ ] No changes required to existing run folders to view them (additive only).

---

## Open questions (to resolve before implementation)
- Which keybindings do you prefer for:
  - Extend run: `e` on Dashboard is requested (confirm).
  - Replay screen: `r` on Dashboard vs on Performance screen?
  - Oracle screen: `o`?
- Extend run: should we support “copy logs” as an option, or always start fresh logs?
- Replay traces: do we want per-game JSON files or a single NDJSON stream?
- Fixed oracle set: how many states (N) is “cheap enough” per iteration on CPU?

