# PRD v1 — High-throughput AlphaZero Yatzy (Rust + Python) with Oracle-Compatible Rules, Scalable Inference, and Built‑in Profiling

This PRD consolidates **v0.1–v0.3** into a single, self-contained **v1** spec. It is written to be implementation-guiding (module boundaries, protocols, state machines, parallelism points, logging/profiling), and to support a **lean CLI-first** rollout so you can validate the full workflow early.

---

## 1) Product vision

Build a **high-throughput AlphaZero-style system** for **2-player Yatzy**:

* **Rust**: game engine, state encoding, MCTS (PUCT), self-play + gating runners, replay + event logs, profiling hooks.
* **Python/PyTorch**: inference server (batched, CPU now / GPU later), trainer (replay → candidate), optional orchestration.
* **Oracle integration**: the provided Rust oracle (`swedish_yatzy_dp`) is used for **solitaire evaluation** and **diagnostics**, ensuring rules/actions match a solved baseline. 

The architecture must run well:

* **locally** on MBP (CPU inference),
* later on **Runpod GPU** (TCP inference + large dynamic batches + multi-worker).

---

## 2) Goals, success metrics, and non-goals

### Goals

1. **Correctness & compatibility**

   * Exact Swedish/Scandinavian Yatzy category order, masks, and keep-mask semantics compatible with the oracle. 
2. **Throughput**

   * Rust hot loops: no per-sim allocations, cache-friendly node storage, minimal synchronization.
   * Inference: dynamic batching + async leaf evaluation.
3. **Futureproof scaling**

   * Swap inference endpoint from `unix://` (local) to `tcp://` (Runpod) without changing MCTS logic.
   * Scale via many workers + one inference server per GPU.
4. **Observability-first**

   * Always-on iteration logs (NDJSON) + sampled MCTS root logs.
   * Metrics & profiling integrated from day 1 (not a later refactor).

### Success metrics (practical acceptance targets)

* **Oracle baseline**: running oracle simulation yields expected score ~248.44, mean ~248–249, bonus rate ~89% (sanity/regression). 
* **Deterministic gating**: same config + same fixed seeds ⇒ identical gating results (win-rate, score-diff summary, sampled diagnostics).
* **Inference batching works**: server reports median batch sizes > 1 (local) and large batches on GPU (later).
* **MCTS diagnostics exist**: fallback rate, pi entropy/max-prob, visits, etc. logged and queryable.

### Non-goals (v1)

* Full distributed orchestration (K8s, job queues). We’ll ship “worker + inference server + trainer” that can be glued with scripts.
* Fully general stochastic tree enumeration. We use stochastic transitions (sampled dice) + batched eval.

---

## 3) Oracle-compatibility is a first-class requirement

Your oracle defines the *canonical low-level interface we align to*, including:

### 3.1 Category indices and ordering

Use exactly **15 categories** indexed `0..14` in this order:
`ones, twos, threes, fours, fives, sixes, pair, two_pairs, three_kind, four_kind, small_straight, large_straight, house, chance, yatzy`. 

### 3.2 Availability mask convention

`avail_mask: u16` where bit `(14 - cat)` is **1 if category `cat` is available**. 

### 3.3 Upper total convention

`upper_total` is **clamped to 63** in state (values above treated as 63). 
Bonus rule (from oracle code): when marking an upper category causes `upper_total` to cross from `<63` to `>=63`, add **+50 bonus** once.

### 3.4 Dice and rerolls

* `dice: [u8;5]`, values `1..=6` (oracle sorts internally, but we will keep sorted always). 
* `rerolls_left: 0..=2`. 

### 3.5 Action semantics: Mark and KeepMask

Oracle returns:

* `Action::Mark { cat }`
* `Action::KeepMask { mask }` where mask refers to the **sorted** dice; bit `(4 - i)` corresponds to `dice[i]`. 

**Important nuance**: when rerolls remain, “mark now” is legal (stop early). The oracle will sometimes choose to mark rather than reroll further.

---

## 4) v1 system architecture

### 4.1 Components

1. **Rust binary: `yz`**

   * Self-play (MCTS + inference client)
   * Gating (candidate vs best)
   * Oracle eval suite (solitaire baseline + diagnostics)
   * Bench + profiling commands
   * Replay writer + NDJSON event logger

2. **Python package: `yatzy_az`**

   * `infer-server`: batched PyTorch inference service (UDS/TCP)
   * `train`: replay → candidate training
   * (optional) `controller`: run one iteration end-to-end

### 4.2 Data flows

* Rust self-play generates `(features, mask, pi, z, [z_margin])` → replay shards.
* Python trainer reads shards → writes candidate checkpoint.
* Rust gating compares candidate vs best → promotion decision.
* Oracle suite produces extra metrics and guardrails (optional promotion constraints).

---

## 5) Core domain model (rules, state, actions)

### 5.1 Rulesets

`Ruleset::SwedishScandinavian` (oracle-compatible) is the only supported ruleset in v1.

### 5.2 Action space: fixed indexed `A = 47`

We use oracle’s **KeepMask** semantics in our AZ policy space:

* **Indices `0..=31`**: `KeepMask(mask)`
* **Indices `32..=46`**: `Mark(cat)` where `cat = idx - 32`

Total `A = 32 + 15 = 47`.

**Legal mask rules**

* If `rerolls_left == 0`: only `Mark(cat)` actions are legal.
* If `rerolls_left > 0`:

  * `Mark(cat)` legal if category available
  * `KeepMask(mask)` legal for `0..=30`
  * `KeepMask(31)` (“keep all”) is **illegal** (dominated; wastes a reroll)

### 5.3 Game state representation (1v1)

A 1v1 match is “two solitaire boards + comparison”. We represent:

**Per-player**

* `avail_mask: u16` (oracle bit convention)
* `upper_total_cap: u8 (0..=63)`
* `total_score: i16/i32` (for win/loss/margin)
* optional: `filled_count: u8` (derived from mask)

**Turn state (shared)**

* `dice: [u8;5]` always sorted
* `rerolls_left: u8`
* `player_to_move: u8 (0|1)`

Terminal when both players have filled 15 categories.

### 5.4 Scoring (must match oracle)

Implement `scores_for_dice(dice)->[i32;15]` consistent with oracle:

* small straight = 15, large straight = 20, yatzy = 50, etc.
* upper bonus +50 when crossing 63 from below.

**Implementation policy**

* Prefer *sharing* the oracle scoring implementation as a crate/module to avoid drift.
* If reimplemented, add property tests comparing against oracle for large random samples.

---

## 6) Chance model and determinism policy

You discovered in Python that action-path-dependent seeding makes “fixed seeds” less meaningful. In v1:

### 6.1 Deterministic “event-keyed” dice stream (for eval/gating)

We define dice outcomes by **episode seed + structural event**, not by evolving state.

**Event key**

* `(episode_seed, player, round_idx, roll_idx)` where `roll_idx ∈ {0,1,2}`.

For each event key, deterministically generate a sequence of 5 die values (uniform in 1..6).
When rerolling `k` dice, take the first `k` values from that event sequence.

This is critical to avoid “duplicate dice index exploits”: rerolling one of two identical dice should not change outcomes.

### 6.2 Modes

* **Self-play mode**: can use fast RNG seeded per episode; determinism not required.
* **Gating/eval mode**: determinism is an **option** (recommended for model selection experiments and debugging, but not required):

  * **Optional deterministic chance stream** (event-keyed). Use this when you want reproducible comparisons / ablations.
  * For long-running training pipelines, avoid making chance deterministic globally to reduce the risk of overfitting to specific seed streams.
  * no Dirichlet noise
  * temperature=0
  * deterministic tie-breaking

---

## 7) MCTS (AlphaZero-style PUCT) spec

### 7.1 Algorithm

PUCT selection with NN priors and value:

* priors `P[a]` from policy logits masked + softmax over legal moves
* value `V(s) ∈ [-1,1]` for backup (win/loss value head)

Selection: `argmax_a (Q(s,a) + U(s,a))`
with `U = c_puct * P[a] * sqrt(N_sum) / (1 + N[a])`.

### 7.2 Key knobs (config)

* `c_puct`
* root Dirichlet `(alpha, epsilon)` **self-play only**
* `max_inflight_per_game` (async leaf evals)
  * Definition: maximum number of concurrent leaf-eval inference requests **per game**.
  * Note: this is **not** the inference batch size; batch size is controlled by server-side `inference.max_batch` and `inference.max_wait_us`.
* `budget_reroll`, `budget_mark`
  * Definition: MCTS **simulations per move**, split by decision type:
    * `budget_reroll`: sims used when `rerolls_left > 0` (keepmask vs mark decisions)
    * `budget_mark`: sims used when `rerolls_left == 0` (must choose a mark)
  * These are the primary per-move compute budget knobs in v1.

### 7.3 Temperature rule (from your learnings)

* `pi` target = normalized visit counts
* temperature affects only **executed action sampling**, not replay targets

**Temperature schedules (explicitly supported)**

We support making temperature a function of the move/ply within a game/turn to control exploration:

* Define a temperature schedule `T(t)` used only for **executed action sampling** (never for `pi` targets stored in replay).
* Recommended defaults:
  * **Self-play**: higher `T` early (more exploration), lower `T` later (more exploitation).
  * **Gating/eval**: typically `T=0` (greedy), unless explicitly experimenting.

Implementation guidance:
* `t` can be defined as a global ply index (number of actions taken so far) or as per-player filled-count / remaining-categories.
* Supported schedule types: constant, step, linear decay, exponential decay.
* Always log the chosen `T(t)` alongside executed moves for reproducibility.

### 7.4 Batched leaf evaluation drift handling

Because inference is batched/async:

* implement **in-flight reservation / virtual loss**
* prevent repeated selection of the same pending leaf
* track collision/pending stats

### 7.5 Robust normalization + fallback

If `pi` becomes degenerate/invalid:

* fallback to uniform over legal moves
* log fallback count/rate (iteration + sampled roots)

### 7.6 Node storage (performance constraint)

Per-game tree:

* arena `Vec<Node>`
* children stored by `ActionIdx` (dense arrays or sparse “legal list”)
* no heap allocations in select/backup loops
* store NodeId indices, not references

### 7.7 Stochastic dice transitions inside MCTS (no explicit chance nodes)

Yatzy transitions are stochastic because dice outcomes are part of state.
In v1, we keep the **fixed action space `A=47`** and do **not** introduce explicit chance nodes in the tree.

Instead:

* MCTS stores **decision nodes only** (states with `player_to_move`, `dice`, `rerolls_left`, per-player boards/totals).
* When traversing an edge for action `a`, the environment performs a stochastic transition:
  * self-play: sample dice outcomes using RNG
  * eval/gating (option): use the event-keyed deterministic chance stream
* The realized next state `s'` (including the new dice) is the child. Over many simulations, a single action edge may lead to **multiple distinct realized children**.
* PUCT selection is still performed over actions `a ∈ {0..A-1}`; stochasticity is in `step(s, a) -> s'`.

---

## 8) Parallelism and throughput strategy (explicit optimization points)

### 8.1 Where we parallelize (v1)

1. **Across worker processes** (scale-out)
2. **Across threads within a worker** (scale-up)
3. **Across games within a thread** (scheduler multiplexing)
4. **Within a move** via async leaf evals (in-flight requests)
5. **Inference batching** (server-side)
6. **Training** (data loader + GPU/CPU parallelism)
7. **Gating evals** (many games in parallel)

### 8.2 Worker scheduler: many games per thread

Each worker thread runs a loop:

* pick a runnable `GameTask`
* call `step(max_work)` to advance its state machine
* rotate to next runnable game

This ensures a thread doesn’t idle while waiting for inference.

### 8.3 Inference is a service (futureproof)

Local:

* `unix:///tmp/yatzy_infer.sock`

Runpod:

* `tcp://host:port`

The client interface is identical.

---

## 9) Inference interface and protocol (Rust ↔ Python)

### 9.1 Server requirements

* Dynamic batching:

  * flush when `batch == max_batch` OR `oldest_wait >= max_wait_us`
* Multi-model routing:

  * requests include `model_id` (best/candidate) so gating can be served by one process

### 9.2 Request/response contract

Inputs:

* `request_id: u64`
* `model_id: u32`
* `feature_schema_id: u32`
* `features: float32[F]` contiguous
* `legal_mask`: packed bits or `uint8[A]`

Outputs:

* `policy_logits: float32[A]`
* `value: float32`
* optional: `margin: float32` (aux head)

### 9.3 Backpressure

Client must cap in-flight requests:

* per game: `max_inflight_per_game`
* per worker: `max_pending_total`

If queue grows, scheduler should prioritize “apply inference + backup” over “enqueue more”.

### 9.4 Optional PyO3 backend

Allowed, but not required in v1:

* embedded inference backend for local experiments
* still batched; single thread owns Python model and GIL

---

## 10) Replay, checkpoints, and logs (crash-resilient)

### 10.1 Replay sample schema

Per decision:

* `features: f32[F]`
* `legal_mask: u8[...]`
* `pi: f32[A]` from MCTS visits
* `z: f32` win/loss from POV of encoded player-to-move
* optional `z_margin: f32 = tanh(score_diff / scale)`

### 10.2 Storage

* replay shards: `safetensors` + small `.meta.json`
* append-only NDJSON logs:

  * `iteration_stats.ndjson` (always-on)
  * `mcts_roots.ndjson` (sampled)

### 10.3 Versioning

Every shard/checkpoint/log includes:

* `protocol_version`
* `feature_schema_id`
* `action_space_id = oracle_keepmask_v1`
* `ruleset_id = swedish_scandinavian_v1`

### 10.4 Replay retention / capacity knobs (status)

We currently store replay as append-only shards under `runs/<id>/replay/`. To avoid unbounded disk growth and to support stable "replay buffer" semantics, we define the following knobs:

* **Replay buffer capacity**: `replay.capacity_shards` (or `replay.capacity_samples`)
  * Definition: keep at most N shards (or M samples) worth of replay for training; prune older shards beyond capacity.
  * Pruning policy should be deterministic and recorded (e.g. keep newest by shard sequence/mtime).
  * **Status:** ✅ **Implemented** (E13.1S1). Pruning keeps newest N shards by filename index; emits `replay_prune` event to `logs/metrics.ndjson`.
  * **AC:** running multiple iterations does not grow replay storage unbounded; pruning is logged and reproducible.

---

## 11) Training (Python)

### 11.1 Model

`YatzyNet`:

* trunk MLP (or small residual MLP)
* policy head → logits[A]
* value head → scalar in [-1,1]
* optional margin head → scalar in [-1,1]

### 11.2 Losses

* policy: cross-entropy vs `pi`
* value: MSE vs `z`
* optional margin: MSE vs `z_margin` weighted by `aux_margin_lambda`

### 11.3 Candidate := Best boundary

When starting a new iteration:

* initialize candidate weights from best
* **reset optimizer state** (Adam moments etc.) — hard requirement from your prior bugfix

Artifacts:

* `best.pt`, `candidate.pt`
* `.meta.json` with config hash, schema ids, git hash, train step counters

### 11.4 Training knobs (status)

This section enumerates training-related config knobs and their implementation status.

* **epochs**: `training.epochs` (supported in config schema)
* **Weight decay (L2)**: `training.weight_decay` (**implemented**)
  * Standard L2 regularization term applied by AdamW/SGD.
  * **Status:** implemented end-to-end (schema + trainer uses AdamW).
* **Training steps per iteration**: `training.steps_per_iteration` (optional alternative to epochs)
  * Definition: number of optimizer updates per iteration (one “step” = one optimizer update on one batch).
  * Relationship to epochs: `steps ≈ epochs * ceil(num_samples / batch_size)`; steps is dataset-size independent.
  * Policy: if `steps_per_iteration` is set, it takes precedence over `epochs`.
  * **Status:** `steps_per_iteration` is implemented end-to-end (schema + trainer reads it when present).
* **Total iterations**: `controller.total_iterations` (or `run.total_iterations`)
  * Definition: how many full iteration cycles to run (self-play → train → gate).
  * This is orchestration/controller config (not a per-step training knob).
  * Default: `1` (single iteration). Set higher for multi-iteration training runs.
  * **Status:** ✅ **Implemented** (E13.1S2). Controller runs N iterations, updates `run.json` with progress, and stops when `controller_iteration_idx >= total_iterations`.
  * **AC:** controller stops after N iterations and records progress in `run.json`.

---

## 12) Gating (candidate vs best) + oracle suite

### 12.1 Gating modes

1. **Regular gating** (fast, higher variance)
2. **Paired-seed, side-swapped** (default)

   * for each seed: play 2 games swapping who goes first
   * reduces variance + cancels first-player advantage

Optional: fixed dev seed sets (persisted lists) for repeatable evaluation sets.

### 12.2 Promotion decision (v1)

* primary: candidate win_rate ≥ threshold
* log score-diff mean + SE (paired)
* optional guardrail: candidate must not regress in oracle solitaire mean by > δ

### 12.3 Oracle evaluation suite

Use oracle as a competence/regression lens:

* run N solitaire games:

  * oracle policy baseline (sanity)
  * best model
  * candidate model
* compare mean/median/std, bonus rate
* baseline expectation for oracle itself: expected score ~248.44 and mean ~248–249 with ~89% bonus rate. 

---

## 13) CLI-first workflows (lean and testable)

### 13.1 Rust CLI: `yz`

Core commands (v1):

* `yz oracle expected` — print oracle expected score
* `yz oracle sim --games N --seed S` — oracle solitaire simulation + histogram
* `yz selfplay --config cfg.yaml --infer unix:///... --out runs/<id>/`
* `yz train` (optional wrapper that shells out to python)
* `yz gate --config cfg.yaml --best best.pt --cand cand.pt`
* `yz oracle-eval --config cfg.yaml --best ... --cand ...`
* `yz bench` — microbenches
* `yz profile selfplay ...` — run with profiler hooks enabled

### 13.2 Python CLI: `python -m yatzy_az ...`

* `... infer-server --model best.pt --bind unix:///tmp/y.sock --device cpu`
* `... train --replay runs/<id>/replay --best best.pt --out runs/<id>/models`

### 13.3 Lean “workflow smoke test” (what you should be able to do early)

1. `yz oracle expected`
2. Start inference server with a tiny dummy model:

   * `python -m yatzy_az infer-server --best dummy --cand dummy --bind unix:///tmp/y.sock`
3. Generate a small replay:

   * `yz selfplay --config configs/local_cpu.yaml --infer unix:///tmp/y.sock --out runs/smoke/ --games 50`
4. Train candidate for a few steps:

   * `python -m yatzy_az train --replay runs/smoke/replay --best runs/smoke/models/best.pt --out runs/smoke/models`
5. Gate candidate vs best:

   * `yz gate --config configs/local_cpu.yaml --best ... --cand ...`

---

## 14) Project structure (monorepo)

### Rust workspace

* `yz-core`: rules, scoring, state, RNG, action mapping (oracle-compatible)
* `yz-features`: feature schema + canonical encoding
* `yz-mcts`: PUCT, tree arena, async leaf pipeline
* `yz-runtime`: schedulers, GameTask state machines, worker threads
* `yz-infer`: socket protocol + client
* `yz-replay`: safetensors shards + readers
* `yz-eval`: gating games + stats aggregation
* `yz-oracle`: adapter to `swedish_yatzy_dp` + oracle suite tools
* `yz-logging`: NDJSON events + metrics/tracing setup
* `yz-cli`: `yz` binary

Include the oracle crate as a workspace member (or submodule):

* `swedish_yatzy_dp` (depends on `rayon`, `rustc-hash`) 

### Python package

* `yatzy_az/server/`
* `yatzy_az/model/`
* `yatzy_az/trainer/`
* `yatzy_az/replay_dataset.py`
* `yatzy_az/config.py`

---

## 15) Profiling and performance budgets (v1 requirement)

### 15.1 Always-on metrics

Rust worker:

* sims/sec, expansions/sec
* pending inference, latency histogram
* fallback_rate
* average pi entropy / max_prob (sampled)

Inference server:

* evals/sec
* batch size histogram
* queue depth
* per-model stats (best vs candidate)

Training:

* samples/sec, step time, loss summaries

### 15.2 Bench suite (Criterion)

* scoring (`scores_for_dice`)
* legal mask generation
* PUCT selection inner loop
* protocol encode/decode

### 15.3 Compile profile guidance

The oracle crate uses aggressive opt settings even in dev/test for DP speed. 
We should mirror this in our workspace (at least for key crates) to keep iteration fast.

---

# 16) Implementation plan: epics & stories (CLI-first, “lean vertical slices”)

Below are epics with implementable stories. Each story includes **deliverable + acceptance criteria**. Build order is designed so you can test the workflow early.

---

## Epic E0 — Repo bootstrap & developer UX

**Goal:** you can build, run, test, profile from day 1.

**Stories**

1. **Workspace + package skeleton**

   * Rust workspace + Python package scaffold + configs folder
   * **AC:** `cargo test` and `python -m yatzy_az --help` both succeed.

2. **Unified config schema**

   * YAML config validated in Rust (`serde`) and Python (`pydantic`)
   * **AC:** same `configs/local_cpu.yaml` loads in both.

3. **CI + formatting**

   * Rust fmt/clippy + Python lint
   * **AC:** PR checks run green.

---

## Epic E1 — Oracle integration + golden compatibility tests

**Goal:** lock in the oracle-aligned conventions early.

**Stories**

1. **Vendor oracle crate**

   * Add `swedish_yatzy_dp` to workspace
   * **AC:** `yz oracle expected` prints expected score ~248.44. 

2. **Golden tests for masks/actions**

   * test category order, `avail_mask` bit convention, keepmask bit mapping
   * **AC:** tests pass and match oracle spec. 

3. **Oracle sim command**

   * `yz oracle sim --games N` prints mean/bonus rate/histogram
   * **AC:** mean near 248–249 and bonus rate near 89% for large N (sanity). 

---

## Epic E2 — Core rules engine (yz-core) + deterministic chance stream

**Goal:** correct transitions and reproducible evaluation.

**Stories**

1. **Action mapping `A=47`**

   * implement index <-> action conversion and legal masks
   * **AC:** legal mask matches phase rules; keep-all illegal when rerolls_left>0.

2. **Scoring parity**

   * implement `scores_for_dice` + bonus logic
   * **AC:** property test: for random dice, our scores match oracle’s scorer exactly (or use shared module).

3. **Deterministic event-keyed dice**

   * implement deterministic dice generation for eval/gating
   * **AC:** same seed produces identical full game trajectories under deterministic policy mode.

---

## Epic E3 — Feature schema + canonical encoding

**Goal:** stable NN inputs and artifact compatibility.

**Stories**

1. **FeatureSchema v1**

   * define `feature_schema_id`, `F`, and encoding contract
   * **AC:** replay shards record schema id; loaders reject mismatched ids.

2. **Canonical player-to-move encoding**

   * encode from POV of current player
   * **Value/sign contract:** NN `value` is interpreted from the POV of the **encoded player-to-move**. Therefore, when traversing an edge that changes `player_to_move`, MCTS backup must **negate** the child value when expressing it in the parent node’s POV (avoid double-flips by defining one convention and sticking to it everywhere).
   * **AC:** swapping players produces consistent mirrored encoding.
   * **AC (swap sanity):** for any state `s`, define `swap_players(s)` as: swap per-player boards/totals, flip `player_to_move`, keep turn state (`dice`, `rerolls_left`) identical. Then `encode(s)` and `encode(swap_players(s))` must be consistent mirror/permute views.\n+\n+     **Value/sign note:** if your value target is defined from the POV of the **encoded player-to-move**, then swapping players *and* flipping `player_to_move` leaves that POV invariant. In that convention, terminal/backup targets satisfy `z(s) = z(swap_players(s))` (and similarly `V(s) ≈ V(swap_players(s))`).\n+\n+     Antisymmetry holds when comparing the **same fixed POV** across swapped states (e.g. player-0 POV), not when also flipping the POV with `player_to_move`.

---

## Epic E3.5 — Game rules engine (state transitions)

**Goal:** implement the actual game mechanics so self-play/MCTS can advance `GameState` correctly.

**Stories**

1. **State transition function**

   * Implement `apply_action(state, action) -> next_state` for the oracle-compatible ruleset:
     * `KeepMask(mask)`:
       * apply to **sorted** dice (bit `(4-i)` refers to `dice[i]`)
       * reroll the 0-bit dice, decrement `rerolls_left`
       * keep dice sorted
     * `Mark(cat)`:
       * compute raw category score from `scores_for_dice(dice)`
       * apply upper bonus (+50 when crossing 63 from below), clamp upper total to 63
       * update `avail_mask` (mark category as filled), update `total_score`
       * reset turn to next player: `player_to_move` flips, `rerolls_left=2`, new dice rolled
   * Deterministic chance must be supported as an **option** (e.g. used for model selection experiments), but training runs should be able to use non-deterministic RNG to avoid overfitting to fixed seed streams.
   * **AC:** a random playout from the initial state always terminates after 30 marks (15 per player), never produces illegal states, and respects PRD legality rules.

2. **Terminal & outcome**

   * Define terminal condition: both players have filled all 15 categories.
   * Define winner/draw: compare `total_score` at terminal (and define tie handling).
   * Provide helper to compute `z` from POV of encoded player-to-move (used by replay writer later).
   * **AC:** for terminal outcomes, if `z` is defined from the POV of `player_to_move`, then `z(s) = z(swap_players(s))` (because `swap_players` also flips `player_to_move`). Antisymmetry holds only when comparing the same fixed POV across swapped states.

3. **Golden transition tests**

   * Add targeted tests for:
     * keepmask bit mapping on sorted dice
     * upper bonus boundary and clamping behavior
     * determinism option: same `episode_seed` + event-keyed mode ⇒ identical trajectories
   * **AC:** tests pass and align with oracle conventions in Section 3.

---

## Epic E4 — MCTS core (yz-mcts) with stub inference

**Goal:** get MCTS correctness + diagnostics before Python integration.

**Stories**

1. **Arena tree + PUCT**

   * selection/expand/backup with dense action indexing
   * **AC:** produces valid pi distributions; no panics; deterministic in eval mode.

2. **Root noise + temperature semantics**

   * noise only in self-play; temperature only for executed move
   * **Root Dirichlet noise (self-play only)**:
     * compute legal action set `L`
     * sample `η ~ Dirichlet(α)` over `|L|` legal actions
     * mix at root: `P_noisy[a] = (1-ε) * P_raw[a] + ε * η[a]` for `a ∈ L`, and `0` for illegal
     * keep `P_raw` as the “clean” priors; use `P_noisy` only for root exploration in self-play
     * log both `P_raw` and `P_noisy` at root
   * **Temperature (executed move only)**:
     * replay target `pi_target` is always normalized visit counts from MCTS (temperature must not alter this)
     * executed distribution:
       * if `T>0`: \(\\pi_{exec}(a) \\propto (\\pi_{target}(a))^{1/T}\\) over legal actions
       * if `T=0`: choose `argmax_a pi_target[a]` with deterministic tie-break
     * temperature schedules `T(t)` are supported (see §7.3); `t` is a caller-provided global ply index
   * **AC:** logs show raw vs noisy priors; pi target unaffected by temp.

3. **Fallback detection**

   * uniform fallback over legal moves + fallback counter
   * **AC:** fallback rate logged; can be triggered in tests.

4. **Async leaf pipeline scaffolding**

   * “in-flight” accounting even with stub inference
   * **AC:** pending leaf collisions decrease when virtual loss enabled.

---

## Epic E5 — Rust inference protocol + client (yz-infer)

**Goal:** the futureproof boundary is in place.

**Stories**

1. **Binary protocol v1**

   * length-delimited frames; request_id routing; schema ids
   * **AC:** round-trip unit tests with a dummy echo server.

2. **InferenceClient with background IO**

   * tickets + response routing + latency histograms
   * **AC:** handles thousands of concurrent requests without deadlock.

3. **Integrate InferenceClient into yz-mcts**

   * Replace the “stub inference” path in `yz-mcts` with `yz-infer::InferenceClient` for leaf evaluation:
     * `yz-mcts` submits leaf evaluation requests (features + legal mask + schema ids) to the client
     * `yz-mcts` receives `(policy_logits[A], value)` responses and applies masked-softmax priors + backup sign convention
   * This integration must support both:
     * **Self-play mode** (Dirichlet root noise enabled, temperature schedule for executed move)
     * **Eval/gating mode** (no root noise, temperature=0, deterministic tie-breaking; deterministic chance optional per gating config)
   * Provide a **dummy inference server** (Rust) for tests that returns:
     * uniform logits over legal moves
     * value = 0
     * preserves `request_id` routing
   * **AC:** `cargo test --package yz-mcts` exercises the async leaf path using the dummy server (including multiple in-flight requests) and produces a valid `pi`.
   * **AC:** end-to-end roundtrip: `yz-mcts` → `InferenceClient` → dummy server → `InferenceClient` → `yz-mcts` works without deadlocks and logs latency histograms.

---

## Epic E6 — Python inference server (batched, multi-model)

**Goal:** throughput lever: batching.

**Stories**

1. **UDS server + dynamic batching**

   * `max_batch` or `max_wait_us` flush rules
   * **AC:** batch histogram shows >1 median with multiple games in flight.

2. **Multi-model routing**

   * `model_id` selects best vs candidate
   * **AC:** gating can query both models via one server.

3. **Metrics endpoint**

   * Prometheus metrics for eval/sec, batch sizes, queue depth
   * **AC:** metrics scrape works locally.

---

## Epic E6.5 — Proper inference server (real PyTorch model: YatzyNet)

**Goal:** replace dummy inference with a **real PyTorch-backed inference path** so self-play, training, and gating can run end-to-end with actual NN policy/value outputs (CPU now, GPU later) while keeping protocol v1 and batching unchanged.

**Motivation / current status**

We already have a Python asyncio server with:

* protocol v1 compatibility (Rust ↔ Python)
* dynamic batching + multi-model routing (best vs candidate)
* Prometheus `/metrics`

However, the current server-side “models” are **dummy/stub** implementations. This epic adds the **real model backend**:

* load `YatzyNet` checkpoints (`best.pt`, `candidate.pt`)
* run forward passes on the configured device
* return `(policy_logits[A], value)` (and optional margin later)

**Stories**

1. **Checkpoint format contract for inference**

   * Define the exact checkpoint payload expected by the inference server:
     * `model` state_dict
     * minimal `config` (hidden/blocks, schema ids) or embed enough metadata to reconstruct `YatzyNet`
     * **`checkpoint_version`** (required, v1 = `1`)
     * **Required meta compatibility fields:** `protocol_version`, `feature_schema_id`, `ruleset_id`, `action_space_a`
   * **AC:** inference server can load `runs/<id>/models/best.pt` written by the trainer and validate schema compatibility.

2. **TorchModel backend in inference server**

   * Implement a `TorchModel` (or similar) in `python/yatzy_az/server/` that:
     * loads `YatzyNet` from a checkpoint path
     * moves model to `--device cpu|cuda`
     * runs batched inference: input `[B, F]` → outputs `policy_logits[B, A]`, `value[B]`
   * Note: CLI wiring for `path:...` model specs is handled in **E6.5S3**; this story focuses on the backend implementation + tests.
   * **AC:** unit test verifies shapes, dtypes, finiteness, and deterministic outputs for fixed weights.

3. **Server CLI: real model specs**

   * Extend `python -m yatzy_az infer-server` to accept real checkpoint specs, e.g.:
     * `--best path:/abs/to/best.pt`
     * `--cand path:/abs/to/candidate.pt`
     * retain `dummy:...` for testing
   * **Spec grammar (v1):**
     * `dummy` or `dummy:<float>`
     * `path:<checkpoint_path>`
   * **AC:** server boots with real checkpoints and serves both model_ids correctly.

4. **“New run” bootstrap: initialize best model if missing**

   * Provide a small utility to create a fresh `best.pt` for a new run, e.g.:
     * `python -m yatzy_az model-init --out runs/<id>/models/best.pt --hidden ... --blocks ...`
   * Note: v1 implements **explicit bootstrap** via `model-init` (no infer-server auto-init flag).
   * **AC:** a brand new run directory can be started without preexisting checkpoints.

5. **Performance + batching correctness**

   * Ensure the TorchModel path integrates with the existing batcher efficiently:
     * no per-item python overhead inside the model forward
     * uses `torch.inference_mode()`
     * uses contiguous tensors and single device transfer per batch
     * optional: expose `--torch-threads` / `--torch-interop-threads` to stabilize CPU inference throughput
   * **AC:** Prometheus shows meaningful batch sizes and eval/sec with multiple concurrent Rust games.

6. **Compatibility tests (Rust client ↔ Python real model)**

   * Add an integration test that starts the Python server with a tiny real model and runs a short Rust MCTS search end-to-end.
   * Note: the test is **opt-in** via `YZ_PY_E2E=1` to keep CI stable.
   * **AC:** `cargo test --workspace` includes at least one end-to-end test proving the real inference path works.

---

## Epic E6.6 — Inference perf + protocol evolution (Option 1 → Option 2 → bitpacking)

**Goal:** reduce the end-to-end **inference-per-leaf cost** (especially Python decode/tensorization and response encoding) while keeping the dynamic batching architecture intact, and then evolve the wire protocol in a controlled way to support packed tensors and legal-mask bitpacking.

**Background / motivation**

We have a high-QPS inference workload driven by many concurrent MCTS leaf evaluations. In practice, the bottleneck is often **Python-side overhead per request**, not the model forward pass:

* **Decode overhead:** materializing features as Python `list[float]` (or list-of-lists) and building tensors via `torch.as_tensor(...)` is slow and allocation-heavy.
* **Encode overhead:** converting outputs to Python lists with `.tolist()` is similarly expensive.
* **Batching correctness:** small batches (or broken batching) can dominate total runtime even if everything else is optimized; protocol changes must not regress batching.

This epic explicitly sequences work into three steps:

1) **Option 1:** keep the existing protocol, but make Python inference **buffer/tensor-native** internally (big ROI, low risk).
2) **Option 2:** add a **Protocol v2** for packed `f32` payloads end-to-end (cleaner contract, further overhead reduction).
3) **Bitpacking:** shrink `legal_mask` from bytes to a compact bitset (6 bytes for `A=47`) and update both ends.

**Constraints / non-goals**

* Do not change MCTS semantics, exploration, or in-flight caps; specifically: **no increase in in-flight virtual loss per game** (this is a protocol/perf change, not a search change).
* Do not “fix” batching by increasing client-side in-flight; batching must remain primarily a **server-side** responsibility (`max_batch`/`max_wait_us`).
* Keep rollout safe: we must be able to validate correctness + performance with targeted benchmarks and end-to-end runs.

**Stories**

1. **Option 1: Tensor-native decode/encode under Protocol v1 (no wire changes)**

   * **Scope:**
     * Keep the existing Rust↔Python framing and field shapes (features length `F`, action space `A=47`, legal mask format as currently shipped).
     * Rewrite the Python server hot path so it never constructs Python float lists per request and never uses `.tolist()` for responses.
     * Decode request payloads into **contiguous** buffers/views and build batched tensors efficiently.
     * Encode responses by writing raw bytes (or writing into a preallocated buffer) instead of list conversions.
   * **Implementation guidance:**
     * Prefer `memoryview`/`numpy.frombuffer` (or equivalent) to create zero-copy views of incoming `f32` feature bytes.
     * Build a batch tensor with one contiguous allocation (`[B, F]`) and a single copy from the input buffer(s) (or minimal copies).
     * Ensure output encoding is also contiguous and cheap (avoid per-element Python loops).
     * Preserve multi-model routing behavior (best/candidate) and existing batching invariants.
   * **Observability / benchmarks:**
     * Add a focused microbench that measures:
       * decode+batch tensorization time (per item and per batch),
       * model forward time,
       * encode time,
       * end-to-end request latency distribution under load.
     * Ensure Prometheus metrics still reflect batch sizes correctly (and add optional timing counters if useful).
   * **AC:**
     * Correctness: outputs are bitwise-stable (or within expected floating-point tolerance) compared to the previous implementation for a fixed model + fixed inputs.
     * Performance: measurable reduction in Python CPU time spent in decode/encode at the same QPS (measured via profiler and/or timing counters).
     * No batching regression: median batch size under representative self-play load does not decrease; eval/sec does not regress.

2. **Option 2: Protocol v2 for packed float tensors (features + logits/value) with safe rollout**

   * **Scope:**
     * Introduce a new protocol version (“v2”) where:
       * features are sent as **packed little-endian float32 bytes** (contiguous),
       * outputs are returned as **packed float32 bytes** for `policy_logits[A]` and `value` (and any future heads).
     * Maintain backward compatibility during rollout:
       * Python server accepts **both v1 and v2** requests (either via an explicit `protocol_version` field in the frame header, a capability handshake, or a per-connection negotiated mode).
       * Rust client can be switched between v1 and v2 via config/flag, enabling A/B perf testing.
   * **Implementation guidance:**
     * Define a precise byte-level spec for v2:
       * endianness, alignment expectations, exact tensor shapes, and how `F`/`A` are encoded/validated.
     * Add strict validation:
       * reject mismatched `feature_schema_id`, mismatched feature byte length, or mismatched action space id.
     * Keep server batching behavior identical across v1 and v2 (a batch is still a list of requests with the same `model_id` and compatible schema).
   * **Testing:**
     * Roundtrip tests for v2 framing in Rust + Python.
     * Mixed-mode test: start server, send interleaved v1 and v2 requests, ensure both work concurrently.
   * **AC:**
     * Correctness: v2 outputs match v1 outputs for identical inputs (within tolerance) for the same model weights.
     * Rollout safety: server can run in “dual stack” mode; client can be toggled without code changes to the server.
     * Performance: further reduction in per-request overhead vs Option 1 alone (especially at high QPS), demonstrated by microbench + E2E run.

3. **Bitpacking: legal mask as a compact bitset (6 bytes for A=47), layered on top of Protocol v2**

   * **Scope:**
     * After v2 is stable, add bitpacking for the `legal_mask`:
       * represent legal actions as a **bitset** of length `A=47` (6 bytes).
       * define exact mapping: bit `i` corresponds to action index `i` (explicitly documented in the protocol spec).
     * Update both ends:
       * Rust: pack legal mask bits when sending; unpack (or apply bitset masking) when needed.
       * Python: unpack to a boolean/uint8 mask (or directly apply bitset masking efficiently) for logits masking.
   * **Compatibility strategy:**
     * Either:
       * define a v2 capability/flag `legal_mask=bitset`, or
       * bump protocol to v3 for mask changes.
     * Server should be able to accept:
       * v2 with byte-mask, and
       * v2 with bitset-mask (gated by flag/version),
       during the transition.
   * **Performance guidance:**
     * Bitpacking is primarily about reducing bandwidth and memory traffic; it should not introduce expensive per-request Python loops.
     * Prefer vectorized unpacking (e.g. `numpy.unpackbits`) or a small optimized routine.
   * **AC:**
     * Correctness: legality masking matches exactly (no illegal actions become legal, and vice versa) across a large randomized corpus of states.
     * Performance: reduced bytes/request and no throughput regression; end-to-end eval/sec does not drop.
     * Stability: no changes to MCTS in-flight limits; no increase in in-flight virtual loss per game.

## Epic E7 — Self-play runtime + replay + NDJSON logs (yz-runtime/yz-replay/yz-logging)

**Goal:** first end-to-end pipeline without training.

**Stories**

1. **GameTask state machine**

   * incremental stepping with caps (`S` steps/tick, `K` inflight)
   * **AC:** many-games-per-thread works; CPU stays busy.

2. **Replay shards**

   * safetensors shard writer + meta
   * **AC:** `yz selfplay` writes shards incrementally; crash doesn’t corrupt previous shards.

3. **NDJSON logging**

   * iteration summaries always-on; root logs sampled
   * **AC:** you can post-mortem a run using logs alone.

---

## Epic E8 — Training (Python)

**Goal:** candidate training loop + checkpoint semantics.

**Stories**

1. **Dataset loader**

   * reads shards; yields batches; respects schema ids
   * **AC:** loads and trains on small run output.

2. **Model + losses**

   * policy/value (+ optional margin head)
   * **AC:** training reduces loss on a tiny dataset (sanity).

3. **Optimizer reset boundary**

   * candidate initialized from best with optimizer reset
   * **AC:** explicit test prevents “reuse Adam moments” regression.

---

## Epic E8.5 — Iteration controller (crash-safe runs + resume)

**Goal:** make full training iterations (self-play → train → gate) **restartable** and **auditable** using run artifacts alone.

This epic exists because a “resumable trainer” requires more than just model weights:
- training needs **periodic autosave** of model+optimizer state
- we need a stable **run manifest** that pins which replay data was used, plus git/config hashes
- resume behavior must be explicit (continue mid-train vs start a fresh iteration from best)

**Stories**

1. **Run directory + manifest (single source of truth)**

   * Define a run layout under `runs/<run_id>/` (replay, logs, models, manifests).
   * Add `run.json` (or `iteration.json`) containing:
     * `git_hash`, `config_hash`, schema ids (`protocol_version`, `feature_schema_id`, `action_space_id`, `ruleset_id`)
     * paths/ids for inputs (replay shard directory, seed set id if applicable)
     * counters: `selfplay_games`, `train_step`, `gate_games`
   * **AC:** you can reconstruct “what happened” for a run without external state.

2. **Trainer autosave + atomic checkpoints**

   * Trainer writes `candidate.pt` that includes:
     * model weights
     * optimizer state
     * `train_step` and training hyperparams
   * Autosave every N steps and write via temp+rename (crash-safe).
   * **AC:** killing training mid-run leaves a valid last checkpoint that can be resumed.

3. **Explicit resume semantics**

   * Support:
     * **Resume**: continue candidate training from `candidate.pt` + optimizer state.
     * **New iteration**: initialize candidate from best **with optimizer reset** (never reuse moments).
   * **AC:** tests cover both codepaths; “reuse Adam moments” regression is impossible.

4. **Replay snapshot semantics (resume-friendly)**

   * Define what it means to “train on replay”:
     * either snapshot a fixed list of shard filenames at iteration start, or
     * support incremental training with a persisted “cursor” / last-seen shard index.
   * **AC:** resuming training does not silently change the training dataset without recording it in the manifest.

---

## Epic E9 — Gating (yz-eval) + promotion

**Goal:** stable evaluation + promotion logic.

**Stories**

1. **Paired seed side-swapped gating**

   * schedule (seed, swap) pairs deterministically
   * **AC:** results reproducible; lower variance vs regular gating.

2. **Fixed dev seed sets**

   * generate + persist seed lists
   * **AC:** gating can run against “seed set id”.

3. **Promotion decision**

   * win-rate threshold + optional score-diff CI
   * **AC:** summary event includes all numbers needed for analysis.

---

## Epic E10 — Oracle suite integration (yz-oracle)

**Goal:** solved baseline as a regression guardrail.

**Stories**

1. **Solitaire oracle suite**

   * oracle baseline + candidate/best solitaire runs
   * **AC:** oracle baseline matches expected stats; candidate/best metrics logged.

2. **Oracle diagnostics in gating**

   * Compare agent action vs oracle best action
   * **AC:** logs include “oracle_match_rate” (and later regret).

3. **(Optional) Regret API**

   * extend oracle with `action_value(...)` for EV(action)
   * **AC:** can compute regret distribution and log it.

---

## Epic E10.5 — Experiment tracking: W&B-compatible logging + full config snapshots

**Goal:** make learning progress **understandable**, **comparable**, and **reproducible** across runs by emitting a W&B-friendly event stream and storing the exact config used for a run.

**Motivation:** We already emit NDJSON + a run manifest, but without a standardized “metrics stream” and a full config snapshot it’s too easy to:

* lose the context of *which knobs produced which curves*
* compare apples-to-oranges across runs (different seed sets, budgets, temperature schedules)
* miss regressions (policy collapse, illegal action mass, throughput drops)
**Stories**

**Status (implementation note):** E10.5S1–E10.5S4 are implemented. The only item still not implemented is the **optional** “per-seed results” payload in `gate_report.json` (E10.5S3).

1. **Full training config snapshot (run-local, crash-safe)**

   * On `yz selfplay` start, write `runs/<id>/config.yaml` as an exact copy of the YAML config used.
     * write via temp + atomic rename
   * Include a reference in `runs/<id>/run.json` (e.g. `config_snapshot: "config.yaml"`).
   * **AC:** given only `runs/<id>/`, you can recover the exact config bytes that produced the run.
2. **Standardized metrics stream (NDJSON → W&B)**

   * Define a stable NDJSON “metrics event” schema for:
     * self-play iteration stats
     * sampled MCTS root stats
     * training step stats
     * gating/eval summaries
     * oracle suite metrics (solitaire baseline + best/candidate vs oracle)
     * oracle diagnostics metrics (e.g. gating `oracle_match_rate_*`)
   * Provide a small Python utility `yatzy_az wandb-sync --run runs/<id>/` that:
     * tails NDJSON + run.json
     * emits W&B scalars/histograms consistently (or prints JSON for other tools)
   * **AC:** you can point W&B at a run directory and see live-updating curves.

   * **Schema guidance (v1, minimal but sufficient):** every metrics event includes:
     * `event`: string (e.g. `selfplay_iter`, `train_step`, `gate_summary`, `oracle_eval`)
     * `ts_ms`: u64
     * `run_id`: string
     * `v`: `{protocol_version, feature_schema_id, action_space_id, ruleset_id}`
     * `git_hash` (optional) and `config_hash` or `config_snapshot` reference
     * `step`: monotonic counter per stream (e.g. `train_step`, `global_ply`, `gate_game_idx`, `oracle_eval_idx`)

   * **Schema guidance (oracle fields):**
     * For oracle solitaire evaluation: `mean`, `std`, `median`, `min`, `max`, `bonus_rate`, `games`, `policy_id` in `{oracle,best,candidate}`
     * For oracle diagnostics in gating: `oracle_match_rate_overall`, `oracle_match_rate_mark`, `oracle_match_rate_reroll`, `oracle_keepall_ignored`
3. **Gating reproducibility fields**

   * Record `gating.seeds_hash` (hash of the exact ordered seed list used for gating; derived schedule or seed_set contents).
   * Persist a compact gating report JSON in the run dir:
     * wins/losses/draws
     * mean score diff
     * oracle diagnostics summary (oracle_match_rate + ignored-count)
     * per-seed results (optional)
   * **AC:** gating results are comparable across runs and auditable from artifacts.
4. **Training stats: minimal but sufficient**

   * Log (at least): `loss_total`, `loss_policy`, `loss_value`, `entropy`, `lr`, `throughput_steps_s`.
   * **AC:** training can be diagnosed from logs alone (no console scraping).
---

## Epic E11 — Profiling & perf regression harness

**Goal:** performance is a feature, not luck.

**Stories**

1. **Criterion microbenches**

   * hot loop benches for scoring/PUCT/protocol
   * **AC:** `yz bench` runs and reports.

2. **E2E perf benchmark**

   * fixed dummy model server; measure sims/sec and eval/sec
   * **AC:** benchmark output is stable enough to detect regressions.

3. **Flamegraph docs + commands**

   * `yz profile` wrapper + docs for `cargo flamegraph`, `torch.profiler`
   * **AC:** one-command workflow to generate flamegraphs.

---

## Epic E12 — Runpod GPU deployment & scale-out

**Goal:** same design, bigger hardware.

**Stories**

1. **Docker images**

   * inference server + trainer + worker
   * **AC:** container run locally; runpod-ready entrypoints.

2. **Multi-GPU inference**

   * one inference server per GPU; load balancing from workers
   * **AC:** worker can spread load across endpoints.

3. **Monitoring**

   * Prometheus scrape + logs
   * **AC:** you can see batch sizes, eval/sec, queue depths live.

---

## Epic E13 — Terminal UI (ratatui) + in-process iteration controller

**Goal:** provide a practical, fast feedback loop for local development and experimentation by offering a **terminal UI** to (a) configure runs, (b) start an iteration, and (c) monitor progress across self-play, training, and gating.

**Background / motivation**

The CLI pipeline works end-to-end, but it’s still friction-heavy to:

* assemble configs safely without editing YAML by hand
* keep a single run directory as the source of truth (config snapshot + `run.json` + metrics stream)
* monitor iteration health (throughput, loss curves, gating summary) without “tailing files” manually

This epic adds a **ratatui-based UI** and a small **Rust controller** that updates the run manifest fields so the UI can present a coherent view of progress.

**What we have implemented so far (v1 scaffolding)**

* `yz tui` command exists and starts a ratatui app (`yz-tui`)
* Run picker + run directory creation under `runs/`
* A full config editor screen (hybrid input) for all current fields in `yz_core::Config`
  * saves overwriteably to `runs/<id>/config.draft.yaml` and reloads it on open
* A simple dashboard view that tails `runs/<id>/logs/metrics.ndjson`
* A controller crate (`yz-controller`) that can:
  * create/ensure `run.json` + `config.yaml`
  * update `run.json` controller fields (`controller_phase`, `controller_status`, timestamps)
  * run **selfplay → gate** in-process
  * run **selfplay → train → gate**, where **training is currently invoked as a Python subprocess**

**Stories**

1. **TUI scaffold + run picker (done)**

   * Create `yz-tui` ratatui application with basic screens (Home/Config/Dashboard).
   * **AC:** `cargo run --bin yz --features tui -- tui` starts, lists/creates runs in `runs/`, and exits without leaving terminal broken.

2. **Config editor: full `yz_core::Config` form (done)**

   * Full config form with hybrid input (steppers + typed fields) and validation.
   * Persist overwriteably to `runs/<id>/config.draft.yaml` (TUI-owned) and reload on open.
   * **AC:** user can configure a run without hand-editing YAML.

3. **TUI starts an iteration (done)**

   * Add UI actions to start/cancel an iteration using the controller.
   * Run controller work on a background thread/task so the UI remains responsive.
   * Persist progress to `runs/<id>/run.json` (phase/status + timestamps) so the UI can recover after restart.
   * **AC:** from the TUI, pressing “Start” begins an iteration and `run.json` shows phase transitions (`selfplay → train → gate → done`) with status strings.

4. **Dashboard: iteration history + live progress (done)**

   * Replace “raw metrics tail only” with a two-panel dashboard driven by `run.json`:
     * **Left panel (across iterations):** show per-iteration summaries: promotion decision, loss summaries, oracle match rates, win_rate.
     * **Right panel (live):** show phase-specific progress bars:
       * self-play: games completed / target
       * gating: games completed / target
       * (optional) training: steps completed / target when `training.steps_per_iteration` is set
   * Introduce `controller_iteration_idx` + `iterations[]` in `run.json` as the UI’s primary source of truth for iteration history.
   * **AC:** dashboard shows iteration-to-iteration learning signals on the left, and a clean live progress view on the right without log tailing.

5. **Training orchestration decision (remaining)**

   * Decide and document how training should be driven by the controller:
     * **Chosen (v1):** invoke Python trainer as a subprocess (simple, robust, isolates failures, no tensor IPC).
     * **Optional later:** embed Python (PyO3) and run training in-process (single-process UX, heavier integration).
   * **Clarification:** v1 uses an in-process Rust controller for orchestration, but training itself is run as a Python subprocess that reads replay shards from disk and writes checkpoints/metrics to the run directory. This avoids any per-batch tensor serialization across processes.
   * **AC:**
     * Controller sets `controller_phase="train"` with meaningful `controller_status` while training runs.
     * Training start/end is visible in `runs/<id>/run.json` (iteration train timestamps).
     * Training failure is reflected in `run.json` via `controller_error` and logs are captured under `runs/<id>/logs/`.
     * TUI dashboard shows training progress (`steps_completed/steps_target` when configured) and last loss scalars from `run.json`.

---

## Epic E13.1 — Integrate newly added knobs (finish runtime behavior) *(complete)*

**Goal:** complete recently added config knobs by implementing their runtime behavior (not just schema/UI).

**Stories**

1. **Replay pruning (`replay.capacity_shards`)** *(done)*

   * Implement deterministic pruning of `runs/<id>/replay/` after shard flush/close.
   * Emit a metrics event (e.g. `replay_prune`) to `logs/metrics.ndjson` summarizing what was removed.
   * **Pruning policy (v1):** shard files are named `shard_{idx:06}.safetensors` + `shard_{idx:06}.meta.json` and pruning keeps the **newest N shards by filename index**.
     * Writer must **resume** at `max_existing_idx + 1` to avoid overwriting shards when continuing self-play in the same run directory.
   * **AC:**
     * replay directory growth is bounded across iterations (keeps newest N shards)
     * pruning behavior is reproducible/auditable (deterministic by shard idx)
     * `logs/metrics.ndjson` includes `event="replay_prune"` with before/after/deleted counts
   * **Status:** Implemented in `yz-replay::prune_shards_by_idx()`. Called by `yz-runtime` after shard close when `capacity_shards > 0`.

2. **Controller iteration loop (`controller.total_iterations`)** *(done)*

   * Implement a controller loop that runs N full iterations (selfplay → train → gate), updating `run.json` counters/status.
   * **Semantics (v1):** `controller.total_iterations` is an **absolute cap** per run directory.
     * `runs/<id>/run.json:controller_iteration_idx` tracks how many iterations have completed.
     * If `controller_iteration_idx >= total_iterations`, starting the controller is a no-op (immediately `done`).
     * If `controller_iteration_idx = k < total_iterations`, the controller runs only the **remaining** iterations `[k..total_iterations)`.
   * **AC:** controller stops after N total iterations and the run is auditable from `run.json` + metrics.
   * **Status:** Implemented in `yz-controller::spawn_iteration()`. Loop runs remaining iterations; tested.

3. **Epochs vs steps semantics** *(done)*

   * Make trainer behavior explicit and deterministic:
     * if `training.steps_per_iteration` is set, run exactly that many optimizer steps
     * else derive steps from `training.epochs` and the replay snapshot size
       * `steps_per_epoch = ceil(replay_snapshot.total_samples / training.batch_size)`
       * `steps_target = training.epochs * steps_per_epoch`
     * epochs-mode requires replay snapshot semantics (or explicit `--steps`) for determinism
   * **Precedence:** CLI `--steps` > `training.steps_per_iteration` > epochs-derived
   * **AC:**
     * epochs/steps behavior is deterministic and logged
     * `runs/<id>/run.json` includes `iterations[].train.steps_target` so the TUI can show training progress
     * `logs/metrics.ndjson` includes a `train_plan` event capturing the derived target inputs
   * **Status:** Implemented in Python trainer. Emits `train_plan` event; records `steps_target` in `run.json`.

---

## Epic E13.2 — Close the loop: end-to-end iterations from TUI

**Goal:** make a single TUI session capable of running multiple iterations (selfplay → train → gate) **without manual intervention**, including automatic model bootstrap, promotion, and inference server lifecycle management.

**Background / motivation**

The TUI + controller can already orchestrate phases, but several gaps prevent a truly hands-off multi-iteration loop:

1. **Training requires `--best`** — the controller launches `python -m yatzy_az train ...` without `--best`, causing the training phase to fail on a fresh run.
2. **No "new run" bootstrap** — a fresh run needs an initial `runs/<id>/models/best.pt` (via `model-init`), but the TUI/controller doesn't create it automatically.
3. **Promotion not applied** — after gating, `run.json` records promote/reject, but `candidate.pt` is not copied to `best.pt` automatically.
4. **Inference server lifecycle** — gating needs *both* best + candidate models loaded concurrently; candidate doesn't exist until after training; we need model hot-reload or controller-managed restart.

This epic addresses all four gaps, choosing **model hot-reload** for the inference server (option b).

**Stories**

1. **Controller passes `--best` to trainer (done)**

   * Update `yz-controller::run_train_subprocess` to always pass `--best runs/<id>/models/best.pt`.
   * Ensure the trainer fails cleanly if `best.pt` doesn't exist (Story 2 handles creation).
   * **AC:**
     * `controller_phase="train"` invokes trainer with correct `--best` path.
     * Training succeeds when `best.pt` exists; fails with a clear error when it doesn't.
   * **Status:** `--best` is always passed; new-run bootstrap handled by E13.2S2.

2. **TUI/controller bootstraps initial `best.pt` (done)**

   * When creating a new run (or starting an iteration on a run with no `models/best.pt`), automatically invoke `python -m yatzy_az model-init --out runs/<id>/models/best.pt` using network architecture from config.
   * Add config knobs (if not present): `model.hidden_dim`, `model.num_blocks` — these are passed to `model-init`.
   * **AC:**
     * Starting an iteration on a fresh run creates `models/best.pt` before self-play begins.
     * Network shape matches config; `best.pt` is loadable by the inference server.
   * **Status:** `model.hidden_dim` and `model.num_blocks` added to shared config schema; controller calls `ensure_best_pt()` which invokes `model-init` when `best.pt` is missing.

3. **Automatic promotion after gating (done)**

   * After gating completes with `promoted=true`, the controller copies `candidate.pt` → `best.pt` atomically.
   * If `promoted=false`, `candidate.pt` is left in place (or optionally deleted) and `best.pt` remains unchanged.
   * Update `run.json` with `iterations[].promoted` flag (already exists) and a `best_model_path` or similar for traceability.
   * **AC:**
     * Promotion is automatic; next iteration uses the promoted model as `best`.
     * `run.json` and `logs/metrics.ndjson` record the promotion event.
   * **Status:** `finalize_iteration` in `yz-controller` performs atomic copy (`copy_atomic`) when `promoted=true` and emits a `MetricsPromotionV1` event to `logs/metrics.ndjson`.

4. **Inference server model hot-reload (done)**

   * Extend the Python inference server to support hot-reloading models without restart:
     * Add a control endpoint (e.g. UDS command or HTTP `/reload`) that accepts `{"model_id": "best"|"cand", "path": "..."}`.
     * Server loads the new checkpoint into memory and swaps atomically (no requests fail during reload).
   * Controller calls the reload endpoint:
     * Before self-play: reload `best` with current `best.pt`.
     * Before gating: reload `best` + load `cand` with the newly trained `candidate.pt`.
   * **AC:**
     * Server stays running across phases; models are swapped via reload endpoint.
     * Prometheus metrics show `model_reloads_total` counter.
     * Integration test: start server → reload model → verify inference uses new weights.
   * **Status:**
     * Python: `POST /reload` endpoint in `metrics_server.py`; `Batcher.replace_model()` for atomic swap; `yatzy_infer_model_reloads_total` counter in Prometheus.
     * Rust: `reload_model()` HTTP client using `ureq`; controller calls `reload_best_for_selfplay()` and `reload_models_for_gating()`.
     * Config: `inference.metrics_bind` added to shared config schema (default `127.0.0.1:18080`).

5. **TUI preflight checks + status** *(done)*

   * Before starting an iteration, the TUI verifies:
     * Inference server is reachable (existing preflight).
     * Server supports hot-reload (version check or capability flag).
   * Dashboard shows model reload events in the live panel.
   * **AC:**
     * TUI prevents starting if server doesn't support hot-reload (or prompts user to restart server).
     * Model reload success/failure is visible in TUI.
   * **Status:**
     * Python: `GET /capabilities` endpoint in `metrics_server.py` returns `{"version": "1", "hot_reload": true|false}`.
     * Rust: `check_server_supports_hot_reload()` function in `yz-tui` calls `/capabilities` and validates response.
     * TUI: `start_iteration()` calls preflight check before spawning controller; fails with clear message if hot-reload not supported.
     * Dashboard: Shows `model_reloads: N` counter from `run.json` (incremented by controller after each reload).
     * Tests: Python tests for `/capabilities` endpoint; Rust tests for capabilities parsing and connection error handling.

---

## Epic R1 — Refactor: Code quality & tech debt

**Goal:** Address accumulated tech debt and code quality issues without changing functionality.

**Stories**

1. **Fix oracle clippy warnings**

   * Remove `#![allow(...)]` directives from `swedish_yatzy_dp` and fix the underlying clippy issues
   * Refactor `needless_range_loop`, `comparison_chain`, `manual_contains`, `new_without_default`, `wrong_self_convention` warnings
   * **AC:** `cargo clippy --workspace -- -D warnings` passes without any `#![allow(...)]` in the oracle crate.

---

## 17) Risks & mitigations (v1)

* **Rules drift**: mitigated by oracle-aligned tests + scorer parity tests.
* **Batching drift / collisions**: virtual loss + inflight caps + collision metrics.
* **Over-threading**: control PyTorch thread counts; avoid mixing rayon-heavy oracle DP build with worker threads in the same process; consider separate process for oracle suite.
* **Artifact incompatibility**: schema ids + strict versioning embedded everywhere.
* **Unobservable learning / non-reproducible runs**: mitigated by a full config snapshot (`runs/<id>/config.yaml`), seed-set hashing for gating, and W&B-compatible structured metrics (Epic E10.5).

---

If you want a next artifact that directly accelerates implementation, I can also draft:

* `configs/local_cpu.yaml` and `configs/runpod_gpu.yaml` templates with reasonable starting knobs,
* the concrete byte-level protocol definition (“Protocol v1”) and request/response structs,
* and the exact `GameTask::step()` pseudocode for the scheduler + inflight inference pipeline.

---

## Epic E6.7 — MCTS reuse + transpositions (CPU throughput lever; stochastic-safe)

**Goal:** reduce **inference calls per move** and total MCTS work on CPU by reusing search work across moves, without changing the core semantics of the search (PUCT, virtual loss caps, temperature/noise rules).

**Background / motivation**

We have already reduced Rust↔Python protocol overhead (v2 packed tensors + optional legal-mask bitset). On CPU, the dominant cost is still typically **inference-per-leaf** and the surrounding MCTS loops. A common next lever is to **reuse** portions of the search tree between consecutive decisions (\"root reuse\"), and/or to deduplicate identical states inside a search via a **transposition table**.

Yatzy is stochastic (dice), so reuse must be **stochastic-safe**:
* after applying an action, the realized next state depends on chance
* reuse should only occur when the **exact realized child state** exists in the previous tree
* if reuse misses, fall back to fresh-root behavior (no correctness regression)

**Constraints / non-goals**
* Do not increase `mcts.max_inflight_per_game` or change virtual loss behavior (this is a reuse optimization, not a semantics change).
* Self-play exploration rules unchanged: Dirichlet noise root-only; temperature affects only executed action sampling; gating remains deterministic (when configured) with `T=0`, no noise.
* Maintain oracle-compatibility invariants (state/action mapping and deterministic chance mode).

### Story 1 — Instrumentation: measure reuse opportunities and expected ROI

**Scope**
* Add always-on counters to quantify:
  * **root_reuse_attempts**, **root_reuse_hits**
  * distribution of reused-root strength: e.g. `reused_child_root_visits` (child `N_sum` at reuse time)
  * inference request volume: `infer_sent` per game (already present); add a per-game or per-move breakdown if needed
* Log the above in a low-volume way (per-worker aggregated stats in `progress.json` and/or periodic `worker_stats.ndjson` events).

**AC**
* A single short run can answer:
  * \"What % of moves could reuse the exact realized child?\"
  * \"When reuse hits, how warm is the child subtree (visit count distribution)?\"
  * \"Does reuse plausibly reduce inference calls per move?\"

### Story 2 — Root/tree reuse across moves (safe fallback)

**Scope**
* Implement \"advance root\" after executing a move:
  * after choosing action `a*` and applying it to obtain the realized next state `s'`,
    attempt to locate an existing child node for `(root_id, a*, state_key(s'))`
  * on hit: set the child as the new root and **retain** the reachable subtree
  * on miss: fall back to current behavior (fresh root / reset tree)
* Ensure root-only behaviors remain correct:
  * Dirichlet noise is applied only at the current root (self-play only)
  * temperature affects only executed action sampling, not stored targets
* Pruning strategy:
  * simplest safe first version: keep arena/children map but treat unreachable nodes as garbage (periodic rebuild optional later)

**AC**
* Correctness: gating/eval determinism is unchanged for a fixed seed set.
* Safety: if reuse misses, behavior matches current baseline.
* Performance: instrumentation shows `root_reuse_hits > 0` on representative self-play runs.

### Story 3 — Eval caching / transpositions

**Phase 1 (low risk): inference-result cache**
* Cache inference outputs keyed by `state_key(state)` (must include all state needed to identify the exact observation: dice, rerolls_left, avail_mask, player_to_move, totals).
* Cache stores **raw logits + value** (never Dirichlet-noised priors).
* Scope: per-game cache first (bounded), optionally worker-local LRU later.

**Phase 2 (higher complexity): transposition table / DAG MCTS**
* Share nodes by state-key within a search (dedupe expansions and optionally share visit stats).
* Requires careful accounting to avoid double counting across multiple parents.

**AC**
* Phase 1: measurable reduction in inference calls on workloads with repeated states (even low single-digit %).
* Phase 2 (if implemented): measurable reduction in node count / expansions without correctness regressions.

### Story 4 — CPU perf regression harness for reuse changes

**Scope**
* Add a repeatable micro+E2E perf harness that reports:
  * secs/game, infer_sent/game, median batch size (server), p50/p95 inference latency
  * reuse hit rate and reused-root visit strength
* Document \"expected ranges\" and add a guardrail checklist (\"do not break batching again\" style) for reuse changes.

**AC**
* One command can run a small, reproducible benchmark and print the above metrics.
* Reuse changes do not regress throughput on CPU beyond noise, and ideally reduce secs/game.
