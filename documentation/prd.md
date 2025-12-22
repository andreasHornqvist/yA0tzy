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
* **Gating/eval mode**: strict determinism:

  * deterministic chance stream
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
* `budget_reroll`, `budget_mark` and optional progressive rules

### 7.3 Temperature rule (from your learnings)

* `pi` target = normalized visit counts
* temperature affects only **executed action sampling**, not replay targets

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

   * `python -m yatzy_az.infer_server --model dummy --bind unix:///tmp/y.sock --device cpu`
3. Generate a small replay:

   * `yz selfplay --config configs/local_cpu.yaml --infer unix:///tmp/y.sock --out runs/smoke/ --games 50`
4. Train candidate for a few steps:

   * `python -m yatzy_az.train --replay runs/smoke/replay --best runs/smoke/models/best.pt --out runs/smoke/models`
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

* `yatzy_az/infer_server/`
* `yatzy_az/model.py`
* `yatzy_az/train.py`
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
   * **AC:** swapping players produces consistent mirrored encoding.

---

## Epic E4 — MCTS core (yz-mcts) with stub inference

**Goal:** get MCTS correctness + diagnostics before Python integration.

**Stories**

1. **Arena tree + PUCT**

   * selection/expand/backup with dense action indexing
   * **AC:** produces valid pi distributions; no panics; deterministic in eval mode.

2. **Root noise + temperature semantics**

   * noise only in self-play; temperature only for executed move
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

   * sample roots and compare agent action vs oracle best action
   * **AC:** logs include “oracle_match_rate” (and later regret).

3. **(Optional) Regret API**

   * extend oracle with `action_value(...)` for EV(action)
   * **AC:** can compute regret distribution and log it.

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

## 17) Risks & mitigations (v1)

* **Rules drift**: mitigated by oracle-aligned tests + scorer parity tests.
* **Batching drift / collisions**: virtual loss + inflight caps + collision metrics.
* **Over-threading**: control PyTorch thread counts; avoid mixing rayon-heavy oracle DP build with worker threads in the same process; consider separate process for oracle suite.
* **Artifact incompatibility**: schema ids + strict versioning embedded everywhere.

---

If you want a next artifact that directly accelerates implementation, I can also draft:

* `configs/local_cpu.yaml` and `configs/runpod_gpu.yaml` templates with reasonable starting knobs,
* the concrete byte-level protocol definition (“Protocol v1”) and request/response structs,
* and the exact `GameTask::step()` pseudocode for the scheduler + inflight inference pipeline.
