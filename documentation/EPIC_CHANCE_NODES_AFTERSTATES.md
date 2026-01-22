# Epic: Explicit Chance Nodes with Implicit Afterstates (Stochastic MCTS)

## Summary

This epic refactors `yz-mcts` from “implicit stochastic transitions” (dice sampled inside `step(s,a)`) to **explicit chance nodes** so that **PUCT values converge to expectations over dice outcomes**, while still keeping **gating / eval episodes reproducible**.

Key idea:

- **Decision node**: pick an action \(a\) (KeepMask / Mark) with PUCT.
- **Afterstate**: for KeepMask, deterministically compute the post-decision, pre-roll state \(as = \phi(s,a)\).
- **Chance node**: sample dice outcome \(o \sim p(\cdot\mid as)\), apply \(s' = T(as,o)\), and continue.
- **Backup**: unchanged; chance nodes just accumulate Monte Carlo samples.

This doc is intentionally “math-first” but anchored to the current codebase (files + invariants).

---

## Epic stories (implementation plan)

This section is the implementation-ready breakdown: each story is sized to land as a single PR/stack of commits, with concrete touchpoints, acceptance criteria, and end-to-end verification steps.

### Story S0 — Prep: isolate chance sampling from `yz_core::apply_action`

**Goal**

Make it possible for `yz-mcts` to advance a reroll step **without** calling `yz_core::apply_action` for KeepMask transitions (since KeepMask will become deterministic at the decision edge).

**Scope / implementation details**

- Add a small set of pure helpers to build next `GameState` from:
  - a base `GameState` (or afterstate)
  - an outcome histogram `roll_hist[6]`
- Keep all changes inside `yz-mcts` for now (preferred), to avoid destabilizing `yz-core`.

**Touchpoints**

- `rust/yz-mcts/src/` (new module, e.g. `afterstate.rs` / `chance_hist.rs`)
- `rust/yz-core/src/action.rs` (use existing `canonicalize_keepmask` – already exported)
- `rust/yz-features` unchanged (still consumes `dice_sorted`).

**Tests**

- Unit tests in `rust/yz-mcts/src/mcts_tests.rs` or a new `afterstate_tests.rs`:
  - histogram materialization produces sorted dice
  - sum constraints hold (`sum(counts)=5`, `sum(roll_hist)=k_to_roll`)
  - deterministic sampling under fixed seed (stable outputs)

**Acceptance criteria**

- [ ] A helper exists to construct `kept_hist` from `(dice_sorted, canonical_keepmask)`.
- [ ] A helper exists to sample `roll_hist` for a given `k_to_roll` from a seeded PRNG.
- [ ] A helper exists to build `dice_sorted` from a histogram.
- [ ] Unit tests cover the above and pass.

**End-to-end verification**

- No functional behavior change yet; `cargo test -p yz-mcts` passes.

---

### Story S1 — NodeKind + AfterState: Decision→Chance factoring for KeepMask

**Goal**

Introduce explicit chance nodes keyed by afterstate for KeepMask decisions:

- Decision node picks action with PUCT.
- If KeepMask: deterministically compute `AfterState` and descend into a `Chance` node.
- Chance node samples an outcome and transitions to a Decision node with realized dice.

**Key design choice**

- NN is evaluated on **decision states only** (i.e., on `GameState`), not on afterstates/chance nodes.

**Scope / implementation details**

**Implementation status**

- Implemented in `yz-mcts` behind `MctsConfig.chance_nodes` (default: `false`).
- Instrumentation counters are exposed via `SearchStats`:
  - `chance_nodes_created`, `chance_visits`, `chance_children_created`, `chance_k_hist`.

1) **Node representation**
   - Refactor `rust/yz-mcts/src/node.rs` so the arena can store:
     - `Decision` (existing arrays: `n,w,p,vl_*`)
     - `Chance` (afterstate + aggregate stats like `visits`, `w_sum`, `num_children`)
   - Keep terminal detection via `yz_core::is_terminal(&GameState)` (no explicit terminal node necessary).

2) **AfterState definition** (Option A baked into chance node)
   - Fields:
     - `players: [yz_core::PlayerState; 2]`
     - `player_to_act: u8`
     - `rerolls_left: u8` (already decremented)
     - `kept_hist: [u8; 6]`
     - `k_to_roll: u8`

3) **Child maps**
   - Split child lookup into:
     - decision→chance: `(decision_node_id, keepmask_action_idx) -> chance_node_id`
     - chance→decision: `(chance_node_id, outcome_hist_key) -> decision_node_id`
     - decision→decision for Mark: keep existing `(decision_node_id, action_idx, StateKey(next_state)) -> decision_node_id`

4) **Traversal**
   - Update `select_leaf_and_reserve`:
     - At Decision:
       - if Mark: keep today’s path (still uses `yz_core::apply_action`).
       - if KeepMask:
         - compute afterstate (canonical keepmask)
         - descend into chance node (create if missing)
     - At Chance:
       - sample `roll_hist` from true distribution (iid uniform dice)
       - compute next `GameState` and descend into decision child keyed by outcome key

5) **Backup + virtual loss**
   - Decision edges: unchanged stats update (N/W).
   - Chance nodes: accumulate sample mean (`visits += 1`, `w_sum += g`).
   - Virtual loss only applies to decision edges (PUCT-selection edges).

**Seeding (search reproducibility)**

Chance sampling inside tree uses a deterministic seed derived from:

- base: search seed (`ChanceMode::Rng { seed }`)
- mix: leaf selection counter (`sel_counter`)
- mix: chance node id (or stable afterstate hash)

This keeps search stochastic but repeatable under gating/fixed-oracle eval.

**Touchpoints**

- `rust/yz-mcts/src/node.rs`
- `rust/yz-mcts/src/mcts.rs`
- `rust/yz-mcts/src/state_key.rs` (add an `OutcomeHistKey` type; decision `StateKey` remains)
- `rust/yz-mcts/src/mcts_tests.rs`

**Acceptance criteria**

- [ ] KeepMask transitions in-tree no longer call `yz_core::apply_action` to sample dice.
- [ ] For each `(decision_state, KeepMask action)`, there is exactly **one** chance node keyed by afterstate.
- [ ] Chance node sampling uses iid dice, but is deterministic for fixed `search_seed`.
- [ ] Existing async inference pipeline remains intact (no deadlocks; `max_inflight` logic still works).
- [ ] Add instrumentation counters (at least in `SearchStats`) for:
  - chance nodes created
  - chance visits
  - chance outcome children created
  - (optional) histogram of `k_to_roll` encountered
- [ ] The explicit chance-node implementation is gated behind a config flag (default off until stable).

**End-to-end verification**

- [ ] `cargo test -p yz-core -p yz-mcts -p yz-eval -p yz-cli` passes.
- [ ] Gating seeded test still passes: `cargo test -p yz-cli --test gate`.
- [ ] Fixed oracle eval unit tests still pass: `cargo test -p yz-eval`.
- [ ] Determinism check (gating/eval): run the same gating seed set twice and verify the produced `gate_report.json` (or summary metrics) is identical.
  - Suggested workflow:
    - use `configs/seed_sets/dev_small_v1.txt`
    - enable `gating.deterministic_chance=true`
    - run twice and compare the reports (byte-identical) or compare a stable hash.

**Manual E2E (optional, “real run”)**

This is the closest end-to-end check (Rust workers + Python infer-server) without needing training:

- Start an inference server (CPU dummy models are fine for plumbing checks):

```bash
cd python
uv run python -m yatzy_az infer-server --bind unix:///tmp/yatzy_infer.sock --device cpu --best dummy --cand dummy
```

- Run gating twice with the same seed set and deterministic chance:
  - Use any small gating config (e.g. one of the `configs/*smoke*.yaml`) and ensure it points at:
    - `infer_endpoint: unix:///tmp/yatzy_infer.sock`
    - `gating.seed_set_id: dev_small_v1`
    - `gating.deterministic_chance: true`
    - low `gating.games` for speed
  - Compare the generated `gate_report.json` (or the `gate_summary` metrics) across the two runs.

- Optional quick “MCTS-in-the-loop” smoke via the existing E2E bench harness:

```bash
cargo run -p yz-cli --bin yz -- bench e2e -- --seconds 5 --parallel 2 --simulations 16 --max-inflight 2 --chance deterministic
```

---

### Story S2 — Progressive widening at chance nodes (bounded tree growth)

**Goal**

Prevent chance nodes from accumulating too many stored outcome children (especially when `k_to_roll=5` → 252 hist outcomes).

**Implementation status**

- Implemented in `yz-mcts` behind `MctsConfig.chance_pw_*` (default: disabled).
- Exposed via shared YAML + Python validation + TUI config editor under `mcts.chance_pw.*`.
- Unbiased behavior is supported: when widening blocks storing a sampled-new outcome, MCTS does a **transient (non-stored) leaf eval + backup**.
- Instrumentation counters are exposed via `SearchStats`:
  - `chance_transient_evals`, `chance_pw_blocked` (in addition to S1 counters).

**Scope / implementation details**

- Add per-chance-node counters:
  - `visits`
  - `num_children`
- Add config knobs to `MctsConfig`:
  - `chance_pw_enabled: bool`
  - `chance_pw_c: f32`
  - `chance_pw_alpha: f32`
  - optionally `chance_pw_max_children: u16` as a hard cap
- Widening rule:
  - allow at most \(K(N)=c N^{\alpha}\) stored outcome children
  - on a sampled new outcome:
    - if below cap: create + store child
    - else: do **transient** evaluation (no insertion)

**Async inference integration note**

Transient evaluation needs a “leaf eval without storing a node” pathway. Two viable implementations:

- **S2A (simpler, first)**: cap children, but still **store** new nodes until cap; once cap hit, **reuse** an existing child by sampling again until hit (approximate; biased) — NOT preferred for “true distribution”.
- **S2B (correct)**: implement ephemeral pending-eval items (store ticket + leaf_state + path) without inserting a node into the arena. This keeps sampling unbiased.

This epic prefers **S2B**, but it’s more code.

**Acceptance criteria**

- [ ] A chance node does not exceed configured child limit schedule.
- [ ] Sampling remains unbiased w.r.t. true distribution (no “resample until hit”).
- [ ] Metrics include chance widening counters (created children vs transient outcomes).

**End-to-end verification**

- [ ] Run gating on a seed set with high reroll rate; verify chance child counts remain bounded.
- [ ] Compare `SearchStats.node_count` / expansions before vs after (should not regress badly).
- [ ] Verify unbiasedness: sample counts over outcomes at a fixed afterstate should match the multinomial distribution (statistical test / sanity bands; non-flaky).

---

### Story S3 — Hybrid exact enumeration for small k (variance reduction)

**Goal**

When `k_to_roll` is small (e.g. 1–2), compute exact expectation at the chance node rather than Monte Carlo sampling.

**Scope / implementation details**

- Add config: `chance_exact_k_max: u8` (default 0/off; typical 2).
- For `k_to_roll <= chance_exact_k_max`:
  - enumerate all hist outcomes `h` (or all sequences, but hist enumeration is smaller)
  - compute \(p(h)\) via multinomial coefficient
  - evaluate downstream values `V(T(as,h))`
  - back up expectation

**Important integration note**

Exact enumeration implies multiple leaf evals per chance visit. With async batching this can be a net win, but requires careful budgeting to avoid blowing `max_inflight`. A clean first version:

- only do exact enumeration when the resulting decision states are terminal or already expanded/cacheable, otherwise fall back to sampling.

**Acceptance criteria**

- [ ] When enabled and `k_to_roll` small, chance value matches exact expectation (unit-tested on toy states).
- [ ] No regressions in async pipeline / inflight caps.

**End-to-end verification**

- [ ] Compare variance of root value estimates on a fixed state with `k_to_roll=1/2` with vs without exact enumeration (diagnostic logging).

---

### Story S4 — Full fidelity: explicit chance for fresh-roll after `Mark`

**Goal**

Extend explicit chance separation to the “fresh roll” chance event after `Mark`.

**Scope**

- Decision(Mark) → Chance(FreshRoll) → Decision(next player with dice)
- This makes the entire turn structure “decision + chance alternating” and removes remaining implicit chance from MCTS.

**Acceptance criteria**

- [ ] No dice sampling occurs inside decision edges (neither KeepMask nor Mark).
- [ ] Gating/eval determinism of played episodes remains unchanged.
- [ ] Tree size remains controlled (likely requires progressive widening for fresh-roll chance nodes too).

**End-to-end verification**

- [ ] Same as S1 plus performance regression check on representative selfplay/gating workloads.

---

## Goals

- **G1: Decision/Chance separation**: introduce explicit chance nodes so MCTS estimates \(\mathbb{E}[\cdot]\) over dice outcomes.
- **G2: Afterstate factorization**: represent “post-decision, pre-roll” deterministically to reduce tree fragmentation.
- **G3: Canonical dice representation**: do chance outcomes and keys in histogram space (canonical), not permutations.
- **G4: Works in all MCTS phases**: selfplay, gating, fixed-oracle evaluation.
- **G5: Preserve gating/eval determinism where it matters**: the *played game* remains deterministic given episode seed + actions, while *search* remains reproducible but stochastic.

## Non-goals (initial implementation)

- **NG1: Explicit chance for fresh-roll after `Mark`**. (This is a follow-up extension; see “Fidelity extensions”.)
- **NG2: Full transposition-table / DAG MCTS**. (Orthogonal; can be layered later.)
- **NG3: NN evaluated on afterstates or chance nodes**. (NN stays on decision states.)

---

## Current behavior (baseline, v1)

Today, `yz-mcts` stores **decision nodes only** and “bakes chance into transitions”:

- Traversal calls `yz_core::apply_action(state, action, ctx)` which samples dice (in `ChanceMode::Rng`) and returns a realized `GameState`.
- Child identity includes dice via `StateKey(next_state)` (`rust/yz-mcts/src/state_key.rs`), so one decision edge can lead to many realized children over time.

Dice determinism in played environment:

- `yz-core` supports event-keyed deterministic dice (`rust/yz-core/src/chance.rs`) used in gating/eval episode rollouts.

Search reproducibility in gating/eval:

- gating derives a deterministic per-ply search seed (`gating_search_seed(...)` in `rust/yz-eval/src/lib.rs`) and passes `ChanceMode::Rng { seed }` into MCTS.

---

## Why explicit chance nodes (what fidelity we gain)

The implicit approach is “correct” in expectation but has two practical issues:

1) **Tree fragmentation**: KeepMask edges spawn many realized children keyed by full dice; the tree spends budget rediscovering the same afterstate.

2) **Poor control knobs**: progressive widening / exact enumeration / outcome caching are awkward when chance is entangled with decision edges.

Explicit chance nodes keep the semantics (“sample dice, back up value”) but reorganize the tree so the stochasticity is localized and controllable.

---

## Seeding behavior (the contract)

We need two different kinds of determinism:

### A) Played-game determinism (gating / eval)

Requirement: in gating and fixed-oracle eval, if the *played game* reaches the same `GameState` and chooses the same action, the realized next dice must be identical across runs.

This is already provided by `yz-core` deterministic chance (`rust/yz-core/src/chance.rs`), where dice are a function of:

- `episode_seed`
- `player`
- `round_idx`
- `roll_idx`

This determinism applies to **episode rollout** (the environment), not necessarily the *search tree*.

### B) Search reproducibility (stochastic but repeatable)

Requirement: the MCTS search should be “real” stochastic sampling over dice outcomes but **reproducible**.

Implementation rule:

- Inside the MCTS tree, **chance outcomes are sampled iid uniform** (the “true distribution”), but using a deterministic PRNG seed derived from the search seed.

This matches the two-layer randomness split:

- Played environment (gating/eval): event-keyed deterministic.
- Search-internal chance: iid but seeded.

This preserves both:

- **fidelity**: MCTS approximates expectation over dice outcomes,
- **reproducibility**: repeated runs with same seeds/config produce the same search behavior.

---

## Model + notation (decision → afterstate → chance)

Let \(s\) be a **decision state** (current player acts). In current code, this is `yz_core::GameState` which includes:

- `players[..]` scorecards/totals
- `dice_sorted: [u8; 5]`
- `rerolls_left`
- `player_to_move`

Decision action \(a\):

- `Action::KeepMask(mask)` on rolls 1–2
- `Action::Mark(cat)` on roll 3

For KeepMask, define a deterministic afterstate:

\[
as = \phi(s, a)
\]

Then chance happens:

\[
o \sim p(\cdot \mid as) \\
s' = T(as, o)
\]

Backup uses a return \(g\) (leaf NN value or terminal outcome).

---

## Proposed tree structure

### Node kinds

Add node kinds in `yz-mcts` (name is illustrative):

- `NodeKind::Decision(...)` (existing semantics)
- `NodeKind::Chance(AfterState)` (new)
- `NodeKind::Terminal` (optional explicit; today terminal is detected by `yz_core::is_terminal(&state)`).

### Decision node expansion

When traversing at a decision node:

- Select action by PUCT over fixed action space \(A=47\).
- Apply deterministically:
  - If `KeepMask`: compute afterstate \(as\) and descend into a chance child keyed by \(as\).
  - If `Mark`: apply the action and go directly to the next player’s decision state.

Important: “apply deterministically” here means **no dice sampling** for KeepMask at the decision node; dice sampling moves to the chance node.

### Chance node traversal

At a chance node (afterstate \(as\)):

- sample dice outcome \(o\) from the true distribution
- apply outcome to afterstate to produce next decision state \(s'\)
- continue

### Backup

Unchanged in principle:

- Decision edges accumulate \((N,W,Q)\) stats used by PUCT.
- Chance nodes accumulate samples (simple Monte Carlo mean of downstream values).

---

## AfterState (Option A: baked into chance node)

Minimum fields (adapted to current code):

- `kept_hist: [u8; 6]` (counts of locked dice faces)
- `k_to_roll: u8` (\(0..=5\))
- `rerolls_left: u8` (already decremented)
- `player_to_act: u8` (same player)
- `players: [yz_core::PlayerState; 2]` (scorecards snapshot needed for downstream value)
- (optional) any other turn metadata required for correctness (today most of it is derivable from `players` + `player_to_move`).

Deriving `kept_hist` from current encoding:

- Base representation is `dice_sorted` and `KeepMask(mask)` where bit `(4-i)` refers to `dice_sorted[i]` (`yz-core`).
- There is already keepmask canonicalization (`yz_core::canonicalize_keepmask`) and legal pruning to canonical masks (`yz-mcts`).

Implementation invariant:

- Always compute afterstate from the **canonical keepmask** used by the search (so equivalent masks map to the same afterstate).

---

## Canonical dice representation (histograms)

We will use histogram representation for chance outcomes:

- Dice configuration: `counts[6]` with sum = 5.

This is already compatible with the NN:

- Feature schema v1 already uses `dice_counts` (6 floats) derived from `dice_sorted` (`rust/yz-features/src/encode.rs`).

So: **no NN schema change is required** to start using histograms internally in MCTS; we can materialize `dice_sorted` from `counts` when building `GameState`.

---

## Chance outcome generation in histogram space

For rerolling \(k\) dice:

- sample \(k\) iid faces in \(\{1..6\}\)
- accumulate into `roll_hist[6]`
- next dice hist = `kept_hist + roll_hist`

This produces canonical outcomes by construction.

Math note (useful later if we enumerate outcomes):

Distinct histograms count:

\[
\#\{h \in \mathbb{N}^6 : \sum h_i = k\} = \binom{k+5}{5}
\]

For \(k=5\): 252 outcomes.

Probability of a histogram \(h\) under fair dice:

\[
p(h) = \frac{k!}{\prod_i h_i!}\left(\frac{1}{6}\right)^k
\]

---

## Sampling vs storing children at chance nodes

Recommended starting mode:

On each visit to a chance node:

1) sample one outcome histogram `h`
2) if `h` exists as a stored child: traverse it
3) else:
   - either create and store the child, then traverse, or
   - do a transient evaluation without storing (requires careful integration with async inference)
4) back up returned value

---

## Progressive widening at chance nodes (optional but likely needed)

Even with histograms, \(k=5\) yields 252 distinct outcomes. Progressive widening controls tree growth.

Maintain per chance node:

- `N(c)` visits
- `num_children` distinct stored outcome children

Widening schedule:

\[
K(N) = c N^{\alpha}
\]

Typical:

- \(\alpha \in [0.5, 0.8]\)
- \(c \in [1, 4]\)

Rule on sampled new outcome `h`:

- if `num_children < K(N)`: store it
- else: do **not** store; do transient evaluation and back up.

Important correctness note:

- Do **not** resample until hitting an existing child (that biases the estimator away from the true distribution).

Implementation note (async inference):

Transient evaluation means: compute `s'` and run leaf eval without creating/storing a node. In the current async pipeline (multiple in-flight pending evals), we should stage this as a *future enhancement* unless we implement a clean “ephemeral leaf” pending-eval path.

---

## Exact enumeration hybrid (optional fidelity/perf knob)

For small \(k\) (e.g. \(k \le 2\)):

- enumerate all outcomes exactly and compute an exact expectation at the chance node.

Hybrid rule (example):

- if `k_to_roll <= K_exact`: enumerate
- else: sample + (optional) progressive widening

This can reduce variance without exploding compute.

---

## Integration with current MCTS architecture (concrete plan)

### Touch points (files)

- `rust/yz-mcts/src/mcts.rs`: traversal (`select_leaf_and_reserve`), backup, child maps, stats.
- `rust/yz-mcts/src/node.rs`: node storage refactor (Decision vs Chance stats).
- `rust/yz-mcts/src/state_key.rs`: decision-state key remains; add a chance/outcome key type.
- `rust/yz-core/src/engine.rs`: **no functional change required** for played game; but we must stop using `apply_action` to sample dice for KeepMask inside tree.
- `rust/yz-eval/src/lib.rs`: ensure gating/fixed-oracle search seeds are still passed correctly.

### New keys and maps

We need three relations:

1) `Decision + KeepMask(action_idx) -> ChanceNodeId` (deterministic mapping)
2) `ChanceNodeId + OutcomeHistKey -> DecisionNodeId`
3) `Decision + Mark(action_idx) + StateKey(next_state) -> DecisionNodeId` (existing mapping)

### PRNG seeding inside chance nodes

Search seed sources today:

- selfplay uses `ChanceMode::Rng { seed }` (varies per game / per ply)
- gating uses `gating_search_seed(...)` to produce deterministic per-ply seed
- fixed-oracle eval derives seed deterministically from the fixed state

We will define a stable derivation for chance sampling inside a traversal:

- base: `mode.seed`
- mix: `sel_counter` (unique per leaf selection in one search)
- mix: `chance_node_id` (or a stable afterstate hash)

This yields deterministic but “real” iid sampling.

### Virtual loss and async inference considerations

Current MCTS uses virtual loss on decision edges to prevent duplicate pending work. With chance nodes:

- Keep virtual loss only on **decision action edges** (where a PUCT choice is made).
- Chance node sampling should not need virtual loss (it is not “selected” by PUCT), but it does affect which downstream leaf is reached.

Path representation should distinguish:

- decision edge (node_id, action_idx)
- chance edge (chance_node_id, outcome_key)

Backup should update:

- decision edge stats \(N,W,Q\)
- chance node aggregate stats (optional but useful)

---

## Fidelity extensions (future stories)

These address “we lose some important fidelity” concerns.

### F1) Explicit chance for fresh-roll after `Mark`

Today `Action::Mark` rolls fresh dice for next player inside `yz_core::apply_action` (that is a chance event). If we want full explicit chance separation, we can model:

- Decision(Mark) -> Chance(FreshRollAfterMark) -> Decision(next player with dice)

This is more faithful but increases chance branching at every turn boundary. A staged approach is recommended:

- Stage 1: explicit chance for KeepMask only (highest benefit / lowest disruption).
- Stage 2: add fresh-roll chance nodes if needed for evaluation fidelity or variance reduction.

### F2) “m samples per chance visit”

Variance reduction knob:

- sample \(m\) outcomes per chance-node visit and back up the average.

This is harder with the current async driver; likely implemented only after the basic chance-node plumbing is stable.

---

## Acceptance criteria (initial epic)

### Correctness + determinism

- [ ] **AC1**: In gating with `deterministic_chance=true`, repeated runs on the same seed set produce identical game outcomes (same scores/winners).
- [ ] **AC2**: In fixed-oracle-set eval, repeated runs with the same config produce identical oracle-match metrics.
- [ ] **AC3**: Selfplay runs complete without regressions (no panics, valid replays).

### Tree semantics

- [ ] **AC4**: For KeepMask, MCTS creates a chance node keyed by afterstate and samples outcomes from the true distribution.
- [ ] **AC5**: Decision node PUCT uses expected values (via chance sampling) rather than accidental “single realized child” values.

### Performance / size

- [ ] **AC6**: In a reroll-heavy workload, node count and/or duplication drops vs baseline implicit stochastic transitions (measure via existing `SearchStats.node_count` plus new chance-node counters).

### Testing

- [ ] **AC7**: Add unit tests proving afterstate construction and histogram sampling correctness (sum constraints, dice materialization, determinism under fixed seed).
- [ ] **AC8**: Update/extend `yz-cli` gating tests to ensure seeded gating still passes.

---

## Implementation checklist (ordered)

1) **Introduce AfterState + histogram utilities** (new `yz-mcts` module)
   - build `kept_hist` from `dice_sorted` + canonical keepmask
   - implement `sample_roll_hist(k)` and `dice_sorted_from_hist`
   - implement `OutcomeHistKey` packing/unpacking

2) **Refactor arena node representation**
   - add `NodeKind::{Decision,Chance}`
   - Decision uses existing arrays; Chance stores afterstate + aggregate stats

3) **Update traversal loop**
   - Decision node: KeepMask → Chance node; Mark → current transition
   - Chance node: sample outcome → Decision state

4) **Update child maps + path + backup**
   - add decision→chance map
   - add chance→decision outcome map
   - extend virtual loss bookkeeping to decision edges only

5) **Instrumentation**
   - counters: chance_nodes_created, chance_visits, chance_children_created
   - optional distribution by `k_to_roll`

6) **Enable across phases**
   - config flag in `MctsConfig` (explicit chance on/off)
   - default off until tests pass; then enable for selfplay/gating/fixed-oracle

7) **Progressive widening (optional stage 2)**
   - implement fixed cap first
   - then `K(N)=c*N^alpha`

