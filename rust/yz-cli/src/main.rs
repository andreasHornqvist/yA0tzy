//! yz: CLI binary for AlphaZero Yatzy.
//!
//! Subcommands (PRD section 13):
//! - oracle expected
//! - oracle sim
//! - selfplay
//! - gate
//! - oracle-eval
//! - bench
//! - profile

use std::env;
use std::path::PathBuf;
use std::process;
use std::process::Command;
use std::time::Duration;

/// Print the oracle's optimal expected score for a fresh game.
fn cmd_oracle_expected() {
    println!("Building oracle DP table...");
    let info = yz_oracle::get_expected_score();
    println!();
    println!("Optimal expected score: {:.4}", info.expected_score);
    println!("Build time: {:.2}s", info.build_time_secs);
}

fn cmd_oracle_sim(args: &[String]) {
    let mut games: usize = 10_000;
    let mut seed: u64 = 0;
    let mut no_hist = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz oracle sim

USAGE:
    yz oracle sim [--games N] [--seed S] [--no-hist]

OPTIONS:
    --games N    Number of games to simulate (default: 10000)
    --seed S     RNG seed (default: 0)
    --no-hist    Skip printing histogram
"#
                );
                return;
            }
            "--games" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --games");
                    process::exit(1);
                }
                games = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --games value: {}", args[i + 1]);
                    process::exit(1);
                });
                i += 2;
            }
            "--seed" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --seed");
                    process::exit(1);
                }
                seed = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --seed value: {}", args[i + 1]);
                    process::exit(1);
                });
                i += 2;
            }
            "--no-hist" => {
                no_hist = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown option for `yz oracle sim`: {}", other);
                eprintln!("Run `yz oracle sim --help` for usage.");
                process::exit(1);
            }
        }
    }

    println!("Building oracle DP table...");
    let _ = yz_oracle::oracle(); // build+cache
    println!("Running simulation...");

    let report = yz_oracle::simulate(games, seed);
    let s = report.summary;

    println!();
    println!("Evaluation:");
    println!("  - Games: {}", games);
    println!(
        "  - Score: mean={:.2}, median={}, std={:.2}, min={}, max={}",
        s.mean, s.median, s.std_dev, s.min, s.max
    );
    println!("  - Upper bonus rate: {:.1}%", report.bonus_rate * 100.0);

    if !no_hist {
        yz_oracle::print_histogram(&report.scores);
    }
}

fn print_help() {
    eprintln!(
        r#"yz - AlphaZero Yatzy CLI

USAGE:
    yz <COMMAND> [OPTIONS]

COMMANDS:
    oracle expected     Print oracle expected score (~248.44)
    oracle sim          Run oracle solitaire simulation
    oracle-set-gen      Generate a deterministic fixed oracle state set (random-but-stratified)
    start-run           Create a run from a config file and start it (foreground by default)
    extend-run          Fork an existing run into a new run (copy config + optionally replay)
    controller          Run a run-dir (runs/<id>) in the foreground and print iteration summaries
    selfplay            Run self-play with MCTS + inference
    selfplay-worker     Internal: run one self-play worker process (spawned by controller)
    gate                Gate candidate vs best model (paired seed + side swap)
    gate-worker         Internal: run one gating worker process (spawned by controller)
    oracle-eval         Evaluate models against oracle baseline
    oracle-fixed-worker Internal: fixed-set oracle diagnostics worker (spawned by controller)
    bench               Run Criterion micro-benchmarks (wrapper around cargo bench)
    tui                 Terminal UI (Ratatui) for configuring + monitoring runs
    profile             Run with profiler hooks enabled

OPTIONS:
    -h, --help          Print this help message
    -V, --version       Print version

For more information, see the PRD or run `yz <COMMAND> --help`.
"#
    );
}

fn cmd_oracle_set_gen(args: &[String]) {
    let mut id: Option<String> = None;
    let mut n: u32 = 4096;
    let mut seed: u64 = 0x5EED_0BAD_CAFE_BEEF;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz oracle-set-gen

USAGE:
    yz oracle-set-gen --id ID [--n N] [--seed SEED] [--out PATH]

OPTIONS:
    --id ID         Set id (output is configs/oracle_sets/<id>.json by default) (required)
    --n N           Number of states to generate (default: 4096)
    --seed SEED     RNG seed u64 (default: 0x5EED_0BAD_CAFE_BEEF)
    --out PATH      Output path (default: configs/oracle_sets/<id>.json)
"#
                );
                return;
            }
            "--id" => {
                id = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--n" => {
                n = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(n);
                i += 2;
            }
            "--seed" => {
                seed = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(seed);
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(args.get(i + 1).cloned().unwrap_or_default()));
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz oracle-set-gen`: {other}");
                eprintln!("Run `yz oracle-set-gen --help` for usage.");
                process::exit(1);
            }
        }
    }

    let id = id.unwrap_or_else(|| {
        eprintln!("Missing --id");
        process::exit(1);
    });
    if id.trim().is_empty() {
        eprintln!("--id must be non-empty");
        process::exit(1);
    }
    let out_path = out_path.unwrap_or_else(|| PathBuf::from(format!("configs/oracle_sets/{id}.json")));

    // Random-but-stratified generation.
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_u64(rng: &mut u64) -> u64 {
        *rng = splitmix64(*rng);
        *rng
    }
    fn next_u32(rng: &mut u64) -> u32 {
        (next_u64(rng) >> 32) as u32
    }
    fn next_usize(rng: &mut u64, m: usize) -> usize {
        if m == 0 {
            0
        } else {
            (next_u32(rng) as usize) % m
        }
    }
    fn pick_range_u8(rng: &mut u64, lo: u8, hi: u8) -> u8 {
        if lo >= hi {
            return lo;
        }
        let span = (hi - lo + 1) as u32;
        lo + ((next_u32(rng) % span) as u8)
    }
    let mut rng = seed;

    // Buckets: turns (filled categories), rounds (upper_total_cap), and roll stage (rerolls_left).
    let roll_buckets: [u8; 3] = [2, 1, 0];
    let turn_buckets: [(u8, u8); 3] = [(0, 4), (5, 9), (10, 14)];
    let upper_buckets: [(u8, u8); 3] = [(0, 20), (21, 42), (43, 63)];

    use std::collections::HashSet;
    let mut seen: HashSet<yz_eval::OracleSliceStateV1> = HashSet::new();
    let mut out: Vec<yz_eval::OracleSliceStateV1> = Vec::with_capacity(n as usize);

    let oracle = yz_oracle::oracle();

    #[derive(Clone, Copy)]
    struct ActionSpec {
        action: yz_core::Action,
        allowed_rolls: &'static [u8],
    }
    const ROLLS_KEEP: &[u8] = &[2, 1];
    const ROLLS_MARK: &[u8] = &[2, 1, 0];

    let mut actions: Vec<ActionSpec> = Vec::new();
    for idx in 0..(yz_core::A as u8) {
        let action = yz_core::index_to_action(idx);
        if matches!(action, yz_core::Action::KeepMask(31)) {
            // Skip keep-all (ignored in eval when rerolls_left>0).
            continue;
        }
        let allowed_rolls = match action {
            yz_core::Action::KeepMask(_) => ROLLS_KEEP,
            yz_core::Action::Mark(_) => ROLLS_MARK,
        };
        actions.push(ActionSpec { action, allowed_rolls });
    }

    let action_count = actions.len().max(1) as u32;
    let base = n / action_count;
    let mut rem = n - base * action_count;

    fn upper_max_from_avail_mask(avail_mask: u16) -> u8 {
        let mut sum: u16 = 0;
        for cat in 0u8..=5u8 {
            let bit = yz_core::avail_bit_for_cat(cat);
            if (avail_mask & bit) == 0 {
                sum = sum.saturating_add(5 * (cat as u16 + 1));
            }
        }
        sum.min(63) as u8
    }

    fn build_avail_mask(
        rng: &mut u64,
        available: u8,
        must_cat: Option<u8>,
    ) -> Option<u16> {
        if available == 0 || available > yz_core::NUM_CATS as u8 {
            return None;
        }
        let mut cats: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        if let Some(cat) = must_cat {
            if (cat as usize) < cats.len() {
                let pos = cats.iter().position(|c| *c == cat).unwrap_or(0);
                cats.swap(0, pos);
            }
        }
        for j in 1..(available as usize) {
            let k = j + next_usize(rng, 15 - j);
            cats.swap(j, k);
        }
        let mut avail_mask: u16 = 0;
        for &cat in cats.iter().take(available as usize) {
            avail_mask |= yz_core::avail_bit_for_cat(cat);
        }
        if let Some(cat) = must_cat {
            if (avail_mask & yz_core::avail_bit_for_cat(cat)) == 0 {
                return None;
            }
        }
        Some(avail_mask)
    }

    fn should_ignore_oracle_action(oa: yz_oracle::Action, rerolls_left: u8) -> bool {
        matches!(oa, yz_oracle::Action::KeepMask { mask: 31 } if rerolls_left > 0)
    }

    fn canonicalize_keepmask(dice_sorted: [u8; 5], mask: u8) -> u8 {
        debug_assert!(dice_sorted.windows(2).all(|w| w[0] <= w[1]));
        debug_assert!(mask < 32);

        let mut need = [0u8; 6];
        for i in 0..5usize {
            let bit = 1u8 << (4 - i);
            if (mask & bit) != 0 {
                let face = dice_sorted[i] as usize;
                debug_assert!((1..=6).contains(&face));
                need[face - 1] = need[face - 1].saturating_add(1);
            }
        }

        let mut out: u8 = 0;
        for i in (0..5usize).rev() {
            let bit = 1u8 << (4 - i);
            let face = dice_sorted[i] as usize;
            let slot = face - 1;
            if need[slot] > 0 {
                need[slot] -= 1;
                out |= bit;
            }
        }
        out
    }

    let mut action_counts: Vec<u32> = vec![0; actions.len()];
    let mut roll_counts: [u32; 3] = [0, 0, 0];
    let mut turn_counts: [u32; 3] = [0, 0, 0];
    let mut upper_counts: [u32; 3] = [0, 0, 0];

    let mark_actions = actions
        .iter()
        .filter(|a| matches!(a.action, yz_core::Action::Mark(_)))
        .count() as u32;
    let _keep_actions = action_count.saturating_sub(mark_actions);
    let mark_target = (n as u64 * (mark_actions as u64) / (action_count as u64)) as u32;
    let _keep_target = n.saturating_sub(mark_target);

    // Roll-stage targets: keep r1/r2 balanced and keep r0 lower to respect mark/keep ratio.
    let r0_target = (mark_target / 2).max(1);
    let r1_target = (n.saturating_sub(r0_target)) / 2;
    let r2_target = n.saturating_sub(r0_target).saturating_sub(r1_target);
    let roll_targets: [u32; 3] = [r2_target, r1_target, r0_target]; // order matches roll_buckets [2,1,0]

    let base_turn = n / 3;
    let mut rem_turn = n - base_turn * 3;
    let mut turn_targets = [base_turn; 3];
    for t in &mut turn_targets {
        if rem_turn == 0 {
            break;
        }
        *t += 1;
        rem_turn -= 1;
    }

    let base_upper = n / 3;
    let mut rem_upper = n - base_upper * 3;
    let mut upper_targets = [base_upper; 3];
    for t in &mut upper_targets {
        if rem_upper == 0 {
            break;
        }
        *t += 1;
        rem_upper -= 1;
    }

    fn pick_bucket_idx(rng: &mut u64, counts: &[u32], targets: &[u32], candidates: &[usize]) -> usize {
        let deficit: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&i| counts[i] < targets[i])
            .collect();
        if deficit.is_empty() {
            let j = next_usize(rng, candidates.len());
            return candidates[j];
        }
        let j = next_usize(rng, deficit.len());
        deficit[j]
    }

    for (ai, a) in actions.iter().enumerate() {
        let target = base + if rem > 0 { rem -= 1; 1 } else { 0 };
        if target == 0 {
            continue;
        }
        let mut attempts = 0u32;
        while action_counts[ai] < target {
            attempts += 1;
            if attempts > 1_500_000 {
                eprintln!(
                    "oracle-set-gen: too many attempts for action {} (added={}/{})",
                    ai, action_counts[ai], target
                );
                process::exit(2);
            }
            let relax = attempts > 800_000;
            let roll_candidates: Vec<usize> = a
                .allowed_rolls
                .iter()
                .filter_map(|r| roll_buckets.iter().position(|x| x == r))
                .collect();
            if roll_candidates.is_empty() {
                continue;
            }
            let r_idx = pick_bucket_idx(&mut rng, &roll_counts, &roll_targets, &roll_candidates);
            let rerolls_left = roll_buckets[r_idx];

            let turn_candidates: Vec<usize> = (0..turn_buckets.len()).collect();
            let t_idx = pick_bucket_idx(&mut rng, &turn_counts, &turn_targets, &turn_candidates);
            let turn_bucket = turn_buckets[t_idx];

            let upper_candidates: Vec<usize> = (0..upper_buckets.len()).collect();
            let u_idx = pick_bucket_idx(&mut rng, &upper_counts, &upper_targets, &upper_candidates);
            let upper_bucket = upper_buckets[u_idx];

            let filled = if relax {
                pick_range_u8(&mut rng, 0, (yz_core::NUM_CATS as u8).saturating_sub(1))
            } else {
                pick_range_u8(&mut rng, turn_bucket.0, turn_bucket.1)
            };
            if filled >= yz_core::NUM_CATS as u8 {
                continue;
            }
            let available = (yz_core::NUM_CATS as u8).saturating_sub(filled);
            if available == 0 {
                continue;
            }
            let must_cat = match a.action {
                yz_core::Action::Mark(cat) => Some(cat),
                _ => None,
            };
            let Some(avail_mask) = build_avail_mask(&mut rng, available, must_cat) else {
                continue;
            };
            let upper_max = upper_max_from_avail_mask(avail_mask);
            let upper_total_cap = if relax {
                pick_range_u8(&mut rng, 0, upper_max)
            } else {
                let u_lo = upper_bucket.0;
                let u_hi = upper_bucket.1.min(upper_max);
                if u_hi < u_lo {
                    continue;
                }
                pick_range_u8(&mut rng, u_lo, u_hi)
            };

            let mut dice = [0u8; 5];
            for d in &mut dice {
                *d = 1 + (next_u32(&mut rng) % 6) as u8;
            }
            dice.sort();

            let st = yz_eval::OracleSliceStateV1 {
                avail_mask,
                upper_total_cap,
                dice_sorted: dice,
                rerolls_left,
            };
            if seen.contains(&st) {
                continue;
            }
            let (oa, _ev) =
                oracle.best_action(st.avail_mask, st.upper_total_cap, st.dice_sorted, st.rerolls_left);
            if should_ignore_oracle_action(oa, st.rerolls_left) {
                continue;
            }
            let expected = match oa {
                yz_oracle::Action::Mark { cat } => yz_core::Action::Mark(cat),
                yz_oracle::Action::KeepMask { mask } => {
                    yz_core::Action::KeepMask(canonicalize_keepmask(st.dice_sorted, mask))
                }
            };
            if expected != a.action {
                continue;
            }
            if seen.insert(st) {
                out.push(st);
                action_counts[ai] += 1;
                roll_counts[r_idx] += 1;
                let tb = if filled <= turn_buckets[0].1 {
                    0
                } else if filled <= turn_buckets[1].1 {
                    1
                } else {
                    2
                };
                turn_counts[tb] += 1;
                let ub = if upper_total_cap <= upper_buckets[0].1 {
                    0
                } else if upper_total_cap <= upper_buckets[1].1 {
                    1
                } else {
                    2
                };
                upper_counts[ub] += 1;
            }
        }
    }

    if out.len() != n as usize {
        eprintln!(
            "oracle-set-gen: generated {} states but expected {}",
            out.len(),
            n
        );
        process::exit(2);
    }

    // Stable output order: sort by a deterministic key.
    out.sort_by_key(|s| {
        let mut k: u64 = s.avail_mask as u64;
        k ^= (s.upper_total_cap as u64) << 16;
        k ^= (s.rerolls_left as u64) << 24;
        k ^= (s.dice_sorted[0] as u64) << 32;
        k ^= (s.dice_sorted[1] as u64) << 40;
        k ^= (s.dice_sorted[2] as u64) << 48;
        k ^= (s.dice_sorted[3] as u64) << 56;
        k
    });

    let bytes = serde_json::to_vec_pretty(&out).unwrap();
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = out_path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", tmp.display());
        process::exit(1);
    });
    std::fs::rename(&tmp, &out_path).unwrap_or_else(|e| {
        eprintln!("Failed to rename {}: {e}", out_path.display());
        process::exit(1);
    });
    println!("wrote {} states to {}", out.len(), out_path.display());
}

fn cmd_oracle_fixed_worker(args: &[String]) {
    let mut run_dir: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut cand_id: Option<u32> = None;
    let mut iter_idx: Option<u32> = None;
    let mut set_id: Option<String> = None;
    let mut shard_idx: u32 = 0;
    let mut num_shards: u32 = 1;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz oracle-fixed-worker

USAGE:
    yz oracle-fixed-worker --run-dir DIR --infer ENDPOINT --cand-id ID --iter-idx N --set-id SET [--shard-idx I --num-shards K]
"#
                );
                return;
            }
            "--run-dir" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--cand-id" => {
                cand_id = args.get(i + 1).and_then(|s| s.parse::<u32>().ok());
                i += 2;
            }
            "--iter-idx" => {
                iter_idx = args.get(i + 1).and_then(|s| s.parse::<u32>().ok());
                i += 2;
            }
            "--set-id" => {
                set_id = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--shard-idx" => {
                shard_idx = args.get(i + 1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                i += 2;
            }
            "--num-shards" => {
                num_shards = args.get(i + 1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1);
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz oracle-fixed-worker`: {other}");
                process::exit(1);
            }
        }
    }

    let run_dir = PathBuf::from(run_dir.unwrap_or_else(|| {
        eprintln!("Missing --run-dir");
        process::exit(1);
    }));
    let infer = infer.unwrap_or_else(|| {
        eprintln!("Missing --infer");
        process::exit(1);
    });
    let cand_id = cand_id.unwrap_or_else(|| {
        eprintln!("Missing --cand-id");
        process::exit(1);
    });
    let iter_idx = iter_idx.unwrap_or_else(|| {
        eprintln!("Missing --iter-idx");
        process::exit(1);
    });
    let set_id = set_id.unwrap_or_else(|| {
        eprintln!("Missing --set-id");
        process::exit(1);
    });
    if num_shards < 1 {
        eprintln!("--num-shards must be >= 1");
        process::exit(1);
    }
    if shard_idx >= num_shards {
        eprintln!("--shard-idx must be in [0, --num-shards)");
        process::exit(1);
    }

    let cancel_file = run_dir.join("cancel.request");
    if cancel_file.exists() {
        return;
    }

    let cfg_path = run_dir.join("config.yaml");
    let cfg = yz_core::Config::load(&cfg_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config.yaml: {e}");
        process::exit(1);
    });
    // Prefer new location: gating.fixed_oracle.* ; fall back to legacy cfg.oracle.*.
    let enabled = cfg.gating.fixed_oracle.enabled
        || cfg.oracle.as_ref().map(|o| o.fixed_set_enabled).unwrap_or(false);
    let cfg_set_id = if cfg.gating.fixed_oracle.enabled {
        cfg.gating.fixed_oracle.set_id.as_deref()
    } else {
        cfg.oracle.as_ref().and_then(|o| o.fixed_set_id.as_deref())
    };
    if !(enabled && cfg_set_id == Some(set_id.as_str())) {
        // Disabled or mismatched set id → no-op.
        return;
    }

    let mut states = yz_eval::load_oracle_set(&set_id).unwrap_or_else(|e| {
        eprintln!("Failed to load oracle set: {e}");
        process::exit(1);
    });

    // Apply fixed-oracle N before sharding so progress/aggregation are accurate.
    let set_len = states.len();
    let max_n = cfg
        .gating
        .fixed_oracle
        .n
        .map(|x| x as usize)
        .unwrap_or(states.len())
        .min(states.len());
    if let Some(n) = cfg.gating.fixed_oracle.n {
        if (n as usize) > set_len {
            eprintln!(
                "[oracle-fixed-worker] warning: fixed_oracle.n={} > set size {}; using {}",
                n, set_len, max_n
            );
        }
    }

    fn state_hash(st: &yz_eval::OracleSliceStateV1) -> u64 {
        let mut k: u64 = st.avail_mask as u64;
        k ^= (st.upper_total_cap as u64) << 16;
        k ^= (st.rerolls_left as u64) << 24;
        k ^= (st.dice_sorted[0] as u64) << 32;
        k ^= (st.dice_sorted[1] as u64) << 40;
        k ^= (st.dice_sorted[2] as u64) << 48;
        k ^= (st.dice_sorted[3] as u64) << 56;
        k
    }

    fn turn_bucket_from_avail(avail_mask: u16) -> u8 {
        let available = avail_mask.count_ones() as i32;
        let filled = (yz_core::NUM_CATS as i32 - available).clamp(0, yz_core::NUM_CATS as i32);
        if filled <= 4 {
            0
        } else if filled <= 9 {
            1
        } else {
            2
        }
    }

    fn upper_bucket_from_cap(upper_total_cap: u8) -> u8 {
        if upper_total_cap <= 20 {
            0
        } else if upper_total_cap <= 42 {
            1
        } else {
            2
        }
    }

    fn roll_bucket_from_rerolls(r: u8) -> u8 {
        match r {
            2 => 0,
            1 => 1,
            _ => 2,
        }
    }

    fn should_ignore_oracle_action(oa: yz_oracle::Action, rerolls_left: u8) -> bool {
        matches!(oa, yz_oracle::Action::KeepMask { mask: 31 } if rerolls_left > 0)
    }

    #[derive(Clone, Copy)]
    struct Meta {
        st: yz_eval::OracleSliceStateV1,
        action_idx: u8,
        roll_bucket: u8,
        turn_bucket: u8,
        upper_bucket: u8,
        hash: u64,
    }

    if max_n < states.len() {
        use std::collections::BTreeMap;

        let oracle = yz_oracle::oracle();
        let mut metas: Vec<Meta> = Vec::with_capacity(states.len());
        for st in &states {
            let (oa, _ev) =
                oracle.best_action(st.avail_mask, st.upper_total_cap, st.dice_sorted, st.rerolls_left);
            if should_ignore_oracle_action(oa, st.rerolls_left) {
                continue;
            }
            let expected = match oa {
                yz_oracle::Action::Mark { cat } => yz_core::Action::Mark(cat),
                yz_oracle::Action::KeepMask { mask } => yz_core::Action::KeepMask(mask),
            };
            let action_idx = yz_core::action_to_index(expected);
            metas.push(Meta {
                st: *st,
                action_idx,
                roll_bucket: roll_bucket_from_rerolls(st.rerolls_left),
                turn_bucket: turn_bucket_from_avail(st.avail_mask),
                upper_bucket: upper_bucket_from_cap(st.upper_total_cap),
                hash: state_hash(st),
            });
        }

        let mut by_action: Vec<Vec<Meta>> = vec![Vec::new(); yz_core::A];
        for m in metas {
            by_action[m.action_idx as usize].push(m);
        }
        let mut action_indices: Vec<usize> = by_action
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v.is_empty() { None } else { Some(i) })
            .collect();
        action_indices.sort_unstable();

        let action_count = action_indices.len().max(1);
        let base = max_n / action_count;
        let mut rem = max_n - base * action_count;

        let mut selected: Vec<Meta> = Vec::with_capacity(max_n);
        let mut remaining: Vec<Meta> = Vec::new();

        for &ai in &action_indices {
            let target = base + if rem > 0 { rem -= 1; 1 } else { 0 };
            if target == 0 {
                continue;
            }
            let mut buckets: BTreeMap<(u8, u8, u8), Vec<Meta>> = BTreeMap::new();
            for m in &by_action[ai] {
                buckets
                    .entry((m.roll_bucket, m.turn_bucket, m.upper_bucket))
                    .or_default()
                    .push(*m);
            }
            for b in buckets.values_mut() {
                b.sort_by_key(|m| m.hash);
            }

            let keys: Vec<(u8, u8, u8)> = buckets.keys().copied().collect();
            let bcount = keys.len().max(1);
            let base_b = target / bcount;
            let mut rem_b = target - base_b * bcount;
            let mut selected_action = 0usize;
            let mut leftover_action: Vec<Meta> = Vec::new();

            for key in keys {
                let take = base_b + if rem_b > 0 { rem_b -= 1; 1 } else { 0 };
                let bucket = buckets.get_mut(&key).unwrap();
                let n_take = take.min(bucket.len());
                selected.extend(bucket.drain(0..n_take));
                selected_action += n_take;
                leftover_action.extend(bucket.drain(..));
            }

            if selected_action < target {
                leftover_action.sort_by_key(|m| m.hash);
                let need = target - selected_action;
                let n_take = need.min(leftover_action.len());
                selected.extend(leftover_action.drain(0..n_take));
            }
            remaining.extend(leftover_action);
        }

        if selected.len() < max_n {
            remaining.sort_by_key(|m| m.hash);
            let need = max_n - selected.len();
            selected.extend(remaining.into_iter().take(need));
        }

        if selected.len() < max_n {
            eprintln!(
                "[oracle-fixed-worker] warning: requested n={} but only selected {} states",
                max_n,
                selected.len()
            );
        }
        selected.sort_by_key(|m| m.hash);
        states = selected.into_iter().map(|m| m.st).collect();
    } else {
        states.sort_by_key(state_hash);
    }

    // Shard states deterministically by index mod num_shards.
    let shard_states: Vec<yz_eval::OracleSliceStateV1> = states
        .iter()
        .enumerate()
        .filter(|(i, _)| (*i as u32) % num_shards == shard_idx)
        .map(|(_, s)| *s)
        .collect();

    // Match gate-worker client + MCTS settings (no dirichlet).
    let client_opts = yz_infer::ClientOptions {
        max_inflight_total: 64,
        max_outbound_queue: 256,
        request_id_start: 1,
        protocol_version: cfg.inference.protocol_version,
        legal_mask_bitset: cfg.inference.legal_mask_bitset && cfg.inference.protocol_version == 2,
    };
    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: 0.0,
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss_mode: match cfg.mcts.virtual_loss_mode.as_str() {
            "q_penalty" => yz_mcts::VirtualLossMode::QPenalty,
            "n_virtual_only" => yz_mcts::VirtualLossMode::NVirtualOnly,
            "off" => yz_mcts::VirtualLossMode::Off,
            _ => yz_mcts::VirtualLossMode::QPenalty,
        },
        virtual_loss: cfg.mcts.virtual_loss.max(0.0),
        expansion_lock: cfg.mcts.katago.expansion_lock,
    };

    if cancel_file.exists() {
        return;
    }

    let t0 = std::time::Instant::now();
    eprintln!(
        "[oracle-fixed-worker] start run_dir={} iter_idx={} set_id={} shard={}/{} n={} assigned={} mcts=mark:{} reroll:{} endpoint={}",
        run_dir.display(),
        iter_idx,
        set_id,
        shard_idx,
        num_shards,
        cfg.gating
            .fixed_oracle
            .n
            .map(|x| x.to_string())
            .unwrap_or_else(|| "all".to_string()),
        shard_states.len(),
        cfg.mcts.budget_mark,
        cfg.mcts.budget_reroll,
        infer
    );

    // Progress + report files (per-shard, per-iteration).
    let out_dir = run_dir
        .join("oracle_fixed")
        .join(format!("iter_{iter_idx:03}"));
    let _ = std::fs::create_dir_all(&out_dir);
    let progress_path = out_dir.join(format!("progress_{shard_idx:03}.json"));
    let report_path = out_dir.join(format!("shard_{shard_idx:03}.json"));

    #[derive(serde::Serialize)]
    struct OracleFixedProgressV1 {
        iter_idx: u32,
        shard_idx: u32,
        num_shards: u32,
        done: u64,
        total: u64,
        pid: u32,
        ts_ms: u64,
    }
    fn write_progress_atomic(path: &PathBuf, p: &OracleFixedProgressV1) {
        let tmp = path.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec(p) {
            let _ = std::fs::write(&tmp, &bytes);
            let _ = std::fs::rename(&tmp, path);
        }
    }
    write_progress_atomic(
        &progress_path,
        &OracleFixedProgressV1 {
            iter_idx,
            shard_idx,
            num_shards,
            done: 0,
            total: shard_states.len() as u64,
            pid: std::process::id(),
            ts_ms: yz_logging::now_ms(),
        },
    );

    let mut last_progress_write = std::time::Instant::now();
    let mut progress_cb = |done: u64, total: u64| {
        let should_write = done == total || last_progress_write.elapsed().as_millis() >= 500;
        if should_write {
            write_progress_atomic(
                &progress_path,
                &OracleFixedProgressV1 {
                    iter_idx,
                    shard_idx,
                    num_shards,
                    done,
                    total,
                    pid: std::process::id(),
                    ts_ms: yz_logging::now_ms(),
                },
            );
            last_progress_write = std::time::Instant::now();
        }
    };

    let rep = yz_eval::eval_fixed_oracle_set(
        &cfg,
        &infer,
        cand_id,
        &client_opts,
        mcts_cfg,
        &set_id,
        &shard_states,
        Some(&mut progress_cb),
    )
    .unwrap_or_else(|e| {
        eprintln!("oracle-fixed-worker failed: {e}");
        process::exit(1);
    });

    if cancel_file.exists() {
        return;
    }

    // Final progress snapshot + write shard report (controller aggregates and emits NDJSON).
    write_progress_atomic(
        &progress_path,
        &OracleFixedProgressV1 {
            iter_idx,
            shard_idx,
            num_shards,
            done: rep.total,
            total: shard_states.len() as u64,
            pid: std::process::id(),
            ts_ms: yz_logging::now_ms(),
        },
    );
    let bytes = serde_json::to_vec_pretty(&rep).unwrap();
    let tmp = report_path.with_extension("json.tmp");
    let _ = std::fs::write(&tmp, &bytes);
    let _ = std::fs::rename(&tmp, &report_path);

    eprintln!(
        "[oracle-fixed-worker] done iter_idx={} set_id={} shard={}/{} total={} overall={:.4} mark={:.4} reroll={:.4} elapsed={:.2}s",
        iter_idx,
        set_id,
        shard_idx,
        num_shards,
        rep.total,
        rep.match_rate_overall(),
        rep.match_rate_mark(),
        rep.match_rate_reroll(),
        t0.elapsed().as_secs_f64()
    );
}

fn cmd_extend_run(args: &[String]) {
    let mut src: Option<String> = None;
    let mut dst: Option<String> = None;
    let mut runs_dir: String = "runs".to_string();
    let mut copy_replay = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz extend-run

USAGE:
    yz extend-run --src SRC_RUN_ID --dst DST_RUN_NAME [--runs-dir DIR] [--copy-replay]

OPTIONS:
    --src SRC_RUN_ID      Source run id (directory under runs/)
    --dst DST_RUN_NAME    New run name (directory under runs/; timestamp appended if exists)
    --runs-dir DIR        Runs directory (default: runs)
    --copy-replay         Copy replay/ from source into destination (default: off)
"#
                );
                return;
            }
            "--src" => {
                src = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--dst" => {
                dst = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--runs-dir" => {
                runs_dir = args.get(i + 1).cloned().unwrap_or_else(|| "runs".to_string());
                i += 2;
            }
            "--copy-replay" => {
                copy_replay = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown option for `yz extend-run`: {other}");
                eprintln!("Run `yz extend-run --help` for usage.");
                process::exit(1);
            }
        }
    }

    let src = src.unwrap_or_else(|| {
        eprintln!("Missing --src");
        process::exit(1);
    });
    let dst = dst.unwrap_or_else(|| {
        eprintln!("Missing --dst");
        process::exit(1);
    });
    if src.trim().is_empty() {
        eprintln!("--src must be non-empty");
        process::exit(1);
    }
    if dst.trim().is_empty() {
        eprintln!("--dst must be non-empty");
        process::exit(1);
    }

    let runs_dir = PathBuf::from(runs_dir);
    std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create runs dir {}: {e}", runs_dir.display());
        process::exit(1);
    });

    let (new_id, new_dir) =
        yz_controller::extend_run(&runs_dir, &src, &dst, copy_replay).unwrap_or_else(|e| {
            eprintln!("extend-run failed: {e}");
            process::exit(1);
        });
    println!("run_id: {new_id}");
    println!("run_dir: {}", new_dir.display());
}

fn sanitize_run_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn ensure_unique_run_dir(runs_dir: &std::path::Path, base_name: &str) -> (String, PathBuf) {
    let ts = yz_logging::now_ms();
    let base = sanitize_run_name(base_name);
    let id = if base.is_empty() {
        format!("run_{ts}")
    } else {
        base
    };
    let mut final_id = id.clone();
    if runs_dir.join(&final_id).exists() {
        final_id = format!("{id}_{ts}");
    }
    let dir = runs_dir.join(&final_id);
    (final_id, dir)
}

fn write_cancel_request(run_dir: &std::path::Path) -> std::io::Result<()> {
    let p = run_dir.join("cancel.request");
    let tmp = run_dir.join("cancel.request.tmp");
    std::fs::write(&tmp, format!("ts_ms: {}\n", yz_logging::now_ms()))?;
    std::fs::rename(&tmp, &p)?;
    Ok(())
}

fn print_iteration_table_row(it: &yz_logging::IterationSummaryV1) {
    let decision = match it.promoted {
        Some(true) => "promote",
        Some(false) => "reject",
        None => "-",
    };
    let wr = it.gate.win_rate.map(|x| format!("{:.3}", x)).unwrap_or("-".to_string());
    let oracle = it
        .oracle
        .match_rate_overall
        .map(|x| format!("{:.3}", x))
        .unwrap_or("-".to_string());
    let lt = it
        .train
        .last_loss_total
        .map(|x| format!("{:.3}", x))
        .unwrap_or("-".to_string());
    let lp = it
        .train
        .last_loss_policy
        .map(|x| format!("{:.3}", x))
        .unwrap_or("-".to_string());
    let lv = it
        .train
        .last_loss_value
        .map(|x| format!("{:.3}", x))
        .unwrap_or("-".to_string());
    let avg_score = it
        .gate
        .mean_cand_score
        .map(|x| format!("{:.1}", x))
        .unwrap_or("-".to_string());
    let avg_best = it
        .gate
        .mean_best_score
        .map(|x| format!("{:.1}", x))
        .unwrap_or("-".to_string());

    println!(
        "{:>4}  {:<7}  {:>7}  {:>7}  {:>5}/{:>5}/{:>5}  {:>7}/{:>7}",
        it.idx, decision, wr, oracle, lt, lp, lv, avg_score, avg_best
    );
}

fn run_controller_foreground(
    run_dir: PathBuf,
    cfg: yz_core::Config,
    python_exe: String,
    print_iter_table: bool,
    verbose_status: bool,
) -> i32 {
    let infer_endpoint = cfg.inference.bind.clone();
    let handle = yz_controller::spawn_iteration(&run_dir, cfg, infer_endpoint, python_exe);
    let run_json = run_dir.join("run.json");

    let mut last_phase: Option<String> = None;
    let mut last_status: Option<String> = None;
    let mut printed_iters: u32 = 0;

    if print_iter_table {
        println!("Iter  Decision  WinRate  Oracle    Loss(t/p/v)     AvgScore(cand/best)");
    }

    while !handle.is_finished() {
        if let Ok(m) = yz_logging::read_manifest(&run_json) {
            if verbose_status && (m.controller_phase != last_phase || m.controller_status != last_status) {
                last_phase = m.controller_phase.clone();
                last_status = m.controller_status.clone();
                if let (Some(p), Some(s)) = (last_phase.as_deref(), last_status.as_deref()) {
                    println!("[phase={p}] {s}");
                }
            }
            if print_iter_table {
                for it in &m.iterations {
                    // Print each completed iteration at most once.
                    if it.idx >= printed_iters && it.ended_ts_ms.is_some() {
                        print_iteration_table_row(it);
                        printed_iters = it.idx.saturating_add(1);
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(300));
    }

    let code = match handle.join() {
        Ok(()) => 0,
        Err(yz_controller::ControllerError::Cancelled) => 2,
        Err(e) => {
            eprintln!("controller failed: {e}");
            1
        }
    };

    // If the controller finishes very quickly, the polling loop above might not print anything.
    // Do one final read and print any completed iterations we haven't printed yet.
    if print_iter_table {
        if let Ok(m) = yz_logging::read_manifest(&run_json) {
            for it in &m.iterations {
                if it.idx >= printed_iters && it.ended_ts_ms.is_some() {
                    print_iteration_table_row(it);
                    printed_iters = it.idx.saturating_add(1);
                }
            }
        }
    }

    code
}

fn cmd_controller(args: &[String]) {
    let mut run_dir: Option<String> = None;
    let mut python_exe: String = "python".to_string();
    let mut print_iter_table = false;
    let mut verbose = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz controller

USAGE:
    yz controller --run-dir runs/<id> [--python-exe python] [--print-iter-table]

OPTIONS:
    --run-dir DIR           Run directory (contains config.yaml) (required)
    --python-exe PATH       Python executable or \"python\" (uv-managed) (default: python)
    --print-iter-table      Print one summary row per completed iteration to stdout
"#
                );
                return;
            }
            "--run-dir" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--python-exe" => {
                python_exe = args.get(i + 1).cloned().unwrap_or_else(|| "python".to_string());
                i += 2;
            }
            "--print-iter-table" => {
                print_iter_table = true;
                i += 1;
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            "--cancel" => {
                let rd = args.get(i + 1).cloned().unwrap_or_default();
                if rd.is_empty() {
                    eprintln!("Missing value for --cancel");
                    process::exit(1);
                }
                if let Err(e) = write_cancel_request(std::path::Path::new(&rd)) {
                    eprintln!("Failed to write cancel.request: {e}");
                    process::exit(1);
                }
                println!("cancel requested: {}", std::path::Path::new(&rd).display());
                return;
            }
            other => {
                eprintln!("Unknown option for `yz controller`: {other}");
                eprintln!("Run `yz controller --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run_dir = run_dir.unwrap_or_else(|| {
        eprintln!("Missing --run-dir");
        process::exit(1);
    });
    let run_dir = PathBuf::from(run_dir);
    let cfg_path = run_dir.join("config.yaml");
    let cfg = yz_core::Config::load(&cfg_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config at {}: {e}", cfg_path.display());
        process::exit(1);
    });

    let code = run_controller_foreground(run_dir, cfg, python_exe, print_iter_table, verbose);
    process::exit(code);
}

fn cmd_start_run(args: &[String]) {
    let mut run_name: Option<String> = None;
    let mut cfg_path: Option<String> = None;
    let mut runs_dir: String = "runs".to_string();
    let mut python_exe: String = "python".to_string();
    let mut detach = false;
    let mut print_iter_table = true;
    let mut verbose = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz start-run

USAGE:
    yz start-run --run-name NAME --config PATH [--runs-dir DIR] [--python-exe PATH] [--detach]

OPTIONS:
    --run-name NAME         Desired run name (directory under runs/) (required)
    --config PATH           Path to YAML config file (required)
    --runs-dir DIR          Runs directory (default: runs)
    --python-exe PATH       Python executable or \"python\" (uv-managed) (default: python)
    --detach                Spawn controller as a child process and return immediately
    --no-print-iter-table   Disable iteration summary printing when running in foreground
"#
                );
                return;
            }
            "--run-name" => {
                run_name = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--config" => {
                cfg_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--runs-dir" => {
                runs_dir = args.get(i + 1).cloned().unwrap_or_else(|| "runs".to_string());
                i += 2;
            }
            "--python-exe" => {
                python_exe = args.get(i + 1).cloned().unwrap_or_else(|| "python".to_string());
                i += 2;
            }
            "--detach" => {
                detach = true;
                i += 1;
            }
            "--no-print-iter-table" => {
                print_iter_table = false;
                i += 1;
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown option for `yz start-run`: {other}");
                eprintln!("Run `yz start-run --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run_name = run_name.unwrap_or_else(|| {
        eprintln!("Missing --run-name");
        process::exit(1);
    });
    let cfg_path = cfg_path.unwrap_or_else(|| {
        eprintln!("Missing --config");
        process::exit(1);
    });
    if run_name.trim().is_empty() {
        eprintln!("--run-name must be non-empty");
        process::exit(1);
    }
    if cfg_path.trim().is_empty() {
        eprintln!("--config must be non-empty");
        process::exit(1);
    }

    let cfg = yz_core::Config::load(&cfg_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config at {cfg_path}: {e}");
        process::exit(1);
    });

    let runs_dir = PathBuf::from(runs_dir);
    std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create runs dir {}: {e}", runs_dir.display());
        process::exit(1);
    });

    let (run_id, run_dir) = ensure_unique_run_dir(&runs_dir, &run_name);

    // Create the run dir + standard subdirs (nice UX + matches TUI behavior).
    std::fs::create_dir_all(run_dir.join("logs")).unwrap_or_else(|e| {
        eprintln!("Failed to create logs dir: {e}");
        process::exit(1);
    });
    std::fs::create_dir_all(run_dir.join("models")).unwrap_or_else(|e| {
        eprintln!("Failed to create models dir: {e}");
        process::exit(1);
    });
    std::fs::create_dir_all(run_dir.join("replay")).unwrap_or_else(|e| {
        eprintln!("Failed to create replay dir: {e}");
        process::exit(1);
    });

    // Write normalized run config snapshot (same contract as controller/TUI).
    if let Err(e) = yz_logging::write_config_snapshot_atomic(&run_dir, &cfg) {
        eprintln!("Failed to write config.yaml: {e}");
        process::exit(1);
    }

    // Clear any stale cancel request.
    let _ = std::fs::remove_file(run_dir.join("cancel.request"));

    println!("run_id: {run_id}");
    println!("run_dir: {}", run_dir.display());

    if detach {
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("yz"));
        let mut cmd = Command::new(exe);
        cmd.arg("controller");
        cmd.arg("--run-dir").arg(&run_dir);
        cmd.arg("--python-exe").arg(&python_exe);
        if print_iter_table {
            cmd.arg("--print-iter-table");
        }
        if verbose {
            cmd.arg("--verbose");
        }
        cmd.stdout(process::Stdio::inherit());
        cmd.stderr(process::Stdio::inherit());
        let _child = cmd.spawn().unwrap_or_else(|e| {
            eprintln!("Failed to spawn controller: {e}");
            process::exit(1);
        });
        return;
    }

    let total_iters = cfg.controller.total_iterations.unwrap_or(1).max(1);
    if print_iter_table {
        println!();
        println!("Training for {total_iters} iterations:");
        println!();
    }
    let code = run_controller_foreground(run_dir, cfg, python_exe, print_iter_table, verbose);
    process::exit(code);
}

fn cmd_selfplay_worker(args: &[String]) {
    let mut run_dir: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut worker_id: u32 = 0;
    let mut num_workers: u32 = 1;
    let mut games: u32 = 1;
    let mut seed_base: u64 = 0xC0FFEE;
    let mut max_samples_per_shard: usize = 8192;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz selfplay-worker

USAGE:
    yz selfplay-worker --run-dir runs/<id> --infer unix:///tmp/yatzy_infer.sock --worker-id W --num-workers N --games G [--seed-base S] [--max-samples-per-shard N]

OPTIONS:
    --run-dir DIR               Run directory (contains run.json + config.yaml) (required)
    --infer ENDPOINT            Inference endpoint (unix:///... or tcp://host:port) (required)
    --worker-id W               Worker id in [0, N) (required)
    --num-workers N             Total worker processes (required)
    --games G                   Games to complete in this worker (required)
    --seed-base S               Seed base for deterministic uniqueness (default: 0xC0FFEE)
    --max-samples-per-shard N   Samples per replay shard (default: 8192)
"#
                );
                return;
            }
            "--run-dir" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--worker-id" => {
                worker_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --worker-id value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--num-workers" => {
                num_workers = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --num-workers value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--games" => {
                games = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --games value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--seed-base" => {
                seed_base = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --seed-base value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--max-samples-per-shard" => {
                max_samples_per_shard = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --max-samples-per-shard value");
                        process::exit(1);
                    });
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz selfplay-worker`: {other}");
                eprintln!("Run `yz selfplay-worker --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run_dir = run_dir.unwrap_or_else(|| {
        eprintln!("Missing --run-dir");
        process::exit(1);
    });
    let infer = infer.unwrap_or_else(|| {
        eprintln!("Missing --infer");
        process::exit(1);
    });
    if num_workers < 1 {
        eprintln!("--num-workers must be >= 1");
        process::exit(1);
    }
    if worker_id >= num_workers {
        eprintln!("--worker-id must be in [0, num_workers)");
        process::exit(1);
    }
    if games < 1 {
        eprintln!("--games must be >= 1");
        process::exit(1);
    }

    let run_dir = PathBuf::from(run_dir);
    let cfg_path = run_dir.join("config.yaml");
    let cfg = yz_core::Config::load(&cfg_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config at {}: {e}", cfg_path.display());
        process::exit(1);
    });

    let backend = connect_infer_backend(
        &infer,
        cfg.inference.protocol_version,
        cfg.inference.legal_mask_bitset,
    );

    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: cfg.mcts.dirichlet_epsilon,
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss_mode: match cfg.mcts.virtual_loss_mode.as_str() {
            "q_penalty" => yz_mcts::VirtualLossMode::QPenalty,
            "n_virtual_only" => yz_mcts::VirtualLossMode::NVirtualOnly,
            "off" => yz_mcts::VirtualLossMode::Off,
            _ => yz_mcts::VirtualLossMode::QPenalty,
        },
        virtual_loss: cfg.mcts.virtual_loss.max(0.0),
        expansion_lock: cfg.mcts.katago.expansion_lock,
    };

    // Replay output is worker-local to avoid collisions.
    let replay_dir = run_dir
        .join("replay_workers")
        .join(format!("worker_{worker_id:03}"));
    let manifest_path = run_dir.join("run.json");
    let manifest = yz_logging::read_manifest(&manifest_path).unwrap_or_else(|e| {
        eprintln!("Failed to read manifest at {}: {e}", manifest_path.display());
        process::exit(1);
    });

    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir.clone(),
        max_samples_per_shard,
        git_hash: manifest.git_hash.clone(),
        config_hash: manifest.config_hash.clone(),
    })
    .unwrap_or_else(|e| {
        eprintln!("Failed to create shard writer at {}: {e}", replay_dir.display());
        process::exit(1);
    });

    // Worker stats log (per-process).
    let logs_dir = run_dir
        .join("logs_workers")
        .join(format!("worker_{worker_id:03}"));
    let _ = std::fs::create_dir_all(&logs_dir);
    let mut worker_log =
        yz_logging::NdjsonWriter::open_append_with_flush(logs_dir.join("worker_stats.ndjson"), 50)
            .ok();
    let progress_path = logs_dir.join("progress.json");

    #[derive(serde::Serialize)]
    struct WorkerStatsEvent<'a> {
        event: &'a str,
        ts_ms: u64,
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        games_completed: u32,
        wall_ms: u64,
    }

    let t_start = std::time::Instant::now();
    if let Some(w) = worker_log.as_mut() {
        let _ = w.write_event(&WorkerStatsEvent {
            event: "selfplay_worker_start",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target: games,
            games_completed: 0,
            wall_ms: 0,
        });
        // Ensure the start event is visible even if the worker is cancelled early.
        let _ = w.flush();
    }
    // Write initial progress atomically (for controller/TUI live progress).
    #[derive(serde::Serialize)]
    struct WorkerProgress {
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        games_completed: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        first_game_started_ts_ms: Option<u64>,
        pid: u32,
        sched_ticks: u64,
        sched_steps: u64,
        sched_would_block: u64,
        sched_terminal: u64,
        // Inference client stats (helps diagnose throughput bottlenecks).
        infer_inflight: u64,
        infer_sent: u64,
        infer_received: u64,
        infer_errors: u64,
        infer_latency_p50_us: u64,
        infer_latency_p95_us: u64,
        infer_latency_mean_us: f64,
        ts_ms: u64,
    }
    fn write_progress_atomic(path: &PathBuf, p: &WorkerProgress) {
        let tmp = path.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec(p) {
            let _ = std::fs::write(&tmp, &bytes);
            let _ = std::fs::rename(&tmp, path);
        }
    }
    write_progress_atomic(
        &progress_path,
        &WorkerProgress {
            worker_id,
            num_workers,
            games_target: games,
            games_completed: 0,
            first_game_started_ts_ms: None,
            pid: std::process::id(),
            sched_ticks: 0,
            sched_steps: 0,
            sched_would_block: 0,
            sched_terminal: 0,
            infer_inflight: 0,
            infer_sent: 0,
            infer_received: 0,
            infer_errors: 0,
            infer_latency_p50_us: 0,
            infer_latency_p95_us: 0,
            infer_latency_mean_us: 0.0,
            ts_ms: yz_logging::now_ms(),
        },
    );

    // Each process runs `threads_per_worker` concurrent game tasks.
    let parallel_games = cfg.selfplay.threads_per_worker.max(1) as usize;
    let num_workers_u64 = num_workers as u64;
    let worker_id_u64 = worker_id as u64;

    let mut tasks = Vec::with_capacity(parallel_games);
    for slot in 0..parallel_games {
        let game_id = worker_id_u64 + (slot as u64) * num_workers_u64;
        let mut ctx = yz_core::TurnContext::new_rng(seed_base ^ (0xC0FFEE ^ game_id));
        let s = yz_core::initial_state(&mut ctx);
        tasks.push(yz_runtime::GameTask::new(
            game_id,
            s,
            yz_mcts::ChanceMode::Rng {
                seed: seed_base ^ (0xBADC0DE ^ game_id),
            },
            cfg.mcts.temperature_schedule.clone(),
            mcts_cfg,
        ));
    }
    let mut sched = yz_runtime::Scheduler::new(tasks, 64);

    // -------------------------
    // Self-play/search aggregation (worker-local)
    // -------------------------
    use yz_runtime::scheduler::ExecutedMoveObserver;
    const HIST_BINS: usize = 32;

    struct Agg {
        run_id: String,
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        root_sample_every_n: u32,

        // Counts.
        moves_executed: u64,
        games_completed: u32,

        // Visit policy histograms.
        pi_entropy_hist: yz_logging::HistogramV1,
        pi_max_p_hist: yz_logging::HistogramV1,
        pi_eff_actions_hist: yz_logging::HistogramV1,
        visit_entropy_norm_hist: yz_logging::HistogramV1,
        keep_mass_hist: yz_logging::HistogramV1,
        keep_eff_actions_hist: yz_logging::HistogramV1,
        mark_eff_actions_hist: yz_logging::HistogramV1,

        // Search scalars.
        root_value_sum: f64,
        root_value_sumsq: f64,
        root_value_n: u64,
        fallbacks_sum: u64,
        fallbacks_nz: u64,
        pending_collisions_sum: u64,
        pending_count_max_hist: yz_logging::HistogramV1,

        // Search quality.
        delta_root_value_sum: f64,
        delta_root_value_n: u64,
        leaf_eval_submitted_sum: u64,
        leaf_eval_discarded_sum: u64,

        // Prior vs visit.
        prior_kl_hist: yz_logging::HistogramV1,
        prior_kl_sum: f64,
        prior_kl_n: u64,
        prior_n: u64,
        prior_argmax_overturn_n: u64,

        // Noise effect (raw vs noisy priors).
        noise_kl_hist: yz_logging::HistogramV1,
        noise_kl_sum: f64,
        noise_kl_n: u64,
        noise_n: u64,
        noise_argmax_flip_n: u64,

        // Game stats.
        game_ply_hist: yz_logging::HistogramV1,
        game_turn_hist: yz_logging::HistogramV1,
        score_p0_hist: yz_logging::HistogramV1,
        score_p1_hist: yz_logging::HistogramV1,
        score_diff_hist: yz_logging::HistogramV1,

        // Optional sampled root logs (NDJSON, worker-local).
        root_log: Option<yz_logging::NdjsonWriter>,
        v: yz_logging::VersionInfoV1,
        git_hash: Option<String>,
        config_snapshot: Option<String>,
    }

    fn dist_entropy_max_eff_argmax(xs: &[f32; yz_core::A]) -> (f64, f64, f64, u8) {
        let mut ent = 0.0f64;
        let mut max_p = -1.0f64;
        let mut argmax = 0usize;
        let mut sumsq = 0.0f64;
        for (i, &x) in xs.iter().enumerate() {
            let p = x as f64;
            if !(p.is_finite()) || p < 0.0 {
                continue;
            }
            if p > max_p {
                max_p = p;
                argmax = i;
            }
            if p > 0.0 {
                ent -= p * p.ln();
            }
            sumsq += p * p;
        }
        let eff = if sumsq > 0.0 { 1.0 / sumsq } else { 0.0 };
        (ent, max_p.max(0.0), eff, argmax as u8)
    }

    fn kl_div(p: &[f32; yz_core::A], q: &[f32; yz_core::A]) -> Option<f64> {
        let mut sp = 0.0f64;
        let mut sq = 0.0f64;
        for i in 0..yz_core::A {
            let pi = p[i] as f64;
            let qi = q[i] as f64;
            if pi.is_finite() && pi > 0.0 {
                sp += pi;
            }
            if qi.is_finite() && qi > 0.0 {
                sq += qi;
            }
        }
        if sp <= 0.0 || sq <= 0.0 {
            return None;
        }
        let mut kl = 0.0f64;
        for i in 0..yz_core::A {
            let pi0 = p[i] as f64;
            if !(pi0.is_finite()) || pi0 <= 0.0 {
                continue;
            }
            let pi = (pi0 / sp).clamp(0.0, 1.0);
            let qi0 = (q[i] as f64) / sq;
            let qi = qi0.max(1e-12);
            kl += pi * (pi / qi).ln();
        }
        Some(kl.max(0.0))
    }

    impl Agg {
        fn new(
            run_id: String,
            worker_id: u32,
            num_workers: u32,
            games_target: u32,
            root_sample_every_n: u32,
            logs_dir: &PathBuf,
            v: yz_logging::VersionInfoV1,
            git_hash: Option<String>,
            config_snapshot: Option<String>,
        ) -> Self {
            let root_log = if root_sample_every_n > 0 {
                yz_logging::NdjsonWriter::open_append_with_flush(
                    logs_dir.join("mcts_root_sample.ndjson"),
                    50,
                )
                .ok()
            } else {
                None
            };
            Self {
                run_id,
                worker_id,
                num_workers,
                games_target,
                root_sample_every_n,
                moves_executed: 0,
                games_completed: 0,
                pi_entropy_hist: yz_logging::HistogramV1::new(0.0, 4.0, HIST_BINS),
                pi_max_p_hist: yz_logging::HistogramV1::new(0.0, 1.0, HIST_BINS),
                pi_eff_actions_hist: yz_logging::HistogramV1::new(0.0, yz_core::A as f64, HIST_BINS),
                visit_entropy_norm_hist: yz_logging::HistogramV1::new(0.0, 1.0, HIST_BINS),
                keep_mass_hist: yz_logging::HistogramV1::new(0.0, 1.0, HIST_BINS),
                keep_eff_actions_hist: yz_logging::HistogramV1::new(0.0, 32.0, HIST_BINS),
                mark_eff_actions_hist: yz_logging::HistogramV1::new(0.0, 32.0, HIST_BINS),
                root_value_sum: 0.0,
                root_value_sumsq: 0.0,
                root_value_n: 0,
                fallbacks_sum: 0,
                fallbacks_nz: 0,
                pending_collisions_sum: 0,
                pending_count_max_hist: yz_logging::HistogramV1::new(0.0, 32.0, HIST_BINS),
                delta_root_value_sum: 0.0,
                delta_root_value_n: 0,
                leaf_eval_submitted_sum: 0,
                leaf_eval_discarded_sum: 0,
                prior_kl_hist: yz_logging::HistogramV1::new(0.0, 3.0, HIST_BINS),
                prior_kl_sum: 0.0,
                prior_kl_n: 0,
                prior_n: 0,
                prior_argmax_overturn_n: 0,
                noise_kl_hist: yz_logging::HistogramV1::new(0.0, 3.0, HIST_BINS),
                noise_kl_sum: 0.0,
                noise_kl_n: 0,
                noise_n: 0,
                noise_argmax_flip_n: 0,
                game_ply_hist: yz_logging::HistogramV1::new(0.0, 256.0, HIST_BINS),
                game_turn_hist: yz_logging::HistogramV1::new(0.0, 64.0, HIST_BINS),
                score_p0_hist: yz_logging::HistogramV1::new(0.0, 400.0, HIST_BINS),
                score_p1_hist: yz_logging::HistogramV1::new(0.0, 400.0, HIST_BINS),
                score_diff_hist: yz_logging::HistogramV1::new(-400.0, 400.0, HIST_BINS),
                root_log,
                v,
                git_hash,
                config_snapshot,
            }
        }

        fn on_game_terminal(&mut self, ply: u32, turn_idx: u32, s0: i16, s1: i16) {
            self.games_completed = self.games_completed.saturating_add(1);
            self.game_ply_hist.add(ply as f64);
            self.game_turn_hist.add(turn_idx as f64);
            self.score_p0_hist.add((s0 as f64).max(0.0));
            self.score_p1_hist.add((s1 as f64).max(0.0));
            self.score_diff_hist.add((s0 as f64) - (s1 as f64));
        }

        fn finalize(self, wall_ms: u64) -> yz_logging::SelfplayWorkerSummaryV1 {
            yz_logging::SelfplayWorkerSummaryV1 {
                event: "selfplay_worker_summary".to_string(),
                ts_ms: yz_logging::now_ms(),
                run_id: self.run_id,
                worker_id: self.worker_id,
                num_workers: self.num_workers,
                games_target: self.games_target,
                games_completed: self.games_completed,
                moves_executed: self.moves_executed,
                wall_ms,
                root_sample_every_n: self.root_sample_every_n,
                pi_entropy_hist: self.pi_entropy_hist,
                pi_max_p_hist: self.pi_max_p_hist,
                pi_eff_actions_hist: self.pi_eff_actions_hist,
                visit_entropy_norm_hist: Some(self.visit_entropy_norm_hist),
                keep_mass_hist: Some(self.keep_mass_hist),
                keep_eff_actions_hist: Some(self.keep_eff_actions_hist),
                mark_eff_actions_hist: Some(self.mark_eff_actions_hist),
                root_value_sum: self.root_value_sum,
                root_value_sumsq: self.root_value_sumsq,
                root_value_n: self.root_value_n,
                fallbacks_sum: self.fallbacks_sum,
                fallbacks_nz: self.fallbacks_nz,
                pending_collisions_sum: self.pending_collisions_sum,
                pending_count_max_hist: self.pending_count_max_hist,
                delta_root_value_sum: self.delta_root_value_sum,
                delta_root_value_n: self.delta_root_value_n,
                leaf_eval_submitted_sum: self.leaf_eval_submitted_sum,
                leaf_eval_discarded_sum: self.leaf_eval_discarded_sum,
                prior_kl_hist: self.prior_kl_hist,
                prior_kl_sum: self.prior_kl_sum,
                prior_kl_n: self.prior_kl_n,
                prior_n: self.prior_n,
                prior_argmax_overturn_n: self.prior_argmax_overturn_n,
                noise_kl_hist: self.noise_kl_hist,
                noise_kl_sum: self.noise_kl_sum,
                noise_kl_n: self.noise_kl_n,
                noise_n: self.noise_n,
                noise_argmax_flip_n: self.noise_argmax_flip_n,
                game_ply_hist: self.game_ply_hist,
                game_turn_hist: self.game_turn_hist,
                score_p0_hist: self.score_p0_hist,
                score_p1_hist: self.score_p1_hist,
                score_diff_hist: self.score_diff_hist,
            }
        }
    }

    impl ExecutedMoveObserver for Agg {
        fn on_executed_move(&mut self, global_ply: u64, exec: &yz_runtime::game_task::ExecutedMove) {
            self.moves_executed = self.moves_executed.saturating_add(1);

            // Visit-policy shape.
            let (ent, max_p, eff, _argmax) = dist_entropy_max_eff_argmax(&exec.search.pi);
            self.pi_entropy_hist.add(ent);
            self.pi_max_p_hist.add(max_p);
            self.pi_eff_actions_hist.add(eff);

            // Keep vs mark distribution within reroll phase.
            if exec.rerolls_left > 0 {
                let mut keep_mass = 0.0f64;
                let mut mark_mass = 0.0f64;
                for i in 0..yz_core::A {
                    let p = exec.search.pi[i] as f64;
                    if !(p.is_finite()) || p <= 0.0 {
                        continue;
                    }
                    if i < 32 {
                        keep_mass += p;
                    } else {
                        mark_mass += p;
                    }
                }
                let tot = (keep_mass + mark_mass).max(1e-12);
                keep_mass = (keep_mass / tot).clamp(0.0, 1.0);
                self.keep_mass_hist.add(keep_mass);

                // EffA on renormalized subsets.
                let eff_subset = |lo: usize, hi: usize, mass: f64| -> f64 {
                    if mass <= 0.0 {
                        return 0.0;
                    }
                    let mut sumsq = 0.0f64;
                    for i in lo..hi {
                        let p = (exec.search.pi[i] as f64) / mass;
                        if p.is_finite() && p > 0.0 {
                            sumsq += p * p;
                        }
                    }
                    if sumsq > 0.0 { 1.0 / sumsq } else { 0.0 }
                };
                let eff_keep = eff_subset(0, 32, keep_mass);
                let eff_mark = eff_subset(32, yz_core::A, 1.0 - keep_mass);
                self.keep_eff_actions_hist.add(eff_keep);
                self.mark_eff_actions_hist.add(eff_mark);
            }

            // Normalized root visit entropy: H(pi_visit) / log(n_legal) (clamped to [0,1]).
            let n_legal = exec
                .search
                .root_priors_noisy
                .as_ref()
                .or(exec.search.root_priors_raw.as_ref())
                .map(|p| p.iter().filter(|&&x| x.is_finite() && x > 0.0).count())
                .unwrap_or_else(|| exec.search.pi.iter().filter(|&&x| x.is_finite() && x > 0.0).count())
                .max(1);
            let denom = (n_legal as f64).ln().max(1e-12);
            let ent_norm = (ent / denom).clamp(0.0, 1.0);
            self.visit_entropy_norm_hist.add(ent_norm);

            // Root value and search stability.
            let rv = exec.search.root_value as f64;
            self.root_value_sum += rv;
            self.root_value_sumsq += rv * rv;
            self.root_value_n = self.root_value_n.saturating_add(1);

            let fb = exec.search.fallbacks as u64;
            self.fallbacks_sum = self.fallbacks_sum.saturating_add(fb);
            if fb > 0 {
                self.fallbacks_nz = self.fallbacks_nz.saturating_add(1);
            }
            self.pending_collisions_sum =
                self.pending_collisions_sum.saturating_add(exec.search.pending_collisions as u64);
            self.pending_count_max_hist.add(exec.search.pending_count_max as f64);

            // Search quality proxies.
            let d = exec.search.delta_root_value as f64;
            if d.is_finite() {
                self.delta_root_value_sum += d;
                self.delta_root_value_n = self.delta_root_value_n.saturating_add(1);
            }
            self.leaf_eval_submitted_sum = self
                .leaf_eval_submitted_sum
                .saturating_add(exec.search.leaf_eval_submitted as u64);
            self.leaf_eval_discarded_sum = self
                .leaf_eval_discarded_sum
                .saturating_add(exec.search.leaf_eval_discarded as u64);

            // Search improvement: compare noisy priors to visit pi (when priors present).
            if let Some(prior_noisy) = &exec.search.root_priors_noisy {
                self.prior_n = self.prior_n.saturating_add(1);
                if let Some(kl) = kl_div(prior_noisy, &exec.search.pi) {
                    self.prior_kl_hist.add(kl);
                    self.prior_kl_sum += kl;
                    self.prior_kl_n = self.prior_kl_n.saturating_add(1);
                }
                let (_, _, _, a_prior) = dist_entropy_max_eff_argmax(prior_noisy);
                let (_, _, _, a_visit) = dist_entropy_max_eff_argmax(&exec.search.pi);
                if a_prior != a_visit {
                    self.prior_argmax_overturn_n = self.prior_argmax_overturn_n.saturating_add(1);
                }
            }

            // Noise impact: raw vs noisy priors.
            if let (Some(raw), Some(noisy)) = (&exec.search.root_priors_raw, &exec.search.root_priors_noisy) {
                self.noise_n = self.noise_n.saturating_add(1);
                if let Some(kl) = kl_div(raw, noisy) {
                    self.noise_kl_hist.add(kl);
                    self.noise_kl_sum += kl;
                    self.noise_kl_n = self.noise_kl_n.saturating_add(1);
                }
                let (_, _, _, a_raw) = dist_entropy_max_eff_argmax(raw);
                let (_, _, _, a_noisy) = dist_entropy_max_eff_argmax(noisy);
                if a_raw != a_noisy {
                    self.noise_argmax_flip_n = self.noise_argmax_flip_n.saturating_add(1);
                }
            }

            // Optional sampled root logs (worker-local NDJSON).
            if self.root_sample_every_n > 0
                && global_ply % (self.root_sample_every_n as u64) == 0
            {
                if let Some(w) = self.root_log.as_mut() {
                    let (ent_s, maxp_s, _eff_s, argmax_s) = dist_entropy_max_eff_argmax(&exec.search.pi);
                    let ev = yz_logging::MetricsMctsRootSampleV1 {
                        event: "mcts_root_sample",
                        ts_ms: yz_logging::now_ms(),
                        v: self.v.clone(),
                        run_id: self.run_id.clone(),
                        git_hash: self.git_hash.clone(),
                        config_snapshot: self.config_snapshot.clone(),
                        global_ply,
                        game_id: exec.game_id,
                        game_ply: exec.game_ply,
                        player_to_move: exec.player_to_move,
                        rerolls_left: exec.rerolls_left,
                        dice: exec.dice,
                        chosen_action: exec.chosen_action,
                        root_value: exec.search.root_value,
                        fallbacks: exec.search.fallbacks,
                        pending_count_max: exec.search.pending_count_max as u64,
                        pending_collisions: exec.search.pending_collisions,
                        pi: yz_logging::PiSummaryV1 {
                            entropy: ent_s as f32,
                            max_p: maxp_s as f32,
                            argmax_a: argmax_s,
                        },
                    };
                    let _ = w.write_event(&ev);
                }
            }
        }
    }

    let root_sample_every_n = cfg.selfplay.root_sample_every_n;
    let v = yz_logging::VersionInfoV1 {
        protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
        feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
        action_space_id: "oracle_keepmask_v2",
        ruleset_id: "swedish_scandinavian_mark_at_r3_v1",
    };
    let mut agg = Agg::new(
        manifest.run_id.clone(),
        worker_id,
        num_workers,
        games,
        root_sample_every_n,
        &logs_dir,
        v,
        manifest.git_hash.clone(),
        manifest.config_snapshot.clone(),
    );

    let mut completed_games: u32 = 0;
    let mut next_seq: u64 = parallel_games as u64;
    let mut last_progress_write = std::time::Instant::now();
    let mut first_game_started_ts_ms: Option<u64> = None;
    while completed_games < games {
        if first_game_started_ts_ms.is_none() {
            first_game_started_ts_ms = Some(yz_logging::now_ms());
            if let Some(w) = worker_log.as_mut() {
                let _ = w.write_event(&WorkerStatsEvent {
                    event: "selfplay_worker_first_game_start",
                    ts_ms: first_game_started_ts_ms.unwrap(),
                    worker_id,
                    num_workers,
                    games_target: games,
                    games_completed: completed_games,
                    wall_ms: t_start.elapsed().as_millis() as u64,
                });
                let _ = w.flush();
            }
        }
        let before_steps = sched.stats().steps;
        let before_terminal = sched.stats().terminal;
        sched.tick_and_write_observe(&backend, &mut writer, None, Some(&mut agg))
            .unwrap_or_else(|e| {
                eprintln!("worker {worker_id}: scheduler failed: {e}");
                process::exit(1);
            });
        let after_steps = sched.stats().steps;
        let after_terminal = sched.stats().terminal;
        // Avoid busy-spinning when all tasks are waiting on inference.
        // This yields CPU to the Python inference server and improves throughput.
        if after_steps == before_steps && after_terminal == before_terminal {
            backend.wait_for_progress(std::time::Duration::from_micros(200));
        }

        // Heartbeat progress even before games complete, so we can prove workers are running.
        // Rate-limit to avoid excessive filesystem churn.
        if last_progress_write.elapsed() >= std::time::Duration::from_millis(500) {
            let s = sched.stats();
            let inf = backend.stats_snapshot();
            write_progress_atomic(
                &progress_path,
                &WorkerProgress {
                    worker_id,
                    num_workers,
                    games_target: games,
                    games_completed: completed_games,
                    first_game_started_ts_ms,
                    pid: std::process::id(),
                    sched_ticks: s.ticks,
                    sched_steps: s.steps,
                    sched_would_block: s.would_block,
                    sched_terminal: s.terminal,
                    infer_inflight: inf.inflight as u64,
                    infer_sent: inf.sent,
                    infer_received: inf.received,
                    infer_errors: inf.errors,
                    infer_latency_p50_us: inf.latency_us.summary.p50_us,
                    infer_latency_p95_us: inf.latency_us.summary.p95_us,
                    infer_latency_mean_us: inf.latency_us.summary.mean_us,
                    ts_ms: yz_logging::now_ms(),
                },
            );
            last_progress_write = std::time::Instant::now();
        }

        for t in sched.tasks_mut() {
            if yz_core::is_terminal(&t.state) {
                // Record game summary before resetting.
                agg.on_game_terminal(
                    t.ply,
                    t.turn_idx,
                    t.state.players[0].total_score,
                    t.state.players[1].total_score,
                );
                completed_games += 1;
                // Progress update: cheap atomic write. Rate-limit a bit.
                if completed_games == games || (completed_games % 5 == 0) {
                    write_progress_atomic(
                        &progress_path,
                        &WorkerProgress {
                            worker_id,
                            num_workers,
                            games_target: games,
                            games_completed: completed_games,
                            first_game_started_ts_ms,
                            pid: std::process::id(),
                            // Note: we can't borrow `sched` here because we already hold a mutable
                            // borrow from `tasks_mut()`. These fields will be updated by the
                            // periodic heartbeat above.
                            sched_ticks: 0,
                            sched_steps: 0,
                            sched_would_block: 0,
                            sched_terminal: 0,
                            infer_inflight: 0,
                            infer_sent: 0,
                            infer_received: 0,
                            infer_errors: 0,
                            infer_latency_p50_us: 0,
                            infer_latency_p95_us: 0,
                            infer_latency_mean_us: 0.0,
                            ts_ms: yz_logging::now_ms(),
                        },
                    );
                }
                if completed_games >= games {
                    break;
                }
                // Reset task for next game; ensure unique game_id across processes.
                let game_id = worker_id_u64 + next_seq * num_workers_u64;
                next_seq += 1;
                let mut ctx = yz_core::TurnContext::new_rng(seed_base ^ (0xC0FFEE ^ game_id));
                let s = yz_core::initial_state(&mut ctx);
                *t = yz_runtime::GameTask::new(
                    game_id,
                    s,
                    yz_mcts::ChanceMode::Rng {
                        seed: seed_base ^ (0xBADC0DE ^ game_id),
                    },
                    cfg.mcts.temperature_schedule.clone(),
                    mcts_cfg,
                );
            }
        }
    }

    writer.finish().unwrap_or_else(|e| {
        eprintln!("worker {worker_id}: failed to finish replay writer: {e}");
        process::exit(1);
    });
    // Final progress.
    let s = sched.stats();
    let inf = backend.stats_snapshot();
    write_progress_atomic(
        &progress_path,
        &WorkerProgress {
            worker_id,
            num_workers,
            games_target: games,
            games_completed: completed_games,
            first_game_started_ts_ms,
            pid: std::process::id(),
            sched_ticks: s.ticks,
            sched_steps: s.steps,
            sched_would_block: s.would_block,
            sched_terminal: s.terminal,
            infer_inflight: inf.inflight as u64,
            infer_sent: inf.sent,
            infer_received: inf.received,
            infer_errors: inf.errors,
            infer_latency_p50_us: inf.latency_us.summary.p50_us,
            infer_latency_p95_us: inf.latency_us.summary.p95_us,
            infer_latency_mean_us: inf.latency_us.summary.mean_us,
            ts_ms: yz_logging::now_ms(),
        },
    );

    if let Some(w) = worker_log.as_mut() {
        let _ = w.write_event(&WorkerStatsEvent {
            event: "selfplay_worker_done",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target: games,
            games_completed: completed_games,
            wall_ms: t_start.elapsed().as_millis() as u64,
        });
        let _ = w.flush();
    }

    // Finalize and write worker-local selfplay summary (best-effort).
    {
        let wall_ms = t_start.elapsed().as_millis() as u64;
        let summary = agg.finalize(wall_ms);
        let path = logs_dir.join("selfplay_worker_summary.json");
        let tmp = path.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec(&summary) {
            let _ = std::fs::write(&tmp, &bytes);
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

fn print_version() {
    println!("yz {}", env!("CARGO_PKG_VERSION"));
}

fn cmd_selfplay(args: &[String]) {
    let mut config_path: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut out: Option<String> = None;
    let mut games: u32 = 10;
    let mut max_samples_per_shard: usize = 8192;
    let mut root_log_every: u64 = 50;
    let mut log_flush_every: u64 = 100;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz selfplay

USAGE:
    yz selfplay --config cfg.yaml --infer unix:///tmp/yatzy_infer.sock --out runs/<id>/ [--games N] [--max-samples-per-shard N]

OPTIONS:
    --config PATH               Path to YAML config (required)
    --infer ENDPOINT            Inference endpoint (unix:///... or tcp://host:port) (required)
    --out DIR                   Output directory (required)
    --games N                   Number of games to play (default: 10)
    --max-samples-per-shard N   Samples per replay shard (default: 8192)
    --root-log-every N          Log one MCTS root every N executed moves (default: 50)
    --log-flush-every N         Flush NDJSON logs every N lines (0 disables) (default: 100)
"#
                );
                return;
            }
            "--config" => {
                config_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--out" => {
                out = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--games" => {
                games = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --games value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--max-samples-per-shard" => {
                max_samples_per_shard = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --max-samples-per-shard value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--root-log-every" => {
                root_log_every =
                    args.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Invalid --root-log-every value");
                            process::exit(1);
                        });
                i += 2;
            }
            "--log-flush-every" => {
                log_flush_every =
                    args.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Invalid --log-flush-every value");
                            process::exit(1);
                        });
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz selfplay`: {}", other);
                eprintln!("Run `yz selfplay --help` for usage.");
                process::exit(1);
            }
        }
    }

    let config_path = config_path.unwrap_or_else(|| {
        eprintln!("Missing --config");
        process::exit(1);
    });
    let infer = infer.unwrap_or_else(|| {
        eprintln!("Missing --infer");
        process::exit(1);
    });
    let out = out.unwrap_or_else(|| {
        eprintln!("Missing --out");
        process::exit(1);
    });

    let cfg = yz_core::Config::load(&config_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config: {e}");
        process::exit(1);
    });

    let replay_dir = PathBuf::from(&out).join("replay");
    let _ = yz_replay::cleanup_tmp_files(&replay_dir);

    let logs_dir = PathBuf::from(&out).join("logs");
    std::fs::create_dir_all(&logs_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create logs dir: {e}");
        process::exit(1);
    });

    let models_dir = PathBuf::from(&out).join("models");
    std::fs::create_dir_all(&models_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create models dir: {e}");
        process::exit(1);
    });

    // E8.5.1: run manifest (runs/<id>/run.json).
    let run_json = PathBuf::from(&out).join("run.json");
    let config_bytes = std::fs::read(&config_path).unwrap_or_else(|e| {
        eprintln!("Failed to read config file: {e}");
        process::exit(1);
    });
    let config_hash = yz_logging::hash_config_bytes(&config_bytes);
    let run_id = PathBuf::from(&out)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&out)
        .to_string();
    let mut manifest = yz_logging::RunManifestV1 {
        run_manifest_version: yz_logging::RUN_MANIFEST_VERSION,
        run_id,
        created_ts_ms: yz_logging::now_ms(),
        protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
        feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
        action_space_id: "oracle_keepmask_v2".to_string(),
        ruleset_id: "swedish_scandinavian_mark_at_r3_v1".to_string(),
        git_hash: yz_logging::try_git_hash(),
        config_hash: Some(config_hash),
        config_snapshot: None,
        config_snapshot_hash: None,
        replay_dir: "replay".to_string(),
        logs_dir: "logs".to_string(),
        models_dir: "models".to_string(),
        selfplay_games_completed: 0,
        train_step: 0,
        best_checkpoint: None,
        candidate_checkpoint: None,
        best_promoted_iter: None,
        train_last_loss_total: None,
        train_last_loss_policy: None,
        train_last_loss_value: None,
        promotion_decision: None,
        promotion_ts_ms: None,
        gate_games: None,
        gate_win_rate: None,
        gate_draw_rate: None,
        gate_wins: None,
        gate_losses: None,
        gate_draws: None,
        gate_ci95_low: None,
        gate_ci95_high: None,
        gate_sprt: None,
        gate_seeds_hash: None,
        gate_oracle_match_rate_overall: None,
        gate_oracle_match_rate_mark: None,
        gate_oracle_match_rate_reroll: None,
        controller_phase: None,
        controller_status: None,
        controller_last_ts_ms: None,
        controller_error: None,
        model_reloads: 0,
        controller_iteration_idx: 0,
        iterations: Vec::new(),
    };
    // If a manifest already exists (resume), keep its created_ts_ms/run_id.
    if let Ok(existing) = yz_logging::read_manifest(&run_json) {
        manifest.created_ts_ms = existing.created_ts_ms;
        manifest.run_id = existing.run_id;
        manifest.train_step = existing.train_step;
        manifest.best_checkpoint = existing.best_checkpoint;
        manifest.candidate_checkpoint = existing.candidate_checkpoint;
        manifest.train_last_loss_total = existing.train_last_loss_total;
        manifest.train_last_loss_policy = existing.train_last_loss_policy;
        manifest.train_last_loss_value = existing.train_last_loss_value;
        manifest.promotion_decision = existing.promotion_decision;
        manifest.promotion_ts_ms = existing.promotion_ts_ms;
        manifest.gate_games = existing.gate_games;
        manifest.gate_win_rate = existing.gate_win_rate;
        manifest.gate_draw_rate = existing.gate_draw_rate;
        manifest.gate_seeds_hash = existing.gate_seeds_hash;
        manifest.gate_oracle_match_rate_overall = existing.gate_oracle_match_rate_overall;
        manifest.gate_oracle_match_rate_mark = existing.gate_oracle_match_rate_mark;
        manifest.gate_oracle_match_rate_reroll = existing.gate_oracle_match_rate_reroll;
        manifest.config_snapshot = existing.config_snapshot;
        manifest.config_snapshot_hash = existing.config_snapshot_hash;
        manifest.controller_phase = existing.controller_phase;
        manifest.controller_status = existing.controller_status;
        manifest.controller_last_ts_ms = existing.controller_last_ts_ms;
        manifest.controller_error = existing.controller_error;
        manifest.controller_iteration_idx = existing.controller_iteration_idx;
        manifest.iterations = existing.iterations;
    }

    // E10.5S1: run-local config snapshot (normalized).
    if manifest.config_snapshot.is_none() || !PathBuf::from(&out).join("config.yaml").exists() {
        if let Ok((rel, h)) = yz_logging::write_config_snapshot_atomic(&out, &cfg) {
            manifest.config_snapshot = Some(rel);
            manifest.config_snapshot_hash = Some(h);
        }
    }
    yz_logging::write_manifest_atomic(&run_json, &manifest).unwrap_or_else(|e| {
        eprintln!("Failed to write run manifest: {e:?}");
        process::exit(1);
    });

    let backend = connect_infer_backend(
        &infer,
        cfg.inference.protocol_version,
        cfg.inference.legal_mask_bitset,
    );
    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir.clone(),
        max_samples_per_shard,
        git_hash: None,
        config_hash: None,
    })
    .unwrap_or_else(|e| {
        eprintln!("Failed to create shard writer: {e}");
        process::exit(1);
    });

    let parallel = cfg.selfplay.threads_per_worker.max(1) as usize;
    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: cfg.mcts.dirichlet_epsilon,
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss_mode: match cfg.mcts.virtual_loss_mode.as_str() {
            "q_penalty" => yz_mcts::VirtualLossMode::QPenalty,
            "n_virtual_only" => yz_mcts::VirtualLossMode::NVirtualOnly,
            "off" => yz_mcts::VirtualLossMode::Off,
            _ => yz_mcts::VirtualLossMode::QPenalty,
        },
        virtual_loss: cfg.mcts.virtual_loss.max(0.0),
        expansion_lock: cfg.mcts.katago.expansion_lock,
    };

    let mut tasks = Vec::new();
    for gid in 0..parallel as u64 {
        let mut ctx = yz_core::TurnContext::new_rng(0xC0FFEE ^ gid);
        let s = yz_core::initial_state(&mut ctx);
        tasks.push(yz_runtime::GameTask::new(
            gid,
            s,
            yz_mcts::ChanceMode::Rng {
                seed: 0xBADC0DE ^ gid,
            },
            cfg.mcts.temperature_schedule.clone(),
            mcts_cfg,
        ));
    }
    let mut sched = yz_runtime::Scheduler::new(tasks, 64);

    let run_id = out.clone();
    let v = yz_logging::VersionInfoV1 {
        protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
        feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
        action_space_id: "oracle_keepmask_v2",
        ruleset_id: "swedish_scandinavian_mark_at_r3_v1",
    };
    let iter_log_path = logs_dir.join("iteration_stats.ndjson");
    let roots_log_path = logs_dir.join("mcts_roots.ndjson");
    let mut loggers = yz_runtime::RunLoggers {
        run_id,
        v,
        git_hash: manifest.git_hash.clone(),
        config_snapshot: manifest.config_snapshot.clone(),
        root_log_every_n: root_log_every,
        iter: yz_logging::NdjsonWriter::open_append_with_flush(iter_log_path, log_flush_every)
            .unwrap_or_else(|e| {
                eprintln!("Failed to create iteration log: {e:?}");
                process::exit(1);
            }),
        roots: yz_logging::NdjsonWriter::open_append_with_flush(roots_log_path, log_flush_every)
            .unwrap_or_else(|e| {
                eprintln!("Failed to create root log: {e:?}");
                process::exit(1);
            }),
        metrics: yz_logging::NdjsonWriter::open_append_with_flush(
            logs_dir.join("metrics.ndjson"),
            log_flush_every,
        )
        .unwrap_or_else(|e| {
            eprintln!("Failed to create metrics log: {e:?}");
            process::exit(1);
        }),
    };

    let mut completed_games: u32 = 0;
    let mut next_game_id: u64 = parallel as u64;
    while completed_games < games {
        sched
            .tick_and_write(&backend, &mut writer, Some(&mut loggers))
            .unwrap_or_else(|e| {
                eprintln!("Replay write error: {e}");
                process::exit(1);
            });

        for t in sched.tasks_mut() {
            if yz_core::is_terminal(&t.state) {
                completed_games += 1;
                if completed_games.is_multiple_of(10) || completed_games == games {
                    manifest.selfplay_games_completed = completed_games as u64;
                    let _ = yz_logging::write_manifest_atomic(&run_json, &manifest);
                }
                if completed_games >= games {
                    break;
                }
                // Reset task for next game.
                let mut ctx = yz_core::TurnContext::new_rng(0xC0FFEE ^ next_game_id);
                let s = yz_core::initial_state(&mut ctx);
                *t = yz_runtime::GameTask::new(
                    next_game_id,
                    s,
                    yz_mcts::ChanceMode::Rng {
                        seed: 0xBADC0DE ^ next_game_id,
                    },
                    cfg.mcts.temperature_schedule.clone(),
                    mcts_cfg,
                );
                next_game_id += 1;
            }
        }
    }

    writer.finish().unwrap_or_else(|e| {
        eprintln!("Failed to flush writer: {e}");
        process::exit(1);
    });
    let _ = loggers.iter.flush();
    let _ = loggers.roots.flush();

    // E13.1S1: replay pruning (optional) + metrics event.
    if let Some(cap) = cfg.replay.capacity_shards {
        if cap > 0 {
            match yz_replay::prune_shards_by_idx(&replay_dir, cap as usize) {
                Ok(rep) => {
                    let ev = yz_logging::MetricsReplayPruneV1 {
                        event: "replay_prune",
                        ts_ms: yz_logging::now_ms(),
                        v: loggers.v.clone(),
                        run_id: loggers.run_id.clone(),
                        git_hash: loggers.git_hash.clone(),
                        config_snapshot: loggers.config_snapshot.clone(),
                        capacity_shards: cap,
                        before_shards: rep.before_shards as u32,
                        after_shards: rep.after_shards as u32,
                        deleted_shards: rep.deleted_shards as u32,
                        deleted_min_idx: rep.deleted_min_idx,
                        deleted_max_idx: rep.deleted_max_idx,
                    };
                    let _ = loggers.metrics.write_event(&ev);
                }
                Err(e) => eprintln!("Replay prune error: {e}"),
            }
        }
    }
    let _ = loggers.metrics.flush();

    // Final manifest update.
    manifest.selfplay_games_completed = completed_games as u64;
    let _ = yz_logging::write_manifest_atomic(&run_json, &manifest);

    println!("Self-play complete. Games={games} out={out}");
}

fn cmd_gate_worker(args: &[String]) {
    use std::path::PathBuf;

    let mut run_dir: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut worker_id: u32 = 0;
    let mut num_workers: u32 = 1;
    let mut best_id: u32 = 0;
    let mut cand_id: u32 = 1;
    let mut schedule_file: Option<String> = None;
    let mut out_path: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz gate-worker

USAGE:
    yz gate-worker --run-dir runs/<id> --infer unix:///tmp/yatzy_infer.sock --worker-id W --num-workers N --best-id 0 --cand-id 1 --schedule-file PATH --out PATH

NOTES:
    - Internal command spawned by the controller for parallel gating.
    - Does NOT write replay shards; only writes results/progress.

OPTIONS:
    --run-dir DIR         Run directory (contains run.json + config.yaml) (required)
    --infer ENDPOINT      Inference endpoint (unix:///... or tcp://host:port) (required)
    --worker-id W         Worker id in [0, N) (required)
    --num-workers N       Total worker processes (required)
    --best-id N           model_id for best (default 0)
    --cand-id N           model_id for candidate (default 1)
    --schedule-file PATH  JSON schedule file for this worker (required)
    --out PATH            Output JSON result file path (required)
"#
                );
                return;
            }
            "--run-dir" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--worker-id" => {
                worker_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --worker-id value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--num-workers" => {
                num_workers = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --num-workers value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--best-id" => {
                best_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(best_id);
                i += 2;
            }
            "--cand-id" => {
                cand_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(cand_id);
                i += 2;
            }
            "--schedule-file" => {
                schedule_file = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--out" => {
                out_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz gate-worker`: {other}");
                eprintln!("Run `yz gate-worker --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run_dir = run_dir.unwrap_or_else(|| {
        eprintln!("Missing --run-dir");
        process::exit(1);
    });
    let infer = infer.unwrap_or_else(|| {
        eprintln!("Missing --infer");
        process::exit(1);
    });
    let schedule_file = schedule_file.unwrap_or_else(|| {
        eprintln!("Missing --schedule-file");
        process::exit(1);
    });
    let out_path = out_path.unwrap_or_else(|| {
        eprintln!("Missing --out");
        process::exit(1);
    });

    let run_dir = PathBuf::from(run_dir);
    let cfg_path = run_dir.join("config.yaml");
    let mut cfg = yz_core::Config::load(&cfg_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config.yaml: {e}");
        process::exit(1);
    });

    #[derive(serde::Deserialize)]
    struct GameSpecJson {
        episode_seed: u64,
        swap: bool,
    }
    let sched_bytes = std::fs::read(&schedule_file).unwrap_or_else(|e| {
        eprintln!("Failed to read schedule file: {e}");
        process::exit(1);
    });
    let sched: Vec<GameSpecJson> = serde_json::from_slice(&sched_bytes).unwrap_or_else(|e| {
        eprintln!("Failed to parse schedule JSON: {e}");
        process::exit(1);
    });
    let schedule: Vec<yz_eval::GameSpec> = sched
        .into_iter()
        .map(|s| yz_eval::GameSpec {
            episode_seed: s.episode_seed,
            swap: s.swap,
        })
        .collect();

    let games_target = schedule.len() as u32;
    let parallel_games = yz_eval::effective_gate_parallel_games(&cfg);

    // Bounded inflight to avoid flooding the inference server.
    let client_opts = yz_infer::ClientOptions {
        max_inflight_total: 64,
        max_outbound_queue: 256,
        request_id_start: 1,
        protocol_version: cfg.inference.protocol_version,
        legal_mask_bitset: cfg.inference.legal_mask_bitset && cfg.inference.protocol_version == 2,
    };
    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: 0.0, // gating: no root noise
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss_mode: match cfg.mcts.virtual_loss_mode.as_str() {
            "q_penalty" => yz_mcts::VirtualLossMode::QPenalty,
            "n_virtual_only" => yz_mcts::VirtualLossMode::NVirtualOnly,
            "off" => yz_mcts::VirtualLossMode::Off,
            _ => yz_mcts::VirtualLossMode::QPenalty,
        },
        virtual_loss: cfg.mcts.virtual_loss.max(0.0),
        expansion_lock: cfg.mcts.katago.expansion_lock,
    };

    // Progress + output directories (mirrors selfplay-worker layout, but under logs_gate_workers).
    let logs_dir = run_dir
        .join("logs_gate_workers")
        .join(format!("worker_{worker_id:03}"));
    let _ = std::fs::create_dir_all(&logs_dir);
    let progress_path = logs_dir.join("progress.json");

    #[derive(serde::Serialize)]
    struct GateWorkerProgress {
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        games_completed: u32,
        #[serde(default)]
        wins: u32,
        #[serde(default)]
        losses: u32,
        #[serde(default)]
        draws: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        first_game_started_ts_ms: Option<u64>,
        pid: u32,
        ts_ms: u64,
    }

    fn write_progress_atomic(path: &PathBuf, p: &GateWorkerProgress) {
        let tmp = path.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec(p) {
            let _ = std::fs::write(&tmp, &bytes);
            let _ = std::fs::rename(&tmp, path);
        }
    }

    // Initial progress file for live aggregation.
    let first_game_started_ts_ms = yz_logging::now_ms();
    write_progress_atomic(
        &progress_path,
        &GateWorkerProgress {
            worker_id,
            num_workers,
            games_target,
            games_completed: 0,
            wins: 0,
            losses: 0,
            draws: 0,
            // Gate-worker doesn't have a scheduler heartbeat by default; record first-game-start
            // immediately so the controller/TUI can show setup time accurately.
            first_game_started_ts_ms: Some(first_game_started_ts_ms),
            pid: std::process::id(),
            ts_ms: yz_logging::now_ms(),
        },
    );

    struct ProgressSink {
        progress_path: PathBuf,
        stop_path: PathBuf,
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        parallel_games: u32,
        worker_stats: Option<yz_logging::NdjsonWriter>,
        t_start: std::time::Instant,
        first_game_started_ts_ms: Option<u64>,
        first_done_logged: bool,
        last_progress_write: std::time::Instant,
        last_stop_check: std::time::Instant,
        stop_seen: bool,
        wins: u32,
        losses: u32,
        draws: u32,
    }
    impl yz_eval::GateProgress for ProgressSink {
        fn on_game_completed(&mut self, completed: u32, _total: u32) {
            write_progress_atomic(
                &self.progress_path,
                &GateWorkerProgress {
                    worker_id: self.worker_id,
                    num_workers: self.num_workers,
                    games_target: self.games_target,
                    games_completed: completed,
                    wins: self.wins,
                    losses: self.losses,
                    draws: self.draws,
                    first_game_started_ts_ms: self.first_game_started_ts_ms,
                    pid: std::process::id(),
                    ts_ms: yz_logging::now_ms(),
                },
            );

            if !self.first_done_logged && completed >= 1 {
                if let Some(w) = self.worker_stats.as_mut() {
                    let _ = w.write_event(&GateWorkerStatsEvent {
                        event: "gate_worker_first_game_done",
                        ts_ms: yz_logging::now_ms(),
                        worker_id: self.worker_id,
                        num_workers: self.num_workers,
                        games_target: self.games_target,
                        wall_ms: self.t_start.elapsed().as_millis() as u64,
                    });
                    let _ = w.flush();
                }
                self.first_done_logged = true;
            }
        }

        fn on_tick(&mut self, stats: &yz_eval::GateTickStats) {
            self.wins = stats.cand_wins;
            self.losses = stats.cand_losses;
            self.draws = stats.draws;
            // Heartbeat progress even before any game finishes, so the controller can surface
            // "waiting for first game" setup vs running time accurately.
            if self.last_progress_write.elapsed() >= std::time::Duration::from_millis(500) {
                write_progress_atomic(
                    &self.progress_path,
                    &GateWorkerProgress {
                        worker_id: self.worker_id,
                        num_workers: self.num_workers,
                        games_target: self.games_target,
                        games_completed: stats.games_completed,
                        wins: stats.cand_wins,
                        losses: stats.cand_losses,
                        draws: stats.draws,
                        first_game_started_ts_ms: self.first_game_started_ts_ms,
                        pid: std::process::id(),
                        ts_ms: yz_logging::now_ms(),
                    },
                );
                self.last_progress_write = std::time::Instant::now();
            }
            if let Some(w) = self.worker_stats.as_mut() {
                #[derive(serde::Serialize)]
                struct GateWorkerTickEvent<'a> {
                    event: &'a str,
                    ts_ms: u64,
                    worker_id: u32,
                    num_workers: u32,
                    games_target: u32,
                    parallel_games: u32,
                    games_completed: u32,
                    wall_ms: u64,
                    ticks: u64,
                    would_block: u64,
                    progress: u64,
                    terminal: u64,
                    best_inflight: u64,
                    cand_inflight: u64,
                }
                let _ = w.write_event(&GateWorkerTickEvent {
                    event: "gate_worker_tick",
                    ts_ms: yz_logging::now_ms(),
                    worker_id: self.worker_id,
                    num_workers: self.num_workers,
                    games_target: self.games_target,
                    parallel_games: self.parallel_games,
                    games_completed: stats.games_completed,
                    wall_ms: self.t_start.elapsed().as_millis() as u64,
                    ticks: stats.ticks,
                    would_block: stats.would_block,
                    progress: stats.progress,
                    terminal: stats.terminal,
                    best_inflight: stats.best_inflight,
                    cand_inflight: stats.cand_inflight,
                });
                let _ = w.flush();
            }
        }

        fn should_stop(&mut self) -> bool {
            if self.stop_seen {
                return true;
            }
            // Avoid excessive filesystem polling.
            if self.last_stop_check.elapsed() < std::time::Duration::from_millis(200) {
                return false;
            }
            self.last_stop_check = std::time::Instant::now();
            if self.stop_path.exists() {
                self.stop_seen = true;
                return true;
            }
            false
        }
    }

    // Oracle diag log (ndjson, best-effort).
    let mut oracle_log =
        yz_logging::NdjsonWriter::open_append_with_flush(logs_dir.join("oracle_diag.ndjson"), 200)
            .ok();
    // Worker timing stats (ndjson, best-effort).
    let mut worker_stats =
        yz_logging::NdjsonWriter::open_append_with_flush(logs_dir.join("worker_stats.ndjson"), 50)
            .ok();
    #[derive(serde::Serialize)]
    struct GateWorkerStatsEvent<'a> {
        event: &'a str,
        ts_ms: u64,
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        wall_ms: u64,
    }
    let t_start = std::time::Instant::now();
    if let Some(w) = worker_stats.as_mut() {
        let _ = w.write_event(&GateWorkerStatsEvent {
            event: "gate_worker_start",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target,
            wall_ms: 0,
        });
        let _ = w.flush();
    }

    struct OracleSink<'a> {
        w: &'a mut yz_logging::NdjsonWriter,
    }
    impl yz_eval::OracleDiagSink for OracleSink<'_> {
        fn on_step(&mut self, ev: &yz_eval::OracleDiagEvent) {
            // Best-effort; don't crash worker for logging.
            let _ = self.w.write_event(ev);
        }
    }
    let mut oracle_sink = oracle_log.as_mut().map(|w| OracleSink { w });

    // Gate replay traces (JSON per game, best-effort).
    // These are used by the TUI Replay screen to step through seed-swapped pairs.
    let (run_id, iter_idx) = yz_logging::read_manifest(&run_dir.join("run.json"))
        .map(|m| (m.run_id, m.controller_iteration_idx))
        .unwrap_or_else(|_| (run_dir.file_name().unwrap_or_default().to_string_lossy().to_string(), 0));
    let gate_replays_dir = run_dir
        .join("gate_replays")
        .join(format!("iter_{iter_idx:03}"));
    let _ = std::fs::create_dir_all(&gate_replays_dir);

    #[derive(serde::Serialize)]
    struct GateReplayFileV1 {
        trace_version: u32,
        run_id: String,
        iter_idx: u32,
        best_model_id: u32,
        cand_model_id: u32,
        episode_seed: u64,
        swap: bool,
        cand_seat: u8,
        best_seat: u8,
        steps: Vec<yz_eval::GateReplayStepV1>,
        terminal: Option<yz_eval::GateReplayTerminalV1>,
    }

    struct ReplaySink {
        dir: PathBuf,
        run_id: String,
        iter_idx: u32,
        best_model_id: u32,
        cand_model_id: u32,
    }
    impl yz_eval::GateReplaySink for ReplaySink {
        fn on_game(&mut self, trace: yz_eval::GateReplayTraceV1) {
            let swap_i = if trace.swap { 1 } else { 0 };
            let path = self
                .dir
                .join(format!("seed_{}_swap_{}.json", trace.episode_seed, swap_i));
            let tmp = path.with_extension("json.tmp");
            let out = GateReplayFileV1 {
                trace_version: trace.trace_version,
                run_id: self.run_id.clone(),
                iter_idx: self.iter_idx,
                best_model_id: self.best_model_id,
                cand_model_id: self.cand_model_id,
                episode_seed: trace.episode_seed,
                swap: trace.swap,
                cand_seat: trace.cand_seat,
                best_seat: trace.best_seat,
                steps: trace.steps,
                terminal: trace.terminal,
            };
            if let Ok(bytes) = serde_json::to_vec(&out) {
                let _ = std::fs::write(&tmp, &bytes);
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }
    let mut replay_sink = ReplaySink {
        dir: gate_replays_dir,
        run_id,
        iter_idx,
        best_model_id: best_id,
        cand_model_id: cand_id,
    };

    if let Some(w) = worker_stats.as_mut() {
        let _ = w.write_event(&GateWorkerStatsEvent {
            event: "gate_worker_first_game_start",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target,
            wall_ms: t_start.elapsed().as_millis() as u64,
        });
        let _ = w.flush();
    }

    let mut sink = ProgressSink {
        progress_path: progress_path.clone(),
        stop_path: run_dir.join("gate_workers").join("stop.json"),
        worker_id,
        num_workers,
        games_target,
        parallel_games,
        worker_stats,
        t_start,
        first_game_started_ts_ms: Some(first_game_started_ts_ms),
        first_done_logged: false,
        last_progress_write: std::time::Instant::now(),
        last_stop_check: std::time::Instant::now(),
        stop_seen: false,
        wins: 0,
        losses: 0,
        draws: 0,
    };

    // IMPORTANT: gate-worker runs a fixed schedule slice. SPRT decisions are made centrally by the controller.
    // Disable local SPRT logic inside each worker to avoid per-slice early-stop bias.
    cfg.gating.katago.sprt = false;

    let partial = yz_eval::gate_schedule_subset(
        &cfg,
        yz_eval::GateOptions {
            infer_endpoint: infer,
            best_model_id: best_id,
            cand_model_id: cand_id,
            client_opts,
            mcts_cfg,
        },
        &schedule,
        Some(&mut sink),
        oracle_sink
            .as_mut()
            .map(|s| s as &mut dyn yz_eval::OracleDiagSink),
        Some(&mut replay_sink),
    )
    .unwrap_or_else(|e| {
        eprintln!("gate-worker failed: {e}");
        process::exit(1);
    });

    // Ensure final progress is complete.
    write_progress_atomic(
        &progress_path,
        &GateWorkerProgress {
            worker_id,
            num_workers,
            games_target,
            games_completed: partial.games,
            wins: partial.cand_wins,
            losses: partial.cand_losses,
            draws: partial.draws,
            first_game_started_ts_ms: sink.first_game_started_ts_ms,
            pid: std::process::id(),
            ts_ms: yz_logging::now_ms(),
        },
    );

    if let Some(w) = sink.worker_stats.as_mut() {
        let _ = w.write_event(&GateWorkerStatsEvent {
            event: "gate_worker_done",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target,
            wall_ms: t_start.elapsed().as_millis() as u64,
        });
        let _ = w.flush();
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct GateWorkerResult {
        worker_id: u32,
        games: u32,
        cand_wins: u32,
        cand_losses: u32,
        draws: u32,
        cand_score_diff_sum: i64,
        cand_score_diff_sumsq: f64,
        cand_score_sum: i64,
        best_score_sum: i64,
    }

    let out_path = PathBuf::from(out_path);
    let tmp = out_path.with_extension("json.tmp");
    let res = GateWorkerResult {
        worker_id,
        games: partial.games,
        cand_wins: partial.cand_wins,
        cand_losses: partial.cand_losses,
        draws: partial.draws,
        cand_score_diff_sum: partial.cand_score_diff_sum,
        cand_score_diff_sumsq: partial.cand_score_diff_sumsq,
        cand_score_sum: partial.cand_score_sum,
        best_score_sum: partial.best_score_sum,
    };
    let bytes = serde_json::to_vec(&res).expect("serialize gate worker result");
    std::fs::write(&tmp, bytes).unwrap_or_else(|e| {
        eprintln!("Failed to write gate worker result: {e}");
        process::exit(1);
    });
    std::fs::rename(&tmp, &out_path).unwrap_or_else(|e| {
        eprintln!("Failed to rename gate worker result: {e}");
        process::exit(1);
    });
}

fn cmd_gate(args: &[String]) {
    let mut config_path: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut best_id: u32 = 0;
    let mut cand_id: u32 = 1;
    let mut run_dir: Option<String> = None;
    let mut out_path: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz gate

USAGE:
    yz gate --config cfg.yaml [--infer unix:///tmp/yatzy_infer.sock] [--best-id 0] [--cand-id 1] [--run runs/<id>/]

NOTES:
    - Candidate vs best are selected by model_id routing on the inference server.
    - If gating.paired_seed_swap=true, gating.games must be even (two games per seed, side-swapped).

OPTIONS:
    --config PATH       Config YAML path
    --infer ENDPOINT    Override inference endpoint (unix:///... or tcp://host:port). Defaults to config.inference.bind
    --best-id N         model_id for best (default 0)
    --cand-id N         model_id for candidate (default 1)
    --run DIR           Optional run dir to update runs/<id>/run.json with gate stats
"#
                );
                return;
            }
            "--config" => {
                config_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--best-id" => {
                best_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(best_id);
                i += 2;
            }
            "--cand-id" => {
                cand_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(cand_id);
                i += 2;
            }
            "--run" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--out" => {
                out_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz gate`: {other}");
                eprintln!("Run `yz gate --help` for usage.");
                process::exit(1);
            }
        }
    }

    let config_path = config_path.unwrap_or_else(|| {
        eprintln!("Missing --config");
        process::exit(1);
    });

    let cfg = yz_core::Config::load(&config_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config: {e}");
        process::exit(1);
    });

    let infer_ep = infer.unwrap_or_else(|| cfg.inference.bind.clone());

    // Use bounded inflight to prevent flooding the inference server.
    // With N workers, system-wide max = N * 64. Server healthy capacity ~50-150.
    let client_opts = yz_infer::ClientOptions {
        max_inflight_total: 64,
        max_outbound_queue: 256,
        request_id_start: 1,
        protocol_version: cfg.inference.protocol_version,
        legal_mask_bitset: cfg.inference.legal_mask_bitset && cfg.inference.protocol_version == 2,
    };

    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: 0.0, // gating: no root noise
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss_mode: match cfg.mcts.virtual_loss_mode.as_str() {
            "q_penalty" => yz_mcts::VirtualLossMode::QPenalty,
            "n_virtual_only" => yz_mcts::VirtualLossMode::NVirtualOnly,
            "off" => yz_mcts::VirtualLossMode::Off,
            _ => yz_mcts::VirtualLossMode::QPenalty,
        },
        virtual_loss: cfg.mcts.virtual_loss.max(0.0),
        expansion_lock: cfg.mcts.katago.expansion_lock,
    };

    let report = yz_eval::gate(
        &cfg,
        yz_eval::GateOptions {
            infer_endpoint: infer_ep.clone(),
            best_model_id: best_id,
            cand_model_id: cand_id,
            client_opts,
            mcts_cfg,
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("Gating failed: {e}");
        process::exit(1);
    });

    let wr = report.win_rate();
    let decision = if cfg.gating.katago.sprt {
        match report.sprt.as_ref().map(|s| s.decision) {
            Some(yz_eval::SprtDecision::AcceptH1) => "promote",
            Some(yz_eval::SprtDecision::AcceptH0) => "reject",
            Some(yz_eval::SprtDecision::InconclusiveMaxGames)
            | Some(yz_eval::SprtDecision::Continue)
            | None => {
                // Fallback to fixed-threshold decision.
                if wr + 1e-12 < cfg.gating.win_rate_threshold {
                    "reject"
                } else {
                    "promote"
                }
            }
        }
    } else if wr + 1e-12 < cfg.gating.win_rate_threshold {
        "reject"
    } else {
        "promote"
    };
    if let Some(id) = cfg.gating.seed_set_id.as_deref() {
        println!(
            "Seed source: seed_set_id={id} requested_games={}",
            if cfg.gating.katago.sprt {
                cfg.gating.katago.sprt_max_games
            } else {
                cfg.gating.games
            }
        );
    } else {
        println!(
            "Seed source: seed={} requested_games={}",
            cfg.gating.seed,
            if cfg.gating.katago.sprt {
                cfg.gating.katago.sprt_max_games
            } else {
                cfg.gating.games
            }
        );
    }
    for w in &report.warnings {
        eprintln!("warning: {w}");
    }
    println!(
        "Gating complete. decision={} games={} wins={} losses={} draws={} win_rate={:.4} win_ci95=[{:.4},{:.4}] mean_score_diff={:.2} se={:.3} ci95=[{:.3},{:.3}] seeds_hash={}",
        decision,
        report.games,
        report.cand_wins,
        report.cand_losses,
        report.draws,
        wr,
        report.win_rate_ci95_low,
        report.win_rate_ci95_high,
        report.mean_score_diff(),
        report.score_diff_se,
        report.score_diff_ci95_low,
        report.score_diff_ci95_high,
        report.seeds_hash
    );

    if let Some(run) = run_dir {
        let run_dir = PathBuf::from(run);
        let run_json = run_dir.join("run.json");
        match yz_logging::read_manifest(&run_json) {
            Ok(mut m) => {
                m.gate_games = Some(report.games as u64);
                m.gate_win_rate = Some(wr);
                m.gate_draw_rate = Some(report.draw_rate);
                m.gate_wins = Some(report.cand_wins as u64);
                m.gate_losses = Some(report.cand_losses as u64);
                m.gate_draws = Some(report.draws as u64);
                m.gate_ci95_low = Some(report.win_rate_ci95_low);
                m.gate_ci95_high = Some(report.win_rate_ci95_high);
                m.gate_sprt = report.sprt.as_ref().map(|s| yz_logging::GateSprtSummaryV1 {
                    enabled: true,
                    min_games: s.min_games as u64,
                    max_games: s.max_games as u64,
                    alpha: s.alpha,
                    beta: s.beta,
                    delta: s.delta,
                    p0: s.p0,
                    p1: s.p1,
                    llr: s.llr,
                    bound_a: s.bound_a,
                    bound_b: s.bound_b,
                    decision: Some(match s.decision {
                        yz_eval::SprtDecision::Continue => "continue",
                        yz_eval::SprtDecision::AcceptH1 => "accept_h1",
                        yz_eval::SprtDecision::AcceptH0 => "accept_h0",
                        yz_eval::SprtDecision::InconclusiveMaxGames => "inconclusive_max_games",
                    }
                    .to_string()),
                    decision_reason: None,
                    games_at_decision: Some(s.games_at_decision as u64),
                });
                m.gate_seeds_hash = Some(report.seeds_hash.clone());
                m.gate_oracle_match_rate_overall = Some(report.oracle_match_rate_overall);
                m.gate_oracle_match_rate_mark = Some(report.oracle_match_rate_mark);
                m.gate_oracle_match_rate_reroll = Some(report.oracle_match_rate_reroll);

                // E10.5S1: ensure run-local config snapshot exists (normalized).
                if m.config_snapshot.is_none() || !run_dir.join("config.yaml").exists() {
                    if let Ok((rel, h)) = yz_logging::write_config_snapshot_atomic(&run_dir, &cfg) {
                        m.config_snapshot = Some(rel);
                        m.config_snapshot_hash = Some(h);
                    }
                }

                let _ = yz_logging::write_manifest_atomic(&run_json, &m);

                // E10.5S2: unified metrics stream.
                let logs_dir = run_dir.join(&m.logs_dir);
                let _ = std::fs::create_dir_all(&logs_dir);
                let metrics_path = logs_dir.join("metrics.ndjson");
                if let Ok(mut w) = yz_logging::NdjsonWriter::open_append(metrics_path) {
                    let ev = yz_logging::MetricsGateSummaryV1 {
                        event: "gate_summary",
                        ts_ms: yz_logging::now_ms(),
                        v: yz_logging::VersionInfoV1 {
                            protocol_version: m.protocol_version,
                            feature_schema_id: m.feature_schema_id,
                            action_space_id: "oracle_keepmask_v2",
                            ruleset_id: "swedish_scandinavian_mark_at_r3_v1",
                        },
                        run_id: m.run_id.clone(),
                        git_hash: m.git_hash.clone(),
                        config_snapshot: m.config_snapshot.clone(),
                        decision: decision.to_string(),
                        games: report.games,
                        wins: report.cand_wins,
                        losses: report.cand_losses,
                        draws: report.draws,
                        win_rate: wr,
                        mean_score_diff: report.mean_score_diff(),
                        mean_cand_score: Some(report.mean_cand_score()),
                        mean_best_score: Some(report.mean_best_score()),
                        score_diff_se: report.score_diff_se,
                        score_diff_ci95_low: report.score_diff_ci95_low,
                        score_diff_ci95_high: report.score_diff_ci95_high,
                        seeds_hash: report.seeds_hash.clone(),
                        oracle_match_rate_overall: report.oracle_match_rate_overall,
                        oracle_match_rate_mark: report.oracle_match_rate_mark,
                        oracle_match_rate_reroll: report.oracle_match_rate_reroll,
                    };
                    let _ = w.write_event(&ev);
                    let _ = w.flush();
                }
            }
            Err(e) => {
                eprintln!("Warning: failed to update run manifest: {e:?}");
            }
        }

        // Write gate_report.json (default under runs/<id>/ unless --out provided).
        let out = out_path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| run_dir.join("gate_report.json"));
        let _ = write_gate_report_atomic(
            &out,
            &GateReportJson {
                decision: decision.to_string(),
                games: report.games,
                wins: report.cand_wins,
                losses: report.cand_losses,
                draws: report.draws,
                win_rate: wr,
                win_rate_threshold: cfg.gating.win_rate_threshold,
                mean_score_diff: report.mean_score_diff(),
                score_diff_se: report.score_diff_se,
                score_diff_ci95_low: report.score_diff_ci95_low,
                score_diff_ci95_high: report.score_diff_ci95_high,
                seeds_hash: report.seeds_hash.clone(),
                seed: cfg.gating.seed,
                seed_set_id: cfg.gating.seed_set_id.clone(),
                warnings: report.warnings.clone(),
                oracle_match_rate_overall: report.oracle_match_rate_overall,
                oracle_match_rate_mark: report.oracle_match_rate_mark,
                oracle_match_rate_reroll: report.oracle_match_rate_reroll,
            },
        );
    }

    if decision == "reject" {
        eprintln!(
            "Candidate rejected: win_rate {:.4} < threshold {:.4}",
            wr, cfg.gating.win_rate_threshold
        );
        process::exit(2);
    }
}

#[derive(serde::Serialize)]
struct GateReportJson {
    decision: String,
    games: u32,
    wins: u32,
    losses: u32,
    draws: u32,
    win_rate: f64,
    win_rate_threshold: f64,
    mean_score_diff: f64,
    score_diff_se: f64,
    score_diff_ci95_low: f64,
    score_diff_ci95_high: f64,
    seeds_hash: String,
    seed: u64,
    seed_set_id: Option<String>,
    warnings: Vec<String>,
    oracle_match_rate_overall: f64,
    oracle_match_rate_mark: f64,
    oracle_match_rate_reroll: f64,
}

fn write_gate_report_atomic(path: &PathBuf, report: &GateReportJson) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(report).expect("serialize gate_report");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn connect_infer_backend(
    endpoint: &str,
    protocol_version: u32,
    legal_mask_bitset: bool,
) -> yz_mcts::InferBackend {
    // Use bounded inflight to prevent flooding the inference server.
    let opts = yz_infer::ClientOptions {
        max_inflight_total: 64,
        max_outbound_queue: 256,
        request_id_start: 1,
        protocol_version,
        legal_mask_bitset: legal_mask_bitset && protocol_version == 2,
    };
    if let Some(rest) = endpoint.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            return yz_mcts::InferBackend::connect_uds(rest, 0, opts).unwrap();
        }
        #[cfg(not(unix))]
        {
            panic!("unix:// endpoints are only supported on unix");
        }
    }
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        return yz_mcts::InferBackend::connect_tcp(rest, 0, opts).unwrap();
    }
    panic!("Unsupported infer endpoint: {endpoint}");
}

fn cmd_iter_finalize(args: &[String]) {
    let mut run: Option<String> = None;
    let mut decision: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz iter finalize

USAGE:
    yz iter finalize --run runs/<id>/ --decision promote|reject

OPTIONS:
    --run DIR            Run directory (contains run.json, models/, replay/, logs/)
    --decision D         promote|reject
"#
                );
                return;
            }
            "--run" => {
                run = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--decision" => {
                decision = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz iter finalize`: {}", other);
                eprintln!("Run `yz iter finalize --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run = run.unwrap_or_else(|| {
        eprintln!("Missing --run");
        process::exit(1);
    });
    let decision = decision.unwrap_or_else(|| {
        eprintln!("Missing --decision");
        process::exit(1);
    });
    if decision != "promote" && decision != "reject" {
        eprintln!("Invalid --decision (expected promote|reject)");
        process::exit(1);
    }

    let run_dir = PathBuf::from(&run);
    let run_json = run_dir.join("run.json");
    let mut manifest = yz_logging::read_manifest(&run_json).unwrap_or_else(|e| {
        eprintln!("Failed to read run manifest: {e:?}");
        process::exit(1);
    });

    let models_dir = run_dir.join(&manifest.models_dir);
    let candidate_pt = models_dir.join("candidate.pt");
    if !candidate_pt.exists() {
        eprintln!("Missing candidate checkpoint: {}", candidate_pt.display());
        process::exit(1);
    }

    // Promote: copy candidate -> best atomically via temp+rename.
    if decision == "promote" {
        let best_pt = models_dir.join("best.pt");
        let tmp_best = models_dir.join("best.pt.tmp");
        let bytes = std::fs::read(&candidate_pt).unwrap_or_else(|e| {
            eprintln!("Failed to read candidate.pt: {e}");
            process::exit(1);
        });
        std::fs::write(&tmp_best, bytes).unwrap_or_else(|e| {
            eprintln!("Failed to write tmp best.pt: {e}");
            process::exit(1);
        });
        std::fs::rename(&tmp_best, &best_pt).unwrap_or_else(|e| {
            eprintln!("Failed to rename best.pt: {e}");
            process::exit(1);
        });

        // best.meta.json: derived from candidate.meta.json if present, else minimal.
        let cand_meta = models_dir.join("candidate.meta.json");
        let best_meta = models_dir.join("best.meta.json");
        if cand_meta.exists() {
            let tmp_meta = models_dir.join("best.meta.json.tmp");
            let meta_bytes = std::fs::read(&cand_meta).unwrap_or_else(|e| {
                eprintln!("Failed to read candidate.meta.json: {e}");
                process::exit(1);
            });
            std::fs::write(&tmp_meta, meta_bytes).unwrap_or_else(|e| {
                eprintln!("Failed to write tmp best.meta.json: {e}");
                process::exit(1);
            });
            std::fs::rename(&tmp_meta, &best_meta).unwrap_or_else(|e| {
                eprintln!("Failed to rename best.meta.json: {e}");
                process::exit(1);
            });
        } else {
            let tmp_meta = models_dir.join("best.meta.json.tmp");
            let meta = serde_json::json!({
                "protocol_version": manifest.protocol_version,
                "feature_schema_id": manifest.feature_schema_id,
                "action_space_id": manifest.action_space_id,
                "ruleset_id": manifest.ruleset_id,
            });
            std::fs::write(&tmp_meta, serde_json::to_vec_pretty(&meta).unwrap()).unwrap_or_else(
                |e| {
                    eprintln!("Failed to write tmp best.meta.json: {e}");
                    process::exit(1);
                },
            );
            std::fs::rename(&tmp_meta, &best_meta).unwrap_or_else(|e| {
                eprintln!("Failed to rename best.meta.json: {e}");
                process::exit(1);
            });
        }

        manifest.best_checkpoint = Some("models/best.pt".to_string());
    }

    manifest.promotion_decision = Some(decision.clone());
    manifest.promotion_ts_ms = Some(yz_logging::now_ms());
    manifest.candidate_checkpoint = Some("models/candidate.pt".to_string());

    yz_logging::write_manifest_atomic(&run_json, &manifest).unwrap_or_else(|e| {
        eprintln!("Failed to write run manifest: {e:?}");
        process::exit(1);
    });

    println!(
        "Finalize complete. decision={decision} run={}",
        run_dir.display()
    );
}

fn cmd_bench(args: &[String]) {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            r#"yz bench

USAGE:
    yz bench [<cargo args>...]
    yz bench e2e -- [OPTIONS]

NOTES:
    - This is a thin wrapper around:
        cargo bench -p yz-bench <cargo args...>
    - E2E benchmark is a separate harness crate:
        cargo run -p yz-bench-e2e -- [OPTIONS]

EXAMPLES:
    yz bench
    yz bench --bench scoring
    yz bench --bench scoring -- --warm-up-time 0.5 --measurement-time 1.0
    yz bench e2e -- --seconds 10 --parallel 8 --simulations 64 --max-inflight 4 --chance deterministic
"#
        );
        return;
    }

    // Subcommand: yz bench e2e -- [opts]
    if args.first().map(|s| s.as_str()) == Some("e2e") {
        let mut cmd = Command::new("cargo");
        cmd.arg("run").arg("-p").arg("yz-bench-e2e").arg("--");
        cmd.args(&args[1..]);
        let status = cmd.status().unwrap_or_else(|e| {
            eprintln!("Failed to run e2e bench harness: {e}");
            eprintln!("Hint: ensure Rust tooling is installed and `cargo` is on PATH.");
            process::exit(1);
        });
        if !status.success() {
            process::exit(status.code().unwrap_or(1));
        }
        return;
    }

    let mut cmd = Command::new("cargo");
    cmd.arg("bench").arg("-p").arg("yz-bench");
    cmd.args(args);

    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("Failed to run cargo bench: {e}");
        eprintln!("Hint: ensure Rust tooling is installed and `cargo` is on PATH.");
        process::exit(1);
    });
    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }
}

fn cargo_flamegraph_available() -> bool {
    let out = Command::new("cargo")
        .args(["flamegraph", "--version"])
        .output();
    matches!(out, Ok(o) if o.status.success())
}

fn spawn_current_exe(args: &[String]) -> process::ExitStatus {
    let exe = env::current_exe().unwrap_or_else(|e| {
        eprintln!("Failed to locate current executable: {e}");
        process::exit(1);
    });
    Command::new(exe).args(args).status().unwrap_or_else(|e| {
        eprintln!("Failed to spawn yz: {e}");
        process::exit(1);
    })
}

fn cmd_profile(args: &[String]) {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            r#"yz profile

USAGE:
    yz profile <selfplay|gate|bench-e2e> -- <args...>

NOTES:
    - This command is a thin wrapper around `cargo flamegraph`.
    - If `cargo flamegraph` is not installed, we fall back to running the underlying command normally.

EXAMPLES:
    yz profile selfplay -- --help
    yz profile bench-e2e -- --seconds 10 --parallel 8 --simulations 64 --max-inflight 4 --chance deterministic

INSTALL:
    cargo install flamegraph
"#
        );
        return;
    }

    let target = args[0].as_str();
    let sep = args.iter().position(|s| s == "--");
    let (before, after) = match sep {
        Some(i) => (&args[1..i], &args[(i + 1)..]),
        None => {
            eprintln!("Missing `--` separator. See `yz profile --help`.");
            process::exit(2);
        }
    };
    if !before.is_empty() {
        eprintln!("Unexpected args before `--`: {before:?}");
        eprintln!("See `yz profile --help`.");
        process::exit(2);
    }

    // Underlying command args (when we fall back).
    let mut underlying: Vec<String> = Vec::new();
    match target {
        "selfplay" => {
            underlying.push("selfplay".to_string());
            underlying.extend_from_slice(after);
        }
        "gate" => {
            underlying.push("gate".to_string());
            underlying.extend_from_slice(after);
        }
        "bench-e2e" => {
            underlying.push("bench".to_string());
            underlying.push("e2e".to_string());
            underlying.push("--".to_string());
            underlying.extend_from_slice(after);
        }
        other => {
            eprintln!("Unknown profile target: {other}");
            eprintln!("See `yz profile --help`.");
            process::exit(2);
        }
    }

    if !cargo_flamegraph_available() {
        eprintln!("warning: `cargo flamegraph` not found (install with: cargo install flamegraph)");
        eprintln!("warning: running underlying command without profiling");
        let status = spawn_current_exe(&underlying);
        if !status.success() {
            process::exit(status.code().unwrap_or(1));
        }
        return;
    }

    // Profile path via cargo flamegraph.
    let mut cmd = Command::new("cargo");
    cmd.arg("flamegraph");
    match target {
        "selfplay" => {
            cmd.args(["--bin", "yz", "--"]);
            cmd.arg("selfplay");
            cmd.args(after);
        }
        "gate" => {
            cmd.args(["--bin", "yz", "--"]);
            cmd.arg("gate");
            cmd.args(after);
        }
        "bench-e2e" => {
            cmd.args(["-p", "yz-bench-e2e", "--bin", "yz-bench-e2e", "--"]);
            cmd.args(after);
        }
        _ => unreachable!(),
    }

    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("Failed to run cargo flamegraph: {e}");
        process::exit(1);
    });
    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        process::exit(0);
    }

    match args[1].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
        }
        "-V" | "--version" => {
            print_version();
        }
        "oracle" => {
            if args.len() < 3 {
                eprintln!("Usage: yz oracle <expected|sim> [OPTIONS]");
                process::exit(1);
            }
            match args[2].as_str() {
                "expected" => {
                    cmd_oracle_expected();
                }
                "sim" => {
                    cmd_oracle_sim(&args[3..]);
                }
                _ => {
                    eprintln!("Unknown oracle subcommand: {}", args[2]);
                    process::exit(1);
                }
            }
        }
        "oracle-set-gen" => {
            cmd_oracle_set_gen(&args[2..]);
        }
        "start-run" => {
            cmd_start_run(&args[2..]);
        }
        "extend-run" => {
            cmd_extend_run(&args[2..]);
        }
        "controller" => {
            cmd_controller(&args[2..]);
        }
        "selfplay" => {
            cmd_selfplay(&args[2..]);
        }
        "selfplay-worker" => {
            cmd_selfplay_worker(&args[2..]);
        }
        "gate-worker" => {
            cmd_gate_worker(&args[2..]);
        }
        "iter" => {
            if args.len() < 3 {
                eprintln!("Usage: yz iter finalize [OPTIONS]");
                process::exit(1);
            }
            match args[2].as_str() {
                "finalize" => {
                    cmd_iter_finalize(&args[3..]);
                }
                other => {
                    eprintln!("Unknown iter subcommand: {other}");
                    eprintln!("Usage: yz iter finalize [OPTIONS]");
                    process::exit(1);
                }
            }
        }
        "gate" => {
            cmd_gate(&args[2..]);
        }
        "oracle-eval" => {
            println!("Oracle evaluation (not yet implemented)");
            println!("Usage: yz oracle-eval --config cfg.yaml --best ... --cand ...");
        }
        "oracle-fixed-worker" => {
            cmd_oracle_fixed_worker(&args[2..]);
        }
        "bench" => {
            cmd_bench(&args[2..]);
        }
        "tui" => {
            if let Err(e) = yz_tui::run() {
                eprintln!("TUI failed: {e}");
                process::exit(1);
            }
        }
        "profile" => {
            cmd_profile(&args[2..]);
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!("Run `yz --help` for usage.");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_compiles() {
        // Basic sanity: the binary compiles and this test runs.
        assert!(true);
    }

    #[test]
    fn sanitize_run_name_replaces_invalid_chars() {
        assert_eq!(sanitize_run_name("abc-DEF_123"), "abc-DEF_123");
        assert_eq!(sanitize_run_name("a b/c"), "a_b_c");
        assert_eq!(sanitize_run_name(""), "");
    }

    #[test]
    fn ensure_unique_run_dir_appends_timestamp_if_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let runs = tmp.path();
        std::fs::create_dir_all(runs.join("foo")).unwrap();
        let (id, dir) = ensure_unique_run_dir(runs, "foo");
        assert!(id.starts_with("foo_"));
        assert_eq!(dir, runs.join(&id));
    }
}
