use yz_core::config::TemperatureSchedule;
use yz_core::Config;

pub fn validate_config(cfg: &Config) -> Result<(), String> {
    // inference
    if cfg.inference.bind.trim().is_empty() {
        return Err("inference.bind must be non-empty".to_string());
    }
    if cfg.inference.device != "cpu" && cfg.inference.device != "cuda" && cfg.inference.device != "mps" {
        return Err("inference.device must be 'cpu', 'cuda', or 'mps'".to_string());
    }
    if cfg.inference.max_batch < 1 {
        return Err("inference.max_batch must be >= 1".to_string());
    }
    if cfg.inference.max_wait_us < 1 {
        return Err("inference.max_wait_us must be >= 1".to_string());
    }
    if let Some(n) = cfg.inference.torch_threads {
        if n < 1 {
            return Err("inference.torch_threads must be >= 1 when set".to_string());
        }
    }
    if let Some(n) = cfg.inference.torch_interop_threads {
        if n < 1 {
            return Err("inference.torch_interop_threads must be >= 1 when set".to_string());
        }
    }

    // mcts
    if cfg.mcts.c_puct <= 0.0 {
        return Err("mcts.c_puct must be > 0".to_string());
    }
    if cfg.mcts.budget_reroll < 1 {
        return Err("mcts.budget_reroll must be >= 1".to_string());
    }
    if cfg.mcts.budget_mark < 1 {
        return Err("mcts.budget_mark must be >= 1".to_string());
    }
    if cfg.mcts.max_inflight_per_game < 1 {
        return Err("mcts.max_inflight_per_game must be >= 1".to_string());
    }
    if cfg.mcts.dirichlet_alpha <= 0.0 {
        return Err("mcts.dirichlet_alpha must be > 0".to_string());
    }
    if !(0.0..=1.0).contains(&cfg.mcts.dirichlet_epsilon) {
        return Err("mcts.dirichlet_epsilon must be in [0,1]".to_string());
    }
    match cfg.mcts.temperature_schedule {
        TemperatureSchedule::Constant { t0 } => {
            if t0 < 0.0 {
                return Err("mcts.temperature_schedule.t0 must be >= 0".to_string());
            }
        }
        TemperatureSchedule::Step {
            t0,
            t1,
            cutoff_turn: _,
        } => {
            if t0 < 0.0 || t1 < 0.0 {
                return Err("mcts.temperature_schedule.t0/t1 must be >= 0".to_string());
            }
        }
    }
    match cfg.mcts.virtual_loss_mode.as_str() {
        "q_penalty" | "n_virtual_only" | "off" => {}
        _ => {
            return Err(
                "mcts.virtual_loss_mode must be 'q_penalty', 'n_virtual_only', or 'off'".to_string(),
            )
        }
    }
    if !(cfg.mcts.virtual_loss.is_finite() && cfg.mcts.virtual_loss >= 0.0) {
        return Err("mcts.virtual_loss must be finite and >= 0".to_string());
    }
    // mcts.chance_pw
    if cfg.mcts.chance_pw.enabled {
        if !(cfg.mcts.chance_pw.c.is_finite() && cfg.mcts.chance_pw.c > 0.0) {
            return Err("mcts.chance_pw.c must be finite and > 0 when enabled".to_string());
        }
        let a = cfg.mcts.chance_pw.alpha;
        if !(a.is_finite() && a > 0.0 && a < 1.0) {
            return Err("mcts.chance_pw.alpha must be in (0,1) when enabled".to_string());
        }
        if cfg.mcts.chance_pw.max_children < 1 {
            return Err("mcts.chance_pw.max_children must be >= 1 when enabled".to_string());
        }
    }

    // model
    match cfg.model.kind.as_str() {
        "residual" | "mlp" => {}
        _ => return Err("model.kind must be 'residual' or 'mlp'".to_string()),
    }

    // selfplay
    if cfg.selfplay.games_per_iteration < 1 {
        return Err("selfplay.games_per_iteration must be >= 1".to_string());
    }
    if cfg.selfplay.workers < 1 {
        return Err("selfplay.workers must be >= 1".to_string());
    }
    if cfg.selfplay.threads_per_worker < 1 {
        return Err("selfplay.threads_per_worker must be >= 1".to_string());
    }

    // training
    if cfg.training.batch_size < 1 {
        return Err("training.batch_size must be >= 1".to_string());
    }
    if cfg.training.learning_rate <= 0.0 {
        return Err("training.learning_rate must be > 0".to_string());
    }
    match cfg.training.optimizer.as_str() {
        "adamw" | "adam" | "sgd" => {}
        _ => return Err("training.optimizer must be 'adamw', 'adam', or 'sgd'".to_string()),
    }
    if cfg.training.weight_decay < 0.0 {
        return Err("training.weight_decay must be >= 0".to_string());
    }
    if cfg.training.epochs < 1 {
        return Err("training.epochs must be >= 1".to_string());
    }
    if let Some(steps) = cfg.training.steps_per_iteration {
        if steps < 1 {
            return Err("training.steps_per_iteration must be >= 1 when set".to_string());
        }
    }

    // gating
    if cfg.gating.games < 1 {
        return Err("gating.games must be >= 1".to_string());
    }
    if !(0.0..=1.0).contains(&cfg.gating.win_rate_threshold) {
        return Err("gating.win_rate_threshold must be in [0,1]".to_string());
    }
    if cfg.gating.katago.sprt {
        let min_games = cfg.gating.katago.sprt_min_games;
        let max_games = cfg.gating.katago.sprt_max_games;
        if min_games < 1 {
            return Err("gating.katago.sprt_min_games must be >= 1".to_string());
        }
        if max_games < min_games {
            return Err("gating.katago.sprt_max_games must be >= sprt_min_games".to_string());
        }
        if cfg.gating.paired_seed_swap {
            if min_games % 2 != 0 {
                return Err(
                    "gating.katago.sprt_min_games must be even when gating.paired_seed_swap=true"
                        .to_string(),
                );
            }
            if max_games % 2 != 0 {
                return Err(
                    "gating.katago.sprt_max_games must be even when gating.paired_seed_swap=true"
                        .to_string(),
                );
            }
        }
        let alpha = cfg.gating.katago.sprt_alpha;
        let beta = cfg.gating.katago.sprt_beta;
        if !(0.0 < alpha && alpha < 1.0) {
            return Err("gating.katago.sprt_alpha must be in (0,1)".to_string());
        }
        if !(0.0 < beta && beta < 1.0) {
            return Err("gating.katago.sprt_beta must be in (0,1)".to_string());
        }
        let delta = cfg.gating.katago.sprt_delta;
        if !(0.0 < delta && delta < 1.0) {
            return Err("gating.katago.sprt_delta must be in (0,1)".to_string());
        }
        let thr = cfg.gating.win_rate_threshold;
        let p0 = thr - delta;
        let p1 = thr + delta;
        if !(p0 > 0.0 && p1 < 1.0 && p0 < p1) {
            return Err(
                "gating.katago.sprt requires p0=thr-delta and p1=thr+delta with 0<p0<p1<1"
                    .to_string(),
            );
        }
    }

    // replay
    if let Some(n) = cfg.replay.capacity_shards {
        if n < 1 {
            return Err("replay.capacity_shards must be >= 1 when set".to_string());
        }
    }

    // controller
    if let Some(n) = cfg.controller.total_iterations {
        if n < 1 {
            return Err("controller.total_iterations must be >= 1 when set".to_string());
        }
    }

    // gating.fixed_oracle
    if cfg.gating.fixed_oracle.enabled {
        let id = cfg
            .gating
            .fixed_oracle
            .set_id
            .as_deref()
            .unwrap_or("")
            .trim()
            .to_string();
        if id.is_empty() {
            return Err(
                "gating.fixed_oracle.set_id must be set when gating.fixed_oracle.enabled=true"
                    .to_string(),
            );
        }
        if let Some(n) = cfg.gating.fixed_oracle.n {
            if n < 1 {
                return Err("gating.fixed_oracle.n must be >= 1 when set".to_string());
            }
        }
    }

    Ok(())
}
