use yz_core::config::TemperatureSchedule;
use yz_core::Config;

pub fn validate_config(cfg: &Config) -> Result<(), String> {
    // inference
    if cfg.inference.bind.trim().is_empty() {
        return Err("inference.bind must be non-empty".to_string());
    }
    if cfg.inference.device != "cpu" && cfg.inference.device != "cuda" {
        return Err("inference.device must be 'cpu' or 'cuda'".to_string());
    }
    if cfg.inference.max_batch < 1 {
        return Err("inference.max_batch must be >= 1".to_string());
    }
    if cfg.inference.max_wait_us < 1 {
        return Err("inference.max_wait_us must be >= 1".to_string());
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
        TemperatureSchedule::Step { t0, t1, cutoff_ply: _ } => {
            if t0 < 0.0 || t1 < 0.0 {
                return Err("mcts.temperature_schedule.t0/t1 must be >= 0".to_string());
            }
        }
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

    Ok(())
}


