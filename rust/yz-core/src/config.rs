//! Unified configuration schema for yA0tzy.
//!
//! This module defines the configuration structure that is shared between
//! Rust and Python components. The same YAML file should load in both.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Configuration loading errors.
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// Root configuration structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Inference server settings.
    pub inference: InferenceConfig,
    /// MCTS algorithm settings.
    pub mcts: MctsConfig,
    /// Self-play settings.
    pub selfplay: SelfplayConfig,
    /// Training settings.
    pub training: TrainingConfig,
    /// Gating evaluation settings.
    pub gating: GatingConfig,
}

/// Inference server configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    /// Bind address (e.g., "unix:///tmp/yatzy_infer.sock" or "tcp://0.0.0.0:9000").
    pub bind: String,
    /// Device to run inference on ("cpu" or "cuda").
    pub device: String,
    /// Maximum batch size before flushing.
    pub max_batch: u32,
    /// Maximum wait time in microseconds before flushing a partial batch.
    pub max_wait_us: u64,
}

/// MCTS algorithm configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MctsConfig {
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Simulation budget for reroll decisions.
    pub budget_reroll: u32,
    /// Simulation budget for mark (category selection) decisions.
    pub budget_mark: u32,
    /// Maximum in-flight inference requests per game.
    pub max_inflight_per_game: u32,
    /// Dirichlet noise alpha (only used in self-play, not gating).
    #[serde(default = "default_dirichlet_alpha")]
    pub dirichlet_alpha: f32,
    /// Dirichlet noise epsilon - fraction of noise to mix in (only used in self-play).
    #[serde(default = "default_dirichlet_epsilon")]
    pub dirichlet_epsilon: f32,

    /// Temperature schedule for executed-move sampling (PRD §7.3 / Epic E4S2).
    ///
    /// Note: temperature never changes replay `pi` targets (visit-count distribution).
    #[serde(default)]
    pub temperature_schedule: TemperatureSchedule,
}

fn default_dirichlet_alpha() -> f32 {
    0.3
}

fn default_dirichlet_epsilon() -> f32 {
    0.25
}

/// Executed-move temperature schedule.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemperatureSchedule {
    /// Constant temperature `t0`.
    Constant { t0: f32 },
    /// Step schedule: use `t0` while `ply < cutoff_ply`, else use `t1`.
    Step { t0: f32, t1: f32, cutoff_ply: u32 },
}

impl Default for TemperatureSchedule {
    fn default() -> Self {
        TemperatureSchedule::Constant { t0: 1.0 }
    }
}

/// Self-play configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SelfplayConfig {
    /// Number of games to play per iteration.
    pub games_per_iteration: u32,
    /// Number of worker processes.
    pub workers: u32,
    /// Number of threads per worker.
    pub threads_per_worker: u32,
}

/// Training configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    /// Batch size for training.
    pub batch_size: u32,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of training epochs per iteration.
    pub epochs: u32,
}

/// Gating (model evaluation) configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GatingConfig {
    /// Number of games to play for gating evaluation.
    pub games: u32,
    /// Base seed for deterministic paired-seed scheduling in gating (E9.1).
    ///
    /// When `paired_seed_swap=true`, the evaluator will deterministically derive per-pair
    /// episode seeds from this base value.
    #[serde(default)]
    pub seed: u64,
    /// Optional fixed dev seed set id (E9.2).
    ///
    /// If set, gating will load `configs/seed_sets/<id>.txt` and schedule games from that list.
    /// This is useful for reproducible evaluation sets. When present, `seed` is ignored for
    /// seed selection.
    #[serde(default = "default_gating_seed_set_id")]
    pub seed_set_id: Option<String>,
    /// Win rate threshold for promotion (e.g., 0.55 = 55%).
    #[serde(default = "default_gating_win_rate_threshold")]
    pub win_rate_threshold: f64,
    /// Whether to use paired seed + side swap for reduced variance.
    pub paired_seed_swap: bool,
    /// Whether to use deterministic event-keyed chance for gating/eval (optional).
    #[serde(default = "default_gating_deterministic_chance")]
    pub deterministic_chance: bool,
}

fn default_gating_deterministic_chance() -> bool {
    true
}

fn default_gating_win_rate_threshold() -> f64 {
    0.55
}

fn default_gating_seed_set_id() -> Option<String> {
    Some("dev_v1".to_string())
}

impl Config {
    /// Load configuration from a YAML file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    /// Load configuration from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, ConfigError> {
        let config: Config = serde_yaml::from_str(yaml)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_local_cpu_yaml() {
        // Load the actual config file from the repo
        let config = Config::load("../../configs/local_cpu.yaml")
            .expect("Failed to load configs/local_cpu.yaml");

        // Verify some expected values
        assert_eq!(config.inference.device, "cpu");
        assert_eq!(config.inference.max_batch, 32);
        assert_eq!(config.mcts.budget_reroll, 100);
        assert_eq!(config.selfplay.workers, 4);
        assert_eq!(config.training.batch_size, 256);
        assert_eq!(config.gating.games, 100);
        assert_eq!(config.gating.seed, 0);
        assert_eq!(config.gating.seed_set_id.as_deref(), Some("dev_v1"));
        assert!(config.gating.paired_seed_swap);
        assert!(config.gating.deterministic_chance);
    }

    #[test]
    fn test_parse_yaml_string() {
        let yaml = r#"
inference:
  bind: "unix:///tmp/test.sock"
  device: "cpu"
  max_batch: 16
  max_wait_us: 500

mcts:
  c_puct: 1.0
  budget_reroll: 50
  budget_mark: 100
  max_inflight_per_game: 2

selfplay:
  games_per_iteration: 10
  workers: 1
  threads_per_worker: 1

training:
  batch_size: 32
  learning_rate: 0.01
  epochs: 5

gating:
  games: 20
  win_rate_threshold: 0.5
  paired_seed_swap: false
"#;

        let config = Config::from_yaml(yaml).expect("Failed to parse YAML");
        assert_eq!(config.inference.max_batch, 16);
        assert_eq!(config.mcts.c_puct, 1.0);
        // Check defaults are applied
        assert_eq!(config.mcts.dirichlet_alpha, 0.3);
        assert_eq!(config.mcts.dirichlet_epsilon, 0.25);
    }

    #[test]
    fn test_invalid_yaml_fails() {
        let invalid_yaml = "this is not: valid: yaml: {{{}}}";
        let result = Config::from_yaml(invalid_yaml);
        assert!(result.is_err());
    }
}
