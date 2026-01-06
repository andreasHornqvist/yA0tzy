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

    /// Replay retention settings.
    #[serde(default)]
    pub replay: ReplayConfig,

    /// Iteration controller / orchestration settings.
    #[serde(default)]
    pub controller: ControllerConfig,

    /// Neural network model architecture settings.
    #[serde(default)]
    pub model: ModelConfig,
}

/// Neural network model architecture configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Hidden layer size for the neural network.
    #[serde(default = "default_model_hidden_dim")]
    pub hidden_dim: u32,
    /// Number of residual blocks in the network.
    #[serde(default = "default_model_num_blocks")]
    pub num_blocks: u32,
}

fn default_model_hidden_dim() -> u32 {
    256
}

fn default_model_num_blocks() -> u32 {
    2
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: default_model_hidden_dim(),
            num_blocks: default_model_num_blocks(),
        }
    }
}

/// Inference server configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    /// Bind address (e.g., "unix:///tmp/yatzy_infer.sock" or "tcp://0.0.0.0:9000").
    pub bind: String,
    /// Device to run inference on ("cpu", "mps", or "cuda").
    pub device: String,
    /// Protocol version to use for the Rust↔Python inference protocol.
    ///
    /// - 1: legacy v1 (float lists / per-f32 codec)
    /// - 2: packed f32 tensors (E6.6 Option 2)
    #[serde(default = "default_inference_protocol_version")]
    pub protocol_version: u32,
    /// If true and protocol_version==2, encode legal_mask as a compact 6-byte bitset (A=47).
    #[serde(default)]
    pub legal_mask_bitset: bool,
    /// Maximum batch size before flushing.
    pub max_batch: u32,
    /// Maximum wait time in microseconds before flushing a partial batch.
    pub max_wait_us: u64,
    /// Optional: torch intra-op threads (Python server CPU perf stability).
    #[serde(default)]
    pub torch_threads: Option<u32>,
    /// Optional: torch inter-op threads (Python server CPU perf stability).
    #[serde(default)]
    pub torch_interop_threads: Option<u32>,
    /// If true, enable debug logging across Rust/Python components for this run.
    ///
    /// This maps to setting `YZ_DEBUG_LOG=1` for spawned subprocesses.
    #[serde(default)]
    pub debug_log: bool,
    /// If true, make infer-server print periodic throughput/batching stats.
    ///
    /// This maps to setting `YZ_INFER_PRINT_STATS=1` and enables
    /// `--print-stats-every-s` on the infer-server process.
    #[serde(default)]
    pub print_stats: bool,
    /// Metrics/control HTTP bind address (e.g., "127.0.0.1:18080").
    /// Used for hot-reloading models (E13.2S4).
    #[serde(default = "default_metrics_bind")]
    pub metrics_bind: String,
}

fn default_metrics_bind() -> String {
    "127.0.0.1:18080".to_string()
}

fn default_inference_protocol_version() -> u32 {
    2
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
    /// Weight decay (L2), applied by AdamW/SGD.
    #[serde(default)]
    pub weight_decay: f64,
    /// Number of training epochs per iteration.
    pub epochs: u32,
    /// Optional number of optimizer steps per iteration (takes precedence over epochs).
    #[serde(default)]
    pub steps_per_iteration: Option<u32>,
    /// Replay sampling mode for training.
    ///
    /// - "sequential": stream samples in shard order (less random, more correlated).
    /// - "random_indexed": build a global index and let DataLoader shuffle indices (recommended).
    #[serde(default = "default_training_sample_mode")]
    pub sample_mode: String,
    /// Torch DataLoader worker processes used by the trainer (0 disables multiprocessing).
    #[serde(default)]
    pub dataloader_workers: u32,
}

fn default_training_sample_mode() -> String {
    "random_indexed".to_string()
}

/// Replay retention configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ReplayConfig {
    /// Keep at most N shards under runs/<id>/replay/ (prune older beyond capacity).
    ///
    /// If None, do not prune automatically.
    #[serde(default)]
    pub capacity_shards: Option<u32>,
}

/// Iteration controller configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ControllerConfig {
    /// Optional number of full iterations to run (selfplay → train → gate).
    ///
    /// If None, controller runs until stopped externally.
    #[serde(default)]
    pub total_iterations: Option<u32>,
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
    /// Optional: number of parallel in-flight games per gate-worker process.
    ///
    /// If None, gating will derive a reasonable default from `selfplay.threads_per_worker`,
    /// typically scaling by 2 because gating uses two model_id streams (best + candidate).
    #[serde(default)]
    pub threads_per_worker: Option<u32>,
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

impl Default for Config {
    fn default() -> Self {
        Self {
            inference: InferenceConfig {
                bind: "unix:///tmp/yatzy_infer.sock".to_string(),
                device: "cpu".to_string(),
                protocol_version: default_inference_protocol_version(),
                legal_mask_bitset: false,
                max_batch: 32,
                max_wait_us: 1000,
                torch_threads: None,
                torch_interop_threads: None,
                debug_log: false,
                print_stats: false,
                metrics_bind: default_metrics_bind(),
            },
            mcts: MctsConfig {
                c_puct: 1.5,
                // Perf default: 400 sims for both decision types (matches common benchmarking target).
                budget_reroll: 400,
                budget_mark: 400,
                // Per-game leaf eval concurrency cap (keeps “leaf batching per game” <= 8).
                max_inflight_per_game: 8,
                dirichlet_alpha: default_dirichlet_alpha(),
                dirichlet_epsilon: default_dirichlet_epsilon(),
                temperature_schedule: TemperatureSchedule::default(),
            },
            selfplay: SelfplayConfig {
                games_per_iteration: 50,
                workers: 1,
                threads_per_worker: 1,
            },
            training: TrainingConfig {
                batch_size: 256,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                epochs: 1,
                steps_per_iteration: None,
                sample_mode: default_training_sample_mode(),
                dataloader_workers: 0,
            },
            gating: GatingConfig {
                games: 50,
                seed: 0,
                seed_set_id: default_gating_seed_set_id(),
                win_rate_threshold: default_gating_win_rate_threshold(),
                paired_seed_swap: true,
                deterministic_chance: default_gating_deterministic_chance(),
                threads_per_worker: None,
            },
            replay: ReplayConfig::default(),
            controller: ControllerConfig::default(),
            model: ModelConfig::default(),
        }
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
        assert_eq!(config.mcts.budget_reroll, 400);
        assert_eq!(config.selfplay.workers, 4);
        assert_eq!(config.training.batch_size, 256);
        assert!((config.training.weight_decay - 0.0001).abs() < 1e-9);
        assert_eq!(config.training.steps_per_iteration, None);
        assert_eq!(config.gating.games, 100);
        assert_eq!(config.gating.seed, 0);
        assert_eq!(config.gating.seed_set_id.as_deref(), Some("dev_v1"));
        assert!(config.gating.paired_seed_swap);
        assert!(config.gating.deterministic_chance);
        assert_eq!(config.gating.threads_per_worker, None);
        assert_eq!(config.replay.capacity_shards, Some(20));
        assert_eq!(config.controller.total_iterations, Some(10));
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
