use std::path::{Path, PathBuf};

use yz_core::Config;

pub const CONFIG_DRAFT_NAME: &str = "config.draft.yaml";
pub const DEFAULT_CONFIG_PATH: &str = "configs/local_cpu.yaml";

pub fn default_cfg_for_new_run() -> Config {
    let mut cfg = load_default_config();

    // Sane perf defaults for the local machine (TUI convenience).
    // - Use multiple OS worker processes (controller now spawns processes)
    // - Keep per-game inflight <= 8 (user preference)
    // - Default to 400 sims for both mark + reroll
    cfg.mcts.budget_reroll = 400;
    cfg.mcts.budget_mark = 400;
    cfg.mcts.max_inflight_per_game = 8;

    // Use most cores, but leave some headroom for the Python inference server + OS.
    // On this machine `available_parallelism()` is reliable and cross-platform.
    if let Ok(n) = std::thread::available_parallelism() {
        let logical = n.get() as u32;
        let workers = logical.saturating_sub(2).max(1);
        cfg.selfplay.workers = workers;
        // Start with a small number of concurrent games per worker process.
        cfg.selfplay.threads_per_worker = if logical >= 8 { 2 } else { 1 };
        // Sane torch thread defaults: estimate physical cores (~logical/2), min 4.
        // torch_interop_threads = 2 is generally a good default for inference.
        if cfg.inference.torch_threads.is_none() {
            cfg.inference.torch_threads = Some((logical / 2).max(4));
        }
        if cfg.inference.torch_interop_threads.is_none() {
            cfg.inference.torch_interop_threads = Some(2);
        }
    } else {
        // Fallback if parallelism detection fails.
        if cfg.inference.torch_threads.is_none() {
            cfg.inference.torch_threads = Some(4);
        }
        if cfg.inference.torch_interop_threads.is_none() {
            cfg.inference.torch_interop_threads = Some(2);
        }
    }
    cfg
}

pub fn draft_path(run_dir: &Path) -> PathBuf {
    run_dir.join(CONFIG_DRAFT_NAME)
}

pub fn snapshot_path(run_dir: &Path) -> PathBuf {
    run_dir.join("config.yaml")
}

pub fn load_cfg_for_run(run_dir: &Path) -> (Config, Option<String>) {
    let d = draft_path(run_dir);
    if d.exists() {
        match Config::load(&d) {
            Ok(cfg) => return (cfg, Some(format!("Loaded {}", CONFIG_DRAFT_NAME))),
            Err(e) => {
                return (
                    load_default_config(),
                    Some(format!("Failed to load {}: {e}", CONFIG_DRAFT_NAME)),
                );
            }
        }
    }
    let s = snapshot_path(run_dir);
    if s.exists() {
        match Config::load(&s) {
            Ok(cfg) => return (cfg, Some("Loaded config.yaml".to_string())),
            Err(e) => {
                return (
                    load_default_config(),
                    Some(format!("Failed to load config.yaml: {e}")),
                )
            }
        }
    }
    // For new runs, try to load from configs/local_cpu.yaml as the default template.
    let default_cfg = default_cfg_for_new_run();
    let msg = if Path::new(DEFAULT_CONFIG_PATH).exists() {
        format!("Loaded {}", DEFAULT_CONFIG_PATH)
    } else {
        "Using built-in defaults".to_string()
    };
    (default_cfg, Some(msg))
}

/// Load the default config from configs/local_cpu.yaml, falling back to built-in defaults.
fn load_default_config() -> Config {
    Config::load(DEFAULT_CONFIG_PATH).unwrap_or_default()
}

pub fn save_cfg_draft_atomic(run_dir: &Path, cfg: &Config) -> Result<(), std::io::Error> {
    std::fs::create_dir_all(run_dir)?;
    let path = draft_path(run_dir);
    let tmp = run_dir.join("config.draft.yaml.tmp");
    let s = serde_yaml::to_string(cfg).map_err(std::io::Error::other)?;
    std::fs::write(&tmp, s)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let mut cfg = Config::default();
        cfg.selfplay.games_per_iteration = 123;
        save_cfg_draft_atomic(run_dir, &cfg).unwrap();
        let (cfg2, _) = load_cfg_for_run(run_dir);
        assert_eq!(cfg2.selfplay.games_per_iteration, 123);
    }
}
