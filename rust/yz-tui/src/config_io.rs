use std::path::{Path, PathBuf};

use yz_core::Config;

pub const CONFIG_DRAFT_NAME: &str = "config.draft.yaml";

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
                    Config::default(),
                    Some(format!("Failed to load {}: {e}", CONFIG_DRAFT_NAME)),
                );
            }
        }
    }
    let s = snapshot_path(run_dir);
    if s.exists() {
        match Config::load(&s) {
            Ok(cfg) => return (cfg, Some("Loaded config.yaml".to_string())),
            Err(e) => return (Config::default(), Some(format!("Failed to load config.yaml: {e}"))),
        }
    }
    (Config::default(), Some("Using Config::default()".to_string()))
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


