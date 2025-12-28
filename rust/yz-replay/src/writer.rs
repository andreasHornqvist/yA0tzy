use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bytemuck::cast_slice;
use safetensors::tensor::{Dtype, TensorView};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use yz_core::A;
use yz_features::schema::F;

use crate::schema::{
    ACTION_SPACE_ID, PROTOCOL_VERSION, RULESET_ID, T_FEATURES, T_LEGAL_MASK, T_PI, T_Z,
};

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("invalid sample: {0}")]
    InvalidSample(&'static str),
}

/// One training decision sample (PRD §10.1).
#[derive(Clone, Debug)]
pub struct ReplaySample {
    pub features: [f32; F],
    pub legal_mask: [u8; A],
    pub pi: [f32; A],
    pub z: f32,
    pub z_margin: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMeta {
    pub protocol_version: u32,
    pub feature_schema_id: u32,
    pub feature_len: usize,
    pub action_space_id: String,
    pub action_space_a: usize,
    pub ruleset_id: String,

    pub num_samples: usize,

    pub git_hash: Option<String>,
    pub config_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ShardWriterConfig {
    pub out_dir: PathBuf,
    pub max_samples_per_shard: usize,
    pub git_hash: Option<String>,
    pub config_hash: Option<String>,
}

pub struct ShardWriter {
    cfg: ShardWriterConfig,
    shard_idx: u64,
    buf: Vec<ReplaySample>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PruneReport {
    pub before_shards: usize,
    pub after_shards: usize,
    pub deleted_shards: usize,
    pub deleted_min_idx: Option<u64>,
    pub deleted_max_idx: Option<u64>,
}

impl ShardWriter {
    pub fn new(cfg: ShardWriterConfig) -> Result<Self, ReplayError> {
        if cfg.max_samples_per_shard == 0 {
            return Err(ReplayError::InvalidSample(
                "max_samples_per_shard must be > 0",
            ));
        }
        fs::create_dir_all(&cfg.out_dir)?;

        // Best-effort cleanup from any prior crash.
        let _ = cleanup_tmp_files(&cfg.out_dir);

        // Resume shard index to avoid overwriting existing shards in the same run dir.
        let shard_idx = next_shard_idx(&cfg.out_dir)?;
        Ok(Self {
            cfg,
            shard_idx,
            buf: Vec::new(),
        })
    }

    pub fn push(&mut self, s: ReplaySample) -> Result<(), ReplayError> {
        self.buf.push(s);
        if self.buf.len() >= self.cfg.max_samples_per_shard {
            self.flush()?;
        }
        Ok(())
    }

    pub fn extend<I: IntoIterator<Item = ReplaySample>>(
        &mut self,
        it: I,
    ) -> Result<(), ReplayError> {
        for s in it {
            self.push(s)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), ReplayError> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let n = self.buf.len();

        // Flatten tensors.
        let mut features = Vec::<f32>::with_capacity(n * F);
        let mut legal = Vec::<u8>::with_capacity(n * A);
        let mut pi = Vec::<f32>::with_capacity(n * A);
        let mut z = Vec::<f32>::with_capacity(n);
        let mut have_margin = false;
        let mut z_margin = Vec::<f32>::with_capacity(n);

        for s in &self.buf {
            features.extend_from_slice(&s.features);
            legal.extend_from_slice(&s.legal_mask);
            pi.extend_from_slice(&s.pi);
            z.push(s.z);
            if let Some(m) = s.z_margin {
                have_margin = true;
                z_margin.push(m);
            } else {
                z_margin.push(0.0);
            }
        }

        let mut tensors: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        tensors.insert(
            T_FEATURES.to_string(),
            TensorView::new(Dtype::F32, vec![n, F], cast_slice(&features))?,
        );
        tensors.insert(
            T_LEGAL_MASK.to_string(),
            TensorView::new(Dtype::U8, vec![n, A], &legal)?,
        );
        tensors.insert(
            T_PI.to_string(),
            TensorView::new(Dtype::F32, vec![n, A], cast_slice(&pi))?,
        );
        tensors.insert(
            T_Z.to_string(),
            TensorView::new(Dtype::F32, vec![n], cast_slice(&z))?,
        );
        if have_margin {
            tensors.insert(
                crate::schema::T_Z_MARGIN.to_string(),
                TensorView::new(Dtype::F32, vec![n], cast_slice(&z_margin))?,
            );
        }

        let final_st = self.safetensors_path(self.shard_idx);
        let tmp_st = final_st.with_extension("safetensors.tmp");
        let final_meta = self.meta_path(self.shard_idx);
        let tmp_meta = final_meta.with_extension("meta.json.tmp");

        // Serialize safetensors to bytes then write atomically via rename.
        let st_bytes = safetensors::serialize(&tensors, &None)?;
        fs::write(&tmp_st, st_bytes)?;
        fs::rename(&tmp_st, &final_st)?;

        let meta = ShardMeta {
            protocol_version: PROTOCOL_VERSION,
            feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
            feature_len: F,
            action_space_id: ACTION_SPACE_ID.to_string(),
            action_space_a: A,
            ruleset_id: RULESET_ID.to_string(),
            num_samples: n,
            git_hash: self.cfg.git_hash.clone(),
            config_hash: self.cfg.config_hash.clone(),
        };
        fs::write(&tmp_meta, serde_json::to_vec_pretty(&meta)?)?;
        fs::rename(&tmp_meta, &final_meta)?;

        self.shard_idx += 1;
        self.buf.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), ReplayError> {
        self.flush()
    }

    fn safetensors_path(&self, idx: u64) -> PathBuf {
        self.cfg.out_dir.join(format!("shard_{idx:06}.safetensors"))
    }

    fn meta_path(&self, idx: u64) -> PathBuf {
        self.cfg.out_dir.join(format!("shard_{idx:06}.meta.json"))
    }
}

fn parse_shard_idx_from_name(name: &str) -> Option<u64> {
    // Expected: shard_{idx:06}.safetensors OR shard_{idx:06}.meta.json
    let rest = name.strip_prefix("shard_")?;
    let digits = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u64>().ok()
}

fn next_shard_idx(dir: &Path) -> Result<u64, ReplayError> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut max_idx: Option<u64> = None;
    for entry in fs::read_dir(dir)? {
        let e = entry?;
        let p = e.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !(name.ends_with(".safetensors") || name.ends_with(".meta.json")) {
            continue;
        }
        if let Some(idx) = parse_shard_idx_from_name(name) {
            max_idx = Some(max_idx.map(|m| m.max(idx)).unwrap_or(idx));
        }
    }
    Ok(max_idx.map(|m| m.saturating_add(1)).unwrap_or(0))
}

fn shard_paths(dir: &Path, idx: u64) -> (PathBuf, PathBuf) {
    (
        dir.join(format!("shard_{idx:06}.safetensors")),
        dir.join(format!("shard_{idx:06}.meta.json")),
    )
}

pub fn prune_shards_by_idx(dir: &Path, keep_n: usize) -> Result<PruneReport, ReplayError> {
    if keep_n == 0 {
        // Caller bug; treat as "keep none" without deleting anything to stay safe.
        return Ok(PruneReport {
            before_shards: 0,
            after_shards: 0,
            deleted_shards: 0,
            deleted_min_idx: None,
            deleted_max_idx: None,
        });
    }
    if !dir.exists() {
        return Ok(PruneReport {
            before_shards: 0,
            after_shards: 0,
            deleted_shards: 0,
            deleted_min_idx: None,
            deleted_max_idx: None,
        });
    }

    let mut idxs: Vec<u64> = Vec::new();
    for entry in fs::read_dir(dir)? {
        let e = entry?;
        let p = e.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !(name.ends_with(".safetensors") || name.ends_with(".meta.json")) {
            continue;
        }
        if let Some(idx) = parse_shard_idx_from_name(name) {
            idxs.push(idx);
        }
    }
    idxs.sort_unstable();
    idxs.dedup();

    let before = idxs.len();
    if before <= keep_n {
        return Ok(PruneReport {
            before_shards: before,
            after_shards: before,
            deleted_shards: 0,
            deleted_min_idx: None,
            deleted_max_idx: None,
        });
    }

    let delete_count = before - keep_n;
    let delete_slice = &idxs[..delete_count];
    let deleted_min_idx = delete_slice.first().copied();
    let deleted_max_idx = delete_slice.last().copied();

    for &idx in delete_slice {
        let (st, meta) = shard_paths(dir, idx);
        let _ = fs::remove_file(st);
        let _ = fs::remove_file(meta);
    }

    Ok(PruneReport {
        before_shards: before,
        after_shards: before - delete_count,
        deleted_shards: delete_count,
        deleted_min_idx,
        deleted_max_idx,
    })
}

pub fn cleanup_tmp_files(dir: &Path) -> Result<(), ReplayError> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let e = entry?;
        let p = e.path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with(".safetensors.tmp") || name.ends_with(".meta.json.tmp") {
                let _ = fs::remove_file(&p);
            }
        }
    }
    Ok(())
}
