use std::fs;

use crate::{ReplaySample, ShardWriter, ShardWriterConfig};
use safetensors::SafeTensors;

#[test]
fn shard_writer_writes_expected_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("replay");
    let mut w = ShardWriter::new(ShardWriterConfig {
        out_dir: out.clone(),
        max_samples_per_shard: 2,
        git_hash: None,
        config_hash: None,
    })
    .unwrap();

    let s = ReplaySample {
        features: [0.0; yz_features::schema::F],
        legal_mask: [1u8; yz_core::A],
        pi: [1.0 / (yz_core::A as f32); yz_core::A],
        z: 1.0,
        z_margin: None,
    };
    w.push(s.clone()).unwrap();
    w.push(s).unwrap(); // triggers flush

    let st_path = out.join("shard_000000.safetensors");
    let meta_path = out.join("shard_000000.meta.json");
    assert!(st_path.exists());
    assert!(meta_path.exists());

    let bytes = fs::read(&st_path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let t_features = st.tensor("features").unwrap();
    assert_eq!(t_features.shape(), &[2, yz_features::schema::F]);

    let t_legal = st.tensor("legal_mask").unwrap();
    assert_eq!(t_legal.shape(), &[2, yz_core::A]);
}

#[test]
fn shard_writer_resumes_index_in_existing_dir() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("replay");
    fs::create_dir_all(&out).unwrap();

    // Create a fake previous shard pair at idx=0.
    fs::write(out.join("shard_000000.safetensors"), b"stub").unwrap();
    fs::write(out.join("shard_000000.meta.json"), b"{}").unwrap();

    let mut w = ShardWriter::new(ShardWriterConfig {
        out_dir: out.clone(),
        max_samples_per_shard: 1,
        git_hash: None,
        config_hash: None,
    })
    .unwrap();

    let s = ReplaySample {
        features: [0.0; yz_features::schema::F],
        legal_mask: [1u8; yz_core::A],
        pi: [1.0 / (yz_core::A as f32); yz_core::A],
        z: 1.0,
        z_margin: None,
    };
    w.push(s).unwrap(); // triggers flush

    // Should not overwrite idx=0; should create idx=1.
    assert!(out.join("shard_000001.safetensors").exists());
    assert!(out.join("shard_000001.meta.json").exists());
}

#[test]
fn prune_keeps_newest_by_idx_and_deletes_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("replay");
    fs::create_dir_all(&out).unwrap();

    // Create 5 shards (pairs).
    for idx in 0u64..5u64 {
        fs::write(out.join(format!("shard_{idx:06}.safetensors")), b"stub").unwrap();
        fs::write(out.join(format!("shard_{idx:06}.meta.json")), b"{}").unwrap();
    }

    let rep = crate::prune_shards_by_idx(&out, 2).unwrap();
    assert_eq!(rep.before_shards, 5);
    assert_eq!(rep.after_shards, 2);
    assert_eq!(rep.deleted_shards, 3);
    assert_eq!(rep.deleted_min_idx, Some(0));
    assert_eq!(rep.deleted_max_idx, Some(2));

    // Oldest 0..2 deleted, newest 3..4 remain.
    for idx in 0u64..3u64 {
        assert!(!out.join(format!("shard_{idx:06}.safetensors")).exists());
        assert!(!out.join(format!("shard_{idx:06}.meta.json")).exists());
    }
    for idx in 3u64..5u64 {
        assert!(out.join(format!("shard_{idx:06}.safetensors")).exists());
        assert!(out.join(format!("shard_{idx:06}.meta.json")).exists());
    }
}
