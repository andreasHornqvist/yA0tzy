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
