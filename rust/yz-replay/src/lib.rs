//! yz-replay: Safetensors shard writers + readers for replay buffers.

pub mod schema;
pub mod writer;

pub use writer::{
    cleanup_tmp_files, ReplayError, ReplaySample, ShardMeta, ShardWriter, ShardWriterConfig,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}

#[cfg(test)]
mod writer_tests;
