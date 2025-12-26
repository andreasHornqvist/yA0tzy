//! Node and edge statistics for PUCT.

use yz_core::A;

pub type NodeId = u32;

#[derive(Clone)]
pub struct Node {
    pub to_play: u8,
    pub is_terminal: bool,
    pub terminal_z: f32,
    pub is_expanded: bool,

    // Stats per action idx.
    pub n: [u32; A],
    pub w: [f32; A],
    pub p: [f32; A],
    pub n_sum: u32,
}

impl Node {
    pub fn new(to_play: u8) -> Self {
        Self {
            to_play,
            is_terminal: false,
            terminal_z: 0.0,
            is_expanded: false,
            n: [0u32; A],
            w: [0.0f32; A],
            p: [0.0f32; A],
            n_sum: 0,
        }
    }

    pub fn q(&self, a: usize) -> f32 {
        let n = self.n[a];
        if n == 0 {
            0.0
        } else {
            self.w[a] / (n as f32)
        }
    }
}
