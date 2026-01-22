//! Node and edge statistics for PUCT.

use crate::afterstate::AfterState;
use yz_core::A;

pub type NodeId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Decision,
    Chance,
}

#[derive(Clone)]
pub struct Node {
    pub kind: NodeKind,
    pub to_play: u8,
    pub is_terminal: bool,
    pub terminal_z: f32,
    pub is_expanded: bool,

    // Stats per action idx.
    pub n: [u32; A],
    pub w: [f32; A],
    pub p: [f32; A],
    pub n_sum: u32,

    // Virtual loss bookkeeping (in-flight leaf eval scaffolding).
    pub vl_n: [u32; A],
    pub vl_w: [f32; A],
    pub vl_sum: u32,

    // Chance-node aggregate stats (used only when kind==Chance).
    pub chance_visits: u32,
    pub chance_w_sum: f32,
    pub chance_num_children: u16,
    pub afterstate: Option<AfterState>,
}

impl Node {
    pub fn new_decision(to_play: u8) -> Self {
        Self {
            kind: NodeKind::Decision,
            to_play,
            is_terminal: false,
            terminal_z: 0.0,
            is_expanded: false,
            n: [0u32; A],
            w: [0.0f32; A],
            p: [0.0f32; A],
            n_sum: 0,
            vl_n: [0u32; A],
            vl_w: [0.0f32; A],
            vl_sum: 0,
            chance_visits: 0,
            chance_w_sum: 0.0,
            chance_num_children: 0,
            afterstate: None,
        }
    }

    pub fn new_chance(afterstate: AfterState) -> Self {
        Self {
            kind: NodeKind::Chance,
            to_play: afterstate.player_to_act,
            is_terminal: false,
            terminal_z: 0.0,
            // Chance nodes are never NN-expanded.
            is_expanded: true,
            n: [0u32; A],
            w: [0.0f32; A],
            p: [0.0f32; A],
            n_sum: 0,
            vl_n: [0u32; A],
            vl_w: [0.0f32; A],
            vl_sum: 0,
            chance_visits: 0,
            chance_w_sum: 0.0,
            chance_num_children: 0,
            afterstate: Some(afterstate),
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

    pub fn q_eff(&self, a: usize, use_virtual_loss: bool) -> f32 {
        if !use_virtual_loss {
            return self.q(a);
        }
        let n = self.n[a].saturating_add(self.vl_n[a]);
        if n == 0 {
            0.0
        } else {
            let w = self.w[a] - self.vl_w[a];
            w / (n as f32)
        }
    }
}
