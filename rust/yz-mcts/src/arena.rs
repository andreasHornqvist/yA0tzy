//! Arena-backed node storage.

use crate::node::{Node, NodeId};

pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn push(&mut self, n: Node) -> NodeId {
        let id = self.nodes.len() as u32;
        self.nodes.push(n);
        id
    }

    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id as usize]
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}
