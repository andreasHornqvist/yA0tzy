//! Inference interface (stub for Epic E4).

use yz_core::{LegalMask, A};

/// Minimal inference interface for E4.
///
/// - `policy_logits[a]` are unnormalized logits for the fixed action space A.
/// - `value` is in [-1,1] from the POV of the encoded player-to-move.
pub trait Inference {
    fn eval(&self, features: &[f32], legal: LegalMask) -> ([f32; A], f32);
}

/// Uniform policy + zero value (baseline stub).
pub struct UniformInference;

impl Inference for UniformInference {
    fn eval(&self, _features: &[f32], legal: LegalMask) -> ([f32; A], f32) {
        let mut logits = [0.0f32; A];
        // Keep illegal actions at a low logit; legal actions get 0.
        for i in 0..A {
            if ((legal >> i) & 1) == 0 {
                logits[i] = -1.0e9;
            }
        }
        (logits, 0.0)
    }
}
