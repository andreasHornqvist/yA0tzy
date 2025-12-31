use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2eBenchJsonV1 {
    pub event: String, // "bench_e2e_v1"
    pub wall_ms: u64,

    pub games_completed: u64,
    pub executed_moves: u64,
    pub simulations_per_move: u32,

    pub sims_per_sec: f64,
    pub evals_per_sec: f64,

    pub infer_sent: u64,
    pub infer_received: u64,
    pub infer_errors: u64,
    pub infer_inflight: u64,
    pub infer_latency_p50_us: u64,
    pub infer_latency_p95_us: u64,
    pub infer_latency_mean_us: f64,

    pub steps: u64,
    pub would_block: u64,
    pub terminal: u64,

    pub mcts_fallbacks: u64,
    pub mcts_pending_collisions: u64,
    pub mcts_pending_count_max: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_shape() {
        let r = E2eBenchJsonV1 {
            event: "bench_e2e_v1".to_string(),
            wall_ms: 123,
            games_completed: 4,
            executed_moves: 50,
            simulations_per_move: 64,
            sims_per_sec: 1.0,
            evals_per_sec: 2.0,
            infer_sent: 3,
            infer_received: 4,
            infer_errors: 0,
            infer_inflight: 1,
            infer_latency_p50_us: 10,
            infer_latency_p95_us: 20,
            infer_latency_mean_us: 15.0,
            steps: 100,
            would_block: 5,
            terminal: 2,
            mcts_fallbacks: 0,
            mcts_pending_collisions: 1,
            mcts_pending_count_max: 3,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: E2eBenchJsonV1 = serde_json::from_str(&s).unwrap();
        assert_eq!(back.event, "bench_e2e_v1");
        assert_eq!(back.games_completed, 4);
    }
}
