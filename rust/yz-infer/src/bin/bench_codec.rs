//! Microbench for yz-infer codec v1 vs v2 (no criterion).
//!
//! Usage:
//!   cargo run -p yz-infer --bin bench_codec --release -- --n 2000000
use std::time::Instant;
use yz_infer::codec::{
    decode_response_any, decode_response_v1, encode_request_v1_into, encode_request_v2_into,
    encode_response_v1, encode_response_v2,
};
use yz_infer::protocol::{InferRequestV1, InferResponseV1, ACTION_SPACE_A, FEATURE_LEN_V1, FEATURE_SCHEMA_ID_V1};

fn main() {
    let mut n: u64 = 2_000_000;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--n" {
            if let Some(v) = args.next() {
                n = v.parse().unwrap_or(n);
            }
        }
    }

    let req = InferRequestV1 {
        request_id: 1,
        model_id: 0,
        feature_schema_id: FEATURE_SCHEMA_ID_V1,
        features: vec![0.0; FEATURE_LEN_V1 as usize],
        legal_mask: vec![1u8; ACTION_SPACE_A as usize],
    };
    let resp = InferResponseV1 {
        request_id: 1,
        policy_logits: vec![0.1; ACTION_SPACE_A as usize],
        value: 0.25,
        margin: None,
    };

    println!("n={n}");

    // Encode request v1
    let mut out = Vec::with_capacity(512);
    let t0 = Instant::now();
    for _ in 0..n {
    encode_request_v1_into(&mut out, &req);
        std::hint::black_box(&out);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("encode_request_v1: {:>10.0} it/s  {:>8.3} ns/it", (n as f64) / dt, (dt * 1e9) / (n as f64));

    // Encode request v2
    let mut out2 = Vec::with_capacity(512);
    let t0 = Instant::now();
    for _ in 0..n {
        encode_request_v2_into(&mut out2, &req, false);
        std::hint::black_box(&out2);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("encode_request_v2: {:>10.0} it/s  {:>8.3} ns/it", (n as f64) / dt, (dt * 1e9) / (n as f64));

    // Encode responses
    let p1 = encode_response_v1(&resp);
    let p2 = encode_response_v2(&resp);

    // Decode response v1
    let t0 = Instant::now();
    for _ in 0..n {
        let r = decode_response_v1(&p1).unwrap();
        std::hint::black_box(r);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("decode_response_v1: {:>10.0} it/s  {:>8.3} ns/it", (n as f64) / dt, (dt * 1e9) / (n as f64));

    // Decode response v2 via version router
    let t0 = Instant::now();
    for _ in 0..n {
        let r = decode_response_any(&p2).unwrap();
        std::hint::black_box(r);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("decode_response_v2: {:>10.0} it/s  {:>8.3} ns/it", (n as f64) / dt, (dt * 1e9) / (n as f64));
}


