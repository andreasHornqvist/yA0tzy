use criterion::{black_box, criterion_group, criterion_main, Criterion};

use yz_infer::codec::{
    decode_request_v1, decode_response_v1, encode_request_v1, encode_response_v1,
};
use yz_infer::protocol::{
    InferRequestV1, InferResponseV1, ACTION_SPACE_A, FEATURE_LEN_V1, FEATURE_SCHEMA_ID_V1,
};

fn make_request() -> InferRequestV1 {
    InferRequestV1 {
        request_id: 123,
        model_id: 0,
        feature_schema_id: FEATURE_SCHEMA_ID_V1,
        features: vec![0.1f32; FEATURE_LEN_V1 as usize],
        legal_mask: vec![1u8; ACTION_SPACE_A as usize],
    }
}

fn make_response() -> InferResponseV1 {
    InferResponseV1 {
        request_id: 123,
        policy_logits: vec![0.0f32; ACTION_SPACE_A as usize],
        value: 0.0,
        margin: None,
    }
}

fn bench_codec(c: &mut Criterion) {
    let req = make_request();
    let resp = make_response();

    c.bench_function("yz_infer_encode_request_v1", |b| {
        b.iter(|| black_box(encode_request_v1(black_box(&req))))
    });

    let req_bytes = encode_request_v1(&req);
    c.bench_function("yz_infer_decode_request_v1", |b| {
        b.iter(|| black_box(decode_request_v1(black_box(&req_bytes)).unwrap()))
    });

    c.bench_function("yz_infer_encode_response_v1", |b| {
        b.iter(|| black_box(encode_response_v1(black_box(&resp))))
    });

    let resp_bytes = encode_response_v1(&resp);
    c.bench_function("yz_infer_decode_response_v1", |b| {
        b.iter(|| black_box(decode_response_v1(black_box(&resp_bytes)).unwrap()))
    });
}

criterion_group!(benches, bench_codec);
criterion_main!(benches);
