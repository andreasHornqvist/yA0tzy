//! yz-infer: Socket protocol + inference client for Rust ↔ Python communication.

pub mod codec;
pub mod frame;
pub mod protocol;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    use super::*;
    use crate::codec::{
        decode_request_v1, decode_response_v1, encode_request_v1, encode_response_v1,
    };
    use crate::frame::{read_frame, write_frame};
    use crate::protocol::{
        InferRequestV1, InferResponseV1, ACTION_SPACE_A, FEATURE_LEN_V1, FEATURE_SCHEMA_ID_V1,
    };

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn roundtrip_over_tcp_dummy_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server = thread::spawn(move || {
            let (mut sock, _peer) = listener.accept().unwrap();
            let payload = read_frame(&mut sock).unwrap();
            let req = decode_request_v1(&payload).unwrap();

            // Dummy response: logits = 0.0 for legal, -1e9 for illegal, value=0.
            let mut logits = vec![0.0f32; ACTION_SPACE_A as usize];
            for (i, &b) in req.legal_mask.iter().enumerate() {
                if b == 0 {
                    logits[i] = -1.0e9;
                }
            }
            let resp = InferResponseV1 {
                request_id: req.request_id,
                policy_logits: logits,
                value: 0.0,
                margin: None,
            };
            let resp_payload = encode_response_v1(&resp);
            write_frame(&mut sock, &resp_payload).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();

        let req = InferRequestV1 {
            request_id: 42,
            model_id: 7,
            feature_schema_id: FEATURE_SCHEMA_ID_V1,
            features: vec![0.0; FEATURE_LEN_V1 as usize],
            legal_mask: vec![1u8; ACTION_SPACE_A as usize],
        };
        let payload = encode_request_v1(&req);
        write_frame(&mut client, &payload).unwrap();

        let resp_payload = read_frame(&mut client).unwrap();
        let resp = decode_response_v1(&resp_payload).unwrap();

        assert_eq!(resp.request_id, req.request_id);
        assert_eq!(resp.policy_logits.len(), ACTION_SPACE_A as usize);
        assert_eq!(resp.value, 0.0);
        assert!(resp.margin.is_none());

        server.join().unwrap();
    }
}
