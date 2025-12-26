//! Length-delimited framing (u32 little-endian length prefix).

use std::io::{Read, Write};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("frame too large: {len} > {max}")]
    TooLarge { len: u32, max: u32 },
    #[error("unexpected EOF while reading frame")]
    UnexpectedEof,
}

pub const MAX_FRAME_LEN: u32 = 64 * 1024 * 1024; // 64 MiB guardrail

pub fn write_frame<W: Write>(w: &mut W, payload: &[u8]) -> Result<(), FrameError> {
    let len: u32 = payload.len().try_into().map_err(|_| FrameError::TooLarge {
        len: u32::MAX,
        max: MAX_FRAME_LEN,
    })?;
    if len > MAX_FRAME_LEN {
        return Err(FrameError::TooLarge {
            len,
            max: MAX_FRAME_LEN,
        });
    }
    w.write_all(&len.to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
}

pub fn read_frame<R: Read>(r: &mut R) -> Result<Vec<u8>, FrameError> {
    let mut len_bytes = [0u8; 4];
    read_exact_or_eof(r, &mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    if len > MAX_FRAME_LEN {
        return Err(FrameError::TooLarge {
            len,
            max: MAX_FRAME_LEN,
        });
    }
    let mut payload = vec![0u8; len as usize];
    read_exact_or_eof(r, &mut payload)?;
    Ok(payload)
}

fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), FrameError> {
    let mut off = 0usize;
    while off < buf.len() {
        match r.read(&mut buf[off..])? {
            0 => return Err(FrameError::UnexpectedEof),
            n => off += n,
        }
    }
    Ok(())
}
