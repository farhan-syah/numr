//! Integration tests for WebGPU integer dtype support (I32, U32).
//!
//! Split by operation shape: elementwise, broadcast, and scalar.

#![cfg(feature = "wgpu")]

mod wgpu_integer_ops;
