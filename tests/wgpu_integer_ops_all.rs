//! Integration tests for WebGPU integer dtype support (I32, U32).
//!
//! Split by operation shape: elementwise, broadcast, and scalar.

#![cfg(feature = "wgpu")]

// `common` carries the WebGPU device lock every test here goes through.
// Concurrent device use loses the device and cascades into the rest of the
// binary, so the lock is mandatory, not a convenience.
mod common;
mod wgpu_integer_ops;
