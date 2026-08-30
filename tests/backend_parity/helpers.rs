//! Shared helpers for backend parity tests: assertion utilities, backend locks, client creation.

// The backend locks and their `with_*` entry points live in `common` so every
// test binary reaches the SAME implementation. A second lock beside this one
// would protect nothing.
#[cfg(feature = "cuda")]
pub use crate::common::backend_lock::with_cuda_backend;
#[cfg(feature = "wgpu")]
pub use crate::common::backend_lock::{with_wgpu_backend, with_wgpu_backend_or_skip};

pub fn assert_parity_f32(a: &[f32], b: &[f32], op: &str) {
    let rtol = 1e-5f32;
    let atol = 1e-7f32;
    assert_eq!(
        a.len(),
        b.len(),
        "parity_f32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();

        if diff > tol {
            panic!(
                "parity_f32[{}] at index {}: {} vs {} (diff={}, tol={})",
                op, i, x, y, diff, tol
            );
        }
    }
}

#[allow(dead_code)]
pub fn assert_parity_f64(a: &[f64], b: &[f64], op: &str) {
    let rtol = 1e-12f64;
    let atol = 1e-14f64;
    assert_eq!(
        a.len(),
        b.len(),
        "parity_f64[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();

        if diff > tol {
            panic!(
                "parity_f64[{}] at index {}: {} vs {} (diff={}, tol={})",
                op, i, x, y, diff, tol
            );
        }
    }
}

#[allow(dead_code)]
pub fn assert_parity_i32(a: &[i32], b: &[i32], op: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "parity_i32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "parity_i32[{}] at index {}: {} vs {}", op, i, x, y);
    }
}

pub fn assert_parity_u32(a: &[u32], b: &[u32], op: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "parity_u32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "parity_u32[{}] at index {}: {} vs {}", op, i, x, y);
    }
}

#[allow(dead_code)]
pub fn assert_parity_bool(a: &[bool], b: &[bool], op: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "parity_bool[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "parity_bool[{}] at index {}: {} vs {}", op, i, x, y);
    }
}
