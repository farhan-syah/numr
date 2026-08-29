//! The parity comparison helper must treat non-finite values by identity.
//!
//! `round` on F16 produces infinity on both CPU and CUDA. The helper computed
//! `(a - b).abs()`, which is NaN for `inf - inf`, and `NaN <= tol` is false, so
//! two backends that agreed were reported as differing.

mod common;

use common::{assert_allclose_f64, values_close};

/// F16 tolerance, the pair that surfaced the defect.
const RTOL: f64 = 0.01;
const ATOL: f64 = 0.1;

#[test]
fn matching_infinities_compare_equal() {
    assert!(values_close(f64::INFINITY, f64::INFINITY, RTOL, ATOL));
    assert!(values_close(
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        RTOL,
        ATOL
    ));
}

#[test]
fn matching_nans_compare_equal() {
    assert!(values_close(f64::NAN, f64::NAN, RTOL, ATOL));
}

#[test]
fn infinity_against_finite_still_fails() {
    assert!(!values_close(f64::INFINITY, 1.0, RTOL, ATOL));
    assert!(!values_close(1.0, f64::INFINITY, RTOL, ATOL));
    assert!(!values_close(f64::INFINITY, f64::NEG_INFINITY, RTOL, ATOL));
}

#[test]
fn nan_against_number_still_fails() {
    assert!(!values_close(f64::NAN, 1.0, RTOL, ATOL));
    assert!(!values_close(1.0, f64::NAN, RTOL, ATOL));
    assert!(!values_close(f64::NAN, f64::INFINITY, RTOL, ATOL));
}

#[test]
fn finite_tolerance_is_unchanged() {
    // 100 * 0.01 + 0.1 = 1.1 is the tolerance at expected = 100.
    assert!(values_close(101.0, 100.0, RTOL, ATOL));
    assert!(!values_close(102.0, 100.0, RTOL, ATOL));
    assert!(values_close(0.05, 0.0, RTOL, ATOL));
    assert!(!values_close(0.2, 0.0, RTOL, ATOL));
}

#[test]
fn slice_helper_accepts_agreeing_infinities() {
    let a = [1.0, f64::INFINITY, f64::NEG_INFINITY];
    let b = [1.0, f64::INFINITY, f64::NEG_INFINITY];
    assert_allclose_f64(&a, &b, RTOL, ATOL, "agreeing infinities");
}

#[test]
#[should_panic(expected = "element 1 differs")]
fn slice_helper_rejects_infinity_against_finite() {
    let a = [1.0, f64::INFINITY];
    let b = [1.0, 2.0];
    assert_allclose_f64(&a, &b, RTOL, ATOL, "infinity vs finite");
}
