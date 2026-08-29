// Backend parity tests for the WebGPU integer generators, `linspace` and
// `arange`, on the bounds where the float-to-integer conversion decides the
// answer.
//
// Both evaluate in f32 whenever the exact 64-bit path does not apply, and WGSL
// leaves `u32(v)` and `i32(v)` IMPLEMENTATION-DEFINED once `v` is outside the
// destination range. A negative f32 converted to u32 is exactly that case, so
// without an explicit guard the U32 result is whatever the driver happens to
// do. CPU has no such freedom: `Element::from_f64` is Rust's `as`, which
// truncates toward zero, clamps to the type's bounds, and maps NaN to 0.
//
// The ordinary in-range cases live in int_ops_wgpu.rs. What is pinned here is
// only the edges: a negative intermediate, and bounds past the dtype's range.
//
// Every test is `#[cfg(feature = "wgpu")]`, so the imports are too - otherwise a
// non-WebGPU build would warn on all of them as unused.
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::{
    assert_parity_i32, assert_parity_u32, with_wgpu_backend_or_skip,
};
#[cfg(feature = "wgpu")]
use crate::common::create_cpu_client;
#[cfg(feature = "wgpu")]
use numr::dtype::DType;
#[cfg(feature = "wgpu")]
use numr::ops::UtilityOps;

// ============================================================================
// linspace
// ============================================================================

/// Fractional bounds take the f32 path, and every sample below zero has to
/// clamp to 0 rather than wrap to a huge u32.
#[cfg(feature = "wgpu")]
#[test]
fn test_linspace_u32_fractional_negative_clamps_to_zero() {
    let cases: Vec<(f64, f64, usize)> = vec![(-3.5, 2.5, 7), (-2.5, -0.5, 3), (-0.5, 4.5, 6)];

    let (cpu_client, _cpu_device) = create_cpu_client();
    let expected: Vec<Vec<u32>> = cases
        .iter()
        .map(|&(start, stop, steps)| {
            cpu_client
                .linspace(start, stop, steps, DType::U32)
                .expect("cpu linspace")
                .to_vec::<u32>()
        })
        .collect();
    assert_eq!(expected[1], vec![0u32, 0, 0], "CPU reference");

    with_wgpu_backend_or_skip(|client, _device| {
        for (&(start, stop, steps), want) in cases.iter().zip(expected.iter()) {
            let got = client
                .linspace(start, stop, steps, DType::U32)
                .expect("wgpu linspace u32");
            assert_parity_u32(
                &got.to_vec::<u32>(),
                want,
                &format!("linspace u32 fractional ({start}, {stop}, {steps})"),
            );
        }
    });
}

/// The signed twin: fractional bounds truncate toward zero on both sides.
#[cfg(feature = "wgpu")]
#[test]
fn test_linspace_i32_fractional_truncates_toward_zero() {
    let cases: Vec<(f64, f64, usize)> = vec![(-3.5, 2.5, 7), (-0.5, 0.5, 3)];

    let (cpu_client, _cpu_device) = create_cpu_client();
    let expected: Vec<Vec<i32>> = cases
        .iter()
        .map(|&(start, stop, steps)| {
            cpu_client
                .linspace(start, stop, steps, DType::I32)
                .expect("cpu linspace")
                .to_vec::<i32>()
        })
        .collect();

    with_wgpu_backend_or_skip(|client, _device| {
        for (&(start, stop, steps), want) in cases.iter().zip(expected.iter()) {
            let got = client
                .linspace(start, stop, steps, DType::I32)
                .expect("wgpu linspace i32");
            assert_parity_i32(
                &got.to_vec::<i32>(),
                want,
                &format!("linspace i32 fractional ({start}, {stop}, {steps})"),
            );
        }
    });
}

/// Bounds outside the dtype's range clamp at the bound, and for U32 a negative
/// bound clamps at zero.
#[cfg(feature = "wgpu")]
#[test]
fn test_linspace_int_saturating_bounds() {
    let u_cases: Vec<(f64, f64, usize)> =
        vec![(0.0, 5_000_000_000.0, 3), (-10.0, 10.0, 5), (-6.0, -2.0, 3)];
    let i_cases: Vec<(f64, f64, usize)> =
        vec![(-5_000_000_000.0, 5_000_000_000.0, 3), (0.0, 3e9, 3)];

    let (cpu_client, _cpu_device) = create_cpu_client();
    let want_u: Vec<Vec<u32>> = u_cases
        .iter()
        .map(|&(start, stop, steps)| {
            cpu_client
                .linspace(start, stop, steps, DType::U32)
                .expect("cpu linspace u32")
                .to_vec::<u32>()
        })
        .collect();
    let want_i: Vec<Vec<i32>> = i_cases
        .iter()
        .map(|&(start, stop, steps)| {
            cpu_client
                .linspace(start, stop, steps, DType::I32)
                .expect("cpu linspace i32")
                .to_vec::<i32>()
        })
        .collect();
    assert_eq!(
        want_u[0],
        vec![0u32, 2_500_000_000, u32::MAX],
        "CPU reference"
    );
    assert_eq!(want_i[0], vec![i32::MIN, 0, i32::MAX], "CPU reference");

    with_wgpu_backend_or_skip(|client, _device| {
        for (&(start, stop, steps), want) in u_cases.iter().zip(want_u.iter()) {
            let got = client
                .linspace(start, stop, steps, DType::U32)
                .expect("wgpu linspace u32");
            assert_parity_u32(
                &got.to_vec::<u32>(),
                want,
                &format!("linspace u32 saturating ({start}, {stop}, {steps})"),
            );
        }
        for (&(start, stop, steps), want) in i_cases.iter().zip(want_i.iter()) {
            let got = client
                .linspace(start, stop, steps, DType::I32)
                .expect("wgpu linspace i32");
            assert_parity_i32(
                &got.to_vec::<i32>(),
                want,
                &format!("linspace i32 saturating ({start}, {stop}, {steps})"),
            );
        }
    });
}

// ============================================================================
// arange
// ============================================================================

/// `arange` has no exact integer path at all - it always evaluates in f32 - so
/// a negative start is the unguarded conversion in its plainest form.
#[cfg(feature = "wgpu")]
#[test]
fn test_arange_u32_negative_start_clamps_to_zero() {
    let (cpu_client, _cpu_device) = create_cpu_client();
    let want = cpu_client
        .arange(-5.0, 5.0, 1.0, DType::U32)
        .expect("cpu arange u32")
        .to_vec::<u32>();
    assert_eq!(want, vec![0u32, 0, 0, 0, 0, 0, 1, 2, 3, 4], "CPU reference");

    with_wgpu_backend_or_skip(|client, _device| {
        let got = client
            .arange(-5.0, 5.0, 1.0, DType::U32)
            .expect("wgpu arange u32");
        assert_parity_u32(&got.to_vec::<u32>(), &want, "arange u32 negative start");
    });
}

/// Starts beyond the dtype's range clamp at the bound rather than wrapping.
#[cfg(feature = "wgpu")]
#[test]
fn test_arange_int_saturating_bounds() {
    let (cpu_client, _cpu_device) = create_cpu_client();
    let want_u = cpu_client
        .arange(5_000_000_000.0, 5_000_000_005.0, 1.0, DType::U32)
        .expect("cpu arange u32")
        .to_vec::<u32>();
    let want_i_hi = cpu_client
        .arange(3_000_000_000.0, 3_000_000_005.0, 1.0, DType::I32)
        .expect("cpu arange i32")
        .to_vec::<i32>();
    let want_i_lo = cpu_client
        .arange(-3_000_000_000.0, -2_999_999_995.0, 1.0, DType::I32)
        .expect("cpu arange i32")
        .to_vec::<i32>();
    assert_eq!(want_u, vec![u32::MAX; 5], "CPU reference");
    assert_eq!(want_i_hi, vec![i32::MAX; 5], "CPU reference");
    assert_eq!(want_i_lo, vec![i32::MIN; 5], "CPU reference");

    with_wgpu_backend_or_skip(|client, _device| {
        let got_u = client
            .arange(5_000_000_000.0, 5_000_000_005.0, 1.0, DType::U32)
            .expect("wgpu arange u32");
        let got_i_hi = client
            .arange(3_000_000_000.0, 3_000_000_005.0, 1.0, DType::I32)
            .expect("wgpu arange i32");
        let got_i_lo = client
            .arange(-3_000_000_000.0, -2_999_999_995.0, 1.0, DType::I32)
            .expect("wgpu arange i32");
        assert_parity_u32(&got_u.to_vec::<u32>(), &want_u, "arange u32 past the range");
        assert_parity_i32(
            &got_i_hi.to_vec::<i32>(),
            &want_i_hi,
            "arange i32 past i32::MAX",
        );
        assert_parity_i32(
            &got_i_lo.to_vec::<i32>(),
            &want_i_lo,
            "arange i32 past i32::MIN",
        );
    });
}
