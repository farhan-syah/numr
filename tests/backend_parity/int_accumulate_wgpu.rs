// Backend parity tests for the WebGPU integer reductions that accumulate:
// `mean` and `scatter_reduce` sum/mean/prod.
//
// These follow the accumulator half of the convention in
// runtime/cpu/kernels/wide_acc.rs: run the total wider than one element, narrow
// exactly once, saturate at the narrow. WGSL has no 64-bit integer, so the
// WebGPU kernels build one out of u32 halves. What that buys is only visible on
// operands the other parity tests do not use:
//
// - a `mean` whose SUM leaves the element range while the mean itself does not,
//   which a kernel accumulating in i32 reports with the wrong sign;
// - a `mean` over more than one dimension, which must divide once by the whole
//   reduced count rather than once per dimension;
// - a `scatter_reduce` sum past the range, which must saturate rather than wrap;
// - a `scatter_reduce` product whose running total sits exactly on a bound,
//   which is exact only for a kernel that keeps the saturation state apart from
//   the value.
//
// Every test is `#[cfg(feature = "wgpu")]`, so the imports are too.
#[cfg(feature = "wgpu")]
use numr::ops::{IndexingOps, ReduceOps, ScatterReduceOp};
#[cfg(feature = "wgpu")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::{
    assert_parity_i32, assert_parity_u32, with_wgpu_backend_or_skip,
};
#[cfg(feature = "wgpu")]
use crate::common::create_cpu_client;

// ============================================================================
// mean
// ============================================================================

/// The sum of each row needs 33 bits, but every mean fits in i32. A kernel that
/// accumulated in the element type would wrap and report a negative mean.
#[cfg(feature = "wgpu")]
#[test]
fn test_mean_i32_sum_overflows_but_mean_does_not() {
    let data = vec![
        2_000_000_000i32,
        2_000_000_000,
        -2_000_000_000,
        -2_000_000_000,
    ];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device).expect("cpu tensor");
    let want = cpu_client
        .mean(&a_cpu, &[1], false)
        .expect("cpu mean")
        .to_vec::<i32>();
    assert_eq!(want, vec![2_000_000_000, -2_000_000_000], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 2], &device).expect("wgpu tensor");
        let got = client.mean(&a, &[1], false).expect("wgpu mean i32");
        assert_parity_i32(
            &got.to_vec::<i32>(),
            &want,
            "mean i32 over an overflowing sum",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_mean_u32_sum_overflows_but_mean_does_not() {
    let data = vec![4_000_000_000u32, 4_000_000_000, 1, 3];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device).expect("cpu tensor");
    let want = cpu_client
        .mean(&a_cpu, &[1], false)
        .expect("cpu mean")
        .to_vec::<u32>();
    assert_eq!(want, vec![4_000_000_000, 2], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 2], &device).expect("wgpu tensor");
        let got = client.mean(&a, &[1], false).expect("wgpu mean u32");
        assert_parity_u32(
            &got.to_vec::<u32>(),
            &want,
            "mean u32 over an overflowing sum",
        );
    });
}

/// An integer mean truncates toward zero, so dividing once per dimension is not
/// the same answer as dividing once by the whole reduced count. `[[0, 3], [0,
/// 3], [0, 0]]` averages to 1 as a single division of 6 by 6, and to 0 chained.
#[cfg(feature = "wgpu")]
#[test]
fn test_mean_i32_multi_dim_divides_once() {
    let data = vec![0i32, 3, 0, 3, 0, 0];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3, 2], &cpu_device).expect("cpu tensor");
    let want = cpu_client
        .mean(&a_cpu, &[0, 1], false)
        .expect("cpu mean")
        .to_vec::<i32>();
    assert_eq!(want, vec![1], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[3, 2], &device).expect("wgpu tensor");
        let got = client.mean(&a, &[0, 1], false).expect("wgpu mean i32");
        assert_parity_i32(&got.to_vec::<i32>(), &want, "mean i32 over two dims");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_mean_int_matches_cpu_on_ordinary_values() {
    let data_i = vec![1i32, 2, 3, 4, -5, -6];
    let data_u = vec![1u32, 2, 3, 4, 5, 7];
    let (cpu_client, cpu_device) = create_cpu_client();
    let ai = Tensor::<CpuRuntime>::from_slice(&data_i, &[2, 3], &cpu_device).expect("cpu i32");
    let au = Tensor::<CpuRuntime>::from_slice(&data_u, &[2, 3], &cpu_device).expect("cpu u32");
    let want_i = cpu_client
        .mean(&ai, &[1], true)
        .expect("cpu mean i32")
        .to_vec::<i32>();
    let want_u = cpu_client
        .mean(&au, &[0], false)
        .expect("cpu mean u32")
        .to_vec::<u32>();

    with_wgpu_backend_or_skip(|client, device| {
        let gi = Tensor::<WgpuRuntime>::from_slice(&data_i, &[2, 3], &device).expect("wgpu i32");
        let gu = Tensor::<WgpuRuntime>::from_slice(&data_u, &[2, 3], &device).expect("wgpu u32");
        let got_i = client.mean(&gi, &[1], true).expect("wgpu mean i32");
        let got_u = client.mean(&gu, &[0], false).expect("wgpu mean u32");
        assert_parity_i32(&got_i.to_vec::<i32>(), &want_i, "mean i32 keepdim");
        assert_parity_u32(&got_u.to_vec::<u32>(), &want_u, "mean u32 over dim 0");
    });
}

/// Integer `sum` accumulates too, so a total past the range saturates instead of
/// wrapping - the same rule, on the reduction the mean is built from.
#[cfg(feature = "wgpu")]
#[test]
fn test_sum_i32_saturates_instead_of_wrapping() {
    let data = vec![i32::MAX, i32::MAX, i32::MIN, i32::MIN];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device).expect("cpu tensor");
    let want = cpu_client
        .sum(&a_cpu, &[1], false)
        .expect("cpu sum")
        .to_vec::<i32>();
    assert_eq!(want, vec![i32::MAX, i32::MIN], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 2], &device).expect("wgpu tensor");
        let got = client.sum(&a, &[1], false).expect("wgpu sum i32");
        assert_parity_i32(&got.to_vec::<i32>(), &want, "sum i32 saturating");
    });
}

// ============================================================================
// scatter_reduce
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_sum_int_parity() {
    let dst_i = vec![0i32, 0, 0, 0];
    let src_i = vec![1i32, 2, 3];
    let dst_u = vec![0u32, 0, 0, 0];
    let src_u = vec![1u32, 2, 3];
    let indices = [0i32, 0, 2];

    let (cpu_client, cpu_device) = create_cpu_client();
    let di = Tensor::<CpuRuntime>::from_slice(&dst_i, &[4], &cpu_device).expect("cpu dst i32");
    let si = Tensor::<CpuRuntime>::from_slice(&src_i, &[3], &cpu_device).expect("cpu src i32");
    let du = Tensor::<CpuRuntime>::from_slice(&dst_u, &[4], &cpu_device).expect("cpu dst u32");
    let su = Tensor::<CpuRuntime>::from_slice(&src_u, &[3], &cpu_device).expect("cpu src u32");
    let ic = Tensor::<CpuRuntime>::from_slice(&indices, &[3], &cpu_device).expect("cpu idx");
    let want_i = cpu_client
        .scatter_reduce(&di, 0, &ic, &si, ScatterReduceOp::Sum, false)
        .expect("cpu scatter sum i32")
        .to_vec::<i32>();
    let want_u = cpu_client
        .scatter_reduce(&du, 0, &ic, &su, ScatterReduceOp::Sum, false)
        .expect("cpu scatter sum u32")
        .to_vec::<u32>();

    with_wgpu_backend_or_skip(|client, device| {
        let gdi = Tensor::<WgpuRuntime>::from_slice(&dst_i, &[4], &device).expect("wgpu dst i32");
        let gsi = Tensor::<WgpuRuntime>::from_slice(&src_i, &[3], &device).expect("wgpu src i32");
        let gdu = Tensor::<WgpuRuntime>::from_slice(&dst_u, &[4], &device).expect("wgpu dst u32");
        let gsu = Tensor::<WgpuRuntime>::from_slice(&src_u, &[3], &device).expect("wgpu src u32");
        let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[3], &device).expect("wgpu idx");
        let got_i = client
            .scatter_reduce(&gdi, 0, &gi, &gsi, ScatterReduceOp::Sum, false)
            .expect("wgpu scatter sum i32");
        let got_u = client
            .scatter_reduce(&gdu, 0, &gi, &gsu, ScatterReduceOp::Sum, false)
            .expect("wgpu scatter sum u32");
        assert_parity_i32(&got_i.to_vec::<i32>(), &want_i, "scatter_reduce sum i32");
        assert_parity_u32(&got_u.to_vec::<u32>(), &want_u, "scatter_reduce sum u32");
    });
}

/// The existing scatter parity tests use single-digit operands, so an element-
/// type atomic passes them while wrapping here. Slot 0 accumulates past
/// `i32::MAX` and must clamp there; slot 1 goes past `i32::MIN`.
#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_sum_i32_saturates() {
    let dst = vec![0i32, 0, 0];
    let src = vec![
        2_000_000_000i32,
        2_000_000_000,
        -2_000_000_000,
        -2_000_000_000,
    ];
    let indices = [0i32, 0, 1, 1];

    let (cpu_client, cpu_device) = create_cpu_client();
    let d = Tensor::<CpuRuntime>::from_slice(&dst, &[3], &cpu_device).expect("cpu dst");
    let s = Tensor::<CpuRuntime>::from_slice(&src, &[4], &cpu_device).expect("cpu src");
    let i = Tensor::<CpuRuntime>::from_slice(&indices, &[4], &cpu_device).expect("cpu idx");
    let want = cpu_client
        .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Sum, false)
        .expect("cpu scatter sum")
        .to_vec::<i32>();
    assert_eq!(want, vec![i32::MAX, i32::MIN, 0], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let gd = Tensor::<WgpuRuntime>::from_slice(&dst, &[3], &device).expect("wgpu dst");
        let gs = Tensor::<WgpuRuntime>::from_slice(&src, &[4], &device).expect("wgpu src");
        let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[4], &device).expect("wgpu idx");
        let got = client
            .scatter_reduce(&gd, 0, &gi, &gs, ScatterReduceOp::Sum, false)
            .expect("wgpu scatter sum i32");
        assert_parity_i32(
            &got.to_vec::<i32>(),
            &want,
            "scatter_reduce sum i32 past the range",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_sum_u32_saturates() {
    let dst = vec![0u32, 0];
    let src = vec![4_000_000_000u32, 4_000_000_000];
    let indices = [0i32, 0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let d = Tensor::<CpuRuntime>::from_slice(&dst, &[2], &cpu_device).expect("cpu dst");
    let s = Tensor::<CpuRuntime>::from_slice(&src, &[2], &cpu_device).expect("cpu src");
    let i = Tensor::<CpuRuntime>::from_slice(&indices, &[2], &cpu_device).expect("cpu idx");
    let want = cpu_client
        .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Sum, false)
        .expect("cpu scatter sum")
        .to_vec::<u32>();
    assert_eq!(want, vec![u32::MAX, 0], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let gd = Tensor::<WgpuRuntime>::from_slice(&dst, &[2], &device).expect("wgpu dst");
        let gs = Tensor::<WgpuRuntime>::from_slice(&src, &[2], &device).expect("wgpu src");
        let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[2], &device).expect("wgpu idx");
        let got = client
            .scatter_reduce(&gd, 0, &gi, &gs, ScatterReduceOp::Sum, false)
            .expect("wgpu scatter sum u32");
        assert_parity_u32(
            &got.to_vec::<u32>(),
            &want,
            "scatter_reduce sum u32 past the range",
        );
    });
}

/// Mean divides once by the contribution count. Slot 0's two contributions sum
/// past `i32::MAX`, so a kernel that clamped before dividing would report
/// `i32::MAX / 2` instead of the true mean.
#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_mean_int_parity() {
    let dst_i = vec![10i32, 20, 30, 40];
    let src_i = vec![2_000_000_000i32, 2_000_000_000, 8];
    let dst_u = vec![10u32, 20, 30, 40];
    let src_u = vec![4_000_000_000u32, 4_000_000_000, 8];
    let indices = [0i32, 0, 2];

    let (cpu_client, cpu_device) = create_cpu_client();
    let di = Tensor::<CpuRuntime>::from_slice(&dst_i, &[4], &cpu_device).expect("cpu dst i32");
    let si = Tensor::<CpuRuntime>::from_slice(&src_i, &[3], &cpu_device).expect("cpu src i32");
    let du = Tensor::<CpuRuntime>::from_slice(&dst_u, &[4], &cpu_device).expect("cpu dst u32");
    let su = Tensor::<CpuRuntime>::from_slice(&src_u, &[3], &cpu_device).expect("cpu src u32");
    let ic = Tensor::<CpuRuntime>::from_slice(&indices, &[3], &cpu_device).expect("cpu idx");

    for include_self in [false, true] {
        let want_i = cpu_client
            .scatter_reduce(&di, 0, &ic, &si, ScatterReduceOp::Mean, include_self)
            .expect("cpu scatter mean i32")
            .to_vec::<i32>();
        let want_u = cpu_client
            .scatter_reduce(&du, 0, &ic, &su, ScatterReduceOp::Mean, include_self)
            .expect("cpu scatter mean u32")
            .to_vec::<u32>();

        with_wgpu_backend_or_skip(|client, device| {
            let gdi =
                Tensor::<WgpuRuntime>::from_slice(&dst_i, &[4], &device).expect("wgpu dst i32");
            let gsi =
                Tensor::<WgpuRuntime>::from_slice(&src_i, &[3], &device).expect("wgpu src i32");
            let gdu =
                Tensor::<WgpuRuntime>::from_slice(&dst_u, &[4], &device).expect("wgpu dst u32");
            let gsu =
                Tensor::<WgpuRuntime>::from_slice(&src_u, &[3], &device).expect("wgpu src u32");
            let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[3], &device).expect("wgpu idx");
            let got_i = client
                .scatter_reduce(&gdi, 0, &gi, &gsi, ScatterReduceOp::Mean, include_self)
                .expect("wgpu scatter mean i32");
            let got_u = client
                .scatter_reduce(&gdu, 0, &gi, &gsu, ScatterReduceOp::Mean, include_self)
                .expect("wgpu scatter mean u32");
            assert_parity_i32(
                &got_i.to_vec::<i32>(),
                &want_i,
                &format!("scatter_reduce mean i32 (include_self={include_self})"),
            );
            assert_parity_u32(
                &got_u.to_vec::<u32>(),
                &want_u,
                &format!("scatter_reduce mean u32 (include_self={include_self})"),
            );
        });
    }
}

// ============================================================================
// scatter_reduce prod
// ============================================================================
//
// The integer product accumulates too, but its state is a magnitude and a sign
// parity rather than a wide sum. A 32-bit atomic cannot carry that: the clamped
// value would double as the saturation state, so a running product of exactly
// `i32::MAX` reports `i32::MIN` after a negative factor. These pin the values
// that distinguish the two.

/// `i32::MAX * -1` is `-i32::MAX`, which the dtype represents exactly. Seeding
/// through `include_self` puts the bound in the accumulator before the negative
/// factor arrives, so the answer does not depend on which order the two land in.
#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_prod_i32_at_the_bound_is_exact() {
    let dst = vec![i32::MAX, 1];
    let src = vec![-1i32];
    let indices = [0i32];

    let (cpu_client, cpu_device) = create_cpu_client();
    let d = Tensor::<CpuRuntime>::from_slice(&dst, &[2], &cpu_device).expect("cpu dst");
    let s = Tensor::<CpuRuntime>::from_slice(&src, &[1], &cpu_device).expect("cpu src");
    let i = Tensor::<CpuRuntime>::from_slice(&indices, &[1], &cpu_device).expect("cpu idx");
    let want = cpu_client
        .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Prod, true)
        .expect("cpu scatter prod")
        .to_vec::<i32>();
    assert_eq!(want, vec![-i32::MAX, 1], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let gd = Tensor::<WgpuRuntime>::from_slice(&dst, &[2], &device).expect("wgpu dst");
        let gs = Tensor::<WgpuRuntime>::from_slice(&src, &[1], &device).expect("wgpu src");
        let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[1], &device).expect("wgpu idx");
        let got = client
            .scatter_reduce(&gd, 0, &gi, &gs, ScatterReduceOp::Prod, true)
            .expect("wgpu scatter prod i32");
        assert_parity_i32(
            &got.to_vec::<i32>(),
            &want,
            "scatter_reduce prod i32 at the bound",
        );
    });
}

/// Slot 0's true product is `1e10`, slot 1's is `-1e10`. Both leave the range,
/// so each clamps to the bound with ITS OWN sign.
#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_prod_i32_saturates_with_the_right_sign() {
    let dst = vec![1i32, 1, 1];
    let src = vec![100_000i32, 100_000, 100_000, -100_000];
    let indices = [0i32, 0, 1, 1];

    let (cpu_client, cpu_device) = create_cpu_client();
    let d = Tensor::<CpuRuntime>::from_slice(&dst, &[3], &cpu_device).expect("cpu dst");
    let s = Tensor::<CpuRuntime>::from_slice(&src, &[4], &cpu_device).expect("cpu src");
    let i = Tensor::<CpuRuntime>::from_slice(&indices, &[4], &cpu_device).expect("cpu idx");
    let want = cpu_client
        .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Prod, false)
        .expect("cpu scatter prod")
        .to_vec::<i32>();
    assert_eq!(want, vec![i32::MAX, i32::MIN, 1], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let gd = Tensor::<WgpuRuntime>::from_slice(&dst, &[3], &device).expect("wgpu dst");
        let gs = Tensor::<WgpuRuntime>::from_slice(&src, &[4], &device).expect("wgpu src");
        let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[4], &device).expect("wgpu idx");
        let got = client
            .scatter_reduce(&gd, 0, &gi, &gs, ScatterReduceOp::Prod, false)
            .expect("wgpu scatter prod i32");
        assert_parity_i32(
            &got.to_vec::<i32>(),
            &want,
            "scatter_reduce prod i32 past the range",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_prod_u32_saturates() {
    let dst = vec![1u32, 1];
    let src = vec![100_000u32, 100_000];
    let indices = [0i32, 0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let d = Tensor::<CpuRuntime>::from_slice(&dst, &[2], &cpu_device).expect("cpu dst");
    let s = Tensor::<CpuRuntime>::from_slice(&src, &[2], &cpu_device).expect("cpu src");
    let i = Tensor::<CpuRuntime>::from_slice(&indices, &[2], &cpu_device).expect("cpu idx");
    let want = cpu_client
        .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Prod, false)
        .expect("cpu scatter prod")
        .to_vec::<u32>();
    assert_eq!(want, vec![u32::MAX, 1], "CPU reference");

    with_wgpu_backend_or_skip(|client, device| {
        let gd = Tensor::<WgpuRuntime>::from_slice(&dst, &[2], &device).expect("wgpu dst");
        let gs = Tensor::<WgpuRuntime>::from_slice(&src, &[2], &device).expect("wgpu src");
        let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[2], &device).expect("wgpu idx");
        let got = client
            .scatter_reduce(&gd, 0, &gi, &gs, ScatterReduceOp::Prod, false)
            .expect("wgpu scatter prod u32");
        assert_parity_u32(
            &got.to_vec::<u32>(),
            &want,
            "scatter_reduce prod u32 past the range",
        );
    });
}

/// Every product here is representable, including the negative ones, so the
/// result is checked exactly rather than at a bound. The 2-D case scatters
/// along the LAST dimension, which is where a per-destination kernel's own
/// lane arithmetic (outer, inner) has to be right.
#[cfg(feature = "wgpu")]
#[test]
fn test_scatter_reduce_prod_int_representable_parity() {
    let dst_i = vec![1i32, 1, 1];
    let src_i = vec![-3i32, 7, 11, -2, -5];
    let dst_u = vec![1u32, 1, 1];
    let src_u = vec![3u32, 7, 11, 2, 5];
    let indices = [0i32, 0, 0, 2, 2];

    let dst2 = vec![2i32, -1, 1, 1, 1, 1];
    let src2 = vec![-4i32, 6, -3, 5];
    let indices2 = [0i32, 2, 1, 1];

    let (cpu_client, cpu_device) = create_cpu_client();
    let di = Tensor::<CpuRuntime>::from_slice(&dst_i, &[3], &cpu_device).expect("cpu dst i32");
    let si = Tensor::<CpuRuntime>::from_slice(&src_i, &[5], &cpu_device).expect("cpu src i32");
    let du = Tensor::<CpuRuntime>::from_slice(&dst_u, &[3], &cpu_device).expect("cpu dst u32");
    let su = Tensor::<CpuRuntime>::from_slice(&src_u, &[5], &cpu_device).expect("cpu src u32");
    let ic = Tensor::<CpuRuntime>::from_slice(&indices, &[5], &cpu_device).expect("cpu idx");
    let d2 = Tensor::<CpuRuntime>::from_slice(&dst2, &[2, 3], &cpu_device).expect("cpu dst 2d");
    let s2 = Tensor::<CpuRuntime>::from_slice(&src2, &[2, 2], &cpu_device).expect("cpu src 2d");
    let i2 = Tensor::<CpuRuntime>::from_slice(&indices2, &[2, 2], &cpu_device).expect("cpu idx 2d");

    for include_self in [false, true] {
        let want_i = cpu_client
            .scatter_reduce(&di, 0, &ic, &si, ScatterReduceOp::Prod, include_self)
            .expect("cpu scatter prod i32")
            .to_vec::<i32>();
        let want_u = cpu_client
            .scatter_reduce(&du, 0, &ic, &su, ScatterReduceOp::Prod, include_self)
            .expect("cpu scatter prod u32")
            .to_vec::<u32>();
        let want_2d = cpu_client
            .scatter_reduce(&d2, 1, &i2, &s2, ScatterReduceOp::Prod, include_self)
            .expect("cpu scatter prod 2d")
            .to_vec::<i32>();

        with_wgpu_backend_or_skip(|client, device| {
            let gdi =
                Tensor::<WgpuRuntime>::from_slice(&dst_i, &[3], &device).expect("wgpu dst i32");
            let gsi =
                Tensor::<WgpuRuntime>::from_slice(&src_i, &[5], &device).expect("wgpu src i32");
            let gdu =
                Tensor::<WgpuRuntime>::from_slice(&dst_u, &[3], &device).expect("wgpu dst u32");
            let gsu =
                Tensor::<WgpuRuntime>::from_slice(&src_u, &[5], &device).expect("wgpu src u32");
            let gi = Tensor::<WgpuRuntime>::from_slice(&indices, &[5], &device).expect("wgpu idx");
            let gd2 =
                Tensor::<WgpuRuntime>::from_slice(&dst2, &[2, 3], &device).expect("wgpu dst 2d");
            let gs2 =
                Tensor::<WgpuRuntime>::from_slice(&src2, &[2, 2], &device).expect("wgpu src 2d");
            let gi2 = Tensor::<WgpuRuntime>::from_slice(&indices2, &[2, 2], &device)
                .expect("wgpu idx 2d");

            let got_i = client
                .scatter_reduce(&gdi, 0, &gi, &gsi, ScatterReduceOp::Prod, include_self)
                .expect("wgpu scatter prod i32");
            let got_u = client
                .scatter_reduce(&gdu, 0, &gi, &gsu, ScatterReduceOp::Prod, include_self)
                .expect("wgpu scatter prod u32");
            let got_2d = client
                .scatter_reduce(&gd2, 1, &gi2, &gs2, ScatterReduceOp::Prod, include_self)
                .expect("wgpu scatter prod 2d");

            assert_parity_i32(
                &got_i.to_vec::<i32>(),
                &want_i,
                &format!("scatter_reduce prod i32 (include_self={include_self})"),
            );
            assert_parity_u32(
                &got_u.to_vec::<u32>(),
                &want_u,
                &format!("scatter_reduce prod u32 (include_self={include_self})"),
            );
            assert_parity_i32(
                &got_2d.to_vec::<i32>(),
                &want_2d,
                &format!("scatter_reduce prod i32 dim 1 (include_self={include_self})"),
            );
        });
    }
}
