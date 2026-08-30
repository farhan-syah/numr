// Backend parity tests for the WebGPU integer kernels that used to be F32-only:
// clamp, linspace, topk, searchsorted, and the three fused elementwise ops.
//
// WebGPU is a first-class backend for F32, I32 and U32 (numr/CLAUDE.md, "Backend
// Status"), so every one of these must answer what CPU answers. Two of them have
// a semantic trap worth pinning here:
//
// - The fused ops must equal the unfused sequence exactly, INCLUDING at the wrap
//   boundary. Integer elementwise ops wrap, accumulators saturate
//   (runtime/cpu/kernels/wide_acc.rs), and a fused kernel that saturated would
//   disagree with `add(mul(a, b), c)` precisely where it matters.
// - Integer `linspace` runs in exact 64-bit integer arithmetic on WebGPU, not in
//   f32, so a value above f32's 24-bit mantissa still lands on the right element.
//
// Every test is `#[cfg(feature = "wgpu")]`, so the imports are too - otherwise a
// non-WebGPU build would warn on all of them as unused.
#[cfg(feature = "wgpu")]
use numr::dtype::DType;
#[cfg(feature = "wgpu")]
use numr::ops::{BinaryOps, ScalarOps, SortingOps, UnaryOps, UtilityOps};
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
// clamp
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_clamp_i32_parity() {
    let data = vec![-2_000_000_000i32, -7, -1, 0, 1, 7, 2_000_000_000];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[7], &cpu_device).expect("cpu tensor");
    let cpu = cpu_client.clamp(&a_cpu, -5.0, 5.0).expect("cpu clamp");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[7], &device).expect("wgpu tensor");
        let got = client.clamp(&a, -5.0, 5.0).expect("wgpu clamp");
        assert_parity_i32(&got.to_vec::<i32>(), &cpu.to_vec::<i32>(), "clamp i32");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_clamp_u32_parity() {
    let data = vec![0u32, 1, 7, 4_000_000_000, u32::MAX];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device).expect("cpu tensor");
    let cpu = cpu_client.clamp(&a_cpu, 2.0, 3_000_000_000.0).expect("cpu");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[5], &device).expect("wgpu tensor");
        let got = client
            .clamp(&a, 2.0, 3_000_000_000.0)
            .expect("wgpu clamp u32");
        assert_parity_u32(&got.to_vec::<u32>(), &cpu.to_vec::<u32>(), "clamp u32");
    });
}

// ============================================================================
// linspace
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_linspace_i32_parity() {
    // The third case is the one f32 cannot do: the exact samples are integers
    // above 2^24, where an f32 intermediate rounds and truncates a step early.
    let cases: Vec<(f64, f64, usize)> = vec![
        (0.0, 10.0, 11),
        (-7.0, 7.0, 5),
        (0.0, 2_000_000_000.0, 9),
        (10.0, 0.0, 4),
        (3.0, 3.0, 3),
        (0.0, 1.0, 5),
    ];

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
                &format!("linspace i32 ({start}, {stop}, {steps})"),
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_linspace_u32_parity() {
    let cases: Vec<(f64, f64, usize)> = vec![
        (0.0, 10.0, 6),
        (0.0, 4_000_000_000.0, 9),
        (100.0, 0.0, 5),
        (0.0, 1.0, 4),
    ];

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

    with_wgpu_backend_or_skip(|client, _device| {
        for (&(start, stop, steps), want) in cases.iter().zip(expected.iter()) {
            let got = client
                .linspace(start, stop, steps, DType::U32)
                .expect("wgpu linspace u32");
            assert_parity_u32(
                &got.to_vec::<u32>(),
                want,
                &format!("linspace u32 ({start}, {stop}, {steps})"),
            );
        }
    });
}

// ============================================================================
// topk / searchsorted
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_topk_i32_parity() {
    let data = vec![5i32, -3, 9, 0, 9, -100, 42, 7];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device).expect("cpu tensor");

    // CPU reports topk indices as I64, WebGPU as I32, so the indices are
    // compared after widening rather than through a dtype-checked helper.
    let expected: Vec<(Vec<i32>, Vec<i64>)> = [true, false]
        .iter()
        .map(|&largest| {
            let (v, i) = cpu_client
                .topk(&a_cpu, 3, 0, largest, true)
                .expect("cpu topk");
            (v.to_vec::<i32>(), i.to_vec::<i64>())
        })
        .collect();

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[8], &device).expect("wgpu tensor");
        for (&largest, (want_vals, want_idx)) in [true, false].iter().zip(expected.iter()) {
            let (vals, idx) = client.topk(&a, 3, 0, largest, true).expect("wgpu topk i32");
            assert_parity_i32(
                &vals.to_vec::<i32>(),
                want_vals,
                &format!("topk i32 values (largest={largest})"),
            );
            let got_idx: Vec<i64> = idx.to_vec::<i32>().iter().map(|&x| x as i64).collect();
            assert_eq!(
                &got_idx, want_idx,
                "topk i32 indices (largest={largest}) WGPU vs CPU"
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_topk_u32_parity() {
    let data = vec![5u32, 3, 9, 0, u32::MAX, 100, 42, 7];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device).expect("cpu tensor");

    let expected: Vec<Vec<u32>> = [true, false]
        .iter()
        .map(|&largest| {
            cpu_client
                .topk(&a_cpu, 3, 0, largest, true)
                .expect("cpu topk")
                .0
                .to_vec::<u32>()
        })
        .collect();

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[8], &device).expect("wgpu tensor");
        for (&largest, want) in [true, false].iter().zip(expected.iter()) {
            let (vals, _) = client.topk(&a, 3, 0, largest, true).expect("wgpu topk u32");
            assert_parity_u32(
                &vals.to_vec::<u32>(),
                want,
                &format!("topk u32 values (largest={largest})"),
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_searchsorted_i32_parity() {
    let seq = vec![-10i32, -3, 0, 0, 4, 4, 4, 99];
    let values = vec![-11i32, -3, 0, 1, 4, 100];
    let (cpu_client, cpu_device) = create_cpu_client();
    let seq_cpu = Tensor::<CpuRuntime>::from_slice(&seq, &[8], &cpu_device).expect("cpu seq");
    let val_cpu = Tensor::<CpuRuntime>::from_slice(&values, &[6], &cpu_device).expect("cpu values");

    // CPU reports the insert positions as I64, WebGPU as I32.
    let expected: Vec<Vec<i64>> = [false, true]
        .iter()
        .map(|&right| {
            cpu_client
                .searchsorted(&seq_cpu, &val_cpu, right)
                .expect("cpu searchsorted")
                .to_vec::<i64>()
        })
        .collect();

    with_wgpu_backend_or_skip(|client, device| {
        let s = Tensor::<WgpuRuntime>::from_slice(&seq, &[8], &device).expect("wgpu seq");
        let v = Tensor::<WgpuRuntime>::from_slice(&values, &[6], &device).expect("wgpu values");
        for (&right, want) in [false, true].iter().zip(expected.iter()) {
            let got = client
                .searchsorted(&s, &v, right)
                .expect("wgpu searchsorted i32");
            let got_i64: Vec<i64> = got.to_vec::<i32>().iter().map(|&x| x as i64).collect();
            assert_eq!(
                &got_i64, want,
                "searchsorted i32 (right={right}) WGPU vs CPU"
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_searchsorted_u32_parity() {
    let seq = vec![0u32, 3, 3, 10, 4_000_000_000, u32::MAX];
    let values = vec![0u32, 3, 4, 4_000_000_000, u32::MAX];
    let (cpu_client, cpu_device) = create_cpu_client();
    let seq_cpu = Tensor::<CpuRuntime>::from_slice(&seq, &[6], &cpu_device).expect("cpu seq");
    let val_cpu = Tensor::<CpuRuntime>::from_slice(&values, &[5], &cpu_device).expect("cpu values");

    // CPU reports the insert positions as I64, WebGPU as I32.
    let expected: Vec<Vec<i64>> = [false, true]
        .iter()
        .map(|&right| {
            cpu_client
                .searchsorted(&seq_cpu, &val_cpu, right)
                .expect("cpu searchsorted")
                .to_vec::<i64>()
        })
        .collect();

    with_wgpu_backend_or_skip(|client, device| {
        let s = Tensor::<WgpuRuntime>::from_slice(&seq, &[6], &device).expect("wgpu seq");
        let v = Tensor::<WgpuRuntime>::from_slice(&values, &[5], &device).expect("wgpu values");
        for (&right, want) in [false, true].iter().zip(expected.iter()) {
            let got = client
                .searchsorted(&s, &v, right)
                .expect("wgpu searchsorted u32");
            let got_i64: Vec<i64> = got.to_vec::<i32>().iter().map(|&x| x as i64).collect();
            assert_eq!(
                &got_i64, want,
                "searchsorted u32 (right={right}) WGPU vs CPU"
            );
        }
    });
}

// ============================================================================
// Fused elementwise
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_ternary_i32_parity() {
    let a = vec![3i32, -4, 7, 0, 11];
    let b = vec![5i32, 6, -2, 9, 1];
    let c = vec![-1i32, 2, 3, 4, -5];

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&a, &[5], &cpu_device).expect("cpu a");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(&b, &[5], &cpu_device).expect("cpu b");
    let c_cpu = Tensor::<CpuRuntime>::from_slice(&c, &[5], &cpu_device).expect("cpu c");
    let cpu_fma = cpu_client
        .fused_mul_add(&a_cpu, &b_cpu, &c_cpu)
        .expect("cpu fma");
    let cpu_fam = cpu_client
        .fused_add_mul(&a_cpu, &b_cpu, &c_cpu)
        .expect("cpu fam");

    with_wgpu_backend_or_skip(|client, device| {
        let a_g = Tensor::<WgpuRuntime>::from_slice(&a, &[5], &device).expect("wgpu a");
        let b_g = Tensor::<WgpuRuntime>::from_slice(&b, &[5], &device).expect("wgpu b");
        let c_g = Tensor::<WgpuRuntime>::from_slice(&c, &[5], &device).expect("wgpu c");
        let fma = client
            .fused_mul_add(&a_g, &b_g, &c_g)
            .expect("wgpu fused_mul_add i32");
        let fam = client
            .fused_add_mul(&a_g, &b_g, &c_g)
            .expect("wgpu fused_add_mul i32");
        assert_parity_i32(
            &fma.to_vec::<i32>(),
            &cpu_fma.to_vec::<i32>(),
            "fused_mul_add i32",
        );
        assert_parity_i32(
            &fam.to_vec::<i32>(),
            &cpu_fam.to_vec::<i32>(),
            "fused_add_mul i32",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_ternary_u32_parity() {
    let a = vec![3u32, 4, 7, 0, 11];
    let b = vec![5u32, 6, 2, 9, 1];
    let c = vec![1u32, 2, 3, 4, 5];

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&a, &[5], &cpu_device).expect("cpu a");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(&b, &[5], &cpu_device).expect("cpu b");
    let c_cpu = Tensor::<CpuRuntime>::from_slice(&c, &[5], &cpu_device).expect("cpu c");
    let cpu_fma = cpu_client
        .fused_mul_add(&a_cpu, &b_cpu, &c_cpu)
        .expect("cpu fma");
    let cpu_fam = cpu_client
        .fused_add_mul(&a_cpu, &b_cpu, &c_cpu)
        .expect("cpu fam");

    with_wgpu_backend_or_skip(|client, device| {
        let a_g = Tensor::<WgpuRuntime>::from_slice(&a, &[5], &device).expect("wgpu a");
        let b_g = Tensor::<WgpuRuntime>::from_slice(&b, &[5], &device).expect("wgpu b");
        let c_g = Tensor::<WgpuRuntime>::from_slice(&c, &[5], &device).expect("wgpu c");
        let fma = client
            .fused_mul_add(&a_g, &b_g, &c_g)
            .expect("wgpu fused_mul_add u32");
        let fam = client
            .fused_add_mul(&a_g, &b_g, &c_g)
            .expect("wgpu fused_add_mul u32");
        assert_parity_u32(
            &fma.to_vec::<u32>(),
            &cpu_fma.to_vec::<u32>(),
            "fused_mul_add u32",
        );
        assert_parity_u32(
            &fam.to_vec::<u32>(),
            &cpu_fam.to_vec::<u32>(),
            "fused_add_mul u32",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_mul_add_scalar_int_parity() {
    let a_i = vec![3i32, -4, 7, 0, 11];
    let a_u = vec![3u32, 4, 7, 0, 11];

    let (cpu_client, cpu_device) = create_cpu_client();
    let ai_cpu = Tensor::<CpuRuntime>::from_slice(&a_i, &[5], &cpu_device).expect("cpu i32");
    let au_cpu = Tensor::<CpuRuntime>::from_slice(&a_u, &[5], &cpu_device).expect("cpu u32");
    let cpu_i = cpu_client
        .fused_mul_add_scalar(&ai_cpu, 3.0, -2.0)
        .expect("cpu scalar fma i32");
    let cpu_u = cpu_client
        .fused_mul_add_scalar(&au_cpu, 3.0, 2.0)
        .expect("cpu scalar fma u32");

    with_wgpu_backend_or_skip(|client, device| {
        let ai = Tensor::<WgpuRuntime>::from_slice(&a_i, &[5], &device).expect("wgpu i32");
        let au = Tensor::<WgpuRuntime>::from_slice(&a_u, &[5], &device).expect("wgpu u32");
        let got_i = client
            .fused_mul_add_scalar(&ai, 3.0, -2.0)
            .expect("wgpu scalar fma i32");
        let got_u = client
            .fused_mul_add_scalar(&au, 3.0, 2.0)
            .expect("wgpu scalar fma u32");
        assert_parity_i32(
            &got_i.to_vec::<i32>(),
            &cpu_i.to_vec::<i32>(),
            "fused_mul_add_scalar i32",
        );
        assert_parity_u32(
            &got_u.to_vec::<u32>(),
            &cpu_u.to_vec::<u32>(),
            "fused_mul_add_scalar u32",
        );
    });
}

/// A fused integer op must equal the unfused sequence exactly, including where
/// the product leaves the dtype's range and wraps. A kernel that saturated
/// instead would pass every well-scaled case above and fail only here.
#[cfg(feature = "wgpu")]
#[test]
fn test_fused_ops_wrap_like_the_unfused_sequence() {
    let a = vec![i32::MAX, i32::MIN, 100_000, -100_000];
    let b = vec![2i32, 2, 100_000, 100_000];
    let c = vec![1i32, -1, i32::MAX, i32::MIN];

    with_wgpu_backend_or_skip(|client, device| {
        let a_g = Tensor::<WgpuRuntime>::from_slice(&a, &[4], &device).expect("wgpu a");
        let b_g = Tensor::<WgpuRuntime>::from_slice(&b, &[4], &device).expect("wgpu b");
        let c_g = Tensor::<WgpuRuntime>::from_slice(&c, &[4], &device).expect("wgpu c");

        let fused = client
            .fused_mul_add(&a_g, &b_g, &c_g)
            .expect("wgpu fused_mul_add");
        let unfused = client
            .add(&client.mul(&a_g, &b_g).expect("mul"), &c_g)
            .expect("add");
        assert_parity_i32(
            &fused.to_vec::<i32>(),
            &unfused.to_vec::<i32>(),
            "fused_mul_add vs add(mul(a, b), c) at the wrap boundary",
        );

        let fused_am = client
            .fused_add_mul(&a_g, &b_g, &c_g)
            .expect("wgpu fused_add_mul");
        let unfused_am = client
            .mul(&client.add(&a_g, &b_g).expect("add"), &c_g)
            .expect("mul");
        assert_parity_i32(
            &fused_am.to_vec::<i32>(),
            &unfused_am.to_vec::<i32>(),
            "fused_add_mul vs mul(add(a, b), c) at the wrap boundary",
        );
    });
}

// ============================================================================
// neg on U32
//
// WGSL rejects `-x` on a u32, so the shader computes `0u - x`, which is defined
// wrapping there. Element-wise integer ops wrap in this crate
// (runtime/cpu/kernels/wide_acc.rs), so that is the contract, not a workaround:
// neg(1u32) is u32::MAX and neg(0) is 0.
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_neg_u32_parity() {
    let data = vec![0u32, 1, 4_000_000_000, u32::MAX];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("cpu tensor");
    let cpu = cpu_client.neg(&a_cpu).expect("cpu neg u32");
    assert_parity_u32(
        &cpu.to_vec::<u32>(),
        &[0, u32::MAX, 294_967_296, 1],
        "neg u32 CPU vs the wrapping contract",
    );

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).expect("wgpu tensor");
        let got = client.neg(&a).expect("wgpu neg u32");
        assert_parity_u32(&got.to_vec::<u32>(), &cpu.to_vec::<u32>(), "neg u32");
    });
}
