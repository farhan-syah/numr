// rand_seeded / randn_seeded reproducibility across CPU/CUDA/WebGPU, plus
// the seed-derivation regression coverage (high-word truncation).

use numr::dtype::DType;
use numr::ops::RandomOps;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "cuda")]
use crate::common::is_dtype_supported;
use crate::common::{DTypeDomain, ToF64, create_cpu_client, parity_dtypes};

use super::distributions::check_normal_stats;

// ============================================================
// rand_seeded reproducibility tests
// ============================================================

#[test]
fn test_rand_seeded_reproducibility_cpu() {
    let (client, _device) = create_cpu_client();

    // Same seed → same output
    let a = client.rand_seeded(&[100], DType::F32, 42).unwrap();
    let b = client.rand_seeded(&[100], DType::F32, 42).unwrap();
    let a_vec: Vec<f32> = a.to_vec();
    let b_vec: Vec<f32> = b.to_vec();
    assert_eq!(a_vec, b_vec, "same seed must produce same output");

    // Different seed → different output
    let c = client.rand_seeded(&[100], DType::F32, 99).unwrap();
    let c_vec: Vec<f32> = c.to_vec();
    assert_ne!(
        a_vec, c_vec,
        "different seeds must produce different output"
    );

    // Values in [0, 1)
    for &v in &a_vec {
        assert!((0.0..1.0).contains(&v), "value out of range: {v}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_rand_seeded_reproducibility_cuda() {
    with_cuda_backend(|client, _device| {
        let a = client.rand_seeded(&[100], DType::F32, 42).unwrap();
        let b = client.rand_seeded(&[100], DType::F32, 42).unwrap();
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        assert_eq!(a_vec, b_vec, "same seed must produce same output on CUDA");

        let c = client.rand_seeded(&[100], DType::F32, 99).unwrap();
        let c_vec: Vec<f32> = c.to_vec();
        assert_ne!(
            a_vec, c_vec,
            "different seeds must produce different output on CUDA"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_rand_seeded_f32_never_reaches_one_cuda() {
    // Regression: rand_f32 narrowed a raw f64 uniform sample to f32 with no
    // clamp. A sample in [1 - 2^-25, 1) rounds to exactly 1.0f32 under
    // round-to-nearest, breaking the documented [0, 1) contract. A large
    // tensor makes this land reliably at the ~2^-25 per-sample probability.
    with_cuda_backend(|client, _device| {
        let a = client.rand_seeded(&[10_000_000], DType::F32, 7).unwrap();
        let a_vec: Vec<f32> = a.to_vec();
        for &v in &a_vec {
            assert!((0.0..1.0).contains(&v), "value out of range: {v}");
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_rand_seeded_reproducibility_wgpu() {
    with_wgpu_backend(|client, _device| {
        let a = client.rand_seeded(&[100], DType::F32, 42).unwrap();
        let b = client.rand_seeded(&[100], DType::F32, 42).unwrap();
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        assert_eq!(a_vec, b_vec, "same seed must produce same output on WebGPU");

        let c = client.rand_seeded(&[100], DType::F32, 99).unwrap();
        let c_vec: Vec<f32> = c.to_vec();
        assert_ne!(
            a_vec, c_vec,
            "different seeds must produce different output on WebGPU"
        );

        // Values in [0, 1)
        for &v in &a_vec {
            assert!((0.0..1.0).contains(&v), "value out of range: {v}");
        }
    });
}

// ============================================================
// randn_seeded reproducibility tests
// ============================================================

/// Check that at least one sampled value falls outside [0, 1). A uniform
/// sampler can never produce this (u64_to_uniform is confined to [0, 1)),
/// so this catches a bug where randn_seeded silently returns uniform values.
fn check_not_uniform<T: ToF64>(vals: &[T]) {
    assert!(
        vals.iter().any(|&x| !(0.0..1.0).contains(&x.to_f64())),
        "all sampled values fell within [0, 1) - looks like a uniform sampler, not normal"
    );
}

#[test]
fn test_randn_seeded_reproducibility_cpu_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        // Skip integer types - randn_seeded() is for floating-point only
        if matches!(dtype, DType::I32 | DType::I64 | DType::U32 | DType::Bool) {
            continue;
        }

        let (client, _device) = create_cpu_client();

        // 10000 samples: with SE ~= 1/sqrt(10000) = 0.01 for the mean, the
        // odds of two independently-seeded 10000-sample normal draws landing
        // on an identical vector by chance are effectively zero.
        let a = client
            .randn_seeded(&[10000], dtype, 42)
            .unwrap_or_else(|e| panic!("CPU randn_seeded failed for {dtype:?}: {e}"));
        let b = client.randn_seeded(&[10000], dtype, 42).unwrap();
        let c = client.randn_seeded(&[10000], dtype, 99).unwrap();

        macro_rules! check {
            ($T:ty) => {{
                let a_vec = a.to_vec::<$T>();
                let b_vec = b.to_vec::<$T>();
                let c_vec = c.to_vec::<$T>();
                assert_eq!(
                    a_vec, b_vec,
                    "randn_seeded[{dtype:?}]: same seed must produce bit-identical output"
                );
                assert_ne!(
                    a_vec, c_vec,
                    "randn_seeded[{dtype:?}]: different seeds must produce different output"
                );
                check_normal_stats(&a_vec, dtype);
                check_not_uniform(&a_vec);
            }};
        }

        match dtype {
            DType::F64 => check!(f64),
            DType::F32 => check!(f32),
            #[cfg(feature = "f16")]
            DType::F16 => check!(half::f16),
            #[cfg(feature = "f16")]
            DType::BF16 => check!(half::bf16),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => check!(numr::dtype::FP8E4M3),
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => check!(numr::dtype::FP8E5M2),
            _ => {}
        }
    }
}

/// randn() called twice without a seed must NOT reproduce - this is what
/// proves the seeded path is doing something the unseeded path does not.
#[test]
fn test_randn_unseeded_differs_across_calls_cpu() {
    let (client, _device) = create_cpu_client();
    let a: Vec<f32> = client.randn(&[10000], DType::F32).unwrap().to_vec();
    let b: Vec<f32> = client.randn(&[10000], DType::F32).unwrap().to_vec();
    assert_ne!(
        a, b,
        "unseeded randn() must not reproduce across independent calls"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_randn_seeded_reproducibility_cuda() {
    // CUDA randn_seeded supports F32, F64, F16, BF16 (same as rand_seeded).
    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        with_cuda_backend(|client, _device| {
            let a = client
                .randn_seeded(&[10000], dtype, 42)
                .unwrap_or_else(|e| panic!("CUDA randn_seeded failed for {dtype:?}: {e}"));
            let b = client.randn_seeded(&[10000], dtype, 42).unwrap();
            let c = client.randn_seeded(&[10000], dtype, 99).unwrap();

            macro_rules! check {
                ($T:ty) => {{
                    let a_vec = a.to_vec::<$T>();
                    let b_vec = b.to_vec::<$T>();
                    let c_vec = c.to_vec::<$T>();
                    assert_eq!(
                        a_vec, b_vec,
                        "randn_seeded[{dtype:?}] CUDA: same seed must produce bit-identical output"
                    );
                    assert_ne!(
                        a_vec, c_vec,
                        "randn_seeded[{dtype:?}] CUDA: different seeds must produce different output"
                    );
                    check_normal_stats(&a_vec, dtype);
                    check_not_uniform(&a_vec);
                }};
            }

            match dtype {
                DType::F64 => check!(f64),
                DType::F32 => check!(f32),
                #[cfg(feature = "f16")]
                DType::F16 => check!(half::f16),
                #[cfg(feature = "f16")]
                DType::BF16 => check!(half::bf16),
                _ => {}
            }
        });
    }
}

/// randn() on CUDA called twice without a seed must NOT reproduce.
#[cfg(feature = "cuda")]
#[test]
fn test_randn_unseeded_differs_across_calls_cuda() {
    with_cuda_backend(|client, _device| {
        let a: Vec<f32> = client.randn(&[10000], DType::F32).unwrap().to_vec();
        let b: Vec<f32> = client.randn(&[10000], DType::F32).unwrap().to_vec();
        assert_ne!(
            a, b,
            "unseeded CUDA randn() must not reproduce across independent calls"
        );
    });
}

/// Edge cases: single-element and empty tensors must not panic and must
/// preserve reproducibility/shape invariants for randn_seeded.
#[test]
fn test_randn_seeded_edge_cases_cpu() {
    let (client, _device) = create_cpu_client();

    // Single element: reproducible, shape preserved.
    let a = client.randn_seeded(&[1], DType::F32, 7).unwrap();
    let b = client.randn_seeded(&[1], DType::F32, 7).unwrap();
    assert_eq!(a.shape(), &[1]);
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());

    // Empty tensor (zero elements): must not panic, shape preserved.
    let empty = client.randn_seeded(&[0], DType::F32, 7).unwrap();
    assert_eq!(empty.shape(), &[0]);
    assert_eq!(empty.numel(), 0);
    assert!(empty.to_vec::<f32>().is_empty());

    // Empty tensor via a zero-sized dimension in a multi-dim shape.
    let empty_2d = client.randn_seeded(&[0, 5], DType::F32, 7).unwrap();
    assert_eq!(empty_2d.shape(), &[0, 5]);
    assert_eq!(empty_2d.numel(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn test_randn_seeded_edge_cases_cuda() {
    with_cuda_backend(|client, _device| {
        let a = client.randn_seeded(&[1], DType::F32, 7).unwrap();
        let b = client.randn_seeded(&[1], DType::F32, 7).unwrap();
        assert_eq!(a.shape(), &[1]);
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());

        let empty = client.randn_seeded(&[0], DType::F32, 7).unwrap();
        assert_eq!(empty.shape(), &[0]);
        assert_eq!(empty.numel(), 0);
    });
}

// ============================================================
// randn_seeded WebGPU tests
// ============================================================

/// Same seed on WebGPU must reproduce, and the output must look like a
/// standard normal draw (finite, roughly mean 0 / var 1, not a uniform
/// sampler in disguise).
#[cfg(feature = "wgpu")]
#[test]
fn test_randn_seeded_reproducibility_wgpu() {
    with_wgpu_backend_or_skip(|client, _device| {
        let a = client.randn_seeded(&[10000], DType::F32, 42).unwrap();
        let b = client.randn_seeded(&[10000], DType::F32, 42).unwrap();
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        assert_eq!(
            a_vec, b_vec,
            "same seed must produce bit-identical output on WebGPU"
        );

        let c = client.randn_seeded(&[10000], DType::F32, 99).unwrap();
        let c_vec: Vec<f32> = c.to_vec();
        assert_ne!(
            a_vec, c_vec,
            "different seeds must produce different output on WebGPU"
        );

        for (i, &v) in a_vec.iter().enumerate() {
            assert!(
                v.is_finite(),
                "randn_seeded[WebGPU] value {i} not finite: {v}"
            );
        }
        check_normal_stats(&a_vec, DType::F32);
        check_not_uniform(&a_vec);
    });
}

/// Two seeds sharing the same low 32 bits but differing in the high 32 bits
/// must produce different output. This is the regression test for the old
/// `seed as u32` truncation, which dropped the high word entirely and made
/// such pairs produce IDENTICAL streams. It must fail against that code.
#[cfg(feature = "wgpu")]
#[test]
fn test_randn_seeded_high_word_not_truncated_wgpu() {
    with_wgpu_backend_or_skip(|client, _device| {
        let seed_a: u64 = 42;
        let seed_b: u64 = 42 | (1u64 << 32); // same low 32 bits, different high 32 bits

        let a = client.randn_seeded(&[1000], DType::F32, seed_a).unwrap();
        let b = client.randn_seeded(&[1000], DType::F32, seed_b).unwrap();
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        assert_ne!(
            a_vec, b_vec,
            "seeds sharing low 32 bits but differing in high 32 bits must \
             produce different output on WebGPU (the high word must not be \
             truncated away)"
        );
    });
}

/// `randn_seeded` on WebGPU only supports F32.
#[cfg(feature = "wgpu")]
#[test]
fn test_randn_seeded_unsupported_dtype_wgpu() {
    with_wgpu_backend_or_skip(|client, _device| {
        let result = client.randn_seeded(&[10], DType::I32, 42);
        assert!(
            result.is_err(),
            "randn_seeded on a non-F32 dtype must error on WebGPU"
        );
    });
}
