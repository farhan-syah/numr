// Backend parity tests for RandomOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes (F32, F64, F16, BF16, FP8).
// Random operations produce backend-specific values - we test shape, dtype, and statistical
// properties rather than exact value parity.

use numr::dtype::DType;
use numr::ops::RandomOps;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{ToF64, create_cpu_client, is_dtype_supported, supported_dtypes};

/// Check uniform distribution: all values in [0, 1) for floating-point dtypes
fn check_uniform_range<T: ToF64>(vals: &[T], dtype: DType) {
    for (i, &v) in vals.iter().enumerate() {
        let f = v.to_f64();
        assert!(
            (0.0..1.0).contains(&f),
            "rand[{dtype:?}] value {i} out of range [0, 1): {f}"
        );
    }
}

/// Check normal distribution: mean ≈ 0, var ≈ 1 for floating-point dtypes
fn check_normal_stats<T: ToF64>(vals: &[T], dtype: DType) {
    let n = vals.len() as f64;
    let mean: f64 = vals.iter().map(|&x| x.to_f64()).sum::<f64>() / n;
    let var: f64 = vals
        .iter()
        .map(|&x| {
            let d = x.to_f64() - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    // Tolerance depends on dtype precision
    let (mean_tol, var_tol) = match dtype {
        DType::F64 => (0.05, 0.1),
        DType::F32 => (0.15, 0.2),
        DType::F16 | DType::BF16 => (0.3, 0.5),
        DType::FP8E4M3 | DType::FP8E5M2 => (1.0, 2.0), // Very coarse
        _ => (0.15, 0.2),
    };

    assert!(
        mean.abs() < mean_tol,
        "randn[{dtype:?}] mean too far from 0: {mean} (tolerance: {mean_tol})"
    );
    assert!(
        (var - 1.0).abs() < var_tol,
        "randn[{dtype:?}] variance too far from 1: {var} (tolerance: {var_tol})"
    );
}

/// Test rand() produces correct shape, dtype, and values in [0, 1) on all backends
#[test]
fn test_rand_invariants_all_backends() {
    for dtype in supported_dtypes("cpu") {
        // Skip integer types - rand() is for floating-point only
        if matches!(dtype, DType::I32 | DType::I64 | DType::U32 | DType::Bool) {
            continue;
        }

        let (cpu_client, _) = create_cpu_client();

        // CPU baseline: verify shape, dtype, range
        let cpu = cpu_client
            .rand(&[4096], dtype)
            .unwrap_or_else(|e| panic!("CPU rand failed for {dtype:?}: {e}"));
        assert_eq!(cpu.shape(), &[4096]);
        assert_eq!(cpu.dtype(), dtype);

        macro_rules! check_cpu {
            ($T:ty) => {{
                let vals = cpu.to_vec::<$T>();
                check_uniform_range(&vals, dtype);
            }};
        }

        match dtype {
            DType::F64 => check_cpu!(f64),
            DType::F32 => check_cpu!(f32),
            #[cfg(feature = "f16")]
            DType::F16 => check_cpu!(half::f16),
            #[cfg(feature = "f16")]
            DType::BF16 => check_cpu!(half::bf16),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => check_cpu!(numr::dtype::FP8E4M3),
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => check_cpu!(numr::dtype::FP8E5M2),
            _ => {}
        }

        // CUDA: verify same invariants
        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _| {
                let result = cuda_client
                    .rand(&[4096], dtype)
                    .unwrap_or_else(|e| panic!("CUDA rand failed for {dtype:?}: {e}"));
                assert_eq!(result.shape(), &[4096]);
                assert_eq!(result.dtype(), dtype);

                macro_rules! check_cuda {
                    ($T:ty) => {{
                        let vals = result.to_vec::<$T>();
                        check_uniform_range(&vals, dtype);
                    }};
                }

                match dtype {
                    DType::F64 => check_cuda!(f64),
                    DType::F32 => check_cuda!(f32),
                    #[cfg(feature = "f16")]
                    DType::F16 => check_cuda!(half::f16),
                    #[cfg(feature = "f16")]
                    DType::BF16 => check_cuda!(half::bf16),
                    #[cfg(feature = "fp8")]
                    DType::FP8E4M3 => check_cuda!(numr::dtype::FP8E4M3),
                    #[cfg(feature = "fp8")]
                    DType::FP8E5M2 => check_cuda!(numr::dtype::FP8E5M2),
                    _ => {}
                }
            });
        }

        // WebGPU: verify same invariants
        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, _| {
                let result = wgpu_client
                    .rand(&[4096], dtype)
                    .unwrap_or_else(|e| panic!("WebGPU rand failed for {dtype:?}: {e}"));
                assert_eq!(result.shape(), &[4096]);
                assert_eq!(result.dtype(), dtype);

                macro_rules! check_wgpu {
                    ($T:ty) => {{
                        let vals = result.to_vec::<$T>();
                        check_uniform_range(&vals, dtype);
                    }};
                }

                if dtype == DType::F32 {
                    check_wgpu!(f32); // WebGPU: F32 only
                }
            });
        }
    }
}

/// Test randn() produces correct shape, dtype, and normal distribution on all backends
#[test]
fn test_randn_invariants_all_backends() {
    for dtype in supported_dtypes("cpu") {
        // Skip integer types - randn() is for floating-point only
        if matches!(dtype, DType::I32 | DType::I64 | DType::U32 | DType::Bool) {
            continue;
        }

        let (cpu_client, _) = create_cpu_client();

        // CPU baseline: verify shape, dtype, normal distribution
        // Use 10000 samples to reduce flakiness (SE ≈ 0.01 vs 0.016 at 4096)
        let cpu = cpu_client
            .randn(&[10000], dtype)
            .unwrap_or_else(|e| panic!("CPU randn failed for {dtype:?}: {e}"));
        assert_eq!(cpu.shape(), &[10000]);
        assert_eq!(cpu.dtype(), dtype);

        macro_rules! check_cpu {
            ($T:ty) => {{
                let vals = cpu.to_vec::<$T>();
                check_normal_stats(&vals, dtype);
            }};
        }

        match dtype {
            DType::F64 => check_cpu!(f64),
            DType::F32 => check_cpu!(f32),
            #[cfg(feature = "f16")]
            DType::F16 => check_cpu!(half::f16),
            #[cfg(feature = "f16")]
            DType::BF16 => check_cpu!(half::bf16),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => check_cpu!(numr::dtype::FP8E4M3),
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => check_cpu!(numr::dtype::FP8E5M2),
            _ => {}
        }

        // CUDA: verify same invariants
        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _| {
                let result = cuda_client
                    .randn(&[4096], dtype)
                    .unwrap_or_else(|e| panic!("CUDA randn failed for {dtype:?}: {e}"));
                assert_eq!(result.shape(), &[4096]);
                assert_eq!(result.dtype(), dtype);

                macro_rules! check_cuda {
                    ($T:ty) => {{
                        let vals = result.to_vec::<$T>();
                        check_normal_stats(&vals, dtype);
                    }};
                }

                match dtype {
                    DType::F64 => check_cuda!(f64),
                    DType::F32 => check_cuda!(f32),
                    #[cfg(feature = "f16")]
                    DType::F16 => check_cuda!(half::f16),
                    #[cfg(feature = "f16")]
                    DType::BF16 => check_cuda!(half::bf16),
                    #[cfg(feature = "fp8")]
                    DType::FP8E4M3 => check_cuda!(numr::dtype::FP8E4M3),
                    #[cfg(feature = "fp8")]
                    DType::FP8E5M2 => check_cuda!(numr::dtype::FP8E5M2),
                    _ => {}
                }
            });
        }

        // WebGPU: verify same invariants
        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, _| {
                let result = wgpu_client
                    .randn(&[4096], dtype)
                    .unwrap_or_else(|e| panic!("WebGPU randn failed for {dtype:?}: {e}"));
                assert_eq!(result.shape(), &[4096]);
                assert_eq!(result.dtype(), dtype);

                macro_rules! check_wgpu {
                    ($T:ty) => {{
                        let vals = result.to_vec::<$T>();
                        check_normal_stats(&vals, dtype);
                    }};
                }

                if dtype == DType::F32 {
                    check_wgpu!(f32); // WebGPU: F32 only
                }
            });
        }
    }
}

/// Test randint() produces correct shape, dtype, and values in [low, high) on all backends
#[test]
fn test_randint_invariants_all_backends() {
    // randint() is I32-only
    let dtype = DType::I32;
    let (cpu_client, _) = create_cpu_client();

    // CPU baseline: verify shape, dtype, range
    let cpu = cpu_client
        .randint(-7, 9, &[2048], dtype)
        .unwrap_or_else(|e| panic!("CPU randint failed for {dtype:?}: {e}"));
    assert_eq!(cpu.shape(), &[2048]);
    assert_eq!(cpu.dtype(), dtype);
    let cpu_vals: Vec<i32> = cpu.to_vec();
    assert!(cpu_vals.iter().all(|&x| (-7..9).contains(&x)));

    // CUDA: verify same invariants
    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, _| {
            let result = cuda_client
                .randint(-7, 9, &[2048], dtype)
                .unwrap_or_else(|e| panic!("CUDA randint failed for {dtype:?}: {e}"));
            assert_eq!(result.shape(), &[2048]);
            assert_eq!(result.dtype(), dtype);
            let vals: Vec<i32> = result.to_vec();
            assert!(vals.iter().all(|&x| (-7..9).contains(&x)));
        });
    }

    // WebGPU: verify same invariants
    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, _| {
            let result = wgpu_client
                .randint(-7, 9, &[2048], dtype)
                .unwrap_or_else(|e| panic!("WebGPU randint failed for {dtype:?}: {e}"));
            assert_eq!(result.shape(), &[2048]);
            assert_eq!(result.dtype(), dtype);
            let vals: Vec<i32> = result.to_vec();
            assert!(vals.iter().all(|&x| (-7..9).contains(&x)));
        });
    }
}

/// Test rand() with multidimensional shapes on all backends
#[test]
fn test_rand_shape_dtype_all_backends() {
    for dtype in supported_dtypes("cpu") {
        // Skip integer types - rand() is for floating-point only
        if matches!(dtype, DType::I32 | DType::I64 | DType::U32 | DType::Bool) {
            continue;
        }

        let (cpu_client, _) = create_cpu_client();

        // CPU baseline
        let cpu = cpu_client
            .rand(&[2, 3, 4], dtype)
            .unwrap_or_else(|e| panic!("CPU rand shape test failed for {dtype:?}: {e}"));
        assert_eq!(cpu.shape(), &[2, 3, 4]);
        assert_eq!(cpu.dtype(), dtype);

        // CUDA
        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _| {
                let result = cuda_client
                    .rand(&[2, 3, 4], dtype)
                    .unwrap_or_else(|e| panic!("CUDA rand shape test failed for {dtype:?}: {e}"));
                assert_eq!(result.shape(), &[2, 3, 4]);
                assert_eq!(result.dtype(), dtype);
            });
        }

        // WebGPU
        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, _| {
                let result = wgpu_client
                    .rand(&[2, 3, 4], dtype)
                    .unwrap_or_else(|e| panic!("WebGPU rand shape test failed for {dtype:?}: {e}"));
                assert_eq!(result.shape(), &[2, 3, 4]);
                assert_eq!(result.dtype(), dtype);
            });
        }
    }
}

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
    for dtype in supported_dtypes("cpu") {
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
