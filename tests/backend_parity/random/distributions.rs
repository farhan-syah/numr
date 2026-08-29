// Unseeded rand/randn/randint invariants: shape, dtype, and statistical
// properties (uniform range, normal mean/variance) across CPU/CUDA/WebGPU.

use numr::dtype::DType;
use numr::ops::RandomOps;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{DTypeDomain, ToF64, create_cpu_client, is_dtype_supported, parity_dtypes};

/// Check uniform distribution: all values in [0, 1) for floating-point dtypes
pub(super) fn check_uniform_range<T: ToF64>(vals: &[T], dtype: DType) {
    for (i, &v) in vals.iter().enumerate() {
        let f = v.to_f64();
        assert!(
            (0.0..1.0).contains(&f),
            "rand[{dtype:?}] value {i} out of range [0, 1): {f}"
        );
    }
}

/// Check normal distribution: mean ≈ 0, var ≈ 1 for floating-point dtypes
pub(super) fn check_normal_stats<T: ToF64>(vals: &[T], dtype: DType) {
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
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
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
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
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
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
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
