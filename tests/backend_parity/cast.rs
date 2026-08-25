// Backend parity tests for TypeConversionOps (cast)
//
// Tests casting between all supported dtype pairs across all backends.
// CPU is the reference; CUDA and WebGPU results must match.
// Comparison reads back in the target dtype natively via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::TypeConversionOps;
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{assert_tensor_allclose, create_cpu_client};

// ============================================================================
// DType Support per Backend for Cast
// ============================================================================

/// All dtypes that participate in cast tests.
/// This is broader than `supported_dtypes` because cast specifically tests
/// conversions between types, including Bool and integer types.
fn cast_dtypes(backend: &str) -> Vec<DType> {
    match backend {
        #[cfg(feature = "wgpu")]
        "wgpu" => vec![DType::F32, DType::I32, DType::U32],
        _ => {
            let mut dtypes = vec![DType::F32, DType::F64, DType::I32, DType::I64, DType::Bool];
            if cfg!(feature = "f16") {
                dtypes.push(DType::F16);
                dtypes.push(DType::BF16);
            }
            if cfg!(feature = "fp8") {
                dtypes.push(DType::FP8E4M3);
                dtypes.push(DType::FP8E5M2);
            }
            dtypes
        }
    }
}

/// Check if a specific cast pair is supported on a backend
fn is_cast_supported(backend: &str, _src: DType, _dst: DType) -> bool {
    let dtypes = cast_dtypes(backend);
    dtypes.contains(&_src) && dtypes.contains(&_dst)
}

// ============================================================================
// Test Data
// ============================================================================

/// Test data covering various value ranges useful for cast verification.
/// Includes positive, negative, zero, fractional, and integer-like values.
const CAST_DATA: &[f64] = &[0.0, 1.0, -1.0, 2.5, -3.5, 42.0, 100.0, 0.125];
const CAST_SHAPE: &[usize] = &[8];

/// Small integer data safe for all dtypes including FP8 (limited range)
const CAST_DATA_SMALL: &[f64] = &[0.0, 1.0, 2.0, 3.0];
const CAST_SHAPE_SMALL: &[usize] = &[4];

/// Bool-oriented data: mix of zero and nonzero values
const BOOL_DATA: &[f64] = &[0.0, 1.0, 0.0, 5.0, -3.0, 0.0, 100.0, 0.0];
const BOOL_SHAPE: &[usize] = &[8];

// ============================================================================
// Core Test Logic
// ============================================================================

fn test_cast_parity(src_dtype: DType, dst_dtype: DType) {
    if src_dtype == dst_dtype {
        return;
    }

    let (cpu_client, cpu_device) = create_cpu_client();

    // Choose test data based on dtype constraints
    let (data, shape) = if dst_dtype == DType::Bool || src_dtype == DType::Bool {
        (BOOL_DATA, BOOL_SHAPE)
    } else if matches!(dst_dtype, DType::FP8E4M3 | DType::FP8E5M2)
        || matches!(src_dtype, DType::FP8E4M3 | DType::FP8E5M2)
    {
        // FP8 has very limited range, use small integers
        (CAST_DATA_SMALL, CAST_SHAPE_SMALL)
    } else {
        (CAST_DATA, CAST_SHAPE)
    };

    // Create source tensor in src_dtype on CPU
    let cpu_src = tensor_from_f64(data, shape, src_dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {src_dtype:?}: {e}"));

    // Cast on CPU (reference)
    let cpu_result = cpu_client
        .cast(&cpu_src, dst_dtype)
        .unwrap_or_else(|e| panic!("CPU cast {src_dtype:?}->{dst_dtype:?} failed: {e}"));

    assert_eq!(
        cpu_result.dtype(),
        dst_dtype,
        "CPU cast output dtype mismatch"
    );

    // CUDA parity
    #[cfg(feature = "cuda")]
    if is_cast_supported("cuda", src_dtype, dst_dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_src = tensor_from_f64(data, shape, src_dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {src_dtype:?}: {e}"));

            let cuda_result = cuda_client
                .cast(&cuda_src, dst_dtype)
                .unwrap_or_else(|e| panic!("CUDA cast {src_dtype:?}->{dst_dtype:?} failed: {e}"));

            assert_eq!(
                cuda_result.dtype(),
                dst_dtype,
                "CUDA cast output dtype mismatch"
            );

            assert_tensor_allclose(
                &cuda_result,
                &cpu_result,
                dst_dtype,
                &format!("cast {src_dtype:?}->{dst_dtype:?} CUDA vs CPU"),
            );
        });
    }

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    if is_cast_supported("wgpu", src_dtype, dst_dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_src = tensor_from_f64(data, shape, src_dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {src_dtype:?}: {e}"));

            let wgpu_result = wgpu_client
                .cast(&wgpu_src, dst_dtype)
                .unwrap_or_else(|e| panic!("WebGPU cast {src_dtype:?}->{dst_dtype:?} failed: {e}"));

            assert_eq!(
                wgpu_result.dtype(),
                dst_dtype,
                "WebGPU cast output dtype mismatch"
            );

            assert_tensor_allclose(
                &wgpu_result,
                &cpu_result,
                dst_dtype,
                &format!("cast {src_dtype:?}->{dst_dtype:?} WebGPU vs CPU"),
            );
        });
    }
}

// ============================================================================
// Float <-> Float Cast Tests
// ============================================================================

#[test]
fn test_cast_f32_f64_parity() {
    test_cast_parity(DType::F32, DType::F64);
}

#[test]
fn test_cast_f64_f32_parity() {
    test_cast_parity(DType::F64, DType::F32);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_f32_f16_parity() {
    test_cast_parity(DType::F32, DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_f16_f32_parity() {
    test_cast_parity(DType::F16, DType::F32);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_f32_bf16_parity() {
    test_cast_parity(DType::F32, DType::BF16);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_bf16_f32_parity() {
    test_cast_parity(DType::BF16, DType::F32);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_f64_f16_parity() {
    test_cast_parity(DType::F64, DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_f64_bf16_parity() {
    test_cast_parity(DType::F64, DType::BF16);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_f16_bf16_parity() {
    test_cast_parity(DType::F16, DType::BF16);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_bf16_f16_parity() {
    test_cast_parity(DType::BF16, DType::F16);
}

// ============================================================================
// FP8 Cast Tests
// ============================================================================

#[test]
#[cfg(feature = "fp8")]
fn test_cast_f32_fp8e4m3_parity() {
    test_cast_parity(DType::F32, DType::FP8E4M3);
}

#[test]
#[cfg(feature = "fp8")]
fn test_cast_fp8e4m3_f32_parity() {
    test_cast_parity(DType::FP8E4M3, DType::F32);
}

#[test]
#[cfg(feature = "fp8")]
fn test_cast_f32_fp8e5m2_parity() {
    test_cast_parity(DType::F32, DType::FP8E5M2);
}

#[test]
#[cfg(feature = "fp8")]
fn test_cast_fp8e5m2_f32_parity() {
    test_cast_parity(DType::FP8E5M2, DType::F32);
}

#[test]
#[cfg(feature = "fp8")]
fn test_cast_fp8e4m3_fp8e5m2_parity() {
    test_cast_parity(DType::FP8E4M3, DType::FP8E5M2);
}

#[test]
#[cfg(feature = "fp8")]
fn test_cast_fp8e5m2_fp8e4m3_parity() {
    test_cast_parity(DType::FP8E5M2, DType::FP8E4M3);
}

// ============================================================================
// Float <-> Integer Cast Tests
// ============================================================================

#[test]
fn test_cast_f32_i32_parity() {
    test_cast_parity(DType::F32, DType::I32);
}

#[test]
fn test_cast_i32_f32_parity() {
    test_cast_parity(DType::I32, DType::F32);
}

#[test]
fn test_cast_f64_i32_parity() {
    test_cast_parity(DType::F64, DType::I32);
}

#[test]
fn test_cast_i32_f64_parity() {
    test_cast_parity(DType::I32, DType::F64);
}

#[test]
fn test_cast_f32_i64_parity() {
    test_cast_parity(DType::F32, DType::I64);
}

#[test]
fn test_cast_i64_f32_parity() {
    test_cast_parity(DType::I64, DType::F32);
}

// ============================================================================
// Bool Cast Tests
// ============================================================================

#[test]
fn test_cast_f32_bool_parity() {
    test_cast_parity(DType::F32, DType::Bool);
}

#[test]
fn test_cast_bool_f32_parity() {
    test_cast_parity(DType::Bool, DType::F32);
}

#[test]
fn test_cast_f64_bool_parity() {
    test_cast_parity(DType::F64, DType::Bool);
}

#[test]
fn test_cast_bool_f64_parity() {
    test_cast_parity(DType::Bool, DType::F64);
}

#[test]
fn test_cast_i32_bool_parity() {
    test_cast_parity(DType::I32, DType::Bool);
}

#[test]
fn test_cast_bool_i32_parity() {
    test_cast_parity(DType::Bool, DType::I32);
}

#[test]
fn test_cast_bool_i64_parity() {
    test_cast_parity(DType::Bool, DType::I64);
}

#[test]
fn test_cast_i64_bool_parity() {
    test_cast_parity(DType::I64, DType::Bool);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_f16_bool_parity() {
    test_cast_parity(DType::F16, DType::Bool);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_bool_f16_parity() {
    test_cast_parity(DType::Bool, DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_bf16_bool_parity() {
    test_cast_parity(DType::BF16, DType::Bool);
}

#[test]
#[cfg(feature = "f16")]
fn test_cast_bool_bf16_parity() {
    test_cast_parity(DType::Bool, DType::BF16);
}

#[test]
#[cfg(feature = "fp8")]
fn test_cast_fp8e4m3_bool_parity() {
    test_cast_parity(DType::FP8E4M3, DType::Bool);
}

#[test]
#[cfg(feature = "fp8")]
fn test_cast_fp8e5m2_bool_parity() {
    test_cast_parity(DType::FP8E5M2, DType::Bool);
}

// ============================================================================
// Exhaustive All-Pairs Test
// ============================================================================

/// Tests all supported cast pairs for each backend.
/// This catches any gaps in the per-pair tests above.
#[test]
fn test_cast_all_pairs_cpu() {
    let dtypes = cast_dtypes("cpu");
    for &src in &dtypes {
        for &dst in &dtypes {
            if src == dst {
                continue;
            }
            test_cast_parity(src, dst);
        }
    }
}

// ============================================================================
// Empty tensor - CUDA parity
// ============================================================================

/// Before the fix, `elementwise_launch_config(0)` produced `grid.x == 0`,
/// which is an invalid CUDA launch (`launch_cast` in
/// `runtime/cuda/kernels/cast.rs` uses this helper directly). Casting a `[0]`
/// F32 tensor to F64 must succeed and return an empty tensor, matching the
/// CPU backend's natural no-op.
#[cfg(feature = "cuda")]
#[test]
fn test_cast_empty_cuda() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let empty = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let result = cuda_client.cast(&empty, DType::F64);
        assert!(
            result.is_ok(),
            "cast on empty CUDA tensor should succeed, got {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.shape(), &[0]);
        assert_eq!(result.dtype(), DType::F64);
        assert_eq!(result.numel(), 0);
    });
}
