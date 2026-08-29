// Backend parity tests for I64 / U64 `cumsum` - CUDA vs CPU.
//
// `supported_dtypes("cpu")` in `../mod.rs` never yields I64/U64, so the
// macro-driven tests there never touch 64-bit integer cumsum on any backend.
// This file fills that hole for CUDA, the backend whose 64-bit `cumsum` used
// to wrap on overflow instead of saturating like CPU (see
// `runtime/cuda/kernels/cumulative.cu`).
//
// Every test below is `#[cfg(feature = "cuda")]`, so these imports are too -
// otherwise a non-CUDA build would warn on all of them as unused.
#[cfg(feature = "cuda")]
use numr::dtype::DType;
#[cfg(feature = "cuda")]
use numr::ops::CumulativeOps;
#[cfg(feature = "cuda")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "cuda")]
use crate::common::create_cpu_client;

// ============================================================================
// cumsum I64 - positive overflow saturates to i64::MAX
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_i64_contiguous_positive_overflow_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [i64::MAX - 1, 10, 5];
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_i64 failed");
        assert_eq!(
            cpu_result.to_vec::<i64>(),
            [i64::MAX - 1, i64::MAX, i64::MAX]
        );

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_i64 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::I64);
        assert_eq!(
            result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 cumsum must match CPU element for element"
        );
    });
}

// ============================================================================
// cumsum I64 - negative overflow saturates to i64::MIN
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_i64_contiguous_negative_overflow_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [i64::MIN + 1, -10, -5];
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_i64 failed");
        assert_eq!(
            cpu_result.to_vec::<i64>(),
            [i64::MIN + 1, i64::MIN, i64::MIN]
        );

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_i64 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::I64);
        assert_eq!(
            result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 cumsum must match CPU element for element"
        );
    });
}

// ============================================================================
// cumsum I64 - overflows, then returns into range: pins whatever CPU produces
//
// CPU's accumulator is i128, wide enough that the running total never
// overflows itself for a scan this short - it only saturates at the point of
// narrowing back to i64. So [i64::MAX - 1, 10, -20] never actually clips: the
// true running totals (i64::MAX - 1, i64::MAX + 9, i64::MAX - 11) narrow to
// (i64::MAX - 1, i64::MAX, i64::MAX - 11), recovering the true value at the
// last step instead of staying pinned at i64::MAX. A per-step saturating add
// on the native 64-bit accumulator cannot reproduce this: once it clamps to
// i64::MAX it has thrown away the +9 headroom, so subtracting 20 lands on
// i64::MAX - 20, not i64::MAX - 11. CUDA's 128-bit `Numr128` accumulator
// keeps that headroom, so it must match CPU here.
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_i64_contiguous_overflow_then_recovers_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [i64::MAX - 1, 10, -20];
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_i64 failed");
        assert_eq!(
            cpu_result.to_vec::<i64>(),
            [i64::MAX - 1, i64::MAX, i64::MAX - 11]
        );

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_i64 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::I64);
        assert_eq!(
            result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 cumsum must recover the true value after CPU does"
        );
    });
}

// ============================================================================
// cumsum U64 - overflow saturates to u64::MAX and stays there
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_u64_contiguous_overflow_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [u64::MAX - 1, 5, 3];
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_u64 failed");
        assert_eq!(
            cpu_result.to_vec::<u64>(),
            [u64::MAX - 1, u64::MAX, u64::MAX]
        );

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_u64 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::U64);
        assert_eq!(
            result.to_vec::<u64>(),
            cpu_result.to_vec::<u64>(),
            "CUDA U64 cumsum must match CPU element for element"
        );
    });
}

// ============================================================================
// cumsum I64 / U64 - non-overflowing baseline
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_i64_baseline_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [1i64, -2, 3, -4, 5];
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_i64 failed");
        assert_eq!(cpu_result.to_vec::<i64>(), [1i64, -1, 2, -2, 3]);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[5], &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_i64 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 cumsum baseline must match CPU"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_u64_baseline_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [1u64, 2, 3, 4, 5];
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_u64 failed");
        assert_eq!(cpu_result.to_vec::<u64>(), [1u64, 3, 6, 10, 15]);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[5], &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_u64 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<u64>(),
            cpu_result.to_vec::<u64>(),
            "CUDA U64 cumsum baseline must match CPU"
        );
    });
}

// ============================================================================
// cumsum I64 / U64 - strided path (scan along a non-last dimension)
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_i64_strided_overflow_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Shape [3, 2], scan along dim 0 (not the last dim), so the kernel
        // takes the strided path. Column 0 overflows and saturates; column 1
        // stays small as a control.
        let data = [i64::MAX - 1, 1, 10, 2, 5, 3];
        let shape = [3usize, 2usize];
        let expected = [i64::MAX - 1, 1, i64::MAX, 3, i64::MAX, 6];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_i64 failed");
        assert_eq!(cpu_result.to_vec::<i64>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &shape, &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_strided_i64 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 strided cumsum must match CPU element for element"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumsum_u64_strided_overflow_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Shape [3, 2], scan along dim 0 (not the last dim). Column 0
        // saturates, column 1 stays small as a control.
        let data = [u64::MAX - 1, 1, 5, 2, 3, 3];
        let shape = [3usize, 2usize];
        let expected = [u64::MAX - 1, 1, u64::MAX, 3, u64::MAX, 6];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device).expect("CPU data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_u64 failed");
        assert_eq!(cpu_result.to_vec::<u64>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &shape, &device).expect("CUDA data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_strided_u64 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<u64>(),
            cpu_result.to_vec::<u64>(),
            "CUDA U64 strided cumsum must match CPU element for element"
        );
    });
}

// ============================================================================
// cumprod I32 / I64 / U32 / U64 - saturating to the true product
//
// `cumprod` must report the true mathematical product clamped to the dtype's
// range, the rule `cumsum` follows. CUDA tracks an exact magnitude plus a sign
// parity (see `runtime/cuda/kernels/cumulative.cu`); CPU accumulates in i128.
// Both must land on the same answer.
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_i32_sign_after_saturation_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // True products are 100_000, 10^10, -10^10. A per-step saturating
        // multiply clamps to i32::MAX first and then reports -i32::MAX.
        let data = [100_000i32, 100_000, -1];
        let expected = [100_000i32, i32::MAX, i32::MIN];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::I32);
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "CUDA I32 cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_i32_zero_after_saturation_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // The zero makes the true product 0 for every later element, even
        // though the running magnitude had already left i32.
        let data = [100_000i32, 100_000, 0, 7];
        let expected = [100_000i32, i32::MAX, 0, 0];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[4], &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "CUDA I32 cumprod must match CPU across a zero factor"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_i32_sign_flips_across_a_saturated_run_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Each further negative factor flips the clamp between MIN and MAX.
        let data = [-100_000i32, 100_000, -1, -1];
        let expected = [-100_000i32, i32::MIN, i32::MAX, i32::MIN];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[4], &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "CUDA I32 cumprod must track the sign across a saturated run"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_u32_saturates_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // U32 has no sign to track, so the product pins at u32::MAX.
        let data = [100_000u32, 100_000, 2];
        let expected = [100_000u32, u32::MAX, u32::MAX];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_u32 failed");
        assert_eq!(cpu_result.to_vec::<u32>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_u32 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::U32);
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "CUDA U32 cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_i64_sign_after_saturation_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [4_000_000_000i64, 4_000_000_000, -1];
        let expected = [4_000_000_000i64, i64::MAX, i64::MIN];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i64 failed");
        assert_eq!(cpu_result.to_vec::<i64>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i64 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::I64);
        assert_eq!(
            result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_u64_saturates_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [10_000_000_000u64, 10_000_000_000, 3];
        let expected = [10_000_000_000u64, u64::MAX, u64::MAX];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_u64 failed");
        assert_eq!(cpu_result.to_vec::<u64>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_u64 should succeed on CUDA");
        assert_eq!(result.dtype(), DType::U64);
        assert_eq!(
            result.to_vec::<u64>(),
            cpu_result.to_vec::<u64>(),
            "CUDA U64 cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_i32_baseline_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // A product that never leaves i32 must be untouched by the new state.
        let data = [2i32, 3, -4, 5];
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), [2i32, 6, -24, -120]);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &[4], &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "CUDA I32 cumprod baseline must match CPU"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_cumprod_i32_strided_saturates_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Shape [3, 2], scan along dim 0 (not the last dim), so the kernel
        // takes the strided path. Column 0 saturates and then flips sign;
        // column 1 stays small as a control.
        let data = [100_000i32, 2, 100_000, 3, -1, 4];
        let shape = [3usize, 2usize];
        let expected = [100_000i32, 2, i32::MAX, 6, i32::MIN, 24];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device).expect("CPU data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<CudaRuntime>::from_slice(&data, &shape, &device).expect("CUDA data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_strided_i32 should succeed on CUDA");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "CUDA I32 strided cumprod must match CPU element for element"
        );
    });
}
