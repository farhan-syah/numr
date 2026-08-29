// Backend parity tests for integer `cumsum` / `cumprod` - WebGPU vs CPU.
//
// `supported_dtypes("cpu")` in `../mod.rs` never yields I32/U32, so the
// macro-driven tests there never touch integer cumulative ops on any
// backend. This file fills that hole for WebGPU, the backend whose integer
// `cumsum` used to wrap instead of saturate (see cumsum_i32.wgsl /
// cumsum_u32.wgsl).

// Every test below is `#[cfg(feature = "wgpu")]`, so these imports are too -
// otherwise a non-WebGPU build would warn on all of them as unused.
#[cfg(feature = "wgpu")]
use numr::dtype::DType;
#[cfg(feature = "wgpu")]
use numr::ops::CumulativeOps;
#[cfg(feature = "wgpu")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "wgpu")]
use crate::common::create_cpu_client;

// ============================================================================
// cumsum I32 - overflow-and-recover, contiguous and strided
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_cumsum_i32_contiguous_overflow_recovers_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Pins the same sequence as CPU's
        // `test_cumsum_i32_accumulates_in_a_wider_integer`: the running total
        // overflows i32 on the second element and comes back into range on
        // the third. A per-step saturating add cannot reproduce this.
        let data = [2_000_000_000i32, 2_000_000_000, -2_000_000_000];
        let expected = [2_000_000_000i32, i32::MAX, 2_000_000_000];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU I32 data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[3], &device).expect("WGPU I32 data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_i32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::I32);
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 cumsum must match CPU element for element"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumsum_i32_strided_overflow_recovers_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Shape [3, 2], scan along dim 0 (not the last dim), so the kernel
        // takes the strided path. Column 0 overflows and recovers like the
        // contiguous case; column 1 stays small as a control.
        let data = [2_000_000_000i32, 1, 2_000_000_000, 2, -2_000_000_000, 3];
        let shape = [3usize, 2usize];
        let expected = [2_000_000_000i32, 1, i32::MAX, 3, 2_000_000_000, 6];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device).expect("CPU I32 data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &shape, &device).expect("WGPU I32 data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_strided_i32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::I32);
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 strided cumsum must match CPU element for element"
        );
    });
}

// ============================================================================
// cumsum U32 - saturation, contiguous and strided
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_cumsum_u32_contiguous_saturates_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Running total crosses u32::MAX and, since U32 inputs never go
        // negative, stays saturated for the rest of the scan.
        let data = [4_000_000_000u32, 1_000_000_000, 5];
        let expected = [4_000_000_000u32, u32::MAX, u32::MAX];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU U32 data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_u32 failed");
        assert_eq!(cpu_result.to_vec::<u32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[3], &device).expect("WGPU U32 data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_u32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::U32);
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "WebGPU U32 cumsum must match CPU element for element"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumsum_u32_strided_saturates_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Shape [3, 2], scan along dim 0 (not the last dim). Column 0
        // saturates, column 1 stays small as a control.
        let data = [4_000_000_000u32, 1, 1_000_000_000, 2, 5, 3];
        let shape = [3usize, 2usize];
        let expected = [4_000_000_000u32, 1, u32::MAX, 3, u32::MAX, 6];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device).expect("CPU U32 data");
        let cpu_result = cpu_client.cumsum(&a_cpu, 0).expect("CPU cumsum_u32 failed");
        assert_eq!(cpu_result.to_vec::<u32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &shape, &device).expect("WGPU U32 data");
        let result = client
            .cumsum(&a, 0)
            .expect("cumsum_strided_u32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::U32);
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "WebGPU U32 strided cumsum must match CPU element for element"
        );
    });
}

// ============================================================================
// cumprod I32 / U32 - non-overflowing baseline
//
// A product that never leaves the dtype must be untouched by the saturating
// state the shaders now carry. The overflow cases follow further down.
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_i32_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [1i32, -2, 3, -4];
        let expected = [1i32, -2, -6, 24];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("CPU I32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).expect("WGPU I32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::I32);
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_u32_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = [1u32, 2, 3, 4];
        let expected = [1u32, 2, 6, 24];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("CPU U32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_u32 failed");
        assert_eq!(cpu_result.to_vec::<u32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).expect("WGPU U32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_u32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::U32);
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "WebGPU U32 cumprod must match CPU element for element"
        );
    });
}

// ============================================================================
// cumprod I32 / U32 - saturating to the true product
//
// `cumprod` must report the true mathematical product clamped to the dtype's
// range, the rule `cumsum` follows. WebGPU tracks an exact magnitude plus a
// sign parity (see cumprod_i32.wgsl and int_saturate.wgsl); CPU accumulates in
// i128. Both must land on the same answer.
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_i32_sign_after_saturation_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // True products are 100_000, 10^10, -10^10. A per-step saturating
        // multiply clamps to i32::MAX first and then reports -i32::MAX.
        let data = [100_000i32, 100_000, -1];
        let expected = [100_000i32, i32::MAX, i32::MIN];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU I32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[3], &device).expect("WGPU I32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::I32);
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_i32_zero_after_saturation_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // The zero makes the true product 0 for every later element, even
        // though the running magnitude had already left i32.
        let data = [100_000i32, 100_000, 0, 7];
        let expected = [100_000i32, i32::MAX, 0, 0];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("CPU I32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).expect("WGPU I32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 shader should exist and succeed on WebGPU");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 cumprod must match CPU across a zero factor"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_i32_sign_flips_across_a_saturated_run_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Each further negative factor flips the clamp between MIN and MAX.
        let data = [-100_000i32, 100_000, -1, -1];
        let expected = [-100_000i32, i32::MIN, i32::MAX, i32::MIN];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).expect("CPU I32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).expect("WGPU I32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_i32 shader should exist and succeed on WebGPU");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 cumprod must track the sign across a saturated run"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_u32_saturates_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // U32 has no sign to track, so the product pins at u32::MAX.
        let data = [100_000u32, 100_000, 2];
        let expected = [100_000u32, u32::MAX, u32::MAX];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device).expect("CPU U32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_u32 failed");
        assert_eq!(cpu_result.to_vec::<u32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[3], &device).expect("WGPU U32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_u32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::U32);
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "WebGPU U32 cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_i32_strided_saturates_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Shape [3, 2], scan along dim 0 (not the last dim), so the kernel
        // takes the strided path. Column 0 saturates and then flips sign;
        // column 1 stays small as a control.
        let data = [100_000i32, 2, 100_000, 3, -1, 4];
        let shape = [3usize, 2usize];
        let expected = [100_000i32, 2, i32::MAX, 6, i32::MIN, 24];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device).expect("CPU I32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &shape, &device).expect("WGPU I32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_strided_i32 shader should exist and succeed on WebGPU");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 strided cumprod must match CPU element for element"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cumprod_u32_strided_saturates_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Shape [3, 2], scan along dim 0. Column 0 saturates, column 1 stays
        // small as a control.
        let data = [100_000u32, 2, 100_000, 3, 2, 4];
        let shape = [3usize, 2usize];
        let expected = [100_000u32, 2, u32::MAX, 6, u32::MAX, 24];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device).expect("CPU U32 data");
        let cpu_result = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("CPU cumprod_u32 failed");
        assert_eq!(cpu_result.to_vec::<u32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&data, &shape, &device).expect("WGPU U32 data");
        let result = client
            .cumprod(&a, 0)
            .expect("cumprod_strided_u32 shader should exist and succeed on WebGPU");
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "WebGPU U32 strided cumprod must match CPU element for element"
        );
    });
}
