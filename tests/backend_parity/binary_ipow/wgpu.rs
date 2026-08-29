// Backend parity tests for integer `pow` / `pow_scalar` - WebGPU vs CPU.
//
// WebGPU is 32-bit only, so I32 and U32 are the whole integer surface here.
// The shaders transliterate `runtime/cpu/kernels/ipow.rs`, so every case below
// compares against CPU rather than against a hand-written constant alone.

// Every test below is `#[cfg(feature = "wgpu")]`, so these imports are too -
// otherwise a non-WebGPU build would warn on all of them as unused.
#[cfg(feature = "wgpu")]
use numr::dtype::DType;
#[cfg(feature = "wgpu")]
use numr::ops::{BinaryOps, ScalarOps};
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
// Integer same-shape pow - WebGPU vs CPU
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_i32_same_shape_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2i32, 3, 4, 5, -2, -3, 0, 1];
        let exps = [0i32, 1, 2, 3, 3, 4, 5, 100];
        let expected = [1i32, 3, 16, 125, -8, 81, 0, 1];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&bases, &[8], &cpu_device).expect("CPU I32 bases");
        let b_cpu =
            Tensor::<CpuRuntime>::from_slice(&exps, &[8], &cpu_device).expect("CPU I32 exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_i32 failed");
        assert_eq!(cpu_result.to_vec::<i32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&bases, &[8], &device).expect("WGPU I32 bases");
        let b =
            Tensor::<WgpuRuntime>::from_slice(&exps, &[8], &device).expect("WGPU I32 exponents");
        let result = client
            .pow(&a, &b)
            .expect("pow_i32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::I32);
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 pow must match CPU element for element"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_u32_same_shape_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2u32, 3, 4, 5, 0, 1, 7];
        let exps = [0u32, 1, 2, 3, 4, 100, 5];
        let expected = [1u32, 3, 16, 125, 0, 1, 16807];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&bases, &[7], &cpu_device).expect("CPU U32 bases");
        let b_cpu =
            Tensor::<CpuRuntime>::from_slice(&exps, &[7], &cpu_device).expect("CPU U32 exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_u32 failed");
        assert_eq!(cpu_result.to_vec::<u32>(), expected);

        let a = Tensor::<WgpuRuntime>::from_slice(&bases, &[7], &device).expect("WGPU U32 bases");
        let b =
            Tensor::<WgpuRuntime>::from_slice(&exps, &[7], &device).expect("WGPU U32 exponents");
        let result = client
            .pow(&a, &b)
            .expect("pow_u32 shader should exist and succeed on WebGPU");
        assert_eq!(result.dtype(), DType::U32);
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "WebGPU U32 pow must match CPU element for element"
        );
    });
}

// ============================================================================
// Broadcast pow - WebGPU vs CPU
//
// The broadcast shaders reach the same helper through different index
// arithmetic, so they need their own coverage.
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_broadcast_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2i32, 3, 4, 5, -6];
        let exp = [3i32];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&bases, &[5], &cpu_device).expect("CPU I32 bases");
        let b_cpu =
            Tensor::<CpuRuntime>::from_slice(&exp, &[1], &cpu_device).expect("CPU I32 exponent");
        let cpu_bcast = cpu_client
            .pow(&a_cpu, &b_cpu)
            .expect("CPU I32 pow broadcast failed");
        assert_eq!(cpu_bcast.to_vec::<i32>(), [8i32, 27, 64, 125, -216]);

        let a = Tensor::<WgpuRuntime>::from_slice(&bases, &[5], &device).expect("WGPU I32 bases");
        let b = Tensor::<WgpuRuntime>::from_slice(&exp, &[1], &device).expect("WGPU I32 exponent");
        let bcast = client
            .pow(&a, &b)
            .expect("broadcast_pow_i32 shader should succeed on WebGPU");
        assert_eq!(
            bcast.to_vec::<i32>(),
            cpu_bcast.to_vec::<i32>(),
            "WebGPU I32 pow broadcast must match CPU"
        );

        // The same-shape and broadcast paths must agree for equivalent inputs.
        let b_full = Tensor::<WgpuRuntime>::from_slice(&[3i32; 5], &[5], &device)
            .expect("WGPU I32 repeated exponent");
        let same_shape = client
            .pow(&a, &b_full)
            .expect("pow_i32 shader should succeed on WebGPU");
        assert_eq!(
            bcast.to_vec::<i32>(),
            same_shape.to_vec::<i32>(),
            "I32 pow same-shape and broadcast paths must agree"
        );

        // U32 takes a separate shader, so exercise its broadcast path too.
        let u_bases = [2u32, 3, 4, 5, 6];
        let u_exp = [4u32];
        let ua_cpu =
            Tensor::<CpuRuntime>::from_slice(&u_bases, &[5], &cpu_device).expect("CPU U32 bases");
        let ub_cpu =
            Tensor::<CpuRuntime>::from_slice(&u_exp, &[1], &cpu_device).expect("CPU U32 exponent");
        let u_cpu_bcast = cpu_client
            .pow(&ua_cpu, &ub_cpu)
            .expect("CPU U32 pow broadcast failed");

        let ua =
            Tensor::<WgpuRuntime>::from_slice(&u_bases, &[5], &device).expect("WGPU U32 bases");
        let ub =
            Tensor::<WgpuRuntime>::from_slice(&u_exp, &[1], &device).expect("WGPU U32 exponent");
        let u_bcast = client
            .pow(&ua, &ub)
            .expect("broadcast_pow_u32 shader should succeed on WebGPU");
        assert_eq!(
            u_bcast.to_vec::<u32>(),
            u_cpu_bcast.to_vec::<u32>(),
            "WebGPU U32 pow broadcast must match CPU"
        );
    });
}

// ============================================================================
// Overflow saturation - WebGPU vs CPU
//
// A wrapping multiply would disagree with CPU on exactly the inputs that
// overflow, so both signs of i32 and the u32 bound are pinned here.
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_i32_overflow_saturates_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2i32, -2, 46341, -46341, 3, 10];
        let exps = [40i32, 41, 2, 3, 30, 10];
        let expected = [i32::MAX, i32::MIN, i32::MAX, i32::MIN, i32::MAX, i32::MAX];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&bases, &[6], &cpu_device).expect("CPU I32 bases");
        let b_cpu =
            Tensor::<CpuRuntime>::from_slice(&exps, &[6], &cpu_device).expect("CPU I32 exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_i32 failed");
        assert_eq!(
            cpu_result.to_vec::<i32>(),
            expected,
            "CPU I32 pow must saturate to the dtype bound with the right sign"
        );

        let a = Tensor::<WgpuRuntime>::from_slice(&bases, &[6], &device).expect("WGPU I32 bases");
        let b =
            Tensor::<WgpuRuntime>::from_slice(&exps, &[6], &device).expect("WGPU I32 exponents");
        let result = client.pow(&a, &b).expect("WebGPU pow_i32 failed");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 pow must saturate exactly as CPU does"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_u32_overflow_saturates_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2u32, 65536, 3, 10];
        let exps = [32u32, 2, 30, 10];
        let expected = [u32::MAX; 4];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&bases, &[4], &cpu_device).expect("CPU U32 bases");
        let b_cpu =
            Tensor::<CpuRuntime>::from_slice(&exps, &[4], &cpu_device).expect("CPU U32 exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_u32 failed");
        assert_eq!(
            cpu_result.to_vec::<u32>(),
            expected,
            "CPU U32 pow must saturate to u32::MAX, not wrap"
        );

        let a = Tensor::<WgpuRuntime>::from_slice(&bases, &[4], &device).expect("WGPU U32 bases");
        let b =
            Tensor::<WgpuRuntime>::from_slice(&exps, &[4], &device).expect("WGPU U32 exponents");
        let result = client.pow(&a, &b).expect("WebGPU pow_u32 failed");
        assert_eq!(
            result.to_vec::<u32>(),
            cpu_result.to_vec::<u32>(),
            "WebGPU U32 pow must saturate exactly as CPU does"
        );
    });
}

// ============================================================================
// Negative exponents - tensor-tensor only
//
// `pow` keeps the integer output dtype because an op's output dtype cannot
// depend on tensor data. CPU computes the true fraction in f64 and truncates
// it; the shader reproduces the four outcomes in integer logic.
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_i32_negative_exponent_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [0i32, 1, -1, -1, 2, -2, 7];
        let exps = [-1i32, -5, -3, -2, -1, -1, -4];
        // 0 ** -n is infinity in f64 and the saturating cast gives i32::MAX. Every
        // base past ±1 has a magnitude below 1, which truncates to 0.
        let expected = [i32::MAX, 1, -1, 1, 0, 0, 0];

        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&bases, &[7], &cpu_device).expect("CPU I32 bases");
        let b_cpu =
            Tensor::<CpuRuntime>::from_slice(&exps, &[7], &cpu_device).expect("CPU I32 exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_i32 failed");
        assert_eq!(
            cpu_result.to_vec::<i32>(),
            expected,
            "CPU is the reference for the negative-exponent truncation"
        );

        let a = Tensor::<WgpuRuntime>::from_slice(&bases, &[7], &device).expect("WGPU I32 bases");
        let b =
            Tensor::<WgpuRuntime>::from_slice(&exps, &[7], &device).expect("WGPU I32 exponents");
        let result = client.pow(&a, &b).expect("WebGPU pow_i32 failed");
        assert_eq!(result.dtype(), DType::I32, "pow keeps the integer dtype");
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "WebGPU I32 pow with a negative exponent must match CPU"
        );
    });
}

// ============================================================================
// pow_scalar - WebGPU vs CPU, and the promotion rule
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_scalar_integer_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let i32_bases = [2i32, -2, 3, 46341, 1, 0];
        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&i32_bases, &[6], &cpu_device).expect("CPU I32 bases");
        let a =
            Tensor::<WgpuRuntime>::from_slice(&i32_bases, &[6], &device).expect("WGPU I32 bases");

        for scalar in [0.0, 1.0, 3.0, 40.0] {
            let cpu_result = cpu_client
                .pow_scalar(&a_cpu, scalar)
                .unwrap_or_else(|e| panic!("CPU I32 pow_scalar({scalar}) failed: {e}"));
            let result = client
                .pow_scalar(&a, scalar)
                .unwrap_or_else(|e| panic!("WebGPU I32 pow_scalar({scalar}) failed: {e}"));
            assert_eq!(cpu_result.dtype(), DType::I32);
            assert_eq!(result.dtype(), DType::I32);
            assert_eq!(
                result.to_vec::<i32>(),
                cpu_result.to_vec::<i32>(),
                "I32 pow_scalar({scalar}) must match CPU"
            );
        }

        let u32_bases = [2u32, 3, 7, 65536, 1, 0];
        let u_cpu =
            Tensor::<CpuRuntime>::from_slice(&u32_bases, &[6], &cpu_device).expect("CPU U32 bases");
        let u =
            Tensor::<WgpuRuntime>::from_slice(&u32_bases, &[6], &device).expect("WGPU U32 bases");

        for scalar in [0.0, 1.0, 3.0, 32.0] {
            let cpu_result = cpu_client
                .pow_scalar(&u_cpu, scalar)
                .unwrap_or_else(|e| panic!("CPU U32 pow_scalar({scalar}) failed: {e}"));
            let result = client
                .pow_scalar(&u, scalar)
                .unwrap_or_else(|e| panic!("WebGPU U32 pow_scalar({scalar}) failed: {e}"));
            assert_eq!(result.dtype(), DType::U32);
            assert_eq!(
                result.to_vec::<u32>(),
                cpu_result.to_vec::<u32>(),
                "U32 pow_scalar({scalar}) must match CPU"
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_pow_scalar_non_integral_exponent_errors_on_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2i32, 9, 7, 1];
        let a = Tensor::<WgpuRuntime>::from_slice(&bases, &[4], &device).expect("WGPU I32 bases");
        let a_cpu =
            Tensor::<CpuRuntime>::from_slice(&bases, &[4], &cpu_device).expect("CPU I32 bases");

        // A negative or fractional exponent has no integer result, so the output
        // dtype is F64 - which WebGPU cannot represent. The check runs on the host,
        // before any dispatch, so such an exponent never reaches the integer shader.
        for scalar in [0.5, 1.5, -0.5, -1.0] {
            assert!(
                client.pow_scalar(&a, scalar).is_err(),
                "WebGPU I32 pow_scalar({scalar}) must refuse the F64 output dtype"
            );
            let cpu_result = cpu_client
                .pow_scalar(&a_cpu, scalar)
                .unwrap_or_else(|e| panic!("CPU I32 pow_scalar({scalar}) failed: {e}"));
            assert_eq!(
                cpu_result.dtype(),
                DType::F64,
                "CPU I32 pow_scalar({scalar}) promotes, which is why WebGPU refuses"
            );
        }
    });
}
