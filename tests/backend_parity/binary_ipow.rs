// Backend parity tests for integer `pow` / `pow_scalar` - CUDA vs CPU.
//
// Split out of `binary.rs`, which holds the float/macro-driven BinaryOps
// parity tests. These tests cover the same-shape integer pow kernels, i32/i64
// overflow saturation, exactness past f64's mantissa, and `pow_scalar`'s
// integer-to-F64 promotion rule - none of which reuse the `BinaryOp` enum or
// `TestCase` machinery from `binary.rs`.

// Every test below is `#[cfg(feature = "cuda")]`, so these imports are too -
// otherwise a non-CUDA build would warn on all of them as unused.
#[cfg(feature = "cuda")]
use numr::dtype::DType;
#[cfg(feature = "cuda")]
use numr::ops::BinaryOps;

#[cfg(feature = "cuda")]
use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "cuda")]
use crate::common::create_cpu_client;

// ============================================================================
// Integer same-shape pow - CUDA parity
//
// `pow_broadcast_i32`/`pow_broadcast_i64` existed, but `pow_i32`/`pow_i64`
// (the same-shape, non-broadcast path) did not. Before the fix, `pow` on
// same-shape I32/I64 CUDA tensors failed with a kernel-not-found error,
// while the broadcast path silently worked - producing different behavior
// for the same logical operation depending on shape.
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_pow_i32_same_shape_cuda() {
    with_cuda_backend(|cuda_client, cuda_device| {
        // bases 2,3,4,5 raised to exponents 0,1,2,3: results are exact and
        // small enough to never overflow i32.
        let bases = [2.0, 3.0, 4.0, 5.0];
        let exps = [0.0, 1.0, 2.0, 3.0];
        let expected = [1i32, 3, 16, 125];

        let a = tensor_from_f64(&bases, &[4], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed for bases");
        let b = tensor_from_f64(&exps, &[4], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed for exponents");

        let result = cuda_client
            .pow(&a, &b)
            .expect("pow_i32 same-shape kernel should exist and succeed on CUDA");
        assert_eq!(result.to_vec::<i32>(), expected);

        // Same-shape result must equal the broadcast-path result for an
        // equivalent input (b broadcast from shape [1]).
        let bases2 = [2.0, 3.0, 4.0, 5.0, 6.0];
        let a2 = tensor_from_f64(&bases2, &[5], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed for bases2");
        let b_scalar = tensor_from_f64(&[3.0], &[1], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed for scalar exponent");
        let b_same_shape = tensor_from_f64(
            &[3.0, 3.0, 3.0, 3.0, 3.0],
            &[5],
            DType::I32,
            &cuda_device,
            &cuda_client,
        )
        .expect("CUDA I32 tensor_from_f64 failed for repeated exponent");

        let broadcast_result = cuda_client
            .pow(&a2, &b_scalar)
            .expect("pow_broadcast_i32 should succeed on CUDA");
        let same_shape_result = cuda_client
            .pow(&a2, &b_same_shape)
            .expect("pow_i32 same-shape kernel should succeed on CUDA");
        assert_eq!(
            broadcast_result.to_vec::<i32>(),
            same_shape_result.to_vec::<i32>(),
            "I32 pow same-shape and broadcast paths must agree"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_pow_i64_same_shape_cuda() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let bases = [2.0, 3.0, 4.0, 5.0];
        let exps = [0.0, 1.0, 2.0, 3.0];
        let expected = [1i64, 3, 16, 125];

        let a = tensor_from_f64(&bases, &[4], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for bases");
        let b = tensor_from_f64(&exps, &[4], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for exponents");

        let result = cuda_client
            .pow(&a, &b)
            .expect("pow_i64 same-shape kernel should exist and succeed on CUDA");
        assert_eq!(result.to_vec::<i64>(), expected);

        // Same-shape result must equal the broadcast-path result for an
        // equivalent input (b broadcast from shape [1]).
        let bases2 = [2.0, 3.0, 4.0, 5.0, 6.0];
        let a2 = tensor_from_f64(&bases2, &[5], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for bases2");
        let b_scalar = tensor_from_f64(&[3.0], &[1], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for scalar exponent");
        let b_same_shape = tensor_from_f64(
            &[3.0, 3.0, 3.0, 3.0, 3.0],
            &[5],
            DType::I64,
            &cuda_device,
            &cuda_client,
        )
        .expect("CUDA I64 tensor_from_f64 failed for repeated exponent");

        let broadcast_result = cuda_client
            .pow(&a2, &b_scalar)
            .expect("pow_broadcast_i64 should succeed on CUDA");
        let same_shape_result = cuda_client
            .pow(&a2, &b_same_shape)
            .expect("pow_i64 same-shape kernel should succeed on CUDA");
        assert_eq!(
            broadcast_result.to_vec::<i64>(),
            same_shape_result.to_vec::<i64>(),
            "I64 pow same-shape and broadcast paths must agree"
        );
    });
}

// ============================================================================
// Integer pow overflow and exactness - CUDA vs CPU parity
//
// Two separate defects met here. CPU computed integer `pow` as `f64::powf`
// and cast back, so any I64 result past 2^53 landed on a neighbouring value.
// CUDA computed it exactly by squaring but multiplied in the element type, so
// it wrapped on overflow while CPU's `as` cast saturated. Both backends now
// compute exactly and saturate, so these compare CUDA against CPU and against
// the true values.
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_pow_i32_overflow_saturates_cuda_matches_cpu() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2.0, -2.0, 46341.0, -46341.0, 3.0, 10.0];
        let exps = [40.0, 41.0, 2.0, 3.0, 30.0, 10.0];
        let expected = [i32::MAX, i32::MIN, i32::MAX, i32::MIN, i32::MAX, i32::MAX];

        let a_cpu = tensor_from_f64(&bases, &[6], DType::I32, &cpu_device, &cpu_client)
            .expect("CPU I32 tensor_from_f64 failed for bases");
        let b_cpu = tensor_from_f64(&exps, &[6], DType::I32, &cpu_device, &cpu_client)
            .expect("CPU I32 tensor_from_f64 failed for exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_i32 failed");
        assert_eq!(
            cpu_result.to_vec::<i32>(),
            expected,
            "CPU I32 pow must saturate, not wrap or round through f64"
        );

        let a_cuda = tensor_from_f64(&bases, &[6], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed for bases");
        let b_cuda = tensor_from_f64(&exps, &[6], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed for exponents");
        let cuda_result = cuda_client
            .pow(&a_cuda, &b_cuda)
            .expect("CUDA pow_i32 failed");
        assert_eq!(
            cuda_result.to_vec::<i32>(),
            cpu_result.to_vec::<i32>(),
            "CUDA I32 pow must saturate exactly as CPU does"
        );

        // The broadcast kernels share the same helper; exercise them too.
        let b_one = tensor_from_f64(&[30.0], &[1], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed for scalar exponent");
        let b_one_cpu = tensor_from_f64(&[30.0], &[1], DType::I32, &cpu_device, &cpu_client)
            .expect("CPU I32 tensor_from_f64 failed for scalar exponent");
        let cuda_bcast = cuda_client
            .pow(&a_cuda, &b_one)
            .expect("CUDA pow_broadcast_i32 failed");
        let cpu_bcast = cpu_client
            .pow(&a_cpu, &b_one_cpu)
            .expect("CPU pow broadcast failed");
        assert_eq!(
            cuda_bcast.to_vec::<i32>(),
            cpu_bcast.to_vec::<i32>(),
            "CUDA I32 pow broadcast must match CPU"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_pow_i64_exact_past_f64_mantissa_cuda_matches_cpu() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // 7^20 needs 57 bits, so the old f64 round trip returned a neighbour.
        let bases = [7.0, 3.0, 2.0, -2.0, 10.0];
        let exps = [20.0, 39.0, 62.0, 63.0, 18.0];
        let expected = [
            79792266297612001i64,
            4052555153018976267,
            4611686018427387904,
            i64::MIN,
            1000000000000000000,
        ];

        let a_cpu = tensor_from_f64(&bases, &[5], DType::I64, &cpu_device, &cpu_client)
            .expect("CPU I64 tensor_from_f64 failed for bases");
        let b_cpu = tensor_from_f64(&exps, &[5], DType::I64, &cpu_device, &cpu_client)
            .expect("CPU I64 tensor_from_f64 failed for exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_i64 failed");
        assert_eq!(
            cpu_result.to_vec::<i64>(),
            expected,
            "CPU I64 pow must be exact past f64's 53-bit mantissa"
        );

        let a_cuda = tensor_from_f64(&bases, &[5], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for bases");
        let b_cuda = tensor_from_f64(&exps, &[5], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for exponents");
        let cuda_result = cuda_client
            .pow(&a_cuda, &b_cuda)
            .expect("CUDA pow_i64 failed");
        assert_eq!(
            cuda_result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 pow must match CPU element for element"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_pow_i64_overflow_saturates_cuda_matches_cpu() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let bases = [2.0, -3.0, 7.0, -2.0];
        let exps = [100.0, 101.0, 30.0, 100.0];
        let expected = [i64::MAX, i64::MIN, i64::MAX, i64::MAX];

        let a_cpu = tensor_from_f64(&bases, &[4], DType::I64, &cpu_device, &cpu_client)
            .expect("CPU I64 tensor_from_f64 failed for bases");
        let b_cpu = tensor_from_f64(&exps, &[4], DType::I64, &cpu_device, &cpu_client)
            .expect("CPU I64 tensor_from_f64 failed for exponents");
        let cpu_result = cpu_client.pow(&a_cpu, &b_cpu).expect("CPU pow_i64 failed");
        assert_eq!(
            cpu_result.to_vec::<i64>(),
            expected,
            "CPU I64 pow must saturate to the dtype bound with the right sign"
        );

        let a_cuda = tensor_from_f64(&bases, &[4], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for bases");
        let b_cuda = tensor_from_f64(&exps, &[4], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed for exponents");
        let cuda_result = cuda_client
            .pow(&a_cuda, &b_cuda)
            .expect("CUDA pow_i64 failed");
        assert_eq!(
            cuda_result.to_vec::<i64>(),
            cpu_result.to_vec::<i64>(),
            "CUDA I64 pow must saturate exactly as CPU does"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_pow_scalar_integer_cuda_matches_cpu() {
    use numr::ops::ScalarOps;

    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        // I32: exact small results, then overflow in both signs.
        let i32_bases = [2.0, -2.0, 3.0, 46341.0, 1.0, 0.0];
        let a_cpu = tensor_from_f64(&i32_bases, &[6], DType::I32, &cpu_device, &cpu_client)
            .expect("CPU I32 tensor_from_f64 failed");
        let a_cuda = tensor_from_f64(&i32_bases, &[6], DType::I32, &cuda_device, &cuda_client)
            .expect("CUDA I32 tensor_from_f64 failed");

        // A non-negative whole exponent keeps the integer dtype on both
        // backends. Negative and fractional exponents promote, and are covered
        // separately below.
        for scalar in [0.0, 1.0, 3.0, 40.0] {
            let cpu_result = cpu_client
                .pow_scalar(&a_cpu, scalar)
                .unwrap_or_else(|e| panic!("CPU pow_scalar({scalar}) failed: {e}"));
            let cuda_result = cuda_client
                .pow_scalar(&a_cuda, scalar)
                .unwrap_or_else(|e| panic!("CUDA pow_scalar({scalar}) failed: {e}"));
            assert_eq!(cpu_result.dtype(), DType::I32);
            assert_eq!(cuda_result.dtype(), DType::I32);
            assert_eq!(
                cuda_result.to_vec::<i32>(),
                cpu_result.to_vec::<i32>(),
                "I32 pow_scalar({scalar}) must match CPU"
            );
        }

        // I64: the exactness case pow_scalar could not reach before.
        let i64_bases = [7.0, 3.0, -2.0, 2.0];
        let b_cpu = tensor_from_f64(&i64_bases, &[4], DType::I64, &cpu_device, &cpu_client)
            .expect("CPU I64 tensor_from_f64 failed");
        let b_cuda = tensor_from_f64(&i64_bases, &[4], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed");

        let cpu_20 = cpu_client
            .pow_scalar(&b_cpu, 20.0)
            .expect("CPU I64 pow_scalar(20) failed");
        assert_eq!(
            cpu_20.to_vec::<i64>()[0],
            79792266297612001i64,
            "CPU I64 pow_scalar must be exact past f64's mantissa"
        );

        for scalar in [0.0, 3.0, 20.0, 100.0] {
            let cpu_result = cpu_client
                .pow_scalar(&b_cpu, scalar)
                .unwrap_or_else(|e| panic!("CPU I64 pow_scalar({scalar}) failed: {e}"));
            let cuda_result = cuda_client
                .pow_scalar(&b_cuda, scalar)
                .unwrap_or_else(|e| panic!("CUDA I64 pow_scalar({scalar}) failed: {e}"));
            assert_eq!(
                cuda_result.to_vec::<i64>(),
                cpu_result.to_vec::<i64>(),
                "I64 pow_scalar({scalar}) must match CPU"
            );
        }

        // A negative or fractional exponent has no integer result, so both
        // backends promote the output to F64 and return the real value.
        let promo_bases = [9.0, 7.0, 2.0, 1.0];
        let p_cpu = tensor_from_f64(&promo_bases, &[4], DType::I64, &cpu_device, &cpu_client)
            .expect("CPU I64 tensor_from_f64 failed");
        let p_cuda = tensor_from_f64(&promo_bases, &[4], DType::I64, &cuda_device, &cuda_client)
            .expect("CUDA I64 tensor_from_f64 failed");

        for scalar in [0.5, 1.5, -0.5, -1.0] {
            let cpu_result = cpu_client
                .pow_scalar(&p_cpu, scalar)
                .unwrap_or_else(|e| panic!("CPU I64 pow_scalar({scalar}) failed: {e}"));
            let cuda_result = cuda_client
                .pow_scalar(&p_cuda, scalar)
                .unwrap_or_else(|e| panic!("CUDA I64 pow_scalar({scalar}) failed: {e}"));
            assert_eq!(
                cpu_result.dtype(),
                DType::F64,
                "CPU I64 pow_scalar({scalar}) must promote to F64"
            );
            assert_eq!(
                cuda_result.dtype(),
                DType::F64,
                "CUDA I64 pow_scalar({scalar}) must promote to F64"
            );
            let cpu_values = cpu_result.to_vec::<f64>();
            let cuda_values = cuda_result.to_vec::<f64>();
            for (i, (&c, &g)) in cpu_values.iter().zip(cuda_values.iter()).enumerate() {
                assert!(
                    (c - g).abs() <= 1e-12 * c.abs().max(1.0),
                    "I64 pow_scalar({scalar}) element {i}: CPU {c} vs CUDA {g}"
                );
            }
        }

        // The promoted values are the real results, not truncated integers.
        let roots = cpu_client
            .pow_scalar(&p_cpu, 0.5)
            .expect("CPU I64 pow_scalar(0.5) failed")
            .to_vec::<f64>();
        assert_eq!(roots[0], 3.0, "9 ** 0.5 is 3.0, not 2 or 3");
        assert!(
            (roots[1] - 2.6457513110645906).abs() < 1e-12,
            "7 ** 0.5 must be the real square root, got {}",
            roots[1]
        );

        let inverse = cpu_client
            .pow_scalar(&p_cpu, -1.0)
            .expect("CPU I64 pow_scalar(-1) failed")
            .to_vec::<f64>();
        assert_eq!(inverse[2], 0.5, "2 ** -1 is 0.5, not 0");

        // The input tensor keeps its dtype; promotion produces a new tensor.
        assert_eq!(p_cpu.dtype(), DType::I64);
    });
}
