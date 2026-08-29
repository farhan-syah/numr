// Bit-exact parity for the float rows of arange, linspace and eye.
//
// `tolerance_for_dtype` allows 1% relative error for F16/BF16 and 1e-5 for F32,
// which hides the whole class of bug these cases are about: an arange or
// linspace built in the wrong precision is off by one ulp, or by one f16 step,
// and passes the tolerant check in `utility.rs` untouched. So CPU and CUDA are
// compared bit for bit here.
//
// The bounds are chosen so that f32 evaluation and f64 evaluation genuinely
// disagree - a `stop` f32 cannot represent, a step count where
// `delta * i / divisor` and `delta * (i / divisor)` differ, a `start` carrying
// more mantissa than the narrower type holds. Each case comment carries the two
// candidate results and why they differ.
//
// WebGPU is checked with the tolerant helper instead. WGSL has no f64, so its
// shaders evaluate in f32 and cannot match an f64 reference bit for bit; that
// is the hardware, not a bug, and the f32 tolerance covers the gap.

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "wgpu")]
use crate::common::assert_tensor_allclose;
use crate::common::{create_cpu_client, is_dtype_supported};
use numr::dtype::DType;
use numr::ops::UtilityOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Read a tensor back as raw element bit patterns.
///
/// Comparing bits rather than values is the point of this module: two f16 that
/// differ by one step, or two f32 that differ by one ulp, are "close" under
/// every tolerance the parity helpers offer.
#[allow(dead_code)]
fn readback_bits<R: Runtime<DType = DType>>(tensor: &Tensor<R>, dtype: DType) -> Vec<u64> {
    match dtype {
        DType::F32 => tensor
            .to_vec::<f32>()
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        DType::F64 => tensor
            .to_vec::<f64>()
            .into_iter()
            .map(f64::to_bits)
            .collect(),
        #[cfg(feature = "f16")]
        DType::F16 => tensor
            .to_vec::<half::f16>()
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        #[cfg(feature = "f16")]
        DType::BF16 => tensor
            .to_vec::<half::bf16>()
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        other => panic!("readback_bits: unsupported dtype {other:?}"),
    }
}

/// Assert two tensors hold identical element bits.
#[allow(dead_code)]
fn assert_bits_eq<R1, R2>(actual: &Tensor<R1>, expected: &Tensor<R2>, dtype: DType, msg: &str)
where
    R1: Runtime<DType = DType>,
    R2: Runtime<DType = DType>,
{
    let a = readback_bits(actual, dtype);
    let e = readback_bits(expected, dtype);
    assert_eq!(
        a.len(),
        e.len(),
        "{msg}: length mismatch ({} vs {})",
        a.len(),
        e.len()
    );
    for (i, (x, y)) in a.iter().zip(e.iter()).enumerate() {
        assert_eq!(
            x, y,
            "{msg}: element {i} differs bit for bit: {x:#x} vs {y:#x}"
        );
    }
}

/// Float dtypes this module compares exactly.
///
/// FP8 is left to the tolerant test: both backends narrow through f32 on the
/// way to an 8-bit float, so an exact comparison here would be measuring the
/// f32 -> FP8 rounding tables against each other, not the arange arithmetic.
fn exact_dtypes() -> Vec<DType> {
    // `mut` is only reached under `f16`, so silence the lint in the builds that
    // compile the base list alone.
    #[allow(unused_mut)]
    let mut dtypes = vec![DType::F32, DType::F64];
    #[cfg(feature = "f16")]
    {
        dtypes.push(DType::F16);
        dtypes.push(DType::BF16);
    }
    dtypes
}

// ============================================================================
// arange
// ============================================================================

/// `(start, stop, step)` per dtype, each picked so a narrowing that is not the
/// CPU's lands on a different element than the CPU's.
fn arange_cases(dtype: DType) -> Vec<(f64, f64, f64)> {
    // The F16 and BF16 starts below are a midpoint plus 2^-30. 1 + 2^-11 is
    // the midpoint between the F16 values 1.0 (0x3c00) and 1.0009765625
    // (0x3c01), and 1 + 2^-8 the midpoint between the BF16 values 1.0 (0x3f80)
    // and 1.0078125 (0x3f81). The 2^-30 pushes each just past its midpoint, so
    // a single IEEE rounding of the f64 would round up and give 0x3c01 / 0x3f81.
    // CPU gives 0x3c00 / 0x3f80 instead, for two different reasons, and CPU is
    // the reference:
    //
    // F16  - `half::f16::from_f64` on x86-64 with F16C rounds to f32 first.
    //        2^-30 is below half an f32 ulp at 1.0 (2^-24), so the f32 stage
    //        drops it, leaving the value exactly on the F16 midpoint, and
    //        ties-to-even takes it back down to 1.0.
    // BF16  - `half::bf16::from_f64` never touches f32. It discards the low 32
    //        mantissa bits of the f64 outright; 2^-30 sits in that discarded
    //        range, so the sticky bits read as zero, the value is again an
    //        exact midpoint, and ties-to-even gives 1.0.
    //
    // These two starts are the whole point of the module: they are the values
    // that tell the three candidate narrowings apart.
    match dtype {
        // start = 2^24 + 1, the first integer f32 cannot represent. CPU builds
        // 16777218 at i=1 in f64 and stores it exactly; the f32 path starts
        // from 16777216, adds 1, and ties back to 16777216.
        DType::F32 => vec![(16_777_217.0, 16_777_222.0, 1.0)],
        // Contraction guard, not a width one. Written as `start + step * idx`
        // the kernel is compiled with --fmad=true and fuses into one rounding:
        // at i=5, fma(0.1, 5, 0.1) is 0.6000000000000001 where the CPU's
        // separate multiply and add give 0.6.
        DType::F64 => vec![(0.1, 1.3, 0.1)],
        #[cfg(feature = "f16")]
        DType::F16 => vec![(1.0 + 1.0 / 2048.0 + 1.0 / 1_073_741_824.0, 4.5, 1.0)],
        #[cfg(feature = "f16")]
        // Second BF16 start: the midpoint plus 2^-23. That bit is still inside
        // the low 32 mantissa bits `half` discards, so CPU again gives 0x3f80,
        // but it is ABOVE half an f32 ulp - an f32 stage would keep it, see a
        // non-zero sticky, and round up to 0x3f81. This one separates `half`'s
        // BF16 rule from the F16 rule; the 2^-30 case above does not.
        DType::BF16 => vec![
            (1.0 + 1.0 / 256.0 + 1.0 / 1_073_741_824.0, 4.5, 1.0),
            (1.0 + 1.0 / 256.0 + 1.0 / 8_388_608.0, 4.5, 1.0),
        ],
        other => panic!("arange_cases: unsupported dtype {other:?}"),
    }
}

fn check_arange_exact(dtype: DType) {
    let (cpu_client, _cpu_device) = create_cpu_client();

    for (start, stop, step) in arange_cases(dtype) {
        let cpu_result = cpu_client
            .arange(start, stop, step, dtype)
            .expect("CPU arange failed");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _cuda_device| {
                let result = cuda_client
                    .arange(start, stop, step, dtype)
                    .expect("CUDA arange failed");
                assert_bits_eq(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("arange({start},{stop},{step}) CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend_or_skip(|wgpu_client, _wgpu_device| {
                let result = wgpu_client
                    .arange(start, stop, step, dtype)
                    .expect("WebGPU arange failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("arange({start},{stop},{step}) WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_arange_bit_exact_vs_cpu() {
    for dtype in exact_dtypes() {
        check_arange_exact(dtype);
    }
}

// ============================================================================
// linspace
// ============================================================================

/// `(start, stop, steps)` per dtype, each an f32-vs-f64 disagreement.
fn linspace_cases(dtype: DType) -> Vec<(f64, f64, usize)> {
    match dtype {
        DType::F32 => vec![
            // stop = 2^24 + 1, which f32 rounds to 2^24. At i = 2 the CPU
            // computes 1 + 16777216 * 2 / 4 = 8388609, exact in f32. The f32
            // path sees delta = 16777215, forms t = 0.5, and lands on
            // 8388608.5 - a tie between 8388608 and 8388609 that rounds to the
            // even 8388608. One whole unit apart.
            (1.0, 16_777_217.0, 5),
            // Pure expression order: no value here needs more than f32 range.
            // At i = 4 the CPU computes 1 + 1 * 4 / 6 and rounds 1.6666666...
            // to 1.6666666269. Forming t = 4/6 first rounds it to
            // 0.66666668653, and 1 + that is exactly halfway between two f32,
            // so ties-to-even gives 1.6666667461 instead.
            (1.0, 2.0, 7),
        ],
        // The f64 row already took f64 scalars; this case guards the divide,
        // which --use_fast_math leaves at full precision for f64 but not f32.
        DType::F64 => vec![(1.0, 16_777_217.0, 5), (0.1, 100.7, 11)],
        #[cfg(feature = "f16")]
        // At i = 1 the exact value is 0.3 + 4097.7 / 6 = 683.25, and f64
        // evaluation lands a hair under it, so it rounds to the F16 value 683.
        // f32 evaluation lands a hair over and rounds to 683.5 - one F16 step,
        // 0.07% relative, far inside the 1% the tolerant helper allows.
        DType::F16 => vec![(0.3, 4098.0, 7)],
        #[cfg(feature = "f16")]
        // At i = 5 the exact value is -1 + 101.7 * 5 / 6 = 83.75, exactly the
        // midpoint between the BF16 values 83.5 and 84.0, so the f64 tie rounds
        // to even and gives 84. f32 evaluation lands just below the midpoint
        // and gives 83.5.
        DType::BF16 => vec![(-1.0, 100.7, 7)],
        other => panic!("linspace_cases: unsupported dtype {other:?}"),
    }
}

fn check_linspace_exact(dtype: DType) {
    let (cpu_client, _cpu_device) = create_cpu_client();

    for (start, stop, steps) in linspace_cases(dtype) {
        let cpu_result = cpu_client
            .linspace(start, stop, steps, dtype)
            .expect("CPU linspace failed");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _cuda_device| {
                let result = cuda_client
                    .linspace(start, stop, steps, dtype)
                    .expect("CUDA linspace failed");
                assert_bits_eq(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("linspace({start},{stop},{steps}) CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend_or_skip(|wgpu_client, _wgpu_device| {
                let result = wgpu_client
                    .linspace(start, stop, steps, dtype)
                    .expect("WebGPU linspace failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("linspace({start},{stop},{steps}) WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_linspace_bit_exact_vs_cpu() {
    for dtype in exact_dtypes() {
        check_linspace_exact(dtype);
    }
}

// ============================================================================
// eye
// ============================================================================

/// Eye stores only 1 and 0, which every float dtype holds exactly, so no bound
/// can make the backends disagree numerically. The exact check is here as the
/// regression guard for the narrowing itself: the row shares the macro that
/// arange and linspace use, and a wrong store would show up as a bit
/// difference rather than as a tolerance failure.
fn check_eye_exact(dtype: DType) {
    let (cpu_client, _cpu_device) = create_cpu_client();

    let cases: Vec<(usize, Option<usize>)> = vec![(3, None), (2, Some(5)), (5, Some(2)), (1, None)];

    for (n, m) in cases {
        let cpu_result = cpu_client.eye(n, m, dtype).expect("CPU eye failed");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _cuda_device| {
                let result = cuda_client.eye(n, m, dtype).expect("CUDA eye failed");
                assert_bits_eq(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("eye({n},{m:?}) CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend_or_skip(|wgpu_client, _wgpu_device| {
                let result = wgpu_client.eye(n, m, dtype).expect("WebGPU eye failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("eye({n},{m:?}) WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_eye_bit_exact_vs_cpu() {
    for dtype in exact_dtypes() {
        check_eye_exact(dtype);
    }
}
