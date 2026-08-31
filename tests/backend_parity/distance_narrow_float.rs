// Distance metrics on a narrow-float tensor accumulate in f32, not in the
// element type.
//
// Every metric in the distance family builds a running total across the `d`
// components of a pair of vectors. A total kept in F16 or BF16 stops growing
// once the accumulator's spacing exceeds twice the increment, so the tail of a
// long vector is dropped outright. CUDA has always accumulated these dtypes in
// f32 — `AccType<__half>` and `AccType<__nv_bfloat16>` in
// `src/runtime/cuda/kernels/distance.cu` are both `float` — so an element-type
// accumulator on CPU is a backend divergence, not just a precision choice.
//
// The trap this file is built to avoid: `tolerance_for_dtype` allows 1%
// relative error for F16 and BF16, which hides an accumulation-width difference
// on a short vector. Each case below therefore spreads its magnitudes far
// enough apart, over enough terms, that the two widths differ by 3% to 6%. The
// test asserts that separation before it asserts the result, so a case that
// stops distinguishing the two widths fails loudly instead of passing vacuously.
//
// Hand verification of case "manhattan":
//
//   * The row is 1024.0 followed by 256 copies of 0.25, against a row of zeros,
//     so the terms summed are 1024 then 256 x 0.25.
//   * F16 has an 11-bit significand, so across [1024, 2048) it steps by
//     1024 * 2^-10 = 1.0. After the first term the accumulator holds 1024;
//     1024 + 0.25 = 1024.25 is nearer 1024 than 1025, so it rounds straight
//     back. Every one of the 256 small terms is lost: the total stays 1024.
//   * BF16 has an 8-bit significand and steps by 1024 * 2^-7 = 8.0 there, so it
//     loses the small terms the same way, and also stays at 1024.
//   * f32 steps by 2^-13 at 1024, so all 256 terms land: 1024 + 64 = 1088.
//   * 1088 is exactly representable in both F16 (a whole number below 2048) and
//     BF16 (1088 = 8 * 136), so the narrowing at write-out adds no error.
//   * 1088 vs 1024 is a 5.9% gap — six times the 1% dtype tolerance.
//
// Case "sqeuclidean" is the same arithmetic one square earlier: 32.0 then 256
// copies of 0.5 square to 1024 then 256 x 0.25, so it lands on the same 1088
// against 1024.
//
// Hand verification of case "minkowski" (p = 1.5, fractional):
//
//   * The row is 256.0 followed by 4096 copies of 0.25. The powered terms are
//     256^1.5 = 4096 and 0.25^1.5 = 0.125.
//   * F16 steps by 4 across [4096, 8192) and BF16 by 32, so every 0.125 term
//     rounds back and an element-type accumulator holds 4096. f32 keeps them:
//     4096 + 4096 * 0.125 = 4608 = 512 * 9.
//   * The exponent compounds the divergence. `p` pre-rounded into the element
//     type is still 1.5, but the final 1/p is not: F16 rounds 0.666... to
//     0.66650390625 and BF16 to 0.66796875.
//   * Element-type accumulator: F16 gives 4096^0.66650390625 = 255.49, BF16
//     gives 4096^0.66796875 = 258.76.
//   * f32 accumulator: 4608^(2/3) = 64 * 81^(1/3) = 276.912.
//   * 276.912 vs 255.49 is 7.7%, vs 258.76 is 6.6% — both far clear of the 1%
//     tolerance. A single-term or short-vector case would land inside it and
//     prove nothing. The BF16 gap is the narrower of the two, so it sets the
//     tail length: 2048 copies leave it at 2.9%, under the 3x margin the
//     separation guard demands.
//
// WebGPU is absent on purpose: it is a 32-bit backend and carries no
// narrow-float dtype at all.

use numr::dtype::DType;
use numr::ops::{DistanceMetric, DistanceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "cuda")]
use crate::common::assert_tensor_allclose;
use crate::common::{create_cpu_client, is_dtype_supported, tolerance_for_dtype};

/// One metric, with the value each accumulator width produces.
struct Case {
    metric: DistanceMetric,
    name: &'static str,
    /// The leading component, large enough to freeze a narrow accumulator.
    head: f64,
    /// The trailing component, repeated `tail_count` times.
    tail: f64,
    tail_count: usize,
    /// The distance an f32 accumulator produces. CUDA's answer, and the one
    /// this file requires from CPU.
    wide: f64,
    /// The distance an element-type accumulator produces, per dtype. The answer
    /// this test exists to reject.
    narrow_f16: f64,
    narrow_bf16: f64,
}

impl Case {
    /// The first row of the input: `head` followed by `tail_count` copies of
    /// `tail`. The second row is zeros, so each component's contribution is the
    /// component itself.
    fn row(&self) -> Vec<f64> {
        let mut row = Vec::with_capacity(1 + self.tail_count);
        row.push(self.head);
        row.extend(std::iter::repeat_n(self.tail, self.tail_count));
        row
    }

    fn dim(&self) -> usize {
        1 + self.tail_count
    }

    fn narrow_for(&self, dtype: DType) -> f64 {
        match dtype {
            DType::BF16 => self.narrow_bf16,
            _ => self.narrow_f16,
        }
    }
}

const CASES: &[Case] = &[
    Case {
        metric: DistanceMetric::Manhattan,
        name: "manhattan",
        head: 1024.0,
        tail: 0.25,
        tail_count: 256,
        wide: 1088.0,
        narrow_f16: 1024.0,
        narrow_bf16: 1024.0,
    },
    Case {
        metric: DistanceMetric::SquaredEuclidean,
        name: "sqeuclidean",
        head: 32.0,
        tail: 0.5,
        tail_count: 256,
        wide: 1088.0,
        narrow_f16: 1024.0,
        narrow_bf16: 1024.0,
    },
    Case {
        metric: DistanceMetric::Minkowski(1.5),
        name: "minkowski p=1.5",
        head: 256.0,
        tail: 0.25,
        tail_count: 4096,
        wide: 276.912,
        narrow_f16: 255.49,
        narrow_bf16: 258.76,
    },
];

/// Every narrow float the CPU backend carries a distance kernel for.
///
/// FP8 is absent because `src/ops/cpu/distance.rs` casts FP8 tensors to F32
/// before the kernel runs, so FP8 never reaches an element-type accumulator.
const NARROW_FLOATS: &[DType] = &[DType::F16, DType::BF16];

/// Read a single-element result back as f64. Both narrow floats widen to f64
/// exactly, so nothing is lost on the way out.
// Both match arms that read `tensor` are `f16`-gated, so the parameter is
// genuinely unused when that feature is off. Gate the allow the same way
// rather than renaming the parameter, which would hide a real unused
// argument if the body ever stops reading it.
#[cfg_attr(not(feature = "f16"), allow(unused_variables))]
fn read_scalar<R: Runtime<DType = DType>>(tensor: &Tensor<R>, dtype: DType) -> f64 {
    match dtype {
        #[cfg(feature = "f16")]
        DType::F16 => tensor.to_vec::<half::f16>()[0].to_f64(),
        #[cfg(feature = "f16")]
        DType::BF16 => tensor.to_vec::<half::bf16>()[0].to_f64(),
        _ => panic!("read_scalar: not a narrow float: {dtype:?}"),
    }
}

/// CPU accumulates a narrow-float distance in f32.
///
/// Fails against a kernel that accumulates in the element type, which is what
/// the CPU distance kernels did before they adopted `WideAcc`'s f32
/// accumulator.
#[test]
fn distance_metrics_accumulate_narrow_floats_in_f32_on_cpu() {
    let (client, device) = create_cpu_client();

    for case in CASES {
        for &dtype in NARROW_FLOATS {
            if !is_dtype_supported("cpu", dtype) {
                continue;
            }

            let (rtol, _) = tolerance_for_dtype(dtype);
            let narrow = case.narrow_for(dtype);
            let separation = (case.wide - narrow).abs() / case.wide.abs();

            // The case is only worth running if the two accumulator widths are
            // further apart than the dtype tolerance. At 1% rtol a short vector
            // would pass under either width and prove nothing.
            assert!(
                separation > 3.0 * rtol,
                "{} [{dtype:?}]: f32 accumulator gives {}, element-type gives {narrow}; \
                 they differ by {:.3}%, which the {:.1}% tolerance would swallow",
                case.name,
                case.wide,
                separation * 100.0,
                rtol * 100.0
            );

            let row = case.row();
            let d = case.dim();
            let zeros = vec![0.0f64; d];

            let x = tensor_from_f64(&row, &[1, d], dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CPU x tensor failed for {dtype:?}: {e}"));
            let y = tensor_from_f64(&zeros, &[1, d], dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CPU y tensor failed for {dtype:?}: {e}"));
            let got = read_scalar(
                &client
                    .cdist(&x, &y, case.metric)
                    .unwrap_or_else(|e| panic!("CPU cdist {} [{dtype:?}]: {e}", case.name)),
                dtype,
            );
            assert!(
                (got - case.wide).abs() / case.wide.abs() <= rtol,
                "cdist {} [{dtype:?}]: got {got}, an f32 accumulator gives {}, \
                 an element-type accumulator gives {narrow}",
                case.name,
                case.wide
            );

            // pdist over the same two rows must agree: it is the same reduction
            // reached through the other entry point.
            let mut stacked = row.clone();
            stacked.extend_from_slice(&zeros);
            let pair = tensor_from_f64(&stacked, &[2, d], dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CPU pair tensor failed for {dtype:?}: {e}"));
            let got_pdist = read_scalar(
                &client
                    .pdist(&pair, case.metric)
                    .unwrap_or_else(|e| panic!("CPU pdist {} [{dtype:?}]: {e}", case.name)),
                dtype,
            );
            assert!(
                (got_pdist - case.wide).abs() / case.wide.abs() <= rtol,
                "pdist {} [{dtype:?}]: got {got_pdist}, an f32 accumulator gives {}, \
                 an element-type accumulator gives {narrow}",
                case.name,
                case.wide
            );
        }
    }
}

/// CUDA and CPU agree on the same cases.
///
/// These inputs are chosen so the two accumulator widths differ by more than
/// the dtype tolerance, which makes this a real parity check rather than one
/// the tolerance would pass either way.
#[cfg(feature = "cuda")]
#[test]
fn distance_metrics_on_narrow_floats_match_cuda() {
    let (cpu_client, cpu_device) = create_cpu_client();

    with_cuda_backend(|cuda_client, cuda_device| {
        for case in CASES {
            for &dtype in NARROW_FLOATS {
                if !is_dtype_supported("cuda", dtype) || !is_dtype_supported("cpu", dtype) {
                    continue;
                }

                let row = case.row();
                let d = case.dim();
                let zeros = vec![0.0f64; d];

                let cpu_x = tensor_from_f64(&row, &[1, d], dtype, &cpu_device, &cpu_client)
                    .unwrap_or_else(|e| panic!("CPU x tensor failed for {dtype:?}: {e}"));
                let cpu_y = tensor_from_f64(&zeros, &[1, d], dtype, &cpu_device, &cpu_client)
                    .unwrap_or_else(|e| panic!("CPU y tensor failed for {dtype:?}: {e}"));
                let cpu_result = cpu_client
                    .cdist(&cpu_x, &cpu_y, case.metric)
                    .unwrap_or_else(|e| panic!("CPU cdist {} [{dtype:?}]: {e}", case.name));

                let x = tensor_from_f64(&row, &[1, d], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA x tensor failed for {dtype:?}: {e}"));
                let y = tensor_from_f64(&zeros, &[1, d], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA y tensor failed for {dtype:?}: {e}"));
                let cuda_result = cuda_client
                    .cdist(&x, &y, case.metric)
                    .unwrap_or_else(|e| panic!("CUDA cdist {} [{dtype:?}]: {e}", case.name));

                assert_tensor_allclose(
                    &cuda_result,
                    &cpu_result,
                    dtype,
                    &format!("cdist {} CUDA vs CPU [{dtype:?}]", case.name),
                );
            }
        }
    });
}
