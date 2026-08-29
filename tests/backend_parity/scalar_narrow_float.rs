// Every case in this module needs a narrow-float dtype, so the whole module
// stands down when neither `f16` nor `fp8` is enabled.
#![cfg(any(feature = "f16", feature = "fp8"))]

// Scalar ops on a narrow-float tensor: the scalar must be rounded ONCE.
//
// A narrow float (F16, BF16, FP8E4M3, FP8E5M2) cannot hold an arbitrary f64
// scalar. A kernel that rounds the scalar into the element type before
// computing rounds twice — once into the element type, once again at the store
// — and lands on a different value than a kernel that computes in f32 against
// the unrounded scalar and narrows once at write-out. One narrowing at
// write-out is the convention `src/runtime/cpu/kernels/wide_acc.rs` states and
// the CUDA FP8 kernels follow (`NUMR_SCALAR_ROW_FP8` in `scalar_ops.cuh`).
//
// `assert_tensor_allclose` alone cannot police this: the FP8E4M3 tolerance is
// 30% relative with atol 2.5, which swallows a whole-ulp disagreement. So the
// cases below pin the exact value each ordering produces and compare bit for
// bit. Every input and expected value is exactly representable in its dtype, so
// exact comparison is the right test, not a tight tolerance.
//
// Hand verification of the first case — FP8E4M3, `add_scalar(0.28125, 0.3)`:
//
//   * FP8E4M3 steps by 2^-5 = 0.03125 across [0.25, 0.5), so 0.3 is NOT
//     representable; the nearest value is 0.3125.
//   * Round once: 0.28125 + 0.3 = 0.58125. FP8E4M3 steps by 0.0625 across
//     [0.5, 1). 0.58125 sits 0.01875 from 0.5625 and 0.04375 from 0.625, so it
//     rounds to 0.5625.
//   * Round twice: 0.28125 + 0.3125 = 0.59375, exactly the midpoint of 0.5625
//     and 0.625. Ties-to-even takes the even mantissa: 0.625 (mantissa field 2)
//     over 0.5625 (mantissa field 1).
//
//   0.5625 vs 0.625 — one ulp apart, and only the second ordering can produce
//   0.625. A scalar that IS representable (5.0, 2.0) agrees under both
//   orderings and would prove nothing.
//
// Hand verification of the F16 case — `add_scalar(0.25048828125, 0.3)`:
//
//   * F16 carries 10 stored mantissa bits, so its step is 2^-12 across
//     [0.25, 0.5) and 2^-11 across [0.5, 1).
//   * 0.25048828125 = 1027 * 2^-12, representable. 0.3 is not: 0.3 * 2^12 =
//     1228.8, so the nearest F16 is 1229 * 2^-12 = 0.300048828125.
//   * Round once: 0.25048828125 + 0.3 = 0.55048828125 = 1127.4 * 2^-11, which
//     rounds to 1127 * 2^-11 = 0.55029296875.
//   * Round twice: 0.25048828125 + 0.300048828125 = 0.550537109375, exactly
//     1127.5 * 2^-11. Ties-to-even takes 1128 * 2^-11 = 0.55078125.
//
// Hand verification of the BF16 case — `add_scalar(0.251953125, 0.3)`:
//
//   * BF16 carries 7 stored mantissa bits, so its step is 2^-9 across
//     [0.25, 0.5) and 2^-8 across [0.5, 1).
//   * 0.251953125 = 129 * 2^-9, representable. 0.3 * 2^9 = 153.6, so the
//     nearest BF16 is 154 * 2^-9 = 0.30078125.
//   * Round once: 0.251953125 + 0.3 = 0.551953125 = 141.3 * 2^-8, which rounds
//     to 141 * 2^-8 = 0.55078125.
//   * Round twice: 0.251953125 + 0.30078125 = 0.552734375, exactly 141.5 *
//     2^-8. Ties-to-even takes 142 * 2^-8 = 0.5546875.
//
//   Both land one ulp apart, and in both the two-rounding answer is the one a
//   1% relative tolerance cannot tell from the correct one.
//
// WebGPU is absent on purpose: it is a 32-bit backend and carries no
// narrow-float dtype at all.

use numr::dtype::DType;
use numr::ops::ScalarOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "cuda")]
use crate::common::assert_tensor_allclose;
use crate::common::{ToF64, create_cpu_client, is_dtype_supported};

/// One scalar op on one dtype, with the value each rounding order produces.
struct Case {
    dtype: DType,
    op: &'static str,
    /// Inputs, each exactly representable in `dtype`.
    input: &'static [f64],
    /// Rounding the result once, against the unrounded scalar. The correct
    /// answer, and what CUDA's FP8 kernels produce.
    once: &'static [f64],
    /// Rounding the scalar into the element type first, then the result. The
    /// answer this test exists to reject.
    twice: &'static [f64],
}

/// Every case uses scalar 0.3, which no narrow float represents exactly.
const SCALAR: f64 = 0.3;

/// Cases found by enumerating every value of each dtype in [0.25, 4] and
/// keeping the ones where the two rounding orders disagree.
const CASES: &[Case] = &[
    Case {
        dtype: DType::FP8E4M3,
        op: "add_scalar",
        input: &[0.28125, 0.40625, 0.875],
        once: &[0.5625, 0.6875, 1.125],
        twice: &[0.625, 0.75, 1.25],
    },
    Case {
        dtype: DType::FP8E4M3,
        op: "mul_scalar",
        input: &[0.34375, 0.375, 0.4375],
        once: &[0.1015625, 0.109375, 0.125],
        twice: &[0.109375, 0.1171875, 0.140625],
    },
    Case {
        dtype: DType::FP8E4M3,
        op: "div_scalar",
        input: &[0.28125, 0.40625, 0.4375],
        once: &[0.9375, 1.375, 1.5],
        twice: &[0.875, 1.25, 1.375],
    },
    Case {
        dtype: DType::FP8E4M3,
        op: "rsub_scalar",
        input: &[0.25, 0.28125, 0.3125],
        once: &[0.05078125, 0.01953125, -0.01171875],
        twice: &[0.0625, 0.03125, 0.0],
    },
    Case {
        dtype: DType::FP8E5M2,
        op: "add_scalar",
        input: &[0.375, 0.625],
        once: &[0.625, 0.875],
        twice: &[0.75, 1.0],
    },
    Case {
        dtype: DType::FP8E5M2,
        op: "mul_scalar",
        input: &[0.375, 0.75, 1.5],
        once: &[0.109375, 0.21875, 0.4375],
        twice: &[0.125, 0.25, 0.5],
    },
    Case {
        dtype: DType::FP8E5M2,
        op: "div_scalar",
        input: &[0.25, 0.5, 1.0],
        once: &[0.875, 1.75, 3.5],
        twice: &[0.75, 1.5, 3.0],
    },
    Case {
        dtype: DType::FP8E5M2,
        op: "rsub_scalar",
        input: &[0.25, 0.3125, 0.375],
        once: &[0.046875, -0.01171875, -0.078125],
        twice: &[0.0625, 0.0, -0.0625],
    },
    Case {
        dtype: DType::F16,
        op: "add_scalar",
        input: &[0.25048828125, 0.25146484375, 0.25244140625],
        once: &[0.55029296875, 0.55126953125, 0.55224609375],
        twice: &[0.55078125, 0.5517578125, 0.552734375],
    },
    Case {
        dtype: DType::F16,
        op: "mul_scalar",
        input: &[0.250732421875, 0.251953125, 0.253173828125],
        once: &[0.0751953125, 0.0755615234375, 0.075927734375],
        twice: &[0.07525634765625, 0.07562255859375, 0.07598876953125],
    },
    Case {
        dtype: DType::F16,
        op: "div_scalar",
        input: &[0.25, 0.250732421875, 0.25146484375],
        once: &[0.83349609375, 0.8359375, 0.83837890625],
        twice: &[0.8330078125, 0.83544921875, 0.837890625],
    },
    Case {
        dtype: DType::F16,
        op: "rsub_scalar",
        input: &[0.25, 0.250244140625, 0.25048828125],
        once: &[0.04998779296875, 0.04974365234375, 0.04949951171875],
        twice: &[0.050048828125, 0.0498046875, 0.049560546875],
    },
    Case {
        dtype: DType::BF16,
        op: "add_scalar",
        input: &[0.251953125, 0.259765625, 0.267578125],
        once: &[0.55078125, 0.55859375, 0.56640625],
        twice: &[0.5546875, 0.5625, 0.5703125],
    },
    Case {
        dtype: DType::BF16,
        op: "mul_scalar",
        input: &[0.255859375, 0.2578125, 0.265625],
        once: &[0.07666015625, 0.0771484375, 0.07958984375],
        twice: &[0.0771484375, 0.07763671875, 0.080078125],
    },
    Case {
        dtype: DType::BF16,
        op: "div_scalar",
        input: &[0.251953125, 0.25390625, 0.2578125],
        once: &[0.83984375, 0.84765625, 0.859375],
        twice: &[0.8359375, 0.84375, 0.85546875],
    },
    Case {
        dtype: DType::BF16,
        op: "rsub_scalar",
        input: &[0.25, 0.251953125, 0.25390625],
        once: &[0.050048828125, 0.048095703125, 0.046142578125],
        twice: &[0.05078125, 0.048828125, 0.046875],
    },
];

fn apply<R: Runtime>(
    client: &impl ScalarOps<R>,
    op: &str,
    tensor: &Tensor<R>,
    scalar: f64,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "add_scalar" => client.add_scalar(tensor, scalar),
        "mul_scalar" => client.mul_scalar(tensor, scalar),
        "div_scalar" => client.div_scalar(tensor, scalar),
        "rsub_scalar" => client.rsub_scalar(tensor, scalar),
        _ => panic!("unknown scalar op: {op}"),
    }
}

/// Read a narrow-float tensor back as `Vec<f64>`. Every narrow float widens to
/// f64 exactly, so the comparison against the case table stays exact.
fn read_back<R: Runtime<DType = DType>>(tensor: &Tensor<R>, dtype: DType) -> Vec<f64> {
    macro_rules! readback {
        ($T:ty) => {
            tensor
                .to_vec::<$T>()
                .iter()
                .map(|x| <$T as ToF64>::to_f64(*x))
                .collect()
        };
    }

    match dtype {
        #[cfg(feature = "f16")]
        DType::F16 => readback!(half::f16),
        #[cfg(feature = "f16")]
        DType::BF16 => readback!(half::bf16),
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => readback!(numr::dtype::FP8E4M3),
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => readback!(numr::dtype::FP8E5M2),
        _ => panic!("read_back: not a narrow float: {dtype:?}"),
    }
}

/// CPU rounds the scalar exactly once.
///
/// Fails against a CPU kernel that pre-rounds the scalar into the element type,
/// which is what `Element::from_f64(scalar)` did for the FP8 dtypes.
#[test]
fn scalar_ops_round_a_narrow_float_scalar_once_on_cpu() {
    let (client, device) = create_cpu_client();

    for case in CASES {
        if !is_dtype_supported("cpu", case.dtype) {
            continue;
        }
        let dtype = case.dtype;
        let op = case.op;

        // The table is only worth running if EVERY element distinguishes the
        // two orderings. An element that agrees would pass whichever ordering
        // the kernel used and quietly weaken the case.
        assert_eq!(case.once.len(), case.input.len());
        assert_eq!(case.twice.len(), case.input.len());
        for (i, (a, b)) in case.once.iter().zip(case.twice.iter()).enumerate() {
            assert_ne!(
                a, b,
                "{op} [{dtype:?}] element {i}: case does not distinguish one rounding from two"
            );
        }

        let shape = [case.input.len()];
        let tensor = tensor_from_f64(case.input, &shape, dtype, &device, &client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let result = apply(&client, op, &tensor, SCALAR)
            .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"));

        let got = read_back(&result, dtype);
        assert_eq!(
            got, case.once,
            "{op} [{dtype:?}] scalar={SCALAR}: expected one rounding {:?}, \
             two roundings would give {:?}",
            case.once, case.twice
        );
    }
}

/// CUDA and CPU agree bit for bit on every narrow float.
///
/// Fails against a CUDA kernel that pre-rounds the scalar into the element
/// type, which is what `NUMR_SCALAR_ROW_FLOAT` did for F16 and BF16 before
/// those two rows moved to `NUMR_SCALAR_ROW_NARROW_FLOAT`. The divergence it
/// leaves is one ulp, well inside the 1% F16/BF16 parity tolerance, so
/// `assert_tensor_allclose` below cannot see it and only this exact comparison
/// can.
#[cfg(feature = "cuda")]
#[test]
fn scalar_ops_on_narrow_floats_match_cuda_bit_for_bit() {
    let (cpu_client, cpu_device) = create_cpu_client();

    with_cuda_backend(|cuda_client, cuda_device| {
        for case in CASES {
            let dtype = case.dtype;
            let op = case.op;
            if !is_dtype_supported("cuda", dtype) {
                continue;
            }

            // Same guard as the CPU test: a case whose two orderings agree
            // would pass whichever one the kernel used.
            for (i, (a, b)) in case.once.iter().zip(case.twice.iter()).enumerate() {
                assert_ne!(
                    a, b,
                    "{op} [{dtype:?}] element {i}: case does not distinguish one rounding from two"
                );
            }

            let shape = [case.input.len()];
            let cpu_tensor = tensor_from_f64(case.input, &shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let cpu_result = apply(&cpu_client, op, &cpu_tensor, SCALAR)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"));

            let cuda_tensor =
                tensor_from_f64(case.input, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = apply(&cuda_client, op, &cuda_tensor, SCALAR)
                .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));

            assert_eq!(
                read_back(&cuda_result, dtype),
                read_back(&cpu_result, dtype),
                "{op} CUDA vs CPU [{dtype:?}] scalar={SCALAR}: one rounding gives {:?}, \
                 two give {:?}",
                case.once,
                case.twice
            );
        }
    });
}

/// The same cases under the standard dtype tolerance, for every narrow float.
///
/// This is the coarse net `assert_tensor_allclose` provides. It cannot see a
/// one-ulp double rounding on FP8, which is why the two tests above compare
/// exactly, but it does catch a backend that diverges by more than an ulp.
#[cfg(feature = "cuda")]
#[test]
fn scalar_ops_on_narrow_floats_match_cuda_within_tolerance() {
    let (cpu_client, cpu_device) = create_cpu_client();

    with_cuda_backend(|cuda_client, cuda_device| {
        for case in CASES {
            let dtype = case.dtype;
            let op = case.op;
            if !is_dtype_supported("cuda", dtype) {
                continue;
            }

            let shape = [case.input.len()];
            let cpu_tensor = tensor_from_f64(case.input, &shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let cpu_result = apply(&cpu_client, op, &cpu_tensor, SCALAR)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"));

            let cuda_tensor =
                tensor_from_f64(case.input, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = apply(&cuda_client, op, &cuda_tensor, SCALAR)
                .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));

            assert_tensor_allclose(
                &cuda_result,
                &cpu_result,
                dtype,
                &format!("{op} CUDA vs CPU [{dtype:?}] scalar={SCALAR}"),
            );
        }
    });
}
