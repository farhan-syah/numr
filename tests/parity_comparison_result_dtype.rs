//! The parity comparison helper must read each tensor by ITS OWN dtype.
//!
//! `pow_scalar` on an I32 tensor with a fractional exponent returns F64 on both
//! backends. The helper read those F64 bytes as I32, so CPU's exact `4.0` (low
//! word `0`) and CUDA's `3.9999999999999996` (low word `0xFFFFFFFF`) compared as
//! `0` vs `-1`. Both results were correct and agreed to within F64 tolerance.

mod common;

use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use common::{assert_tensor_allclose, create_cpu_client};

/// The two f64 bit patterns from the pow_scalar failure.
///
/// Reinterpreted as I32 they are `0` and `-1`; as F64 they differ by one ulp.
const EXACT: f64 = 4.0;
const ONE_ULP_BELOW: f64 = 3.999_999_999_999_999_6;

#[test]
fn promoted_result_is_read_by_its_own_dtype() {
    let (_client, device) = create_cpu_client();
    let actual = Tensor::<CpuRuntime>::from_slice(&[EXACT], &[1], &device).unwrap();
    let expected = Tensor::<CpuRuntime>::from_slice(&[ONE_ULP_BELOW], &[1], &device).unwrap();

    // I32 is the dtype the test was parameterised on; F64 is what the op returned.
    assert_tensor_allclose(&actual, &expected, DType::I32, "promoted pow_scalar");
}

#[test]
#[should_panic(expected = "result dtype divergence")]
fn result_dtype_divergence_is_a_parity_failure() {
    let (client, device) = create_cpu_client();
    let actual = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device).unwrap();
    let expected = client.cast(&actual, DType::F32).unwrap();

    assert_tensor_allclose(&actual, &expected, DType::F64, "diverging result dtype");
}
