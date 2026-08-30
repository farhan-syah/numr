// Backend parity for integer division at its two singular points: a zero
// divisor and `MIN / -1`.
//
// The contract, stated in `src/runtime/cpu/kernels/binary_int.rs`: a zero
// divisor yields 0, and `i32::MIN / -1` yields `i32::MIN` (`wrapping_div`)
// rather than overflowing. CUDA mirrors it in `NUMR_BINOP_INT_DIV_SIGNED` and
// `NUMR_BINOP_INT_DIV_UNSIGNED` (`src/runtime/cuda/kernels/binary_ops.cuh`).
//
// WGSL defines `e1 / 0` as `e1`, so a bare `a / b` in a shader returns the
// dividend instead of 0. Nothing in the `div` sweep in `binary.rs` divides by
// zero, so that answer went unseen. All three shapes that carry an integer
// division are driven here: element-wise, broadcast, and against a scalar.
//
// Each backend is checked against the contract in absolute terms as well as
// against CPU, so the case is pinned even in a build with no GPU feature.

use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::common::is_dtype_supported;
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

// ============================================================================
// Inputs
// ============================================================================

/// Dividends, divisors and the contract's quotients for one dtype.
///
/// The signed rows end with `MIN / -1`, whose wrapping quotient is `MIN`. The
/// unsigned rows have no such case: `(T)-1` is the maximum there, and `a / max`
/// is an ordinary division.
fn case(dtype: DType) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if dtype.is_signed_int() {
        let bits = match dtype {
            DType::I8 => 8,
            DType::I16 => 16,
            DType::I32 => 32,
            DType::I64 => 64,
            _ => panic!("not a signed integer dtype: {dtype:?}"),
        };
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        (
            vec![7.0, 0.0, min as f64, max as f64],
            vec![0.0, 0.0, -1.0, 0.0],
            vec![0.0, 0.0, min as f64, 0.0],
        )
    } else {
        let bits = match dtype {
            DType::U8 => 8,
            DType::U16 => 16,
            DType::U32 => 32,
            DType::U64 => 64,
            _ => panic!("not an unsigned integer dtype: {dtype:?}"),
        };
        let max = (1i128 << bits) - 1;
        (
            vec![7.0, 0.0, max as f64, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        )
    }
}

/// Which shape of the division the case drives.
#[derive(Copy, Clone)]
enum Shape {
    /// `div(a, b)` with both operands the same shape.
    Elementwise,
    /// `div(a, b)` with `b` a length-1 tensor, taking the broadcast kernel.
    Broadcast,
    /// `div_scalar(a, 0.0)`, taking the scalar kernel.
    Scalar,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Elementwise => "div",
            Shape::Broadcast => "div broadcast",
            Shape::Scalar => "div_scalar",
        }
    }
}

fn divide<R: Runtime<DType = DType>, C>(
    client: &C,
    device: &R::Device,
    shape: Shape,
    a: &[f64],
    b: &[f64],
    dtype: DType,
) -> numr::error::Result<Tensor<R>>
where
    C: BinaryOps<R> + ScalarOps<R> + numr::ops::TypeConversionOps<R>,
{
    let len = a.len();
    let a_tensor = tensor_from_f64(a, &[len], dtype, device, client)?;
    match shape {
        Shape::Elementwise => {
            let b_tensor = tensor_from_f64(b, &[len], dtype, device, client)?;
            client.div(&a_tensor, &b_tensor)
        }
        Shape::Broadcast => {
            let b_tensor = tensor_from_f64(&[0.0], &[1], dtype, device, client)?;
            client.div(&a_tensor, &b_tensor)
        }
        Shape::Scalar => client.div_scalar(&a_tensor, 0.0),
    }
}

/// The contract's quotients for `shape`. Broadcast and scalar divide every
/// element by zero, so every quotient is zero.
fn want(shape: Shape, quotients: &[f64]) -> Vec<f64> {
    match shape {
        Shape::Elementwise => quotients.to_vec(),
        Shape::Broadcast | Shape::Scalar => vec![0.0; quotients.len()],
    }
}

// ============================================================================
// Parity Driver
// ============================================================================

fn test_div_zero_parity(shape: Shape) {
    for dtype in parity_dtypes(DTypeDomain::IntsOnly, "cpu") {
        let (a, b, quotients) = case(dtype);
        let expected = want(shape, &quotients);
        let len = a.len();

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_result = divide(&cpu_client, &cpu_device, shape, &a, &b, dtype)
            .unwrap_or_else(|e| panic!("CPU {} failed for {dtype:?}: {e}", shape.label()));
        let cpu_want = tensor_from_f64(&expected, &[len], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        assert_tensor_allclose(
            &cpu_result,
            &cpu_want,
            dtype,
            &format!("{} CPU vs contract [{dtype:?}]", shape.label()),
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let result = divide(&cuda_client, &cuda_device, shape, &a, &b, dtype)
                    .unwrap_or_else(|e| panic!("CUDA {} failed for {dtype:?}: {e}", shape.label()));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{} CUDA vs CPU [{dtype:?}]", shape.label()),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
                let result = divide(&wgpu_client, &wgpu_device, shape, &a, &b, dtype)
                    .unwrap_or_else(|e| {
                        panic!("WebGPU {} failed for {dtype:?}: {e}", shape.label())
                    });
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{} WebGPU vs CPU [{dtype:?}]", shape.label()),
                );
            });
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_div_by_zero_elementwise_parity() {
    test_div_zero_parity(Shape::Elementwise);
}

#[test]
fn test_div_by_zero_broadcast_parity() {
    test_div_zero_parity(Shape::Broadcast);
}

#[test]
fn test_div_by_zero_scalar_parity() {
    test_div_zero_parity(Shape::Scalar);
}
