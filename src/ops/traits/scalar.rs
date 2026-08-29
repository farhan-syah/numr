//! Scalar operations trait for tensor-scalar operations.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::TensorOps;

/// Scalar operations trait for tensor-scalar operations
pub trait ScalarOps<R: Runtime>: TensorOps<R> {
    /// Add scalar to tensor: a + scalar
    fn add_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Subtract scalar from tensor: a - scalar
    fn sub_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Multiply tensor by scalar: a * scalar
    fn mul_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Divide tensor by scalar: a / scalar
    fn div_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Raise tensor to scalar power: a^scalar
    ///
    /// The output dtype depends on the input dtype and the exponent:
    ///
    /// - Float dtype: same dtype in and out, for every exponent.
    /// - Integer dtype, exponent a non-negative whole number: same integer
    ///   dtype, computed exactly and saturating at the dtype's bound.
    /// - Integer dtype, exponent negative, fractional, infinite, or NaN: **F64**.
    ///   The result is a non-integer real, so it cannot be the input dtype.
    ///   `2 ** -1` is `0.5f64` and `9 ** 0.5` is `3.0f64`. The input is not
    ///   modified.
    ///
    /// The exponent is a host-side parameter, so the output dtype is known
    /// before the op runs. The tensor-tensor [`pow`](super::BinaryOps::pow)
    /// keeps the integer dtype instead, because its exponent is data and dtype
    /// inference cannot read data.
    ///
    /// # Errors
    ///
    /// WebGPU is 32-bit only, so the promoting case returns
    /// `Error::UnsupportedDType` for F64 there.
    fn pow_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Reverse subtract: scalar - a
    fn rsub_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Fused multiply-add scalar: a * scale + bias
    ///
    /// Applies an affine transform to each element in a single pass.
    /// Common in normalization (scale + shift) and quantization.
    fn fused_mul_add_scalar(&self, a: &Tensor<R>, scale: f64, bias: f64) -> Result<Tensor<R>>;
}
