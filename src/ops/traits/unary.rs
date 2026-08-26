//! Unary operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Unary operations
///
/// This trait defines element-wise unary operations on tensors.
/// Each operation is applied independently to each element.
pub trait UnaryOps<R: Runtime> {
    // ===== Sign and Absolute =====

    /// Negation: -a
    fn neg(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Absolute value: |a|
    fn abs(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Sign: returns -1 for negative, 0 for zero, 1 for positive
    fn sign(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Power and Root =====

    /// Square root: sqrt(a)
    fn sqrt(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Reciprocal square root: 1/sqrt(a) - critical for normalization layers
    fn rsqrt(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Square: a²
    fn square(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Cube root: cbrt(a)
    fn cbrt(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Reciprocal: 1/a
    fn recip(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Exponential and Logarithmic =====

    /// Exponential: e^a
    fn exp(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Base-2 exponential: 2^a
    fn exp2(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Exponential minus 1: e^a - 1 (numerically stable for small a)
    fn expm1(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Natural logarithm: ln(a)
    fn log(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Base-2 logarithm: log2(a)
    fn log2(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Base-10 logarithm: log10(a)
    fn log10(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Natural log of 1+a: ln(1+a) (numerically stable for small a)
    fn log1p(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Trigonometric =====

    /// Sine: sin(a)
    fn sin(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Cosine: cos(a)
    fn cos(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Tangent: tan(a)
    fn tan(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Arc sine (inverse sine): asin(a), domain [-1,1], range [-π/2, π/2]
    fn asin(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Arc cosine (inverse cosine): acos(a), domain [-1,1], range [0, π]
    fn acos(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Arc tangent (inverse tangent): atan(a)
    fn atan(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Hyperbolic =====

    /// Hyperbolic sine: sinh(a)
    fn sinh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Hyperbolic cosine: cosh(a)
    fn cosh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Hyperbolic tangent: tanh(a)
    fn tanh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Inverse hyperbolic sine: asinh(a)
    fn asinh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Inverse hyperbolic cosine: acosh(a), domain [1, ∞)
    fn acosh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Inverse hyperbolic tangent: atanh(a), domain (-1, 1)
    fn atanh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Rounding =====

    /// Floor: floor(a)
    fn floor(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Ceiling: ceil(a)
    fn ceil(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Round to nearest integer, ties away from zero.
    ///
    /// Delegates to Rust's `f32::round`/`f64::round`: 0.5 becomes 1.0, 2.5
    /// becomes 3.0, and -2.5 becomes -3.0.
    ///
    /// This is NOT what `np.round` and `torch.round` do — they break ties to
    /// even. Use [`UnaryOps::round_ties_even`] to match them.
    fn round(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Round to nearest integer, ties to even (IEEE 754 roundTiesToEven).
    ///
    /// Delegates to Rust's `f32::round_ties_even`/`f64::round_ties_even`: 0.5
    /// becomes 0.0, 2.5 becomes 2.0, and 3.5 becomes 4.0. This matches
    /// `np.round` and `torch.round` element for element.
    ///
    /// Use [`UnaryOps::round`] for half-away-from-zero rounding.
    fn round_ties_even(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Truncate toward zero: trunc(a)
    fn trunc(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Special Checks =====

    /// Check for NaN values: returns U8 tensor (1 if NaN, 0 otherwise)
    fn isnan(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Check for Inf values: returns U8 tensor (1 if Inf, 0 otherwise)
    fn isinf(&self, a: &Tensor<R>) -> Result<Tensor<R>>;
}
