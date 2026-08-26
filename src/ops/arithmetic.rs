//! Arithmetic operations helpers
//!
//! This module contains helper types and functions for arithmetic operations.
//! The actual operations are defined in the `TensorOps` trait.

/// Compute the output shape for binary operations with broadcasting
///
/// Returns None if the shapes are incompatible for broadcasting.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    // Iterate from right to left
    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return None; // Incompatible shapes
        }
    }

    result.reverse();
    Some(result)
}

/// Binary operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: a + b
    Add,
    /// Subtraction: a - b
    Sub,
    /// Multiplication: a * b
    Mul,
    /// Division: a / b
    Div,
    /// Power: a^b
    Pow,
    /// Maximum: max(a, b)
    Max,
    /// Minimum: min(a, b)
    Min,
    /// Two-argument arctangent: atan2(y, x) - angle in radians
    Atan2,
}

/// Unary operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    // === Sign and Absolute ===
    /// Negation: -a
    Neg,
    /// Absolute value: |a|
    Abs,
    /// Sign: returns -1 for negative, 0 for zero, 1 for positive
    Sign,

    // === Power and Root ===
    /// Square root: sqrt(a)
    Sqrt,
    /// Reciprocal square root: 1/sqrt(a) - critical for normalization layers
    Rsqrt,
    /// Square: a^2
    Square,
    /// Cube root: cbrt(a)
    Cbrt,
    /// Reciprocal: 1/a
    Recip,

    // === Exponential and Logarithmic ===
    /// Exponential: e^a
    Exp,
    /// Base-2 exponential: 2^a
    Exp2,
    /// Exponential minus 1: e^a - 1 (numerically stable for small a)
    Expm1,
    /// Natural log: ln(a)
    Log,
    /// Base-2 logarithm: log2(a)
    Log2,
    /// Base-10 logarithm: log10(a)
    Log10,
    /// Natural log of 1+a: ln(1+a) (numerically stable for small a)
    Log1p,

    // === Trigonometric ===
    /// Sine: sin(a)
    Sin,
    /// Cosine: cos(a)
    Cos,
    /// Tangent: tan(a)
    Tan,
    /// Arc sine (inverse sine): asin(a), domain [-1,1], range [-π/2, π/2]
    Asin,
    /// Arc cosine (inverse cosine): acos(a), domain [-1,1], range [0, π]
    Acos,
    /// Arc tangent (inverse tangent): atan(a)
    Atan,

    // === Hyperbolic ===
    /// Hyperbolic sine: sinh(a)
    Sinh,
    /// Hyperbolic cosine: cosh(a)
    Cosh,
    /// Hyperbolic tangent: tanh(a)
    Tanh,
    /// Inverse hyperbolic sine: asinh(a)
    Asinh,
    /// Inverse hyperbolic cosine: acosh(a), domain [1, ∞)
    Acosh,
    /// Inverse hyperbolic tangent: atanh(a), domain (-1, 1)
    Atanh,

    // === Rounding ===
    /// Floor: floor(a)
    Floor,
    /// Ceiling: ceil(a)
    Ceil,
    /// Round to nearest, ties away from zero: round(a).
    ///
    /// Matches Rust's `f32::round`/`f64::round`. A tie such as 2.5 goes to 3.0
    /// and -2.5 goes to -3.0. For the IEEE roundTiesToEven behaviour that
    /// `np.round` and `torch.round` use, see [`UnaryOp::RoundTiesEven`].
    Round,
    /// Round to nearest, ties to even: IEEE 754 roundTiesToEven.
    ///
    /// Matches Rust's `f32::round_ties_even`/`f64::round_ties_even`, and the
    /// semantics of `np.round` and `torch.round`. A tie such as 2.5 goes to 2.0
    /// and 3.5 goes to 4.0.
    RoundTiesEven,
    /// Truncate toward zero: trunc(a)
    Trunc,
}

/// Comparison operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal: a == b
    Eq,
    /// Not equal: a != b
    Ne,
    /// Less than: a < b
    Lt,
    /// Less than or equal: a <= b
    Le,
    /// Greater than: a > b
    Gt,
    /// Greater than or equal: a >= b
    Ge,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape() {
        // Same shapes
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));

        // Broadcasting with 1
        assert_eq!(broadcast_shape(&[2, 3], &[1, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 1], &[2, 3]), Some(vec![2, 3]));

        // Different ranks
        assert_eq!(broadcast_shape(&[3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));

        // Incompatible shapes
        assert_eq!(broadcast_shape(&[2, 3], &[2, 4]), None);
        assert_eq!(broadcast_shape(&[3], &[4]), None);
    }
}
