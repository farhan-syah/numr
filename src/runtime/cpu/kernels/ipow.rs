//! Exact, saturating integer exponentiation for the `Pow` kernels.
//!
//! Computing `base ** exp` as `base.to_f64().powf(exp.to_f64())` loses the
//! result the moment it passes f64's 53-bit mantissa: `7i64 ** 20` is
//! 79792266297612001, and the f64 round trip returns a neighbouring value.
//! Integer dtypes therefore get exponentiation by squaring in i128, which is
//! exact for every product two 64-bit integers can form.
//!
//! Overflow saturates to the output dtype's bound. This is the same convention
//! [`wide_acc`](super::wide_acc) documents for every other integer kernel:
//! saturation matches [`Element::from_f64`], stays a total function, and is what
//! the previous `as` cast already did once the float result left the dtype's
//! range.
//!
//! A negative exponent keeps the float path. The true value is a fraction that
//! truncates to 0 or ±1, and CPU and CUDA already agree on it.
//!
//! `pow_scalar` never reaches these kernels with an integer dtype and a negative
//! or fractional exponent. Such a result is a non-integer real, so the op layer
//! gives it an F64 output instead. The tensor-tensor `pow` keeps its integer
//! output, because an op's output dtype cannot depend on tensor data.

use crate::dtype::Element;

/// `base ** exp` for one element.
///
/// Exact and saturating for integer dtypes, `powf` for every other dtype.
/// `T::DTYPE` is a per-monomorphization constant, so the branch folds away and
/// the float dtypes keep their current code.
#[inline]
pub fn pow_elem<T: Element>(base: T, exp: T) -> T {
    if T::DTYPE.is_int() {
        ipow_saturating(base, exp)
    } else {
        T::from_f64(base.to_f64().powf(exp.to_f64()))
    }
}

/// `base ** scalar` for one element, where the exponent arrives as `f64`.
///
/// # Invariant
///
/// An integer dtype reaches this kernel only with a non-negative whole exponent.
/// [`pow_scalar_output_dtype`](crate::runtime::pow_scalar_output_dtype) gives
/// every other exponent an F64 output at the op layer, so the promoted tensor
/// arrives here as a float and takes `powf`. The kernel returns a value rather
/// than a `Result`, so that decision cannot live here.
#[inline]
pub fn pow_elem_scalar<T: Element>(base: T, scalar: f64) -> T {
    if T::DTYPE.is_int() {
        // Above 1024 the outcome depends only on the base's magnitude and the
        // exponent's parity: magnitude 0 or 1 is already fixed, and anything
        // larger saturates. Capping there preserves parity, which is what a
        // negative base needs, and avoids an `as u128` that would saturate to an
        // odd value. CUDA applies the identical cap.
        let exp = if scalar > 1024.0 {
            1024u128 + u128::from(scalar % 2.0 != 0.0)
        } else {
            scalar as u128
        };
        return ipow_from_parts::<T>(base.to_i128(), exp);
    }
    T::from_f64(base.to_f64().powf(scalar))
}

/// Integer `base ** exp`, exact then saturated to `T`'s range.
fn ipow_saturating<T: Element>(base: T, exp: T) -> T {
    let e = exp.to_i128();
    if e < 0 {
        // A negative exponent is a fraction; keep the float path both backends
        // already agree on.
        return T::from_f64(base.to_f64().powf(exp.to_f64()));
    }
    ipow_from_parts::<T>(base.to_i128(), e as u128)
}

/// Shared body: exact magnitude by squaring, sign applied, then saturated.
#[inline]
fn ipow_from_parts<T: Element>(base: i128, exp: u128) -> T {
    let negative = base < 0 && (exp & 1) == 1;
    match ipow_magnitude(base.unsigned_abs(), exp) {
        Some(m) if m <= i128::MAX as u128 => {
            let signed = if negative { -(m as i128) } else { m as i128 };
            T::from_i128_saturating(signed)
        }
        // Past i128 entirely: saturate to the output dtype's bound.
        _ => T::from_i128_saturating(if negative { i128::MIN } else { i128::MAX }),
    }
}

/// Magnitude by squaring. `None` when it leaves u128.
fn ipow_magnitude(base: u128, mut exp: u128) -> Option<u128> {
    let mut result: u128 = 1;
    let mut acc = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.checked_mul(acc)?;
        }
        exp >>= 1;
        if exp > 0 {
            acc = acc.checked_mul(acc)?;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_pow_is_exact_past_the_f64_mantissa() {
        // The f64 round trip returns a neighbouring value here.
        assert_eq!(pow_elem(7i64, 20i64), 79792266297612001i64);
    }

    #[test]
    fn overflow_saturates_to_the_dtype_bound() {
        assert_eq!(pow_elem(2i32, 40i32), i32::MAX);
        assert_eq!(pow_elem(-2i32, 41i32), i32::MIN);
        assert_eq!(pow_elem(2i64, 200i64), i64::MAX);
        assert_eq!(pow_elem(-3i64, 201i64), i64::MIN);
    }

    #[test]
    fn small_integer_cases_are_unchanged() {
        assert_eq!(pow_elem(2i32, 10i32), 1024i32);
        assert_eq!(pow_elem(5i32, 0i32), 1i32);
        assert_eq!(pow_elem(0i32, 5i32), 0i32);
        assert_eq!(pow_elem(-2i32, 3i32), -8i32);
        assert_eq!(pow_elem(1i32, 100i32), 1i32);
    }

    #[test]
    fn negative_exponents_keep_the_float_truncation() {
        assert_eq!(pow_elem(2i32, -1i32), 0i32);
        assert_eq!(pow_elem(1i32, -5i32), 1i32);
        assert_eq!(pow_elem(-1i32, -3i32), -1i32);
    }

    #[test]
    fn float_dtypes_keep_powf() {
        assert!((pow_elem(2.0f32, 0.5f32) - std::f32::consts::SQRT_2).abs() < 1e-6);
        assert!((pow_elem(9.0f64, 0.5f64) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_exponent_matches_the_tensor_exponent() {
        assert_eq!(pow_elem_scalar::<i64>(7, 20.0), 79792266297612001i64);
        assert_eq!(pow_elem_scalar::<i32>(2, 40.0), i32::MAX);
        assert_eq!(pow_elem_scalar::<i32>(2, 10.0), 1024i32);
        assert_eq!(pow_elem_scalar::<i32>(-2, 3.0), -8i32);
        // A negative or fractional exponent promotes to F64 at the op layer, so
        // an integer dtype never reaches this kernel with one. A float dtype
        // takes `powf` for every exponent.
        assert!((pow_elem_scalar::<f64>(9.0, 0.5) - 3.0).abs() < 1e-12);
        assert_eq!(pow_elem_scalar::<f64>(2.0, -1.0), 0.5);
    }
}
