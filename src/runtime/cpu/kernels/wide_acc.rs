//! Wide accumulators for kernels whose element type cannot hold a running total.
//!
//! A summing kernel that keeps its accumulator in the element type `T` is wrong
//! for every dtype narrower than the total it builds:
//!
//! - A float narrower than F32 (F16, BF16, FP8E4M3, FP8E5M2) stops growing as
//!   soon as the accumulator's spacing exceeds twice the increment, so the sum
//!   converges on a constant or saturates to infinity.
//! - An integer wraps (release) or panics (debug) on overflow, so a total that
//!   crosses the type's range and comes back — a matmul with cancellation, a
//!   mean whose sum overflows but whose average does not — reports a value that
//!   is not merely imprecise but the wrong sign and magnitude.
//!
//! The fix is the same in both cases: accumulate in a wider type and narrow
//! exactly once at write-out. [`WideAcc`] gives the two accumulators those
//! kernels need, so `cumsum`, `matmul`, and the integer reduction epilogues
//! share one definition of "wider" instead of open-coding three.
//!
//! Pick the accumulator with [`WideAcc`]'s two implementors:
//!
//! - `f32` for `DType::is_narrow_float()`. F32 is the accumulator format the
//!   rest of numr already uses for these dtypes (see `reduce_kernel` and
//!   `reduce_epilogue`), and it holds every F16/BF16/FP8 total that the output
//!   dtype can represent.
//! - `i128` for `DType::is_int()`. It is exact for any product or sum of two
//!   64-bit integers, so no integer dtype loses information on the way in.
//!
//! Output dtypes never change: [`WideAcc::to_elem`] narrows back to `T`.
//!
//! # Saturating, not wrapping
//!
//! Integer narrowing clamps to the output dtype's range, and the i128
//! accumulator itself uses saturating add and multiply. This is a deliberate
//! choice, not an accident of the types:
//!
//! - It matches [`Element::from_f64`], which already saturates because Rust's
//!   float-to-int `as` cast saturates. Widening the accumulator therefore does
//!   not introduce a second narrowing convention.
//! - It is the only option that stays a total function. Wrapping reports a
//!   value with the wrong sign, and returning an error would make an overflow
//!   anywhere in a batch fail the whole call.
//!
//! `i64` and `u64` have no wider integer dtype in numr, so their *output* still
//! cannot represent a total that overflows them; saturation is what they get,
//! and the i128 accumulator at least keeps intermediate partial sums exact.
//!
//! # Which side of the line a kernel is on
//!
//! - ACCUMULATORS saturate: `cumsum`, `cumprod`, `matmul`, `mean`, and the
//!   integer reductions. They build a running total wider than one element, so
//!   clamping the total is the only answer that keeps the sign right.
//! - ELEMENTWISE ops wrap: `add`, `sub`, and `mul` on two tensors. Each output
//!   is one machine operation on two elements, and wrapping is what every
//!   array library and every GPU ISA gives there.
//!
//! A new kernel picks the accumulator side only when it carries state across
//! elements.

use crate::dtype::Element;

/// An accumulator wider than the element type being summed.
///
/// Implemented for `f32` (narrow-float elements) and `i128` (integer elements).
pub trait WideAcc: Copy {
    /// Additive identity.
    const ZERO: Self;

    /// Multiplicative identity, the seed for a running product.
    const ONE: Self;

    /// Widen one element into the accumulator.
    fn from_elem<T: Element>(v: T) -> Self;

    /// Narrow the accumulator back to the element type, saturating on overflow.
    fn to_elem<T: Element>(self) -> T;

    /// `self + other`, saturating for integers.
    fn wide_add(self, other: Self) -> Self;

    /// `self * other`, saturating for integers.
    fn wide_mul(self, other: Self) -> Self;
}

impl WideAcc for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_elem<T: Element>(v: T) -> Self {
        v.to_f32()
    }

    #[inline]
    fn to_elem<T: Element>(self) -> T {
        T::from_f32(self)
    }

    #[inline]
    fn wide_add(self, other: Self) -> Self {
        self + other
    }

    #[inline]
    fn wide_mul(self, other: Self) -> Self {
        self * other
    }
}

impl WideAcc for i128 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn from_elem<T: Element>(v: T) -> Self {
        v.to_i128()
    }

    #[inline]
    fn to_elem<T: Element>(self) -> T {
        T::from_i128_saturating(self)
    }

    #[inline]
    fn wide_add(self, other: Self) -> Self {
        self.saturating_add(other)
    }

    #[inline]
    fn wide_mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
}

/// Finish an integer mean: divide a wide sum by the element count.
///
/// The division truncates toward zero, which is what the previous
/// `T::from_f64(sum.to_f64() / count as f64)` epilogue did for every sum it
/// could represent. Doing it in i128 keeps it right for the sums it could not:
/// `mean([2_000_000_000, 2_000_000_000])` as I32 is 2_000_000_000, even though
/// the sum needs 33 bits.
///
/// `count.max(1)` fixes the answer for an integer mean over ZERO elements at 0.
/// A float answers `0 / 0`, which is NaN, and that is what every backend gives
/// it; an integer dtype cannot represent NaN, so 0 is a choice forced by the
/// dtype rather than a mathematical identity. CUDA's `numr128_div_u64_trunc`
/// forces a zero divisor to 1 for the same reason, and WebGPU's
/// `empty_reduce_identity` answers 0 for a non-float `mean`. Do not remove the
/// clamp: without it this divides by zero and panics.
#[inline]
pub fn int_mean_from_sum<T: Element>(sum: i128, count: usize) -> T {
    T::from_i128_saturating(sum / count.max(1) as i128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i128_accumulator_holds_an_i32_sum_that_overflows_i32() {
        // Two i32 values whose sum needs 32 bits plus a sign bit.
        let a = 2_000_000_000i32;
        let sum = i128::ZERO
            .wide_add(i128::from_elem(a))
            .wide_add(i128::from_elem(a));
        assert_eq!(sum, 4_000_000_000i128);
        // Halving it lands back inside i32, which a wrapping accumulator could
        // never do.
        assert_eq!((sum / 2).to_elem::<i32>(), 2_000_000_000i32);
    }

    #[test]
    fn narrowing_saturates_instead_of_wrapping() {
        assert_eq!(4_000_000_000i128.to_elem::<i32>(), i32::MAX);
        assert_eq!((-4_000_000_000i128).to_elem::<i32>(), i32::MIN);
        assert_eq!((-1i128).to_elem::<u8>(), 0u8);
        assert_eq!(300i128.to_elem::<u8>(), u8::MAX);
    }

    #[test]
    fn int_mean_survives_a_sum_that_overflows_the_output_dtype() {
        let sum = 2_000_000_000i128 + 2_000_000_000i128;
        assert_eq!(int_mean_from_sum::<i32>(sum, 2), 2_000_000_000i32);
        assert_eq!(int_mean_from_sum::<i32>(-sum, 2), -2_000_000_000i32);
    }

    #[test]
    fn int_mean_truncates_toward_zero() {
        assert_eq!(int_mean_from_sum::<i32>(7, 2), 3i32);
        assert_eq!(int_mean_from_sum::<i32>(-7, 2), -3i32);
    }

    #[test]
    fn i128_products_of_i64_operands_are_exact() {
        let a = i128::from_elem(3_000_000_000i64);
        let b = i128::from_elem(3_000_000_000i64);
        assert_eq!(a.wide_mul(b), 9_000_000_000_000_000_000i128);
    }
}
