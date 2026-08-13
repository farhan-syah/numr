//! Element trait for mapping Rust types to DType

use super::DType;
use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Sub};

/// Trait for types that can be elements of a tensor
///
/// This trait connects Rust's type system to numr's runtime dtype system.
/// It's implemented for all primitive numeric types.
///
/// # Bounds
/// - `Copy + Clone + Send + Sync + 'static` - Basic trait requirements
/// - `Pod + Zeroable` - Safe memory transmutation (bytemuck)
/// - `Add + Sub + Mul + Div` - Arithmetic operations (Output = Self)
/// - `PartialOrd` - Comparison for min/max operations
///
/// Note: `Neg` is NOT required since unsigned types don't support it.
/// Negation is handled via to_f64/from_f64 conversion in kernels.
pub trait Element:
    Copy
    + Clone
    + Send
    + Sync
    + Pod
    + Zeroable
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
{
    /// The corresponding DType for this Rust type
    const DTYPE: DType;

    /// Convert to f64 for generic numeric operations
    ///
    /// # Complex Number Behavior
    ///
    /// For complex types (Complex64, Complex128), this returns the **magnitude** (|z|),
    /// not the real part. This is consistent with:
    /// - PartialOrd using magnitude for comparison
    /// - The need for a single scalar representation
    ///
    /// If you need the real part, access `.re` directly on the complex type.
    fn to_f64(self) -> f64;

    /// Convert from f64 to this type
    ///
    /// # Complex Number Behavior
    ///
    /// For complex types, this creates a **real number** (imaginary part = 0).
    fn from_f64(v: f64) -> Self;

    /// Convert to f32
    ///
    /// Default implementation goes through f64. Types that have direct f32
    /// conversion (f16, bf16, fp8) override this for efficiency.
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Convert from f32 to this type
    ///
    /// Default implementation goes through f64. Types that have direct f32
    /// conversion (f16, bf16, fp8) override this for efficiency.
    #[inline]
    fn from_f32(v: f32) -> Self {
        Self::from_f64(v as f64)
    }

    /// Total-order comparison used by all sorting operations.
    ///
    /// `PartialOrd` is not total for floats, and a non-transitive comparator makes
    /// both comparison sorts and bitonic networks return unsorted output.
    ///
    /// Ordering matches PyTorch and NumPy: NaN compares greater than every non-NaN
    /// value (last ascending, first descending), NaNs compare equal to each other,
    /// and `-0.0` compares equal to `+0.0`. This differs from [`f32::total_cmp`],
    /// which orders `-0.0` before `+0.0` and negative NaN below `-inf`.
    ///
    /// The default is already correct for integer types, whose `PartialOrd` is total.
    #[inline]
    fn sort_cmp(&self, other: &Self) -> Ordering {
        match self.partial_cmp(other) {
            Some(ord) => ord,
            None => {
                // `partial_cmp` returns `None` only for NaN, the sole value that
                // compares unequal to itself.
                #[allow(clippy::eq_op)]
                let self_nan = self != self;
                #[allow(clippy::eq_op)]
                let other_nan = other != other;
                match (self_nan, other_nan) {
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    _ => Ordering::Equal,
                }
            }
        }
    }

    /// Zero value
    fn zero() -> Self;

    /// One value
    fn one() -> Self;
}

impl Element for f64 {
    const DTYPE: DType = DType::F64;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i64
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i32 {
    const DTYPE: DType = DType::I32;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i32
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i16 {
    const DTYPE: DType = DType::I16;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i16
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i8 {
    const DTYPE: DType = DType::I8;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i8
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u64 {
    const DTYPE: DType = DType::U64;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u64
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u32 {
    const DTYPE: DType = DType::U32;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u32
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u16 {
    const DTYPE: DType = DType::U16;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u16
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u8
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

// Note: bool doesn't implement Pod, so we can't implement Element for it directly.
// Boolean tensors use u8 internally.

// ============================================================================
// Half-precision floating point types (requires "f16" feature)
// ============================================================================

#[cfg(feature = "f16")]
impl Element for half::f16 {
    const DTYPE: DType = DType::F16;

    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        half::f16::from_f64(v)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        half::f16::from_f32(v)
    }

    #[inline]
    fn zero() -> Self {
        half::f16::ZERO
    }

    #[inline]
    fn one() -> Self {
        half::f16::ONE
    }
}

#[cfg(feature = "f16")]
impl Element for half::bf16 {
    const DTYPE: DType = DType::BF16;

    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        half::bf16::from_f64(v)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        half::bf16::from_f32(v)
    }

    #[inline]
    fn zero() -> Self {
        half::bf16::ZERO
    }

    #[inline]
    fn one() -> Self {
        half::bf16::ONE
    }
}

// ============================================================================
// 8-bit floating point types (requires "fp8" feature)
// ============================================================================

impl Element for super::fp8::FP8E4M3 {
    const DTYPE: DType = DType::FP8E4M3;

    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::from_f32(v as f32)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        Self::from_f32(v)
    }

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

impl Element for super::fp8::FP8E5M2 {
    const DTYPE: DType = DType::FP8E5M2;

    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::from_f32(v as f32)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        Self::from_f32(v)
    }

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

// ============================================================================
// Complex types
//
// Complex number conversion semantics:
// - to_f64(): Returns magnitude (|z| = sqrt(re² + im²))
//   This is intentional - a lossy conversion that provides a single scalar.
//   For the real part, use z.re directly.
// - from_f64(): Creates a real number (im = 0)
//
// These semantics are consistent with PartialOrd (compare by magnitude).
// ============================================================================

impl Element for super::complex::Complex64 {
    const DTYPE: DType = DType::Complex64;

    /// Returns magnitude (|z|) - this is a lossy conversion.
    /// For the real part, use `.re` directly.
    #[inline]
    fn to_f64(self) -> f64 {
        self.magnitude() as f64
    }

    /// Creates a real complex number (im = 0)
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::new(v as f32, 0.0)
    }

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

impl Element for super::complex::Complex128 {
    const DTYPE: DType = DType::Complex128;

    /// Returns magnitude (|z|) - this is a lossy conversion.
    /// For the real part, use `.re` directly.
    #[inline]
    fn to_f64(self) -> f64 {
        self.magnitude()
    }

    /// Creates a real complex number (im = 0)
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::new(v, 0.0)
    }

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_dtype() {
        assert_eq!(f64::DTYPE, DType::F64);
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(i32::DTYPE, DType::I32);
        assert_eq!(u8::DTYPE, DType::U8);
    }

    #[test]
    fn test_element_conversions() {
        assert_eq!(f32::from_f64(2.5).to_f64(), 2.5f32 as f64);
        assert_eq!(i32::from_f64(42.0), 42);
    }

    #[test]
    fn test_fp8_element_dtype() {
        use super::super::fp8::{FP8E4M3, FP8E5M2};
        assert_eq!(FP8E4M3::DTYPE, DType::FP8E4M3);
        assert_eq!(FP8E5M2::DTYPE, DType::FP8E5M2);
    }

    #[test]
    fn test_fp8_element_conversions() {
        use super::super::fp8::{FP8E4M3, FP8E5M2};

        // FP8E4M3 roundtrip
        let e4m3 = FP8E4M3::from_f64(2.0);
        assert!((e4m3.to_f64() - 2.0).abs() < 0.1);

        // FP8E5M2 roundtrip
        let e5m2 = FP8E5M2::from_f64(100.0);
        assert!((e5m2.to_f64() - 100.0).abs() < 15.0);

        // Zero and one
        assert_eq!(FP8E4M3::zero().to_f32(), 0.0);
        assert!((FP8E4M3::one().to_f32() - 1.0).abs() < 0.01);
        assert_eq!(FP8E5M2::zero().to_f32(), 0.0);
        assert!((FP8E5M2::one().to_f32() - 1.0).abs() < 0.01);
    }
}
