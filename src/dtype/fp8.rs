//! FP8 (8-bit floating point) types for numr tensors
//!
//! This module provides two FP8 formats commonly used in ML:
//!
//! - **FP8E4M3**: 1 sign + 4 exponent + 3 mantissa bits
//!   - Range: ~[-448, 448]
//!   - Higher precision, smaller range
//!   - Best for: weights, activations
//!
//! - **FP8E5M2**: 1 sign + 5 exponent + 2 mantissa bits
//!   - Range: ~[-57344, 57344]
//!   - Lower precision, larger range
//!   - Best for: gradients (needs larger dynamic range)
//!
//! # Example
//!
//! ```
//! # use numr::dtype::FP8E4M3;
//! // Convert from f32
//! let fp8_val = FP8E4M3::from_f32(1.5);
//!
//! // Convert back to f32
//! let f32_val = fp8_val.to_f32();
//! assert!(fp8_val.to_f32() > 0.0);
//! ```

use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// ============================================================================
// CUDA Compatibility Traits
// ============================================================================

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;
#[cfg(feature = "cuda")]
use cudarc::types::CudaTypeName;

// ============================================================================
// FP8E4M3 Type
// ============================================================================

/// 8-bit floating point with 4 exponent and 3 mantissa bits.
///
/// Format: 1 sign + 4 exponent + 3 mantissa
/// - Bias: 7
/// - Range: ~[-448, 448]
/// - Smallest positive normal: 2^-6 ≈ 0.015625
/// - Precision: ~3 decimal digits
///
/// Best for: weights and activations where precision matters more than range.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FP8E4M3(pub u8);

// Safety: FP8E4M3 is a transparent wrapper around u8, which is Pod
unsafe impl Pod for FP8E4M3 {}
unsafe impl Zeroable for FP8E4M3 {}

impl FP8E4M3 {
    /// Zero value
    pub const ZERO: Self = Self(0x00);
    /// One value (sign=0, exp=0111, mant=000 -> 1.0 * 2^(7-7) = 1.0)
    pub const ONE: Self = Self(0x38);
    /// Negative one
    pub const NEG_ONE: Self = Self(0xB8);
    /// Maximum positive value (~448)
    pub const MAX: Self = Self(0x7E); // sign=0, exp=1111, mant=110
    /// Minimum positive value (smallest normal)
    pub const MIN_POSITIVE: Self = Self(0x08); // sign=0, exp=0001, mant=000
    /// Positive infinity (represented as max value, no inf in E4M3)
    pub const INFINITY: Self = Self::MAX;
    /// NaN (exp=1111, mant=111)
    pub const NAN: Self = Self(0x7F);

    /// Exponent bias for E4M3 format (reserved for future use)
    #[allow(dead_code)]
    const BIAS: i32 = 7;
    /// Number of mantissa bits (reserved for future use)
    #[allow(dead_code)]
    const MANTISSA_BITS: u32 = 3;
    /// Number of exponent bits (reserved for future use)
    #[allow(dead_code)]
    const EXPONENT_BITS: u32 = 4;

    /// Create from raw bits
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Get raw bits
    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert from f32 to FP8E4M3 with saturation.
    ///
    /// Values outside the representable range are clamped:
    /// - |x| > 448 → ±MAX
    /// - |x| ≤ 2^-10 → ±0 (round-to-nearest below the smallest subnormal)
    ///
    /// See [`f32_to_fp8_e4m3`] for detailed conversion semantics.
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self(f32_to_fp8_e4m3(x))
    }

    /// Convert from FP8E4M3 to f32
    #[inline]
    pub fn to_f32(self) -> f32 {
        fp8_e4m3_to_f32(self.0)
    }

    /// Convert from f64 to FP8E4M3
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Self::from_f32(x as f32)
    }

    /// Convert from FP8E4M3 to f64
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Check if this value is NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        // E4M3: NaN is exp=1111, mant=111
        (self.0 & 0x7F) == 0x7F
    }

    /// Check if this value is zero (positive or negative)
    #[inline]
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7F) == 0
    }

    /// Check if this value is negative
    #[inline]
    pub fn is_negative(self) -> bool {
        (self.0 & 0x80) != 0 && !self.is_zero()
    }

    /// Maximum representable value as f32
    pub const fn max_value() -> f32 {
        448.0
    }

    /// Minimum representable value as f32
    pub const fn min_value() -> f32 {
        -448.0
    }
}

impl fmt::Debug for FP8E4M3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FP8E4M3({})", self.to_f32())
    }
}

impl fmt::Display for FP8E4M3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl PartialOrd for FP8E4M3 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// Arithmetic operations via f32 conversion
impl Add for FP8E4M3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub for FP8E4M3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul for FP8E4M3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div for FP8E4M3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl std::ops::AddAssign for FP8E4M3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() + rhs.to_f32());
    }
}

impl std::ops::SubAssign for FP8E4M3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() - rhs.to_f32());
    }
}

impl std::ops::MulAssign for FP8E4M3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() * rhs.to_f32());
    }
}

impl std::ops::DivAssign for FP8E4M3 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() / rhs.to_f32());
    }
}

// ============================================================================
// FP8E5M2 Type
// ============================================================================

/// 8-bit floating point with 5 exponent and 2 mantissa bits.
///
/// Format: 1 sign + 5 exponent + 2 mantissa
/// - Bias: 15
/// - Range: ~[-57344, 57344]
/// - Smallest positive normal: 2^-14 ≈ 0.000061
/// - Precision: ~2 decimal digits
///
/// Best for: gradients where dynamic range matters more than precision.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FP8E5M2(pub u8);

// Safety: FP8E5M2 is a transparent wrapper around u8, which is Pod
unsafe impl Pod for FP8E5M2 {}
unsafe impl Zeroable for FP8E5M2 {}

impl FP8E5M2 {
    /// Zero value
    pub const ZERO: Self = Self(0x00);
    /// One value (sign=0, exp=01111, mant=00 -> 1.0 * 2^(15-15) = 1.0)
    pub const ONE: Self = Self(0x3C);
    /// Negative one
    pub const NEG_ONE: Self = Self(0xBC);
    /// Maximum positive value (~57344)
    pub const MAX: Self = Self(0x7B); // sign=0, exp=11110, mant=11
    /// Minimum positive value (smallest normal)
    pub const MIN_POSITIVE: Self = Self(0x04); // sign=0, exp=00001, mant=00
    /// Positive infinity
    pub const INFINITY: Self = Self(0x7C); // sign=0, exp=11111, mant=00
    /// Negative infinity
    pub const NEG_INFINITY: Self = Self(0xFC);
    /// NaN (exp=11111, mant!=00)
    pub const NAN: Self = Self(0x7F);

    /// Exponent bias for E5M2 format (reserved for future use)
    #[allow(dead_code)]
    const BIAS: i32 = 15;
    /// Number of mantissa bits (reserved for future use)
    #[allow(dead_code)]
    const MANTISSA_BITS: u32 = 2;
    /// Number of exponent bits (reserved for future use)
    #[allow(dead_code)]
    const EXPONENT_BITS: u32 = 5;

    /// Create from raw bits
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Get raw bits
    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert from f32 to FP8E5M2 with saturation.
    ///
    /// Values outside the representable range are clamped:
    /// - |x| > 57344 → ±Infinity
    /// - |x| ≤ 2^-17 → ±0 (round-to-nearest below the smallest subnormal)
    ///
    /// See [`f32_to_fp8_e5m2`] for detailed conversion semantics.
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self(f32_to_fp8_e5m2(x))
    }

    /// Convert from FP8E5M2 to f32
    #[inline]
    pub fn to_f32(self) -> f32 {
        fp8_e5m2_to_f32(self.0)
    }

    /// Convert from f64 to FP8E5M2
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Self::from_f32(x as f32)
    }

    /// Convert from FP8E5M2 to f64
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Check if this value is NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        // E5M2: NaN is exp=11111, mant!=00
        let exp = (self.0 >> 2) & 0x1F;
        let mant = self.0 & 0x03;
        exp == 0x1F && mant != 0
    }

    /// Check if this value is infinite
    #[inline]
    pub fn is_infinite(self) -> bool {
        // E5M2: Inf is exp=11111, mant=00
        (self.0 & 0x7F) == 0x7C
    }

    /// Check if this value is zero (positive or negative)
    #[inline]
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7F) == 0
    }

    /// Check if this value is negative
    #[inline]
    pub fn is_negative(self) -> bool {
        (self.0 & 0x80) != 0 && !self.is_zero()
    }

    /// Maximum representable finite value as f32
    pub const fn max_value() -> f32 {
        57344.0
    }

    /// Minimum representable finite value as f32
    pub const fn min_value() -> f32 {
        -57344.0
    }
}

impl fmt::Debug for FP8E5M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FP8E5M2({})", self.to_f32())
    }
}

impl fmt::Display for FP8E5M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl PartialOrd for FP8E5M2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// Arithmetic operations via f32 conversion
impl Add for FP8E5M2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub for FP8E5M2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul for FP8E5M2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div for FP8E5M2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl std::ops::AddAssign for FP8E5M2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() + rhs.to_f32());
    }
}

impl std::ops::SubAssign for FP8E5M2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() - rhs.to_f32());
    }
}

impl std::ops::MulAssign for FP8E5M2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() * rhs.to_f32());
    }
}

impl std::ops::DivAssign for FP8E5M2 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Self::from_f32(self.to_f32() / rhs.to_f32());
    }
}

// ============================================================================
// CUDA Trait Implementations
// ============================================================================

/// FP8E4M3 maps to numr's custom CUDA type (1 sign + 4 exponent + 3 mantissa)
/// This matches the `numr_fp8_e4m3` struct defined in CUDA kernels.
#[cfg(feature = "cuda")]
impl CudaTypeName for FP8E4M3 {
    const NAME: &'static str = "numr_fp8_e4m3";
}

/// FP8E5M2 maps to numr's custom CUDA type (1 sign + 5 exponent + 2 mantissa)
/// This matches the `numr_fp8_e5m2` struct defined in CUDA kernels.
#[cfg(feature = "cuda")]
impl CudaTypeName for FP8E5M2 {
    const NAME: &'static str = "numr_fp8_e5m2";
}

/// SAFETY: FP8E4M3 is #[repr(transparent)] wrapping u8, which is trivially safe
/// for GPU transfers. The type is Pod and Zeroable.
#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for FP8E4M3 {}

/// SAFETY: FP8E5M2 is #[repr(transparent)] wrapping u8, which is trivially safe
/// for GPU transfers. The type is Pod and Zeroable.
#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for FP8E5M2 {}

// ============================================================================
// Conversion Functions
// ============================================================================

/// Convert f32 to FP8 E4M3 format with saturation semantics.
///
/// This function never panics. All f32 values are accepted and converted using
/// saturation (clamping) for out-of-range values.
///
/// # Saturation Behavior
///
/// - **Overflow** (|x| > 448): Saturates to ±MAX (0x7E / 0xFE)
/// - **Underflow** (|x| ≤ 2^-10): Rounds to ±0
/// - **Infinity**: Saturates to ±MAX (E4M3 has no infinity representation)
/// - **NaN**: Preserved as NaN (0x7F / 0xFF)
/// - **Subnormals**: Converted correctly with gradual underflow
///
/// # Precision Loss
///
/// FP8 E4M3 has only 3 mantissa bits (~1 decimal digit of precision).
/// Rounding uses round-to-nearest-even (banker's rounding).
///
/// # Valid Output Range
///
/// - Normal values: ±[2^-6, 448] (approximately ±[0.015625, 448])
/// - Subnormal values: ±[2^-9, 2^-6) (approximately ±[0.00195, 0.015625))
#[inline]
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    // Handle special cases
    if exp == 255 {
        // NaN or Inf
        if mant != 0 {
            // NaN
            return 0x7F | (sign << 7);
        }
        // Inf -> clamp to max (E4M3 has no inf)
        return 0x7E | (sign << 7);
    }

    if exp == 0 && mant == 0 {
        // Zero
        return sign << 7;
    }

    // Compute unbiased exponent
    // F32 bias = 127, E4M3 bias = 7
    let unbiased_exp = if exp == 0 {
        // Subnormal f32
        -126
    } else {
        exp - 127
    };

    // E4M3 exponent range: -6 to 8 (stored as 1 to 15, with 0 for subnormals)
    // Max exp = 8 (stored as 15), but exp=15 with mant=111 is NaN
    // So max normal is exp=15, mant=110 -> value = 1.75 * 2^8 = 448

    if unbiased_exp > 8 {
        // Overflow -> max value
        return 0x7E | (sign << 7);
    }

    if unbiased_exp < -10 {
        // Below half the smallest subnormal (2^-10), so round-to-nearest gives zero.
        return sign << 7;
    }

    // Handle subnormals in E4M3 (exp stored as 0)
    if unbiased_exp < -6 {
        // E4M3 subnormal: exp=0, implicit leading 0.
        // Value = (fp8_mant / 8) * 2^-6 = fp8_mant * 2^-9.
        // The f32 significand is the value scaled by 2^(23 - unbiased_exp),
        // so fp8_mant = significand >> (14 - unbiased_exp).
        let significand = if exp == 0 {
            // F32 subnormal: no implicit leading 1.
            mant
        } else {
            // F32 normal: add implicit 1.xxx
            0x800000 | mant
        };
        let shift = (14 - unbiased_exp) as u32;
        let truncated = significand >> shift;
        let round_bit = (significand >> (shift - 1)) & 1;
        let remainder = significand & ((1 << (shift - 1)) - 1);
        // Round to nearest even, matching the normal path.
        let fp8_mant = if round_bit != 0 && (remainder != 0 || (truncated & 1) != 0) {
            truncated + 1
        } else {
            truncated
        };
        // A carry out of the 3-bit mantissa lands in the exponent field, which is
        // exactly the smallest normal. No mask, or the carry would be discarded.
        return (sign << 7) | fp8_mant as u8;
    }

    // Normal E4M3 value
    let fp8_exp = (unbiased_exp + 7) as u8;

    // Round mantissa from 23 bits to 3 bits
    let fp8_mant = if exp == 0 {
        // F32 subnormal
        (mant >> (23 - 3)) as u8
    } else {
        // F32 normal - round to nearest even
        let mant_shifted = mant >> (23 - 3 - 1); // Keep 4 bits for rounding
        let round_bit = (mant_shifted & 1) != 0;
        let mant_3bit = (mant_shifted >> 1) as u8;

        if round_bit {
            // Round up if tie breaks to even or if remainder > 0.5
            let remainder = mant & ((1 << (23 - 4)) - 1);
            if remainder != 0 || (mant_3bit & 1) != 0 {
                // Round up
                if mant_3bit == 0x07 {
                    // Mantissa overflow, increment exponent
                    if fp8_exp >= 15 {
                        // Would overflow to NaN, clamp to max
                        return 0x7E | (sign << 7);
                    }
                    return (sign << 7) | ((fp8_exp + 1) << 3);
                }
                mant_3bit + 1
            } else {
                mant_3bit
            }
        } else {
            mant_3bit
        }
    };

    // Check for NaN pattern (exp=15, mant=7)
    if fp8_exp == 15 && fp8_mant == 7 {
        // Clamp to max instead of NaN
        return 0x7E | (sign << 7);
    }

    (sign << 7) | (fp8_exp << 3) | (fp8_mant & 0x07)
}

/// Convert FP8 E4M3 to f32.
///
/// This function never panics. All 256 possible u8 values produce valid f32 results.
///
/// # Special Values
///
/// - 0x00 / 0x80: Positive/negative zero
/// - 0x7F / 0xFF: Positive/negative NaN
/// - 0x7E / 0xFE: Positive/negative MAX (~448)
///
/// # Conversion
///
/// The conversion is exact - no precision is lost when converting FP8 to f32.
#[inline]
pub fn fp8_e4m3_to_f32(x: u8) -> f32 {
    let sign = (x >> 7) & 1;
    let exp = (x >> 3) & 0x0F;
    let mant = x & 0x07;

    // Handle special cases
    if exp == 15 && mant == 7 {
        // NaN
        return f32::NAN;
    }

    if exp == 0 && mant == 0 {
        // Zero
        return if sign != 0 { -0.0 } else { 0.0 };
    }

    let f32_sign = (sign as u32) << 31;

    if exp == 0 {
        // Subnormal: value = 0.mant * 2^(-6)
        // Convert to f32: find the leading 1 bit
        let mant_val = mant as f32 / 8.0; // 0.mant in binary
        let value = mant_val * 2.0f32.powi(-6);
        return if sign != 0 { -value } else { value };
    }

    // Normal: value = 1.mant * 2^(exp - 7)
    // F32 format: 1.mant23 * 2^(exp32 - 127)
    // So exp32 = exp - 7 + 127 = exp + 120
    let f32_exp = ((exp as u32) + 120) << 23;
    let f32_mant = (mant as u32) << 20; // Shift 3-bit mant to 23-bit position

    f32::from_bits(f32_sign | f32_exp | f32_mant)
}

/// Convert f32 to FP8 E5M2 format with saturation semantics.
///
/// This function never panics. All f32 values are accepted and converted using
/// saturation for out-of-range values.
///
/// # Saturation Behavior
///
/// - **Overflow** (|x| > 57344): Saturates to ±Infinity (0x7C / 0xFC)
/// - **Underflow** (|x| ≤ 2^-17): Rounds to ±0
/// - **Infinity**: Preserved as ±Infinity (E5M2 supports infinity)
/// - **NaN**: Preserved as NaN (0x7D-0x7F / 0xFD-0xFF)
/// - **Subnormals**: Converted correctly with gradual underflow
///
/// # Precision Loss
///
/// FP8 E5M2 has only 2 mantissa bits (~0.6 decimal digits of precision).
/// Rounding uses round-to-nearest-even (banker's rounding).
///
/// # Valid Output Range
///
/// - Normal values: ±[2^-14, 57344] (approximately ±[0.000061, 57344])
/// - Subnormal values: ±[2^-16, 2^-14) (approximately ±[0.000015, 0.000061))
/// - Special: ±Infinity, NaN
#[inline]
pub fn f32_to_fp8_e5m2(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    // Handle special cases
    if exp == 255 {
        // NaN or Inf
        if mant != 0 {
            // NaN
            return 0x7F | (sign << 7);
        }
        // Inf
        return 0x7C | (sign << 7);
    }

    if exp == 0 && mant == 0 {
        // Zero
        return sign << 7;
    }

    // Compute unbiased exponent
    // F32 bias = 127, E5M2 bias = 15
    let unbiased_exp = if exp == 0 {
        // Subnormal f32
        -126
    } else {
        exp - 127
    };

    // E5M2 exponent range: -14 to 15 (stored as 1 to 30, with 0 for subnormals, 31 for inf/nan)
    if unbiased_exp > 15 {
        // Overflow -> Inf
        return 0x7C | (sign << 7);
    }

    if unbiased_exp < -17 {
        // Below half the smallest subnormal (2^-17), so round-to-nearest gives zero.
        return sign << 7;
    }

    // Handle subnormals in E5M2 (exp stored as 0)
    if unbiased_exp < -14 {
        // E5M2 subnormal: exp=0, implicit leading 0.
        // Value = (fp8_mant / 4) * 2^-14 = fp8_mant * 2^-16.
        // The f32 significand is the value scaled by 2^(23 - unbiased_exp),
        // so fp8_mant = significand >> (7 - unbiased_exp).
        let significand = if exp == 0 { mant } else { 0x800000 | mant };
        let shift = (7 - unbiased_exp) as u32;
        let truncated = significand >> shift;
        let round_bit = (significand >> (shift - 1)) & 1;
        let remainder = significand & ((1 << (shift - 1)) - 1);
        // Round to nearest even, matching the normal path.
        let fp8_mant = if round_bit != 0 && (remainder != 0 || (truncated & 1) != 0) {
            truncated + 1
        } else {
            truncated
        };
        // A carry out of the 2-bit mantissa lands in the exponent field, which is
        // exactly the smallest normal. No mask, or the carry would be discarded.
        return (sign << 7) | fp8_mant as u8;
    }

    // Normal E5M2 value
    let fp8_exp = (unbiased_exp + 15) as u8;

    // Round mantissa from 23 bits to 2 bits
    let fp8_mant = if exp == 0 {
        (mant >> (23 - 2)) as u8
    } else {
        // Round to nearest even
        let mant_shifted = mant >> (23 - 2 - 1);
        let round_bit = (mant_shifted & 1) != 0;
        let mant_2bit = (mant_shifted >> 1) as u8;

        if round_bit {
            let remainder = mant & ((1 << (23 - 3)) - 1);
            if remainder != 0 || (mant_2bit & 1) != 0 {
                if mant_2bit == 0x03 {
                    if fp8_exp >= 30 {
                        // Overflow to Inf
                        return 0x7C | (sign << 7);
                    }
                    return (sign << 7) | ((fp8_exp + 1) << 2);
                }
                mant_2bit + 1
            } else {
                mant_2bit
            }
        } else {
            mant_2bit
        }
    };

    (sign << 7) | (fp8_exp << 2) | (fp8_mant & 0x03)
}

/// Convert FP8 E5M2 to f32.
///
/// This function never panics. All 256 possible u8 values produce valid f32 results.
///
/// # Special Values
///
/// - 0x00 / 0x80: Positive/negative zero
/// - 0x7C / 0xFC: Positive/negative infinity
/// - 0x7D-0x7F / 0xFD-0xFF: NaN values
/// - 0x7B / 0xFB: Positive/negative MAX (~57344)
///
/// # Conversion
///
/// The conversion is exact - no precision is lost when converting FP8 to f32.
#[inline]
pub fn fp8_e5m2_to_f32(x: u8) -> f32 {
    let sign = (x >> 7) & 1;
    let exp = (x >> 2) & 0x1F;
    let mant = x & 0x03;

    // Handle special cases
    if exp == 31 {
        if mant != 0 {
            // NaN
            return f32::NAN;
        }
        // Inf
        return if sign != 0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }

    if exp == 0 && mant == 0 {
        // Zero
        return if sign != 0 { -0.0 } else { 0.0 };
    }

    let f32_sign = (sign as u32) << 31;

    if exp == 0 {
        // Subnormal: value = 0.mant * 2^(-14)
        let mant_val = mant as f32 / 4.0;
        let value = mant_val * 2.0f32.powi(-14);
        return if sign != 0 { -value } else { value };
    }

    // Normal: value = 1.mant * 2^(exp - 15)
    // F32: exp32 = exp - 15 + 127 = exp + 112
    let f32_exp = ((exp as u32) + 112) << 23;
    let f32_mant = (mant as u32) << 21;

    f32::from_bits(f32_sign | f32_exp | f32_mant)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== FP8E4M3 Tests ==========

    #[test]
    fn test_fp8_e4m3_zero() {
        assert_eq!(FP8E4M3::ZERO.to_f32(), 0.0);
        assert_eq!(FP8E4M3::from_f32(0.0).to_bits(), 0x00);
        assert_eq!(FP8E4M3::from_f32(-0.0).to_bits(), 0x80);
    }

    #[test]
    fn test_fp8_e4m3_one() {
        let one = FP8E4M3::ONE;
        assert!((one.to_f32() - 1.0).abs() < 0.01);

        let from_one = FP8E4M3::from_f32(1.0);
        assert!((from_one.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let test_values = [0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 100.0, 448.0];
        for &val in &test_values {
            let fp8 = FP8E4M3::from_f32(val);
            let back = fp8.to_f32();
            // Allow ~12% error due to 3-bit mantissa
            let rel_error = (back - val).abs() / val.abs().max(1e-6);
            assert!(
                rel_error < 0.2,
                "FP8E4M3 roundtrip failed for {}: got {}, rel_error={}",
                val,
                back,
                rel_error
            );
        }
    }

    #[test]
    fn test_fp8_e4m3_negative() {
        let neg_one = FP8E4M3::from_f32(-1.0);
        assert!(neg_one.to_f32() < 0.0);
        assert!((neg_one.to_f32() + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fp8_e4m3_overflow() {
        let big = FP8E4M3::from_f32(1000.0);
        assert!((big.to_f32() - 448.0).abs() < 1.0); // Clamped to max
    }

    #[test]
    fn test_fp8_e4m3_underflow() {
        let tiny = FP8E4M3::from_f32(1e-10);
        assert_eq!(tiny.to_f32(), 0.0);
    }

    #[test]
    fn test_fp8_e4m3_nan() {
        let nan = FP8E4M3::from_f32(f32::NAN);
        assert!(nan.is_nan());
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_fp8_e4m3_inf() {
        let inf = FP8E4M3::from_f32(f32::INFINITY);
        // E4M3 has no infinity, should clamp to max
        assert!((inf.to_f32() - 448.0).abs() < 1.0);
    }

    #[test]
    fn test_fp8_e4m3_arithmetic() {
        let a = FP8E4M3::from_f32(2.0);
        let b = FP8E4M3::from_f32(3.0);

        let sum = a + b;
        assert!((sum.to_f32() - 5.0).abs() < 0.5);

        let diff = b - a;
        assert!((diff.to_f32() - 1.0).abs() < 0.5);

        let prod = a * b;
        assert!((prod.to_f32() - 6.0).abs() < 0.5);

        let quot = b / a;
        assert!((quot.to_f32() - 1.5).abs() < 0.3);
    }

    // ========== FP8E5M2 Tests ==========

    #[test]
    fn test_fp8_e5m2_zero() {
        assert_eq!(FP8E5M2::ZERO.to_f32(), 0.0);
        assert_eq!(FP8E5M2::from_f32(0.0).to_bits(), 0x00);
    }

    #[test]
    fn test_fp8_e5m2_one() {
        let one = FP8E5M2::ONE;
        assert!((one.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fp8_e5m2_roundtrip() {
        let test_values = [0.5, 1.0, 2.0, 4.0, 8.0, 100.0, 1000.0, 10000.0];
        for &val in &test_values {
            let fp8 = FP8E5M2::from_f32(val);
            let back = fp8.to_f32();
            // Allow ~25% error due to 2-bit mantissa
            let rel_error = (back - val).abs() / val.abs().max(1e-6);
            assert!(
                rel_error < 0.35,
                "FP8E5M2 roundtrip failed for {}: got {}, rel_error={}",
                val,
                back,
                rel_error
            );
        }
    }

    #[test]
    fn test_fp8_e5m2_large_range() {
        // E5M2 should handle larger values than E4M3
        let large = FP8E5M2::from_f32(50000.0);
        assert!(large.to_f32() > 30000.0);
    }

    #[test]
    fn test_fp8_e5m2_inf() {
        let inf = FP8E5M2::from_f32(f32::INFINITY);
        assert!(inf.is_infinite());
        assert!(inf.to_f32().is_infinite());
    }

    #[test]
    fn test_fp8_e5m2_nan() {
        let nan = FP8E5M2::from_f32(f32::NAN);
        assert!(nan.is_nan());
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_fp8_e5m2_arithmetic() {
        let a = FP8E5M2::from_f32(100.0);
        let b = FP8E5M2::from_f32(200.0);

        let sum = a + b;
        assert!((sum.to_f32() - 300.0).abs() < 50.0);

        let diff = b - a;
        assert!((diff.to_f32() - 100.0).abs() < 30.0);
    }

    // ========== Pod/Zeroable Tests ==========

    #[test]
    fn test_fp8_e4m3_bytemuck() {
        let arr = [FP8E4M3::ZERO, FP8E4M3::ONE];
        let bytes: &[u8] = bytemuck::cast_slice(&arr);
        assert_eq!(bytes.len(), 2);

        let back: &[FP8E4M3] = bytemuck::cast_slice(bytes);
        assert_eq!(back[0], FP8E4M3::ZERO);
        assert_eq!(back[1], FP8E4M3::ONE);
    }

    // ========== Subnormal Encoding Tests ==========
    //
    // Derived from the format definitions, not from observed behaviour:
    // E4M3 subnormals are (m / 8) * 2^-6 for m in 1..=7.
    // E5M2 subnormals are (m / 4) * 2^-14 for m in 1..=3.

    /// The m-th E4M3 subnormal, straight from the format definition.
    fn e4m3_subnormal(m: u8) -> f32 {
        (m as f32 / 8.0) * 2.0f32.powi(-6)
    }

    /// The m-th E5M2 subnormal, straight from the format definition.
    fn e5m2_subnormal(m: u8) -> f32 {
        (m as f32 / 4.0) * 2.0f32.powi(-14)
    }

    #[test]
    fn test_fp8_e4m3_every_subnormal_roundtrips_exactly() {
        for m in 1u8..=7 {
            let value = e4m3_subnormal(m);
            let encoded = FP8E4M3::from_f32(value);
            assert_eq!(
                encoded.to_bits(),
                m,
                "E4M3 subnormal {} should encode to mantissa {}",
                value,
                m
            );
            assert_eq!(
                encoded.to_f32(),
                value,
                "E4M3 subnormal roundtrip failed for m={}",
                m
            );

            let neg = FP8E4M3::from_f32(-value);
            assert_eq!(neg.to_bits(), 0x80 | m);
            assert_eq!(neg.to_f32(), -value);
        }
    }

    #[test]
    fn test_fp8_e4m3_smallest_subnormal_survives() {
        // 2^-9 is the smallest E4M3 subnormal and must not flush to zero.
        let smallest = 2.0f32.powi(-9);
        assert_eq!(smallest, e4m3_subnormal(1));
        assert_eq!(FP8E4M3::from_f32(smallest).to_f32(), smallest);
        assert_eq!(FP8E4M3::from_f32(-smallest).to_f32(), -smallest);
    }

    #[test]
    fn test_fp8_e4m3_subnormal_normal_boundary() {
        // Largest subnormal is 7 * 2^-9, the next value up is the smallest normal 2^-6.
        let largest_sub = e4m3_subnormal(7);
        assert_eq!(FP8E4M3::from_f32(largest_sub).to_bits(), 0x07);
        assert_eq!(FP8E4M3::from_f32(largest_sub).to_f32(), largest_sub);

        let smallest_normal = 2.0f32.powi(-6);
        assert_eq!(
            FP8E4M3::from_f32(smallest_normal).to_bits(),
            FP8E4M3::MIN_POSITIVE.to_bits()
        );
        assert_eq!(FP8E4M3::from_f32(smallest_normal).to_f32(), smallest_normal);
        assert_eq!(
            FP8E4M3::from_f32(-smallest_normal).to_f32(),
            -smallest_normal
        );
    }

    #[test]
    fn test_fp8_e4m3_below_smallest_subnormal_flushes_to_zero() {
        // 2^-10 is an exact tie against zero; round-to-nearest-even picks zero.
        for &value in &[2.0f32.powi(-10), 2.0f32.powi(-11), 2.0f32.powi(-20), 1e-10] {
            assert_eq!(FP8E4M3::from_f32(value).to_f32(), 0.0, "value {}", value);
            assert_eq!(FP8E4M3::from_f32(-value).to_f32(), 0.0, "value {}", -value);
        }
    }

    #[test]
    fn test_fp8_e4m3_subnormal_rounds_to_nearest_even() {
        // 0.75 * 2^-9 sits above the midpoint, so it rounds up to the smallest subnormal.
        let value = 0.75 * 2.0f32.powi(-9);
        assert_eq!(FP8E4M3::from_f32(value).to_f32(), 2.0f32.powi(-9));

        // 1.5 * 2^-9 is a tie between m=1 and m=2; even wins.
        let tie = 1.5 * 2.0f32.powi(-9);
        assert_eq!(FP8E4M3::from_f32(tie).to_f32(), 2.0 * 2.0f32.powi(-9));
    }

    #[test]
    fn test_fp8_e4m3_normals_unchanged() {
        for &value in &[0.5f32, 1.0, 1.5, 2.0, 3.0, 16.0, 448.0, 0.015625] {
            assert_eq!(FP8E4M3::from_f32(value).to_f32(), value, "value {}", value);
            assert_eq!(
                FP8E4M3::from_f32(-value).to_f32(),
                -value,
                "value {}",
                value
            );
        }
    }

    #[test]
    fn test_fp8_e5m2_every_subnormal_roundtrips_exactly() {
        for m in 1u8..=3 {
            let value = e5m2_subnormal(m);
            let encoded = FP8E5M2::from_f32(value);
            assert_eq!(
                encoded.to_bits(),
                m,
                "E5M2 subnormal {} should encode to mantissa {}",
                value,
                m
            );
            assert_eq!(
                encoded.to_f32(),
                value,
                "E5M2 subnormal roundtrip failed for m={}",
                m
            );

            let neg = FP8E5M2::from_f32(-value);
            assert_eq!(neg.to_bits(), 0x80 | m);
            assert_eq!(neg.to_f32(), -value);
        }
    }

    #[test]
    fn test_fp8_e5m2_smallest_subnormal_survives() {
        // 2^-16 is the smallest E5M2 subnormal and must not flush to zero.
        let smallest = 2.0f32.powi(-16);
        assert_eq!(smallest, e5m2_subnormal(1));
        assert_eq!(FP8E5M2::from_f32(smallest).to_f32(), smallest);
        assert_eq!(FP8E5M2::from_f32(-smallest).to_f32(), -smallest);
    }

    #[test]
    fn test_fp8_e5m2_subnormal_normal_boundary() {
        let largest_sub = e5m2_subnormal(3);
        assert_eq!(FP8E5M2::from_f32(largest_sub).to_bits(), 0x03);
        assert_eq!(FP8E5M2::from_f32(largest_sub).to_f32(), largest_sub);

        let smallest_normal = 2.0f32.powi(-14);
        assert_eq!(
            FP8E5M2::from_f32(smallest_normal).to_bits(),
            FP8E5M2::MIN_POSITIVE.to_bits()
        );
        assert_eq!(FP8E5M2::from_f32(smallest_normal).to_f32(), smallest_normal);
        assert_eq!(
            FP8E5M2::from_f32(-smallest_normal).to_f32(),
            -smallest_normal
        );
    }

    #[test]
    fn test_fp8_e5m2_below_smallest_subnormal_flushes_to_zero() {
        // 2^-17 is an exact tie against zero; round-to-nearest-even picks zero.
        for &value in &[2.0f32.powi(-17), 2.0f32.powi(-18), 2.0f32.powi(-30), 1e-12] {
            assert_eq!(FP8E5M2::from_f32(value).to_f32(), 0.0, "value {}", value);
            assert_eq!(FP8E5M2::from_f32(-value).to_f32(), 0.0, "value {}", -value);
        }
    }

    #[test]
    fn test_fp8_e5m2_subnormal_rounds_to_nearest_even() {
        // 0.75 * 2^-16 sits above the midpoint, so it rounds up to the smallest subnormal.
        let value = 0.75 * 2.0f32.powi(-16);
        assert_eq!(FP8E5M2::from_f32(value).to_f32(), 2.0f32.powi(-16));

        // 1.5 * 2^-16 is a tie between m=1 and m=2; even wins.
        let tie = 1.5 * 2.0f32.powi(-16);
        assert_eq!(FP8E5M2::from_f32(tie).to_f32(), 2.0 * 2.0f32.powi(-16));
    }

    #[test]
    fn test_fp8_e5m2_normals_unchanged() {
        for &value in &[0.5f32, 1.0, 1.5, 2.0, 3.0, 16.0, 57344.0, 0.25] {
            assert_eq!(FP8E5M2::from_f32(value).to_f32(), value, "value {}", value);
            assert_eq!(
                FP8E5M2::from_f32(-value).to_f32(),
                -value,
                "value {}",
                value
            );
        }
    }

    #[test]
    fn test_fp8_e5m2_bytemuck() {
        let arr = [FP8E5M2::ZERO, FP8E5M2::ONE];
        let bytes: &[u8] = bytemuck::cast_slice(&arr);
        assert_eq!(bytes.len(), 2);

        let back: &[FP8E5M2] = bytemuck::cast_slice(bytes);
        assert_eq!(back[0], FP8E5M2::ZERO);
        assert_eq!(back[1], FP8E5M2::ONE);
    }
}
