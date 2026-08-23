//! Core DType enum and methods.

use std::fmt;

use super::complex::{Complex64, Complex128};
use super::fp8::{FP8E4M3, FP8E5M2};

/// Data types supported by numr tensors
///
/// This enum represents the element type of a tensor at runtime.
/// Using an enum (rather than generics) allows:
/// - Mixed-precision operations
/// - Runtime type selection
/// - Support for quantized types that aren't `Copy`
///
/// # Discriminant Values (Serialization Stability)
///
/// The discriminant values are **stable** for serialization purposes:
/// - Floats: 0-9 (F64=0, F32=1, F16=2, BF16=3, FP8E4M3=4, FP8E5M2=5)
/// - Signed ints: 10-19 (I64=10, I32=11, I16=12, I8=13)
/// - Unsigned ints: 20-29 (U64=20, U32=21, U16=22, U8=23)
/// - Bool: 30
/// - Complex: 40-49 (Complex64=40, Complex128=41)
///
/// New types will use reserved ranges. Existing values are NEVER changed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum DType {
    // Floating point types (0-9)
    /// 64-bit floating point
    F64 = 0,
    /// 32-bit floating point (most common)
    F32 = 1,
    /// 16-bit floating point (IEEE 754)
    F16 = 2,
    /// 16-bit brain floating point
    BF16 = 3,
    /// 8-bit floating point (1 sign + 4 exp + 3 mant), range ~[-448, 448]
    /// Best for: weights, activations (higher precision, smaller range)
    FP8E4M3 = 4,
    /// 8-bit floating point (1 sign + 5 exp + 2 mant), range ~[-57344, 57344]
    /// Best for: gradients (larger dynamic range, lower precision)
    FP8E5M2 = 5,

    // Integer types
    /// 64-bit signed integer
    I64 = 10,
    /// 32-bit signed integer
    I32 = 11,
    /// 16-bit signed integer
    I16 = 12,
    /// 8-bit signed integer
    I8 = 13,

    // Unsigned integer types
    /// 64-bit unsigned integer
    U64 = 20,
    /// 32-bit unsigned integer
    U32 = 21,
    /// 16-bit unsigned integer
    U16 = 22,
    /// 8-bit unsigned integer
    U8 = 23,

    /// Boolean type
    Bool = 30,

    // Complex types
    /// 64-bit complex (two f32: re, im)
    Complex64 = 40,
    /// 128-bit complex (two f64: re, im)
    Complex128 = 41,
}

impl DType {
    /// Size of one element in bytes
    #[inline]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::Complex128 => 16,
            Self::F64 | Self::I64 | Self::U64 | Self::Complex64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::FP8E4M3 | Self::FP8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// Returns true if this is a floating point type
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F64 | Self::F32 | Self::F16 | Self::BF16 | Self::FP8E4M3 | Self::FP8E5M2
        )
    }

    /// Returns true if this is a floating point type narrower than F32.
    ///
    /// F16, BF16, FP8E4M3, and FP8E5M2 are storage formats, not accumulator
    /// formats. A running sum held in one of them stops growing as soon as the
    /// accumulator's spacing exceeds twice the increment, so the sum silently
    /// converges on a constant instead of the true total. Reductions widen
    /// these dtypes to F32 or wider and narrow only the final result.
    #[inline]
    pub const fn is_narrow_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::FP8E4M3 | Self::FP8E5M2)
    }

    /// Returns true if this is a complex number type
    #[inline]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex64 | Self::Complex128)
    }

    /// Returns the underlying float type for complex types
    /// Returns None for non-complex types
    #[inline]
    pub const fn complex_component_dtype(self) -> Option<Self> {
        match self {
            Self::Complex64 => Some(Self::F32),
            Self::Complex128 => Some(Self::F64),
            _ => None,
        }
    }

    /// Returns true if this is a signed integer type
    #[inline]
    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I64 | Self::I32 | Self::I16 | Self::I8)
    }

    /// Returns true if this is an unsigned integer type
    #[inline]
    pub const fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U64 | Self::U32 | Self::U16 | Self::U8)
    }

    /// Returns true if this is any integer type (signed or unsigned)
    #[inline]
    pub const fn is_int(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Returns true if this is a boolean type
    #[inline]
    pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Returns true if this type can represent negative values
    #[inline]
    pub const fn is_signed(self) -> bool {
        self.is_float() || self.is_signed_int() || self.is_complex()
    }

    /// Get the default dtype for floating point operations
    #[inline]
    pub const fn default_float() -> Self {
        Self::F32
    }

    /// Get the default dtype for integer operations
    #[inline]
    pub const fn default_int() -> Self {
        Self::I64
    }

    /// Short name for display (e.g., "f32", "i64")
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::FP8E4M3 => "fp8e4m3",
            Self::FP8E5M2 => "fp8e5m2",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::Bool => "bool",
            Self::Complex64 => "c64",
            Self::Complex128 => "c128",
        }
    }

    /// Minimum value representable by this dtype (as f64)
    ///
    /// For complex types, returns the minimum value of each component
    pub fn min_value(self) -> f64 {
        match self {
            Self::F64 => f64::MIN,
            Self::F32 => f32::MIN as f64,
            Self::F16 => -65504.0,     // IEEE 754 half precision
            Self::BF16 => -3.4e38,     // Approximate
            Self::FP8E4M3 => -448.0,   // 1 sign + 4 exp + 3 mant
            Self::FP8E5M2 => -57344.0, // 1 sign + 5 exp + 2 mant
            Self::I64 => i64::MIN as f64,
            Self::I32 => i32::MIN as f64,
            Self::I16 => i16::MIN as f64,
            Self::I8 => i8::MIN as f64,
            Self::U64 => 0.0,
            Self::U32 => 0.0,
            Self::U16 => 0.0,
            Self::U8 => 0.0,
            Self::Bool => 0.0,
            Self::Complex64 => f32::MIN as f64,
            Self::Complex128 => f64::MIN,
        }
    }

    /// Fill a buffer with `count` elements of this DType set to `value`.
    ///
    /// Returns the raw bytes. Used by generic constructors (zeros, ones, full_scalar).
    pub fn fill_bytes_impl(self, value: f64, count: usize) -> Vec<u8> {
        #[inline]
        fn typed_to_bytes<T: bytemuck::NoUninit>(v: Vec<T>) -> Vec<u8> {
            bytemuck::cast_slice::<T, u8>(&v).to_vec()
        }

        match self {
            DType::F64 => typed_to_bytes(vec![value; count]),
            DType::F32 => typed_to_bytes(vec![value as f32; count]),
            DType::F16 => {
                let bits = crate::dtype::half_from_f32_util(value as f32, true);
                typed_to_bytes(vec![bits; count])
            }
            DType::BF16 => {
                let bits = crate::dtype::half_from_f32_util(value as f32, false);
                typed_to_bytes(vec![bits; count])
            }
            DType::FP8E4M3 => {
                vec![FP8E4M3::from_f32(value as f32).to_bits(); count]
            }
            DType::FP8E5M2 => {
                vec![FP8E5M2::from_f32(value as f32).to_bits(); count]
            }
            DType::I64 => typed_to_bytes(vec![value as i64; count]),
            DType::I32 => typed_to_bytes(vec![value as i32; count]),
            DType::I16 => typed_to_bytes(vec![value as i16; count]),
            DType::I8 => typed_to_bytes(vec![value as i8; count]),
            DType::U64 => typed_to_bytes(vec![value as u64; count]),
            DType::U32 => typed_to_bytes(vec![value as u32; count]),
            DType::U16 => typed_to_bytes(vec![value as u16; count]),
            DType::U8 => vec![value as u8; count],
            DType::Bool => vec![if value != 0.0 { 1u8 } else { 0u8 }; count],
            DType::Complex64 => typed_to_bytes(vec![Complex64::new(value as f32, 0.0); count]),
            DType::Complex128 => typed_to_bytes(vec![Complex128::new(value, 0.0); count]),
        }
    }

    /// Maximum value representable by this dtype (as f64)
    ///
    /// For complex types, returns the maximum value of each component
    pub fn max_value(self) -> f64 {
        match self {
            Self::F64 => f64::MAX,
            Self::F32 => f32::MAX as f64,
            Self::F16 => 65504.0,
            Self::BF16 => 3.4e38,
            Self::FP8E4M3 => 448.0,
            Self::FP8E5M2 => 57344.0,
            Self::I64 => i64::MAX as f64,
            Self::I32 => i32::MAX as f64,
            Self::I16 => i16::MAX as f64,
            Self::I8 => i8::MAX as f64,
            Self::U64 => u64::MAX as f64,
            Self::U32 => u32::MAX as f64,
            Self::U16 => u16::MAX as f64,
            Self::U8 => u8::MAX as f64,
            Self::Bool => 1.0,
            Self::Complex64 => f32::MAX as f64,
            Self::Complex128 => f64::MAX,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}
