//! The dtype gate shared by every indexing launcher.
//!
//! One list, so a gate cannot drift from the instantiation rows in
//! `index_ops.cuh` / `index_nd_ops.cuh`. A gate that admits a dtype with no
//! kernel turns a clean [`Error::UnsupportedDType`] into a launch failure, so
//! this list must stay exactly the set the `.cu` rows instantiate.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Kernel-name suffix for a dtype the indexing kernels are instantiated for.
///
/// `Bool` is one byte per element, so it has its own row alongside U8's.
/// Only the complex dtypes are rejected: they have no indexing kernels.
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] naming `op` for any other dtype.
pub fn index_dtype_suffix(dtype: DType, op: &'static str) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        DType::FP8E4M3 => Ok("fp8_e4m3"),
        DType::FP8E5M2 => Ok("fp8_e5m2"),
        DType::I64 => Ok("i64"),
        DType::I32 => Ok("i32"),
        DType::I16 => Ok("i16"),
        DType::I8 => Ok("i8"),
        DType::U64 => Ok("u64"),
        DType::U32 => Ok("u32"),
        DType::U16 => Ok("u16"),
        DType::U8 => Ok("u8"),
        DType::Bool => Ok("bool"),
        DType::Complex64 | DType::Complex128 => Err(Error::UnsupportedDType { dtype, op }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_integer_dtype_has_a_suffix() {
        for (dtype, expected) in [
            (DType::I64, "i64"),
            (DType::I32, "i32"),
            (DType::I16, "i16"),
            (DType::I8, "i8"),
            (DType::U64, "u64"),
            (DType::U32, "u32"),
            (DType::U16, "u16"),
            (DType::U8, "u8"),
        ] {
            assert_eq!(index_dtype_suffix(dtype, "test").unwrap(), expected);
        }
    }

    #[test]
    fn complex_is_rejected_rather_than_launched() {
        for dtype in [DType::Complex64, DType::Complex128] {
            assert!(index_dtype_suffix(dtype, "test").is_err());
        }
    }
}
