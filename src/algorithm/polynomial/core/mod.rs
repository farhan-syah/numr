//! Core polynomial algorithms shared across all backends
//!
//! This module provides the unified implementation for polynomial operations.
//! All backends (CPU, CUDA, WebGPU) call these functions to ensure numerical parity.
//!
//! # Design
//!
//! These functions operate through the trait interface (LinearAlgebraAlgorithms,
//! BinaryOps, IndexingOps, etc.) using ONLY tensor operations. This ensures all
//! operations stay on the original device without GPU↔CPU transfers.
//!
//! # No GPU↔CPU Transfers
//!
//! All algorithms are implemented using tensor operations only:
//! - `index_select` for accessing individual coefficients
//! - `scatter_reduce` for convolution operations
//! - `arange` and `eye` for tensor construction
//! - Broadcasting for element-wise operations

mod convolve;
mod polyfromroots;
mod polymul;
mod polyroots;
mod polyval;

pub use convolve::convolve_impl;
pub use polyfromroots::polyfromroots_impl;
pub use polymul::polymul_impl;
pub use polyroots::polyroots_impl;
pub use polyval::polyval_impl;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// DType support flags for backend validation
///
/// Used to specify which floating-point dtypes a backend supports,
/// and which index dtype to use for indexing operations.
///
/// # Examples
///
/// ```
/// use numr::algorithm::polynomial::core::DTypeSupport;
/// use numr::dtype::DType;
///
/// // CPU and CUDA support both F32 and F64, use I64 indices
/// let support = DTypeSupport::FULL;
/// assert!(support.check(DType::F64, "polyroots").is_ok());
/// assert_eq!(support.index_dtype, DType::I64);
///
/// // WebGPU only supports F32, uses I32 indices (WGSL has no i64)
/// let support = DTypeSupport::F32_ONLY;
/// assert!(support.check(DType::F64, "polyroots").is_err());
/// assert_eq!(support.index_dtype, DType::I32);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DTypeSupport {
    /// Whether F32 dtype is supported
    pub f32: bool,
    /// Whether F64 dtype is supported
    pub f64: bool,
    /// Index dtype to use for indexing operations
    /// CPU/CUDA use I64, WebGPU uses I32 (WGSL has no i64 type)
    pub index_dtype: DType,
}

impl DTypeSupport {
    /// Full floating-point support (F32 and F64) with I64 indices
    ///
    /// Used for CPU and CUDA backends which support both single and double precision.
    pub const FULL: Self = Self {
        f32: true,
        f64: true,
        index_dtype: DType::I64,
    };

    /// F32 only support with I32 indices
    ///
    /// Used for WebGPU backend since WGSL does not support 64-bit floats or integers.
    pub const F32_ONLY: Self = Self {
        f32: true,
        f64: false,
        index_dtype: DType::I32,
    };

    /// Check if dtype is supported for the given operation
    ///
    /// Returns `Ok(())` if supported, `Err(UnsupportedDType)` otherwise.
    pub fn check(&self, dtype: DType, op: &'static str) -> Result<()> {
        match dtype {
            DType::F32 if self.f32 => Ok(()),
            DType::F64 if self.f64 => Ok(()),
            // F16, BF16, FP8 supported if F32 is supported (they convert to/from F32)
            DType::F16 | DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2 if self.f32 => Ok(()),
            DType::F32 | DType::F64 => Err(Error::UnsupportedDType { dtype, op }),
            _ => Err(Error::UnsupportedDType { dtype, op }),
        }
    }
}

// ============================================================================
// Helper Functions (shared across algorithm implementations)
// ============================================================================

/// Create a single-element index tensor with the specified dtype
///
/// # Arguments
///
/// * `index` - The index value
/// * `index_dtype` - The dtype for the index tensor (I32 or I64)
/// * `device` - The device to create the tensor on
pub(crate) fn create_index_tensor<R: Runtime<DType = DType>>(
    index: usize,
    index_dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    match index_dtype {
        DType::I32 => Tensor::<R>::from_slice(&[index as i32], &[1], device),
        _ => Tensor::<R>::from_slice(&[index as i64], &[1], device),
    }
}

/// Create an arange-like index tensor [start, start+1, ..., end-1]
///
/// # Arguments
///
/// * `start` - Start index (inclusive)
/// * `end` - End index (exclusive)
/// * `index_dtype` - The dtype for the index tensor (I32 or I64)
/// * `device` - The device to create the tensor on
pub(crate) fn create_arange_tensor<R: Runtime<DType = DType>>(
    start: usize,
    end: usize,
    index_dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    match index_dtype {
        DType::I32 => {
            let indices: Vec<i32> = (start..end).map(|i| i as i32).collect();
            Tensor::<R>::from_slice(&indices, &[indices.len()], device)
        }
        _ => {
            let indices: Vec<i64> = (start..end).map(|i| i as i64).collect();
            Tensor::<R>::from_slice(&indices, &[indices.len()], device)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_support() {
        assert!(DTypeSupport::FULL.check(DType::F32, "test").is_ok());
        assert!(DTypeSupport::FULL.check(DType::F64, "test").is_ok());
        assert!(DTypeSupport::FULL.check(DType::I32, "test").is_err());

        assert!(DTypeSupport::F32_ONLY.check(DType::F32, "test").is_ok());
        assert!(DTypeSupport::F32_ONLY.check(DType::F64, "test").is_err());
    }
}
