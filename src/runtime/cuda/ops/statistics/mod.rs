//! Statistics operations for CUDA runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, kurtosis, and mode
//! using native CUDA kernels where possible.
//!
//! This module shares common logic with other backends via the `statistics_common`
//! module to ensure consistency and reduce code duplication.
//!
//! # Module Organization
//!
//! - `quantile` - Quantile, percentile, and median operations
//! - `histogram` - Histogram computation
//! - `moments` - Skewness and kurtosis (higher-order moments)
//! - `mode` - Mode (most frequent value) using native CUDA kernel
//!
//! # Native GPU Operations
//!
//! - **Mode**: Uses native CUDA kernel (`launch_mode_dim`) after GPU-based sorting.
//!   No CPU fallback - entire operation runs on GPU.

mod histogram;
mod mode;
mod moments;
mod quantile;

pub use histogram::histogram_impl;
pub use mode::mode_impl;
pub use moments::{kurtosis_impl, skew_impl};
pub use quantile::{median_impl, percentile_impl, quantile_impl};

// ============================================================================
// Shared Helper Functions
// ============================================================================

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::common::statistics_common::compute_bin_edges_f64;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Create bin edges tensor from computed f64 edges.
pub(crate) fn create_bin_edges(
    client: &CudaClient,
    min_val: f64,
    max_val: f64,
    bins: usize,
    dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let edges_data = compute_bin_edges_f64(min_val, max_val, bins);

    match dtype {
        DType::F32 => {
            let edges_f32: Vec<f32> = edges_data.iter().map(|&v| v as f32).collect();
            Tensor::<CudaRuntime>::from_slice(&edges_f32, &[bins + 1], &client.device)
        }
        DType::F64 => Tensor::<CudaRuntime>::from_slice(&edges_data, &[bins + 1], &client.device),
        _ => {
            // Create as F32 and cast
            let edges_f32: Vec<f32> = edges_data.iter().map(|&v| v as f32).collect();
            let edges = Tensor::<CudaRuntime>::from_slice(&edges_f32, &[bins + 1], &client.device)?;
            client.cast(&edges, dtype)
        }
    }
}

/// Read a single scalar value from GPU tensor using cuMemcpyDtoH_v2.
/// This is used for reading computed min/max values in histogram and statistics operations.
pub(crate) fn read_scalar_f64(t: &Tensor<CudaRuntime>) -> Result<f64> {
    // Ensure we have a single-element tensor
    if t.numel() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "read_scalar_f64 requires a single-element tensor".to_string(),
        });
    }

    let dtype = t.dtype();

    // Ensure contiguous layout
    let tensor = if t.is_contiguous() {
        t.clone()
    } else {
        t.contiguous()?
    };

    // Get GPU buffer pointer
    let ptr = tensor.ptr();

    // Allocate host memory and copy from GPU based on dtype
    let result = match dtype {
        DType::F32 => {
            let mut val: f32 = 0.0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut f32 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<f32>(),
                );
            }
            val as f64
        }
        DType::F64 => {
            let mut val: f64 = 0.0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut f64 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<f64>(),
                );
            }
            val
        }
        DType::I32 => {
            let mut val: i32 = 0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut i32 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<i32>(),
                );
            }
            val as f64
        }
        DType::I64 => {
            let mut val: i64 = 0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut i64 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<i64>(),
                );
            }
            val as f64
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "read_scalar_f64",
            });
        }
    };

    Ok(result)
}
