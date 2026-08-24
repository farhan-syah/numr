//! Statistics operations for WebGPU runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, kurtosis, and mode
//! using native WGSL shaders where possible.
//!
//! This module shares common logic with other backends via the `statistics_common`
//! module to ensure consistency and reduce code duplication.
//!
//! # Module Organization
//!
//! - `quantile` - Quantile, percentile, and median operations
//! - `histogram` - Histogram computation
//! - `moments` - Skewness and kurtosis (higher-order moments)
//! - `mode` - Mode (most frequent value) using native WGSL shader
//!
//! # Native GPU Operations
//!
//! - **Mode**: Uses native WGSL shader (`launch_mode_dim`) after GPU-based sorting.
//!   No CPU fallback for supported dtypes (F32, I32, U32).

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
use crate::runtime::RuntimeClient;
use crate::runtime::common::statistics_common::compute_bin_edges_f64;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Create bin edges tensor from computed f64 edges.
pub(crate) fn create_bin_edges(
    client: &WgpuClient,
    min_val: f64,
    max_val: f64,
    bins: usize,
    dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    let edges_data = compute_bin_edges_f64(min_val, max_val, bins);

    // WebGPU primarily supports F32, so create as F32 first
    let edges_f32: Vec<f32> = edges_data.iter().map(|&v| v as f32).collect();
    let edges = Tensor::<WgpuRuntime>::try_from_slice(&edges_f32, &[bins + 1], client.device())?;

    if dtype == DType::F32 {
        Ok(edges)
    } else {
        client.cast(&edges, dtype)
    }
}

/// Extract scalar f64 value from tensor using proper GPU buffer transfer.
///
/// # Scalar Control-Flow Transfer
///
/// This function reads a single scalar (1-element tensor) from GPU to CPU.
/// This is a **control-flow transfer** - the scalar value is needed to make
/// CPU-side decisions about algorithm behavior (e.g., determining bin ranges
/// for histogram, quantile bounds for sorting).
///
/// Unlike data-parallel transfers that should never happen, scalar control-flow
/// transfers are acceptable and necessary when the algorithm requires CPU-side
/// decisions based on GPU-computed summary values (similar to convergence checks
/// in iterative algorithms).
///
/// The transfer uses WebGPU staging buffers for proper GPU-to-CPU data movement,
/// NOT `.to_vec()` which would be an undocumented side channel.
pub(crate) fn tensor_to_f64(client: &WgpuClient, t: &Tensor<WgpuRuntime>) -> Result<f64> {
    use crate::runtime::wgpu::client::get_buffer;

    let dtype = t.dtype();

    // Verify it's a scalar
    if t.numel() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: format!(
                "tensor_to_f64 requires a scalar (1-element) tensor, got numel={}",
                t.numel()
            ),
        });
    }

    // Get buffer from tensor
    let src_buffer = get_buffer(t.ptr())
        .ok_or_else(|| Error::Internal("Failed to get tensor buffer".to_string()))?;

    // Create staging buffer and copy
    let staging = client.create_staging_buffer("scalar_staging", dtype.size_in_bytes() as u64);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scalar_copy"),
        });
    encoder.copy_buffer_to_buffer(&src_buffer, 0, &staging, 0, dtype.size_in_bytes() as u64);
    client.submit_and_wait(encoder);

    // Read value
    let val = match dtype {
        DType::F32 => {
            let mut data = [0f32; 1];
            client.read_buffer(&staging, &mut data)?;
            data[0] as f64
        }
        DType::I32 => {
            let mut data = [0i32; 1];
            client.read_buffer(&staging, &mut data)?;
            data[0] as f64
        }
        DType::U32 => {
            let mut data = [0u32; 1];
            client.read_buffer(&staging, &mut data)?;
            data[0] as f64
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "tensor_to_f64",
            });
        }
    };

    Ok(val)
}
