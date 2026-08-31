//! Distance computation CUDA kernel launchers.
//!
//! Provides launchers for pairwise distance computation using various metrics.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::DistanceMetric;

/// Module name for distance kernels
pub const DISTANCE_MODULE: &str = "distance";

/// Convert DistanceMetric to kernel index
fn metric_to_index(metric: DistanceMetric) -> u32 {
    match metric {
        DistanceMetric::Euclidean => 0,
        DistanceMetric::SquaredEuclidean => 1,
        DistanceMetric::Manhattan => 2,
        DistanceMetric::Chebyshev => 3,
        DistanceMetric::Minkowski(_) => 4,
        DistanceMetric::Cosine => 5,
        DistanceMetric::Correlation => 6,
        DistanceMetric::Hamming => 7,
        DistanceMetric::Jaccard => 8,
    }
}

/// Get Minkowski p value from metric as f32 (for F32/F16/BF16 kernels)
fn metric_p_value_f32(metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Minkowski(p) => p as f32,
        _ => 2.0, // Default (not used for non-Minkowski)
    }
}

/// Get Minkowski p value from metric as f64 (for F64 kernel)
fn metric_p_value_f64(metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Minkowski(p) => p,
        _ => 2.0, // Default (not used for non-Minkowski)
    }
}

/// Launch cdist kernel - pairwise distances between two point sets.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - x must have shape (n, d), y must have shape (m, d)
/// - out must have shape (n, m)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the tensors
/// * `x_ptr` - Device pointer to first input tensor (n, d)
/// * `y_ptr` - Device pointer to second input tensor (m, d)
/// * `out_ptr` - Device pointer to output tensor (n, m)
/// * `n` - Number of points in x
/// * `m` - Number of points in y
/// * `d` - Dimensionality
/// * `metric` - Distance metric to use
pub unsafe fn launch_cdist(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    n: usize,
    m: usize,
    d: usize,
    metric: DistanceMetric,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, DISTANCE_MODULE)?;
        let func_name = kernel_name("cdist", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let numel = n * m;
        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let metric_idx = metric_to_index(metric);
        let n_u32 = n as u32;
        let m_u32 = m as u32;
        let d_u32 = d as u32;
        let p_f32 = metric_p_value_f32(metric);
        let p_f64 = metric_p_value_f64(metric);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr);
        builder.arg(&y_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n_u32);
        builder.arg(&m_u32);
        builder.arg(&d_u32);
        builder.arg(&metric_idx);

        // AccT is f64 for F64 dtype, f32 for all others
        if dtype == DType::F64 {
            builder.arg(&p_f64);
        } else {
            builder.arg(&p_f32);
        }

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("Failed to launch cdist kernel: {:?}", e)))?;

        Ok(())
    }
}

/// Launch pdist kernel - pairwise distances within one point set (condensed).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - x must have shape (n, d)
/// - out must have shape (n*(n-1)/2,)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the tensors
/// * `x_ptr` - Device pointer to input tensor (n, d)
/// * `out_ptr` - Device pointer to output tensor (n*(n-1)/2,)
/// * `n` - Number of points
/// * `d` - Dimensionality
/// * `metric` - Distance metric to use
pub unsafe fn launch_pdist(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    n: usize,
    d: usize,
    metric: DistanceMetric,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, DISTANCE_MODULE)?;
        let func_name = kernel_name("pdist", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let numel = n * (n - 1) / 2;
        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let metric_idx = metric_to_index(metric);
        let n_u32 = n as u32;
        let d_u32 = d as u32;
        let p_f32 = metric_p_value_f32(metric);
        let p_f64 = metric_p_value_f64(metric);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n_u32);
        builder.arg(&d_u32);
        builder.arg(&metric_idx);

        if dtype == DType::F64 {
            builder.arg(&p_f64);
        } else {
            builder.arg(&p_f32);
        }

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("Failed to launch pdist kernel: {:?}", e)))?;

        Ok(())
    }
}

/// Launch squareform kernel - condensed to square.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - condensed must have shape (n*(n-1)/2,)
/// - square must have shape (n, n)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the tensors
/// * `condensed_ptr` - Device pointer to condensed tensor
/// * `square_ptr` - Device pointer to square output tensor
/// * `n` - Number of points
pub unsafe fn launch_squareform(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    condensed_ptr: u64,
    square_ptr: u64,
    n: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, DISTANCE_MODULE)?;
        let func_name = kernel_name("squareform", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let numel = n * n;
        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&condensed_ptr);
        builder.arg(&square_ptr);
        builder.arg(&n_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("Failed to launch squareform kernel: {:?}", e)))?;

        Ok(())
    }
}

/// Launch squareform_inverse kernel - square to condensed.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - square must have shape (n, n)
/// - condensed must have shape (n*(n-1)/2,)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the tensors
/// * `square_ptr` - Device pointer to square input tensor
/// * `condensed_ptr` - Device pointer to condensed output tensor
/// * `n` - Number of points
pub unsafe fn launch_squareform_inverse(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    square_ptr: u64,
    condensed_ptr: u64,
    n: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, DISTANCE_MODULE)?;
        let func_name = kernel_name("squareform_inverse", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let numel = n * (n - 1) / 2;
        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&square_ptr);
        builder.arg(&condensed_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "Failed to launch squareform_inverse kernel: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
