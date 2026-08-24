//! Moment statistics (skewness, kurtosis) for CPU runtime.

use super::super::helpers::{dispatch_dtype, ensure_contiguous};
use super::super::{CpuClient, CpuRuntime};
use crate::dtype::Element;
use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, StatisticalOps};
use crate::runtime::common::statistics_common::{
    DIVISION_EPSILON, compute_kurtosis, compute_skewness,
};
use crate::tensor::Tensor;

/// Compute skewness (third standardized moment) along dimensions.
///
/// Skewness measures the asymmetry of a distribution.
///
/// # Formula
///
/// ```text
/// skewness = E[(X - mu)^3] / sigma^3
/// ```
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `dims` - Dimensions to reduce (empty = all)
/// * `keepdim` - Whether to keep reduced dimensions
/// * `correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// Tensor containing the skewness values.
pub fn skew_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Handle scalar/global reduction case
    if dims.is_empty() {
        let numel = a.numel();
        let a_contig = ensure_contiguous(a)?;
        let a_ptr = a_contig.ptr();

        let skewness = dispatch_dtype!(dtype, T => {
            unsafe {
                let slice = std::slice::from_raw_parts(a_ptr as *const T, numel);
                compute_skewness(slice, correction)
            }
        }, "skew");

        let out_shape = if keepdim { vec![1; ndim] } else { vec![] };
        let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &client.device)?;
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe { *(out_ptr as *mut T) = T::from_f64(skewness); }
        }, "skew");

        return Ok(out);
    }

    // Use composition for dimensional reduction: mean, centered, std, pow, div
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute third moment: mean((centered)^3)
    let centered_cubed = client.pow_scalar(&centered, 3.0)?;
    let m3 = client.mean(&centered_cubed, dims, keepdim)?;

    // Compute std^3
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_cubed = client.pow_scalar(&std_val, 3.0)?;

    // skew = m3 / std^3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<CpuRuntime>::try_full_scalar(
        std_cubed.shape(),
        dtype,
        DIVISION_EPSILON,
        &client.device,
    )?;
    let std_cubed_safe = client.add(&std_cubed, &epsilon)?;

    client.div(&m3, &std_cubed_safe)
}

/// Compute kurtosis (fourth standardized moment, excess) along dimensions.
///
/// Excess kurtosis measures the "tailedness" of a distribution relative to
/// a normal distribution.
///
/// # Formula
///
/// ```text
/// excess_kurtosis = E[(X - mu)^4] / sigma^4 - 3
/// ```
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `dims` - Dimensions to reduce (empty = all)
/// * `keepdim` - Whether to keep reduced dimensions
/// * `correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// Tensor containing the excess kurtosis values.
pub fn kurtosis_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Handle scalar/global reduction case
    if dims.is_empty() {
        let numel = a.numel();
        let a_contig = ensure_contiguous(a)?;
        let a_ptr = a_contig.ptr();

        let kurtosis = dispatch_dtype!(dtype, T => {
            unsafe {
                let slice = std::slice::from_raw_parts(a_ptr as *const T, numel);
                compute_kurtosis(slice, correction)
            }
        }, "kurtosis");

        let out_shape = if keepdim { vec![1; ndim] } else { vec![] };
        let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &client.device)?;
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe { *(out_ptr as *mut T) = T::from_f64(kurtosis); }
        }, "kurtosis");

        return Ok(out);
    }

    // Use composition for dimensional reduction
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute fourth moment: mean((centered)^4)
    let centered_fourth = client.pow_scalar(&centered, 4.0)?;
    let m4 = client.mean(&centered_fourth, dims, keepdim)?;

    // Compute std^4
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_fourth = client.pow_scalar(&std_val, 4.0)?;

    // kurtosis = m4 / std^4 - 3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<CpuRuntime>::try_full_scalar(
        std_fourth.shape(),
        dtype,
        DIVISION_EPSILON,
        &client.device,
    )?;
    let std_fourth_safe = client.add(&std_fourth, &epsilon)?;

    let ratio = client.div(&m4, &std_fourth_safe)?;
    let three = Tensor::<CpuRuntime>::try_full_scalar(ratio.shape(), dtype, 3.0, &client.device)?;
    client.sub(&ratio, &three)
}
