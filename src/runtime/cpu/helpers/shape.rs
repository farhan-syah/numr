//! Shape transformation operation helpers for CPU tensors

use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::common::shape_ops;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Concatenate tensors along a dimension
pub fn cat_impl(
    client: &CpuClient,
    tensors: &[&Tensor<CpuRuntime>],
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    let params = shape_ops::validate_cat(tensors, dim)?;

    let out = Tensor::<CpuRuntime>::try_empty(&params.out_shape, params.dtype, &client.device)?;
    let out_ptr = out.ptr();
    let elem_size = params.dtype.size_in_bytes();

    // Byte-level copies — memcpy doesn't need type dispatch, and dispatch_dtype!
    // adds measurable branch overhead for small tensors (~25% regression on 1D cat).
    unsafe {
        let mut cat_offset = 0usize;
        for &tensor in tensors {
            let contig_tmp_owned;
            let src_ptr = if tensor.is_contiguous() {
                tensor.ptr() as *const u8
            } else {
                contig_tmp_owned = tensor.contiguous()?;
                contig_tmp_owned.ptr() as *const u8
            };
            let src_cat_size = tensor.shape()[params.dim_idx];
            let src_bytes = src_cat_size * params.inner_size * elem_size;

            if params.outer_size == 1 {
                let dst_offset = cat_offset * params.inner_size * elem_size;
                std::ptr::copy_nonoverlapping(
                    src_ptr,
                    (out_ptr as *mut u8).add(dst_offset),
                    src_bytes,
                );
            } else {
                let row_bytes = params.cat_dim_total * params.inner_size * elem_size;
                for outer in 0..params.outer_size {
                    let src_base = outer * src_bytes;
                    let dst_base = outer * row_bytes + cat_offset * params.inner_size * elem_size;
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(src_base),
                        (out_ptr as *mut u8).add(dst_base),
                        src_bytes,
                    );
                }
            }

            cat_offset += src_cat_size;
        }
    }

    Ok(out)
}

/// Stack tensors along a new dimension
pub fn stack_impl(
    client: &CpuClient,
    tensors: &[&Tensor<CpuRuntime>],
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let _dim_idx = shape_ops::validate_stack(tensors, dim)?;

    // Unsqueeze each tensor and then cat
    // stack(tensors, dim) = cat([t.unsqueeze(dim) for t in tensors], dim)
    let unsqueezed: Vec<Tensor<CpuRuntime>> = tensors
        .iter()
        .map(|t| t.unsqueeze(dim))
        .collect::<Result<_>>()?;

    let unsqueezed_refs: Vec<&Tensor<CpuRuntime>> = unsqueezed.iter().collect();
    cat_impl(client, &unsqueezed_refs, dim)
}

/// Split a tensor into chunks of a given size along a dimension (zero-copy)
pub fn split_impl(
    tensor: &Tensor<CpuRuntime>,
    split_size: usize,
    dim: isize,
) -> Result<Vec<Tensor<CpuRuntime>>> {
    shape_ops::split_impl(tensor, split_size, dim)
}

/// Split a tensor into a specific number of chunks (zero-copy)
pub fn chunk_impl(
    tensor: &Tensor<CpuRuntime>,
    chunks: usize,
    dim: isize,
) -> Result<Vec<Tensor<CpuRuntime>>> {
    shape_ops::chunk_impl(tensor, chunks, dim)
}

/// Repeat tensor along each dimension
pub fn repeat_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    repeats: &[usize],
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = shape_ops::validate_repeat(tensor, repeats)?;

    // Handle case where all repeats are 1 (no-op)
    if repeats.iter().all(|&r| r == 1) {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let in_shape = tensor.shape();
    let out = Tensor::<CpuRuntime>::try_empty(&params.out_shape, dtype, &client.device)?;

    // Make input contiguous
    let tensor_contig = ensure_contiguous(tensor)?;
    let src_ptr = tensor_contig.ptr();
    let dst_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            repeat_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                in_shape,
                &params.out_shape,
                repeats,
            );
        }
    }, "repeat");

    Ok(out)
}

/// Kernel for repeat operation
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn repeat_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    in_shape: &[usize],
    out_shape: &[usize],
    _repeats: &[usize],
) {
    let ndim = in_shape.len();
    let out_numel: usize = out_shape.iter().product();

    // Compute strides for input and output
    let mut in_strides = vec![1usize; ndim];
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // For each output element, compute the corresponding input element
    for out_idx in 0..out_numel {
        // Convert flat index to multi-dimensional indices for output
        let mut remaining = out_idx;
        let mut src_offset = 0usize;

        #[allow(clippy::needless_range_loop)]
        for d in 0..ndim {
            let out_coord = remaining / out_strides[d];
            remaining %= out_strides[d];

            // Map output coordinate to input coordinate using modulo
            let in_coord = out_coord % in_shape[d];
            src_offset += in_coord * in_strides[d];
        }

        *dst.add(out_idx) = *src.add(src_offset);
    }
}

/// Pad tensor with a constant value
pub fn pad_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    padding: &[usize],
    value: f64,
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = shape_ops::validate_pad(tensor, padding)?;

    // Handle case where no padding is added
    if params.pad_per_dim.iter().all(|&(b, a)| b == 0 && a == 0) {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let in_shape = tensor.shape();

    // Create output filled with padding value
    let out =
        Tensor::<CpuRuntime>::try_full_scalar(&params.out_shape, dtype, value, &client.device)?;

    // Make input contiguous
    let tensor_contig = ensure_contiguous(tensor)?;
    let src_ptr = tensor_contig.ptr();
    let dst_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            pad_copy_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                in_shape,
                &params.out_shape,
                &params.pad_per_dim,
            );
        }
    }, "pad");

    Ok(out)
}

/// Kernel for copying input data into padded output
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pad_copy_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    in_shape: &[usize],
    out_shape: &[usize],
    pad_per_dim: &[(usize, usize)],
) {
    let ndim = in_shape.len();
    let in_numel: usize = in_shape.iter().product();

    // Compute strides for input and output
    let mut in_strides = vec![1usize; ndim];
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Copy each input element to its padded position
    for in_idx in 0..in_numel {
        let mut remaining = in_idx;
        let mut dst_offset = 0usize;

        #[allow(clippy::needless_range_loop)]
        for d in 0..ndim {
            let in_coord = remaining / in_strides[d];
            remaining %= in_strides[d];

            // Add padding offset
            let out_coord = in_coord + pad_per_dim[d].0;
            dst_offset += out_coord * out_strides[d];
        }

        *dst.add(dst_offset) = *src.add(in_idx);
    }
}

/// Roll tensor elements along a dimension
pub fn roll_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    shift: isize,
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = shape_ops::validate_roll(tensor, shift, dim)?;

    // Handle case where shift is 0 (no-op)
    if params.shift == 0 {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let shape = tensor.shape();
    let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &client.device)?;

    // Make input contiguous
    let tensor_contig = ensure_contiguous(tensor)?;
    let src_ptr = tensor_contig.ptr();
    let dst_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            roll_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                shape,
                params.dim_idx,
                params.shift,
                params.dim_size,
            );
        }
    }, "roll");

    Ok(out)
}

/// Kernel for roll operation
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn roll_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    shape: &[usize],
    dim_idx: usize,
    shift: usize,
    dim_size: usize,
) {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();

    // Compute strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // For each element, compute rolled position
    for idx in 0..numel {
        let mut remaining = idx;
        let mut dst_offset = 0usize;

        #[allow(clippy::needless_range_loop)]
        for d in 0..ndim {
            let coord = remaining / strides[d];
            remaining %= strides[d];

            if d == dim_idx {
                // Apply circular shift
                let new_coord = (coord + shift) % dim_size;
                dst_offset += new_coord * strides[d];
            } else {
                dst_offset += coord * strides[d];
            }
        }

        *dst.add(dst_offset) = *src.add(idx);
    }
}
