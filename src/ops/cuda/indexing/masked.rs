//! Masked operations for CUDA runtime

use super::helpers::{BroadcastContext, validate_mask_dtype};
use crate::error::Result;
use crate::runtime::cuda::kernels::{
    launch_masked_count, launch_masked_count_broadcast, launch_masked_fill,
    launch_masked_fill_broadcast, launch_masked_prefix_sum, launch_masked_prefix_sum_broadcast,
    launch_masked_select, launch_masked_select_broadcast,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Runtime, ensure_contiguous};
use crate::tensor::Tensor;

/// Execute masked_select operation.
pub fn masked_select(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    mask: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    validate_mask_dtype(mask)?;

    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let mask_contig = ensure_contiguous(mask)?;
    let numel = a.numel();

    let bcast = BroadcastContext::prepare(a, mask, &client.device)?;

    // Phase 1: Count true elements in mask
    let count_bytes = std::mem::size_of::<u32>();
    let count_ptr = CudaRuntime::allocate(count_bytes, &client.device)?;

    let zero: u32 = 0;
    CudaRuntime::copy_to_device(bytemuck::bytes_of(&zero), count_ptr, &client.device)?;

    if bcast.needs_broadcast {
        unsafe {
            launch_masked_count_broadcast(
                &client.context,
                &client.stream,
                client.device.index,
                mask_contig.ptr(),
                count_ptr,
                bcast.strides_ptr(),
                bcast.shape_ptr(),
                bcast.ndim,
                numel,
            )?;
        }
    } else {
        unsafe {
            launch_masked_count(
                &client.context,
                &client.stream,
                client.device.index,
                mask_contig.ptr(),
                count_ptr,
                numel,
            )?;
        }
    }

    let mut count_buf = [0u32; 1];
    CudaRuntime::copy_from_device(
        count_ptr,
        bytemuck::bytes_of_mut(&mut count_buf),
        &client.device,
    )?;
    let count = count_buf[0] as usize;
    CudaRuntime::deallocate(count_ptr, count_bytes, &client.device);

    let out = Tensor::<CudaRuntime>::empty(&[count], dtype, &client.device)?;
    if count == 0 {
        return Ok(out);
    }

    // Phase 2: Compute prefix sum
    let prefix_sum_bytes = numel * std::mem::size_of::<u32>();
    let prefix_sum_ptr = CudaRuntime::allocate(prefix_sum_bytes, &client.device)?;

    if bcast.needs_broadcast {
        unsafe {
            launch_masked_prefix_sum_broadcast(
                &client.context,
                &client.stream,
                client.device.index,
                mask_contig.ptr(),
                prefix_sum_ptr,
                bcast.strides_ptr(),
                bcast.shape_ptr(),
                bcast.ndim,
                numel,
            )?;
        }
    } else {
        unsafe {
            launch_masked_prefix_sum(
                &client.context,
                &client.stream,
                client.device.index,
                mask_contig.ptr(),
                prefix_sum_ptr,
                numel,
            )?;
        }
    }

    // Phase 3: Gather selected elements
    if bcast.needs_broadcast {
        unsafe {
            launch_masked_select_broadcast(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                mask_contig.ptr(),
                out.ptr(),
                prefix_sum_ptr,
                bcast.strides_ptr(),
                bcast.shape_ptr(),
                bcast.ndim,
                numel,
            )?;
        }
    } else {
        unsafe {
            launch_masked_select(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                mask_contig.ptr(),
                out.ptr(),
                prefix_sum_ptr,
                numel,
            )?;
        }
    }

    CudaRuntime::deallocate(prefix_sum_ptr, prefix_sum_bytes, &client.device);
    Ok(out)
}

/// Execute masked_fill operation.
pub fn masked_fill(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    mask: &Tensor<CudaRuntime>,
    value: f64,
) -> Result<Tensor<CudaRuntime>> {
    validate_mask_dtype(mask)?;

    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let mask_contig = ensure_contiguous(mask)?;
    let numel = a.numel();

    let bcast = BroadcastContext::prepare(a, mask, &client.device)?;
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;

    if bcast.needs_broadcast {
        unsafe {
            launch_masked_fill_broadcast(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                mask_contig.ptr(),
                out.ptr(),
                value,
                bcast.strides_ptr(),
                bcast.shape_ptr(),
                bcast.ndim,
                numel,
            )?;
        }
    } else {
        unsafe {
            launch_masked_fill(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                mask_contig.ptr(),
                out.ptr(),
                value,
                numel,
            )?;
        }
    }

    Ok(out)
}
