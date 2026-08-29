//! Element-wise comparison launcher for the CUDA client.
//!
//! Equal shapes take the contiguous kernel; unequal shapes take the strided
//! broadcast kernel, so both stay on the GPU.

use crate::error::Result;
use crate::runtime::cuda::kernels::{launch_broadcast_compare_op, launch_compare_op};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

/// Launch a native comparison operation on GPU.
///
/// # Performance
///
/// - **Same shape**: Uses optimized element-wise kernel (fast)
/// - **Different shapes**: Uses broadcast kernel with strided access (stays on GPU)
pub(crate) fn native_compare_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // A zero-element output has nothing to compute, and an empty launch grid
    // is itself invalid, so skip the launch entirely. Placed after the
    // dtype/shape checks so a mismatched pair still reports its error rather
    // than succeeding because it was empty.
    if out_shape.iter().product::<usize>() == 0 {
        return Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
    }

    // For same-shape tensors, use the optimized element-wise kernel
    if a.shape() == b.shape() {
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_compare_op(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        return Ok(out);
    }

    // For different shapes, use the broadcast kernel (stays on GPU)
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    unsafe {
        launch_broadcast_compare_op(
            &client.context,
            &client.stream,
            client.device.index,
            &client.device,
            op,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            a.shape(),
            b.shape(),
            &out_shape,
        )?;
    }

    Ok(out)
}
