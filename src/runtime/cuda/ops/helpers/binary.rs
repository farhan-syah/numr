//! Element-wise binary op launchers for the CUDA client.
//!
//! Equal shapes take the contiguous kernel; unequal shapes take the strided
//! broadcast kernel, so both stay on the GPU.

use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{launch_binary_op, launch_broadcast_binary_op};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

/// Launch a native binary operation on GPU.
///
/// # Performance
///
/// - **Same shape**: Runs entirely on GPU (fast)
/// - **Different shapes**: Falls back to CPU with GPU↔CPU transfers (slow)
///
/// For broadcasting operations, consider pre-expanding tensors to matching shapes
/// using `broadcast_to()` or similar operations to avoid CPU fallback.
pub(crate) fn native_binary_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // A zero-element output has nothing to compute, and an empty launch grid
    // is itself invalid, so skip the launch entirely.
    if out_shape.iter().product::<usize>() == 0 {
        return Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
    }

    // For same-shape tensors, use the optimized element-wise kernel
    if a.shape() == b.shape() {
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_binary_op(
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
        launch_broadcast_binary_op(
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

/// Launch a native CUDA binary operation into a caller-provided destination.
///
/// Identical to [`native_binary_op`] but writes `op(a, b)` into the existing
/// `out` tensor instead of allocating. Required for destination-passing
/// workflows (e.g. CUDA graph capture) where the output buffer must be
/// allocated outside the captured region so its device address is stable.
///
/// `out` must be contiguous, share the inputs' dtype, and have a shape equal to
/// `broadcast(a, b)`.
pub(crate) fn native_binary_op_into(
    client: &CudaClient,
    out: &Tensor<CudaRuntime>,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<()> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    if out.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: out.dtype(),
        });
    }
    if out.shape() != out_shape.as_slice() {
        return Err(Error::ShapeMismatch {
            expected: out_shape,
            got: out.shape().to_vec(),
        });
    }
    if !out.is_contiguous() {
        return Err(Error::Backend(
            "native_binary_op_into: destination tensor must be contiguous".into(),
        ));
    }

    // A zero-element destination has nothing to compute, and an empty launch
    // grid is itself invalid, so skip the launch entirely.
    if out.numel() == 0 {
        return Ok(());
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    if a.shape() == b.shape() {
        unsafe {
            launch_binary_op(
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
    } else {
        unsafe {
            launch_broadcast_binary_op(
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
    }

    Ok(())
}
