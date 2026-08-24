//! Binary operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::{Error, Result};
use crate::ops::{BinaryOp, Kernel};
use crate::runtime::{compute_broadcast_shape, validate_binary_dtypes};
use crate::tensor::Tensor;

/// Helper for binary operations (add, sub, mul, div, pow, max, min)
pub fn binary_op_impl(
    client: &CpuClient,
    op: BinaryOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // Create output tensor
    let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &client.device)?;

    write_binary_into(client, op, a, b, &out, &out_shape, op_name)?;

    Ok(out)
}

/// Destination-passing binary op: writes `op(a, b)` into the caller-owned `out`
/// tensor instead of allocating. Required for workflows where the output buffer
/// must have a stable address (e.g. CUDA graph capture). `out` must be
/// contiguous and its shape must equal `broadcast(a, b)`.
pub fn binary_op_into_impl(
    client: &CpuClient,
    op: BinaryOp,
    out: &Tensor<CpuRuntime>,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
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
            "binary_op_into: destination tensor must be contiguous".into(),
        ));
    }

    write_binary_into(client, op, a, b, out, &out_shape, op_name)
}

/// Shared computation for both the allocating and destination-passing paths.
/// Writes `op(a, b)` into `out` (which must already have shape `out_shape` and
/// the validated dtype).
fn write_binary_into(
    client: &CpuClient,
    op: BinaryOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    out: &Tensor<CpuRuntime>,
    out_shape: &[usize],
    op_name: &'static str,
) -> Result<()> {
    let dtype = out.dtype();
    let out_ptr = out.ptr();

    // Check if we can use the fast path (same shapes, both contiguous)
    let same_shapes = a.shape() == b.shape() && a.shape() == out_shape;
    let both_contiguous = a.is_contiguous() && b.is_contiguous();

    if same_shapes && both_contiguous {
        // Fast path: no broadcasting needed, use contiguous kernel
        let len = a.numel();
        let a_ptr = a.ptr();
        let b_ptr = b.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::binary_op::<T>(
                    client, op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                );
            }
        }, op_name);
    } else {
        // Broadcasting path: use strided kernel
        // Broadcast both inputs to output shape (zero-copy views with stride 0 for broadcast dims)
        let a_broadcast = a.broadcast_to(out_shape)?;
        let b_broadcast = b.broadcast_to(out_shape)?;

        let a_ptr = a_broadcast.ptr();
        let b_ptr = b_broadcast.ptr();

        // Get strides from broadcast layouts
        let a_strides: Vec<isize> = a_broadcast.layout().strides().to_vec();
        let b_strides: Vec<isize> = b_broadcast.layout().strides().to_vec();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::binary_op_strided_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    out_shape,
                    &a_strides,
                    &b_strides,
                    0,
                    0,
                );
            }
        }, op_name);
    }

    Ok(())
}
