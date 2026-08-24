//! Compare operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::CompareOp;
use crate::runtime::{compute_broadcast_shape, validate_binary_dtypes};
use crate::tensor::Tensor;

/// Helper for comparison operations (eq, ne, lt, le, gt, ge)
pub fn compare_op_impl(
    client: &CpuClient,
    op: CompareOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &client.device)?;
    let out_ptr = out.ptr();

    // Fast path for same shapes, both contiguous
    let same_shapes = a.shape() == b.shape() && a.shape() == out_shape.as_slice();
    let both_contiguous = a.is_contiguous() && b.is_contiguous();

    if same_shapes && both_contiguous {
        let len = a.numel();
        let a_ptr = a.ptr();
        let b_ptr = b.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::compare_op_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                );
            }
        }, op_name);
    } else {
        // Broadcasting path: use strided kernel
        let a_broadcast = a.broadcast_to(&out_shape)?;
        let b_broadcast = b.broadcast_to(&out_shape)?;

        let a_strides: Vec<isize> = a_broadcast.layout().strides().to_vec();
        let b_strides: Vec<isize> = b_broadcast.layout().strides().to_vec();
        let a_ptr = a_broadcast.ptr();
        let b_ptr = b_broadcast.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::compare_op_strided_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    &out_shape,
                    &a_strides,
                    &b_strides,
                    0,
                    0,
                );
            }
        }, op_name);
    }

    Ok(out)
}
