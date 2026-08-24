//! Scalar operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::BinaryOp;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Helper for scalar operations (scalar arithmetic with tensors)
pub fn scalar_op_impl(
    client: &CpuClient,
    op: BinaryOp,
    a: &Tensor<CpuRuntime>,
    scalar: f64,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scalar_op_kernel::<T>(
                op,
                a_ptr as *const T,
                scalar,
                out_ptr as *mut T,
                len,
            );
        }
    }, op_name);

    Ok(out)
}

/// Reverse scalar subtract: scalar - a
pub fn rsub_scalar_op_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    scalar: f64,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::rsub_scalar_kernel::<T>(
                a_ptr as *const T,
                scalar,
                out_ptr as *mut T,
                len,
            );
        }
    }, "rsub_scalar");

    Ok(out)
}
