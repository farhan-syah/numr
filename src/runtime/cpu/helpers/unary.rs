//! Unary operation helpers for CPU tensors

use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::{Kernel, UnaryOp};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Helper for unary operations (neg, abs, sqrt, exp, log, sin, cos, etc.)
pub fn unary_op_impl(
    client: &CpuClient,
    op: UnaryOp,
    a: &Tensor<CpuRuntime>,
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
            <CpuClient as Kernel<CpuRuntime>>::unary_op::<T>(
                client, op,
                a_ptr as *const T,
                out_ptr as *mut T,
                len,
            );
        }
    }, op_name);

    Ok(out)
}
