//! Fused elementwise operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Helper for fused_mul_add: out = a * b + c
pub fn fused_mul_add_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    c: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    if b.dtype() != dtype || c.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: if b.dtype() != dtype {
                b.dtype()
            } else {
                c.dtype()
            },
        });
    }
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: if a.shape() != b.shape() {
                b.shape().to_vec()
            } else {
                c.shape().to_vec()
            },
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let c_contig = ensure_contiguous(c)?;
    let out = Tensor::<CpuRuntime>::try_empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let b_ptr = b_contig.ptr();
    let c_ptr = c_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::fused_mul_add_kernel::<T>(
                a_ptr as *const T,
                b_ptr as *const T,
                c_ptr as *const T,
                out_ptr as *mut T,
                len,
            );
        }
    }, "fused_mul_add");

    Ok(out)
}

/// Helper for fused_add_mul: out = (a + b) * c
pub fn fused_add_mul_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    c: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    if b.dtype() != dtype || c.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: if b.dtype() != dtype {
                b.dtype()
            } else {
                c.dtype()
            },
        });
    }
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: if a.shape() != b.shape() {
                b.shape().to_vec()
            } else {
                c.shape().to_vec()
            },
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let c_contig = ensure_contiguous(c)?;
    let out = Tensor::<CpuRuntime>::try_empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let b_ptr = b_contig.ptr();
    let c_ptr = c_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::fused_add_mul_kernel::<T>(
                a_ptr as *const T,
                b_ptr as *const T,
                c_ptr as *const T,
                out_ptr as *mut T,
                len,
            );
        }
    }, "fused_add_mul");

    Ok(out)
}

/// Helper for fused_mul_add_scalar: out = a * scale + bias
pub fn fused_mul_add_scalar_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    scale: f64,
    bias: f64,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::try_empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::fused_mul_add_scalar_kernel::<T>(
                a_ptr as *const T,
                scale,
                bias,
                out_ptr as *mut T,
                len,
            );
        }
    }, "fused_mul_add_scalar");

    Ok(out)
}
