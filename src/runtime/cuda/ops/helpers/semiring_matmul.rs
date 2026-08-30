//! Semiring matrix multiplication entry points for the CUDA client.
//!
//! The semiring op is a kernel argument, so both forms share one kernel name
//! per dtype.
//!
//! `out_dtype` and `kernel_dtype` are separate because Bool has no kernel of its
//! own: it shares U8's, the two being one byte wide. Selecting that kernel must
//! not relabel the RESULT, whose dtype is fixed by the input's — so the output is
//! allocated as `out_dtype` while the launch dispatches on `kernel_dtype`.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::matmul_output_shape;
use crate::runtime::cuda::kernels::{
    launch_semiring_matmul_batched_kernel, launch_semiring_matmul_kernel,
};
use crate::runtime::cuda::ops::matmul_broadcast::resolve_batched_operands;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Native semiring matrix multiplication using CUDA kernel.
pub(crate) fn semiring_matmul_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    out_dtype: DType,
    kernel_dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    semiring_op: u32,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, out_dtype, &client.device)?;

    unsafe {
        launch_semiring_matmul_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            kernel_dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
            semiring_op,
        )?;
    }

    Ok(out)
}

/// Native batched semiring matrix multiplication using CUDA kernel.
pub(crate) fn semiring_matmul_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    out_dtype: DType,
    kernel_dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    semiring_op: u32,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Pointers and batch counts must come from the same tensors, so both are taken
    // from one resolver rather than derived separately.
    let operands = resolve_batched_operands(a, b, &out_shape)?;
    let (a_contig, b_contig) = operands.contiguous()?;
    let (a_batch, b_batch) = (operands.a_batch, operands.b_batch);

    let out = Tensor::<CudaRuntime>::empty(&out_shape, out_dtype, &client.device)?;

    unsafe {
        launch_semiring_matmul_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            kernel_dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            batch,
            m,
            n,
            k,
            semiring_op,
            a_batch,
            b_batch,
        )?;
    }

    Ok(out)
}
