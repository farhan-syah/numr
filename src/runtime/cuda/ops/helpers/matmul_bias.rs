//! Native fused matmul+bias entry points for the CUDA client.
//!
//! Same tiled GEMM as [`super::matmul`], with the bias folded into the
//! epilogue so the result is not read back for a separate add.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::matmul_bias_output_shape;
use crate::runtime::cuda::kernels::{
    int_matmul_output_dtype, launch_matmul_bias_batched_kernel, launch_matmul_bias_kernel,
};
use crate::runtime::cuda::ops::matmul_broadcast::resolve_batched_operands;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Native fused matmul+bias using tiled CUDA kernel: C = A @ B + bias
///
/// Uses the same tiled GEMM algorithm as matmul_native, but fuses bias addition
/// into the epilogue to avoid an extra memory round-trip.
pub(crate) fn matmul_bias_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let bias_contig = ensure_contiguous(bias)?;

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
        Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        },
    )?;

    // I8 widens to I32 in the fused form exactly as in the plain one, and the
    // validator has already required an I32 bias to seed that accumulator.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_bias_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            bias_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Native batched fused matmul+bias using tiled CUDA kernel:
/// C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] + bias[N]
pub(crate) fn matmul_bias_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let bias_contig = ensure_contiguous(bias)?;

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
        Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        },
    )?;

    // Pointers and batch counts must come from the same tensors, so both are taken
    // from one resolver rather than derived separately.
    let operands = resolve_batched_operands(a, b, &out_shape)?;
    let (a_contig, b_contig) = operands.contiguous()?;
    let (a_batch, b_batch) = (operands.a_batch, operands.b_batch);

    // Widens at I8 like every other matmul entry point here.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_bias_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            bias_contig.ptr(),
            out.ptr(),
            batch,
            m,
            n,
            k,
            a_batch,
            b_batch,
        )?;
    }

    Ok(out)
}
