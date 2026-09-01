//! Native fused GEMM epilogue entry points for the CUDA client.
//!
//! Each prepares operands (contiguity, output allocation) and launches the
//! matching kernel. The launchers themselves pick between the generic tiled
//! kernels and the WMMA tensor-core kernels
//! (`runtime/cuda/kernels/gemm_epilogue/launcher.rs`).

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::GemmActivation;
use crate::runtime::cuda::kernels::{
    launch_gemm_bias_act_batched_kernel, launch_gemm_bias_act_kernel,
    launch_gemm_bias_residual_batched_kernel, launch_gemm_bias_residual_kernel,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// `C = activation(A @ B + bias)` for one 2-D GEMM.
///
/// `out_shape` is the caller's already-validated output shape — the padded one
/// on the WMMA padding path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_bias_act_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    out_shape: &[usize],
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let bias_contig = ensure_contiguous(bias)?;
    let out = Tensor::<CudaRuntime>::empty(out_shape, dtype, &client.device)?;

    unsafe {
        launch_gemm_bias_act_kernel(
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
            activation,
        )?;
    }

    Ok(out)
}

/// Batched `C = activation(A @ B + bias)`. The bias is `[N]` and broadcasts
/// across rows and batch slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_bias_act_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    out_shape: &[usize],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let bias_contig = ensure_contiguous(bias)?;
    let out = Tensor::<CudaRuntime>::empty(out_shape, dtype, &client.device)?;

    unsafe {
        launch_gemm_bias_act_batched_kernel(
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
            activation,
        )?;
    }

    Ok(out)
}

/// `C = A @ B + bias + residual` for one 2-D GEMM. The residual is elementwise
/// over `[M,N]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_bias_residual_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    residual: &Tensor<CudaRuntime>,
    dtype: DType,
    out_shape: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let bias_contig = ensure_contiguous(bias)?;
    let res_contig = ensure_contiguous(residual)?;
    let out = Tensor::<CudaRuntime>::empty(out_shape, dtype, &client.device)?;

    unsafe {
        launch_gemm_bias_residual_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            bias_contig.ptr(),
            res_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Batched `C = A @ B + bias + residual`. The residual carries one `[M,N]`
/// slice per batch index.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_bias_residual_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    residual: &Tensor<CudaRuntime>,
    dtype: DType,
    out_shape: &[usize],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let bias_contig = ensure_contiguous(bias)?;
    let res_contig = ensure_contiguous(residual)?;
    let out = Tensor::<CudaRuntime>::empty(out_shape, dtype, &client.device)?;

    unsafe {
        launch_gemm_bias_residual_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            bias_contig.ptr(),
            res_contig.ptr(),
            out.ptr(),
            batch,
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}
