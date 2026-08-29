//! Native tiled GEMM entry points for the CUDA client.
//!
//! Handles the operand preparation the kernels cannot: transpose-view
//! detection so a `[K,N]` view reaches GEMV without materialising the copy,
//! batch broadcasting, and the integer output-dtype widening.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::matmul_output_shape;
use crate::runtime::cuda::kernels::{
    int_matmul_output_dtype, launch_gemv_kernel_bt_mr, launch_matmul_batched_kernel,
    launch_matmul_kernel,
};
use crate::runtime::cuda::ops::matmul_broadcast::resolve_batched_operands;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Native matrix multiplication using tiled CUDA kernel.
///
/// Uses shared memory tiling for cache efficiency. This is the default
/// implementation that works without any vendor dependencies.
/// Detect if a 2D tensor is a simple transpose of a contiguous [N,K] matrix.
///
/// A tensor with shape [K, N] and strides [1, K] is a transpose view of
/// contiguous [N, K] data. We can pass the raw pointer directly to gemv_bt
/// instead of materializing the transpose (which copies the entire matrix).
fn is_simple_transpose_2d(tensor: &Tensor<CudaRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 2 {
        return false;
    }
    // shape=[K,N], strides=[1,K] means transpose of contiguous [N,K]
    strides[0] == 1 && strides[1] == shape[0] as isize
}

/// Whether this dtype may take the small-M GEMV fast path in a plain matmul.
///
/// FP8 has no GEMV kernel at all. I8 has none either, and could not use one: its
/// matmul widens to I32 (see `int_matmul_output_dtype`) while every GEMV kernel
/// writes the element type. CPU excludes I8 from its own GEMV-BT path for
/// the same reason, so both backends run the tiled kernel at every M.
#[inline]
fn has_gemv_kernel(dtype: DType) -> bool {
    !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2 | DType::I8)
}

pub(crate) fn matmul_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Fast path: if B is a transposed view of contiguous [N,K] and M is small,
    // use gemv_bt kernel directly — avoids copying the entire weight matrix.
    // FP8 and I8 are excluded: see `has_gemv_kernel`.
    if m <= 16 && has_gemv_kernel(dtype) && is_simple_transpose_2d(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_gemv_kernel_bt_mr(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(), // raw [N,K] pointer — no copy!
                out.ptr(),
                1, // batch
                m,
                n,
                k,
                1, // a_batch
                1, // b_batch
            )?;
        }

        return Ok(out);
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    // I8 is the one dtype whose matmul does not write its own dtype: it widens
    // to I32, matching CPU's quantized accumulation. Every other dtype maps to
    // itself here.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Detect if the last two dims of a 3D tensor are a simple transpose.
/// Shape [B, K, N] with strides [B_stride, 1, K] means each batch slice
/// is a transpose of contiguous [N, K].
fn is_batched_transpose_last2(tensor: &Tensor<CudaRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 3 {
        return false;
    }
    let k = shape[1];
    let n = shape[2];
    // strides: [n*k, 1, k] means transpose of contiguous [batch, N, K]
    strides[1] == 1 && strides[2] == k as isize && strides[0] == (n * k) as isize
}

/// Native batched matrix multiplication using tiled CUDA kernel.
pub(crate) fn matmul_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Pointers and batch counts must come from the same tensors, so both are taken
    // from one resolver rather than derived separately.
    let operands = resolve_batched_operands(a, b, &out_shape)?;
    let (a, b) = (&operands.a, &operands.b);
    let (a_batch, b_batch) = (operands.a_batch, operands.b_batch);

    // Fast path: transposed B with small M → gemv_bt. FP8 and I8 are excluded
    // for the same reason as in matmul_native.
    if m <= 16 && has_gemv_kernel(dtype) && is_batched_transpose_last2(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_gemv_kernel_bt_mr(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(),
                out.ptr(),
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            )?;
        }

        return Ok(out);
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    // I8 widens to I32 here too — see `matmul_native`.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
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
