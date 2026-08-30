//! Element-wise unary op launcher for the CUDA client.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::launch_unary_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Launch a native CUDA unary operation (element-wise, single input).
///
/// Dispatches to CUDA kernels for operations like neg, abs, sqrt, exp, log,
/// sin, cos, sigmoid, relu, etc. The operation runs entirely on GPU.
///
/// # Arguments
/// * `op` - Operation name (must match kernel function suffix, e.g., "neg", "exp")
pub(crate) fn native_unary_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();

    // `unary.cu` instantiates no `bool` row, so an unguarded launch would
    // fail kernel lookup with an opaque `Error::Internal`. CPU rejects Bool
    // here too (`dispatch_dtype!` has no Bool arm), so report the same
    // `UnsupportedDType` CPU does instead of a symbol-not-found error.
    if dtype == DType::Bool {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;

    // A zero-element tensor has nothing to compute, and an empty launch grid
    // is itself invalid, so skip the launch entirely.
    if out.numel() == 0 {
        return Ok(out);
    }

    unsafe {
        launch_unary_op(
            &client.context,
            &client.stream,
            client.device.index,
            op,
            dtype,
            a_contig.ptr(),
            out.ptr(),
            out.numel(),
        )?;
    }

    Ok(out)
}
