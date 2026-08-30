//! bincount for CUDA.
//!
//! Two entry points share one validation step and one histogram launch, and
//! differ only in how the output length is obtained: `bincount` derives it from
//! a device max reduction plus a scalar readback, `bincount_with_len` takes it
//! from the caller and performs no device-to-host transfer at all.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, TypeConversionOps};
use crate::runtime::cuda::kernels::{launch_bincount_weighted, launch_fill_with_f64};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Validated form of a bincount call: dtypes settled, weights checked.
struct BincountPlan {
    input_dtype: DType,
    weights_dtype: Option<DType>,
    out_dtype: DType,
}

/// Validate rank, input dtype and weights shape.
///
/// Every rejection uses the variant and payload the CPU reference backend uses,
/// so a caller matching on the error does not have to special-case the backend.
fn bincount_validate(
    input: &Tensor<CudaRuntime>,
    weights: Option<&Tensor<CudaRuntime>>,
) -> Result<BincountPlan> {
    if input.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![input.numel()],
            got: input.shape().to_vec(),
        });
    }

    let input_dtype = input.dtype();
    if !matches!(input_dtype, DType::I32 | DType::I64) {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: input_dtype,
        });
    }

    let weights_dtype = if let Some(w) = weights {
        if w.shape() != input.shape() {
            return Err(Error::ShapeMismatch {
                expected: input.shape().to_vec(),
                got: w.shape().to_vec(),
            });
        }
        Some(w.dtype())
    } else {
        None
    };

    Ok(BincountPlan {
        input_dtype,
        weights_dtype,
        out_dtype: weights_dtype.unwrap_or(DType::I64),
    })
}

/// Zero an output of `output_len` bins and accumulate the histogram into it.
///
/// The kernel already skips any value outside `[0, output_len)`, so both entry
/// points share it unchanged and differ only in how `output_len` is obtained.
fn bincount_accumulate(
    client: &CudaClient,
    plan: &BincountPlan,
    input: &Tensor<CudaRuntime>,
    weights: Option<&Tensor<CudaRuntime>>,
    output_len: usize,
) -> Result<Tensor<CudaRuntime>> {
    let input_contig = ensure_contiguous(input)?;
    let numel = input.numel();

    // Allocate output and zero-initialize
    let out = Tensor::<CudaRuntime>::empty(&[output_len], plan.out_dtype, &client.device)?;

    // Zero bins: an empty allocation yields a null device pointer, and there is
    // nothing to fill or accumulate. Hand the empty histogram back.
    if output_len == 0 {
        return Ok(out);
    }

    // Zero the output buffer
    unsafe {
        launch_fill_with_f64(
            &client.context,
            &client.stream,
            client.device.index,
            plan.out_dtype,
            0.0,
            out.ptr(),
            output_len,
        )?;
    }

    let weights_contig = weights.map(ensure_contiguous).transpose()?;
    let weights_ptr = weights_contig.as_ref().map(|w| w.ptr());

    unsafe {
        launch_bincount_weighted(
            &client.context,
            &client.stream,
            client.device.index,
            plan.input_dtype,
            plan.weights_dtype,
            input_contig.ptr(),
            weights_ptr,
            out.ptr(),
            numel,
            output_len,
        )?;
    }

    Ok(out)
}

/// Execute bincount operation.
pub fn bincount(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weights: Option<&Tensor<CudaRuntime>>,
    minlength: usize,
) -> Result<Tensor<CudaRuntime>> {
    let plan = bincount_validate(input, weights)?;

    // Find the max value on GPU to determine output size.
    // Cast to F64 for max reduction (CUDA reduce kernels support F64 but not integer types),
    // then read the single scalar back to CPU for allocation sizing —
    // this is a necessary system boundary (same as CPU impl computing max first).
    // F64 preserves full i32/i64 precision (up to 2^53), unlike F32 which loses precision past 2^24.
    // `bincount_with_len` below exists to let a caller that already knows the
    // output length skip this sizing sync entirely.
    let input_f64 = client.cast(input, DType::F64)?;
    let max_tensor = client.max(&input_f64, &[0], false)?;
    let max_val = max_tensor.item::<f64>()? as i64;
    if max_val < 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "bincount requires non-negative values".to_string(),
        });
    }
    let output_len = ((max_val as usize) + 1).max(minlength);

    bincount_accumulate(client, &plan, input, weights, output_len)
}

/// Execute bincount into a caller-sized histogram of exactly `len` bins.
///
/// No max reduction and no `item()` readback: nothing on this path moves data
/// from device to host, which is the entire reason it exists. Values outside
/// `[0, len)` — negative or too large — are ignored rather than rejected,
/// because detecting one would need the sync this path avoids. The kernel's own
/// bounds test performs that filtering on device.
pub fn bincount_with_len(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weights: Option<&Tensor<CudaRuntime>>,
    len: usize,
) -> Result<Tensor<CudaRuntime>> {
    let plan = bincount_validate(input, weights)?;
    bincount_accumulate(client, &plan, input, weights, len)
}
