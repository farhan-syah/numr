//! bincount helpers for CPU tensors.
//!
//! Two entry points share one histogram core and differ only in how the output
//! length is obtained: `bincount_impl` derives it from a max scan over the
//! input, `bincount_with_len_impl` takes it from the caller.

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Validated, dtype-normalized form of a bincount call.
struct BincountInput {
    /// Input values widened to i64 regardless of the I32/I64 input dtype.
    values: Vec<i64>,
    /// Output dtype: the weights dtype when weighted, I64 counts otherwise.
    out_dtype: DType,
}

/// Validate rank, input dtype and weights shape, then widen the input to i64.
///
/// Every rejection here uses the same variant and payload as the existing
/// `bincount` path: a caller matching on the error must not have to know which
/// entry point or which backend produced it.
fn prepare(
    input: &Tensor<CpuRuntime>,
    weights: Option<&Tensor<CpuRuntime>>,
) -> Result<BincountInput> {
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

    let out_dtype = if let Some(w) = weights {
        if w.shape() != input.shape() {
            return Err(Error::ShapeMismatch {
                expected: input.shape().to_vec(),
                got: w.shape().to_vec(),
            });
        }
        w.dtype()
    } else {
        DType::I64 // Count output is I64 when no weights
    };

    let input_contig = ensure_contiguous(input)?;
    let numel = input.numel();
    let values: Vec<i64> = if input_dtype == DType::I64 {
        unsafe { std::slice::from_raw_parts(input_contig.ptr() as *const i64, numel).to_vec() }
    } else {
        let i32_slice =
            unsafe { std::slice::from_raw_parts(input_contig.ptr() as *const i32, numel) };
        i32_slice.iter().map(|&x| x as i64).collect()
    };

    Ok(BincountInput { values, out_dtype })
}

/// Accumulate the histogram into a freshly allocated output of `output_len` bins.
///
/// `reject_negative` selects the two contracts: the torch-compatible `bincount`
/// rejects a negative input, while the caller-sized path ignores every value
/// outside `[0, output_len)` because detecting one is exactly the work it exists
/// to skip.
fn accumulate(
    client: &CpuClient,
    prepared: &BincountInput,
    weights: Option<&Tensor<CpuRuntime>>,
    output_len: usize,
    reject_negative: bool,
) -> Result<Tensor<CpuRuntime>> {
    let numel = prepared.values.len();
    let out_dtype = prepared.out_dtype;
    let out = Tensor::<CpuRuntime>::empty(&[output_len], out_dtype, &client.device)?;

    // Zero bins: an empty allocation yields a null pointer, and the kernel would
    // build a slice from it. Nothing to accumulate, so hand the tensor back.
    if output_len == 0 {
        return Ok(out);
    }

    let out_ptr = out.ptr();

    if let Some(w) = weights {
        let w_contig = ensure_contiguous(w)?;
        let w_ptr = w_contig.ptr();

        dispatch_dtype!(out_dtype, T => {
            let success = unsafe {
                kernels::bincount_kernel::<T>(
                    prepared.values.as_ptr(),
                    w_ptr as *const T,
                    out_ptr as *mut T,
                    numel,
                    output_len,
                    reject_negative,
                )
            };
            if !success {
                return Err(Error::InvalidArgument {
                    arg: "input",
                    reason: "bincount requires non-negative values".to_string(),
                });
            }
        }, "bincount");
    } else {
        // No weights - output is I64 counts
        let success = unsafe {
            kernels::bincount_kernel::<i64>(
                prepared.values.as_ptr(),
                std::ptr::null(),
                out_ptr as *mut i64,
                numel,
                output_len,
                reject_negative,
            )
        };
        if !success {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "bincount requires non-negative values".to_string(),
            });
        }
    }

    Ok(out)
}

/// Count occurrences of each value in an integer tensor.
pub fn bincount_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    weights: Option<&Tensor<CpuRuntime>>,
    minlength: usize,
) -> Result<Tensor<CpuRuntime>> {
    let prepared = prepare(input, weights)?;

    // Find max value to determine output size
    let max_val =
        unsafe { kernels::max_i64_kernel(prepared.values.as_ptr(), prepared.values.len()) };
    if max_val < 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "bincount requires non-negative values".to_string(),
        });
    }
    let output_len = (max_val as usize + 1).max(minlength);

    accumulate(client, &prepared, weights, output_len, true)
}

/// Count occurrences into a caller-sized histogram of exactly `len` bins.
///
/// Values outside `[0, len)` — negative or too large — are ignored, not
/// rejected. This is the reference behaviour every backend matches.
pub fn bincount_with_len_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    weights: Option<&Tensor<CpuRuntime>>,
    len: usize,
) -> Result<Tensor<CpuRuntime>> {
    let prepared = prepare(input, weights)?;
    accumulate(client, &prepared, weights, len, false)
}
