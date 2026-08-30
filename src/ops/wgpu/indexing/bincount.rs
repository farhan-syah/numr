//! bincount for WebGPU.
//!
//! Counts occurrences of each value in a 1D integer tensor. The weighted form
//! accumulates F32 weights through float atomics, so it is F32-only.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, ShapeOps, TypeConversionOps};
use crate::runtime::RuntimeClient;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    BincountParams, create_params_buffer, ensure_i32_indices, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::launch_bincount;
use crate::tensor::Tensor;

/// Validated, I32-normalized form of a bincount call.
struct BincountPlan {
    /// Input narrowed to contiguous I32: WebGPU shaders index in i32.
    input: Tensor<WgpuRuntime>,
    weights: Option<Tensor<WgpuRuntime>>,
    /// U32 counts when unweighted, else the weights dtype.
    output_dtype: DType,
}

/// Validate rank, input dtype, weights shape and weights dtype, then narrow the
/// input to contiguous I32.
///
/// CPU is the reference backend, so each rejection uses the variant and payload
/// it uses: a caller matching on the error must not have to special-case which
/// backend produced it.
fn plan_bincount(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    weights: Option<&Tensor<WgpuRuntime>>,
) -> Result<BincountPlan> {
    if input.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![input.numel()],
            got: input.shape().to_vec(),
        });
    }

    if !matches!(input.dtype(), DType::I32 | DType::I64) {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: input.dtype(),
        });
    }

    // Determine output dtype
    let output_dtype = if let Some(w) = weights {
        if w.shape() != input.shape() {
            return Err(Error::ShapeMismatch {
                expected: input.shape().to_vec(),
                got: w.shape().to_vec(),
            });
        }
        // F32 only: the weighted shader accumulates through a compare-and-swap
        // loop over `atomic<u32>`, reading each slot back as an f32. Accepting
        // I32 or U32 here would pass validation and then fail inside
        // `launch_bincount`, which is F32-only, so the two must agree.
        if w.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: w.dtype(),
                op: "bincount weights",
            });
        }
        w.dtype()
    } else {
        DType::U32 // Unweighted bincount returns counts as U32
    };

    // Cast I64→I32 on GPU (WebGPU shaders use i32 indices)
    let input_i32 = ensure_i32_indices(client, input)?;
    let input = ensure_contiguous(&input_i32)?;
    let weights = weights.map(ensure_contiguous).transpose()?;

    Ok(BincountPlan {
        input,
        weights,
        output_dtype,
    })
}

/// Allocate a zeroed histogram of `output_len` bins and dispatch the shader.
///
/// The shader already skips any value outside `[0, output_len)`, so both entry
/// points share it unchanged and differ only in how `output_len` is obtained.
fn accumulate(
    client: &WgpuClient,
    plan: &BincountPlan,
    output_len: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let n = plan.input.numel();

    // Zero bins: WebGPU rejects a zero-sized buffer, so there is nothing to bind
    // a dispatch to. The result is the empty histogram, in the dtype the
    // populated path would have returned.
    if output_len == 0 {
        let dtype = if plan.weights.is_none() {
            DType::I64
        } else {
            plan.output_dtype
        };
        return Tensor::empty(&[0], dtype, client.device());
    }

    // Allocate zero-initialized output buffer.
    // Unweighted: U32 counts. Weighted: same dtype as weights (shader uses atomic<u32> bitcast).
    let output = if plan.output_dtype == DType::U32 {
        let zeros = vec![0u32; output_len];
        Tensor::<WgpuRuntime>::from_slice(&zeros, &[output_len], client.device())?
    } else {
        Tensor::zeros(&[output_len], plan.output_dtype, client.device())?
    };

    // An empty input contributes to no bin, so the zeroed histogram above is
    // already the answer. `get_tensor_buffer` has no buffer to return for the
    // input's zero-byte allocation, so the dispatch must not be reached.
    if n == 0 {
        if plan.weights.is_none() {
            return client.cast(&output, DType::I64);
        }
        return Ok(output);
    }

    // Get buffers
    let input_buf = get_tensor_buffer(&plan.input)?;
    let output_buf = get_tensor_buffer(&output)?;

    let weights_buf = if let Some(ref w) = plan.weights {
        Some(get_tensor_buffer(w)?)
    } else {
        None
    };

    // Create params
    let params = BincountParams {
        n: n as u32,
        minlength: output_len as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_bincount(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &input_buf,
        weights_buf.as_deref(),
        &output_buf,
        &params_buf,
        n,
        plan.weights.as_ref().map(|w| w.dtype()),
    )?;

    // Cast U32 kernel output to I64 for parity with CPU backend (unweighted returns I64)
    if plan.weights.is_none() {
        return client.cast(&output, DType::I64);
    }

    Ok(output)
}

pub(super) fn bincount(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    weights: Option<&Tensor<WgpuRuntime>>,
    minlength: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let plan = plan_bincount(client, input, weights)?;

    // An empty input holds no value at all, so it holds no negative one: the
    // answer is `minlength` zeroed bins, which is what CPU and CUDA return.
    // WebGPU also rejects a zero-sized buffer, so the sizing reduction below
    // must not be reached — there is nothing for it to bind to.
    if plan.input.numel() == 0 {
        return accumulate(client, &plan, minlength);
    }

    // Determine output size: max reduction on GPU, read single scalar back.
    // This is a necessary system boundary (same as CPU/CUDA computing max first).
    //
    // The max is taken in I32, not a float dtype. CUDA casts to F64 because its
    // reduce kernels have no integer path, and F64's 53-bit mantissa holds every
    // i32/i64 index exactly. WebGPU has no F64, and F32's 24-bit mantissa rounds
    // any value above 2^24, which would size the output wrongly. WebGPU does have
    // native I32 reduce kernels, so the max stays in the integer domain and is
    // exact across the whole I32 range. Values beyond I32 are unrepresentable on
    // this backend at all — that is WebGPU's 32-bit dtype limit, and
    // `ensure_i32_indices` in `plan_bincount` is where an I64 input narrows.
    //
    // The minimum rides along with the maximum. Checking `max < 0` alone let a
    // negative sitting beside a positive maximum through to the shader, which
    // silently dropped it. Detecting one needs the minimum, so both reductions
    // run on device and their two scalars are concatenated into a single
    // 2-element tensor: one extra reduction and one extra `cat`, but still
    // exactly one device-to-host readback — the same sizing sync this path
    // already performed.
    //
    // `bincount_with_len` below exists to let a caller that already knows the
    // output length skip this sizing sync entirely.
    let min_tensor = client.min(&plan.input, &[0], true)?;
    let max_tensor = client.max(&plan.input, &[0], true)?;
    let bounds = client.cat(&[&min_tensor, &max_tensor], 0)?;
    let bounds = bounds.to_vec::<i32>();
    let (min_val, max_val) = match bounds.as_slice() {
        [lo, hi] => (*lo as i64, *hi as i64),
        other => {
            return Err(Error::Internal(format!(
                "bincount: the min/max readback returned {} values, expected 2",
                other.len()
            )));
        }
    };
    if min_val < 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "bincount requires non-negative values".to_string(),
        });
    }
    let output_len = ((max_val as usize) + 1).max(minlength);

    accumulate(client, &plan, output_len)
}

/// bincount into a caller-sized histogram of exactly `len` bins.
///
/// No max reduction and no `item()` readback: nothing on this path moves data
/// from device to host, which is the entire reason it exists. Values outside
/// `[0, len)` — negative or too large — are ignored rather than rejected,
/// because detecting one would need the sync this path avoids. The shader's own
/// bounds test performs that filtering on device.
pub(super) fn bincount_with_len(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    weights: Option<&Tensor<WgpuRuntime>>,
    len: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let plan = plan_bincount(client, input, weights)?;
    accumulate(client, &plan, len)
}
