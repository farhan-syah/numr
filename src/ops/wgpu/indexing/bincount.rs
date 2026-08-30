//! bincount for WebGPU.
//!
//! Counts occurrences of each value in a 1D integer tensor. The weighted form
//! accumulates F32 weights through float atomics, so it is F32-only.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, TypeConversionOps};
use crate::runtime::RuntimeClient;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    BincountParams, create_params_buffer, ensure_i32_indices, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::launch_bincount;
use crate::tensor::Tensor;

pub(super) fn bincount(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    weights: Option<&Tensor<WgpuRuntime>>,
    minlength: usize,
) -> Result<Tensor<WgpuRuntime>> {
    // Validate input is 1D integer. CPU is the reference backend, so both
    // rejections use the variant and payload it uses: a caller matching on the
    // error must not have to special-case which backend produced it.
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
        if !matches!(w.dtype(), DType::F32 | DType::I32 | DType::U32) {
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

    let n = input.numel();

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
    // `ensure_i32_indices` above is where an I64 input narrows.
    let max_tensor = client.max(&input, &[0], true)?;
    let max_val = max_tensor.item::<i32>()? as i64;
    if max_val < 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "bincount requires non-negative values".to_string(),
        });
    }
    let output_len = ((max_val as usize) + 1).max(minlength);

    // Allocate zero-initialized output buffer.
    // Unweighted: U32 counts. Weighted: same dtype as weights (shader uses atomic<u32> bitcast).
    let output = if output_dtype == DType::U32 {
        let zeros = vec![0u32; output_len];
        Tensor::<WgpuRuntime>::from_slice(&zeros, &[output_len], client.device())?
    } else {
        Tensor::zeros(&[output_len], output_dtype, client.device())?
    };

    // Get buffers
    let input_buf = get_tensor_buffer(&input)?;
    let output_buf = get_tensor_buffer(&output)?;

    let weights_buf = if let Some(ref w) = weights {
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
        weights.as_ref().map(|w| w.dtype()),
    )?;

    // Cast U32 kernel output to I64 for parity with CPU backend (unweighted returns I64)
    if weights.is_none() {
        return client.cast(&output, DType::I64);
    }

    Ok(output)
}
