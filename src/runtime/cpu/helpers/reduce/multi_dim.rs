//! Fused contiguous multi-dimension reduction

use super::common::{advance_coord, contiguous_strides, out_index_from_coord};
use crate::dispatch_dtype;
use crate::dtype::Element;
use crate::error::Result;
use crate::ops::{AccumulationPrecision, ReduceOp, reduce_output_shape};
use crate::runtime::cpu::kernels::Accumulator;
use crate::runtime::cpu::kernels::wide_acc::int_mean_from_sum;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// Fused contiguous multi-dimension reduction for small tensors.
///
/// Executes reduction in a single pass over input elements and writes directly
/// into output buckets, avoiding intermediate tensors from repeated single-dim
/// reductions.
pub(super) fn reduce_multi_dim_fused(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    precision: AccumulationPrecision,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let shape = a.shape();
    // `Native` on a float narrower than F32 means F32: a fused multi-dim sum
    // keeps one accumulator per output bucket, and in BF16 that accumulator
    // saturates and stalls on a constant. F32, F64, and integers resolve to
    // `Native` and take exactly the same path as before.
    let precision = precision.resolve(a.dtype());
    let out_shape = reduce_output_shape(shape, dims, keepdim);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, a.dtype(), &client.device)?;

    let mut reduce_mask = vec![false; shape.len()];
    for &d in dims {
        reduce_mask[d] = true;
    }

    let kept_axes: Vec<usize> = if keepdim {
        Vec::new()
    } else {
        (0..shape.len())
            .filter(|&axis| !reduce_mask[axis])
            .collect()
    };
    let out_strides = contiguous_strides(&out_shape);
    let reduce_count = dims.iter().fold(1usize, |acc, &d| acc * shape[d]);
    let numel = a.numel();
    let out_numel = out.numel();

    let in_ptr = a.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(a.dtype(), T => {
        unsafe {
            // Integer `mean` keeps one accumulator per output bucket, and in the
            // element type that accumulator wraps on a sum whose average the
            // output dtype could still represent. Sum in i128 and divide there.
            if T::DTYPE.is_int() && matches!(op, ReduceOp::Mean) {
                reduce_multi_dim_fused_int_mean::<T>(
                    in_ptr as *const T,
                    out_ptr as *mut T,
                    numel,
                    out_numel,
                    shape,
                    &reduce_mask,
                    keepdim,
                    &kept_axes,
                    &out_strides,
                    reduce_count,
                );
            } else {
                match precision {
                    AccumulationPrecision::Native => reduce_multi_dim_fused_native::<T>(
                        op,
                        in_ptr as *const T,
                        out_ptr as *mut T,
                        numel,
                        out_numel,
                        shape,
                        &reduce_mask,
                        keepdim,
                        &kept_axes,
                        &out_strides,
                        reduce_count,
                    ),
                    AccumulationPrecision::FP32 | AccumulationPrecision::BF16 => {
                        reduce_multi_dim_fused_acc::<T, f32>(
                            op,
                            in_ptr as *const T,
                            out_ptr as *mut T,
                            numel,
                            out_numel,
                            shape,
                            &reduce_mask,
                            keepdim,
                            &kept_axes,
                            &out_strides,
                            reduce_count,
                        )
                    }
                    AccumulationPrecision::FP64 => reduce_multi_dim_fused_acc::<T, f64>(
                        op,
                        in_ptr as *const T,
                        out_ptr as *mut T,
                        numel,
                        out_numel,
                        shape,
                        &reduce_mask,
                        keepdim,
                        &kept_axes,
                        &out_strides,
                        reduce_count,
                    ),
                }
            }
        }
    }, op_name);

    Ok(out)
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_multi_dim_fused_native<T: Element>(
    op: ReduceOp,
    input: *const T,
    output: *mut T,
    numel: usize,
    out_numel: usize,
    shape: &[usize],
    reduce_mask: &[bool],
    keepdim: bool,
    kept_axes: &[usize],
    out_strides: &[usize],
    reduce_count: usize,
) {
    match op {
        ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Any => {
            for i in 0..out_numel {
                *output.add(i) = T::zero();
            }
        }
        ReduceOp::Prod | ReduceOp::All => {
            for i in 0..out_numel {
                *output.add(i) = T::one();
            }
        }
        ReduceOp::Max | ReduceOp::Min => {}
    }

    let mut initialized = if matches!(op, ReduceOp::Max | ReduceOp::Min) {
        vec![false; out_numel]
    } else {
        Vec::new()
    };

    let mut coord = vec![0usize; shape.len()];
    for linear in 0..numel {
        let out_idx = out_index_from_coord(&coord, reduce_mask, keepdim, kept_axes, out_strides);
        let val = *input.add(linear);

        match op {
            ReduceOp::Sum | ReduceOp::Mean => {
                let acc = *output.add(out_idx);
                *output.add(out_idx) = acc + val;
            }
            ReduceOp::Prod => {
                let acc = *output.add(out_idx);
                *output.add(out_idx) = acc * val;
            }
            ReduceOp::Max => {
                if !initialized[out_idx] {
                    *output.add(out_idx) = val;
                    initialized[out_idx] = true;
                } else {
                    let acc = *output.add(out_idx);
                    *output.add(out_idx) = if val > acc { val } else { acc };
                }
            }
            ReduceOp::Min => {
                if !initialized[out_idx] {
                    *output.add(out_idx) = val;
                    initialized[out_idx] = true;
                } else {
                    let acc = *output.add(out_idx);
                    *output.add(out_idx) = if val < acc { val } else { acc };
                }
            }
            ReduceOp::All => {
                let acc = *output.add(out_idx);
                *output.add(out_idx) = if val.to_f64() != 0.0 && acc.to_f64() != 0.0 {
                    T::one()
                } else {
                    T::zero()
                };
            }
            ReduceOp::Any => {
                let acc = *output.add(out_idx);
                *output.add(out_idx) = if val.to_f64() != 0.0 || acc.to_f64() != 0.0 {
                    T::one()
                } else {
                    T::zero()
                };
            }
        }

        if linear + 1 < numel {
            advance_coord(&mut coord, shape);
        }
    }

    if matches!(op, ReduceOp::Mean) {
        for i in 0..out_numel {
            let scaled = (*output.add(i)).to_f64() / reduce_count as f64;
            *output.add(i) = T::from_f64(scaled);
        }
    }
}

/// Fused multi-dimension integer `mean`, accumulating in i128.
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn reduce_multi_dim_fused_int_mean<T: Element>(
    input: *const T,
    output: *mut T,
    numel: usize,
    out_numel: usize,
    shape: &[usize],
    reduce_mask: &[bool],
    keepdim: bool,
    kept_axes: &[usize],
    out_strides: &[usize],
    reduce_count: usize,
) {
    let mut acc = vec![0i128; out_numel];

    let mut coord = vec![0usize; shape.len()];
    for linear in 0..numel {
        let out_idx = out_index_from_coord(&coord, reduce_mask, keepdim, kept_axes, out_strides);
        acc[out_idx] = acc[out_idx].saturating_add((*input.add(linear)).to_i128());

        if linear + 1 < numel {
            advance_coord(&mut coord, shape);
        }
    }

    for (i, &sum) in acc.iter().enumerate() {
        *output.add(i) = int_mean_from_sum::<T>(sum, reduce_count);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_multi_dim_fused_acc<T: Element, A: Accumulator>(
    op: ReduceOp,
    input: *const T,
    output: *mut T,
    numel: usize,
    out_numel: usize,
    shape: &[usize],
    reduce_mask: &[bool],
    keepdim: bool,
    kept_axes: &[usize],
    out_strides: &[usize],
    reduce_count: usize,
) {
    let mut acc = match op {
        ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Any | ReduceOp::Max | ReduceOp::Min => {
            vec![A::ZERO; out_numel]
        }
        ReduceOp::Prod | ReduceOp::All => vec![A::ONE; out_numel],
    };

    let mut initialized = if matches!(op, ReduceOp::Max | ReduceOp::Min) {
        vec![false; out_numel]
    } else {
        Vec::new()
    };

    let mut coord = vec![0usize; shape.len()];
    for linear in 0..numel {
        let out_idx = out_index_from_coord(&coord, reduce_mask, keepdim, kept_axes, out_strides);
        let val = A::acc_in((*input.add(linear)).to_f64());

        match op {
            ReduceOp::Sum | ReduceOp::Mean => {
                acc[out_idx] = acc[out_idx].acc_add(val);
            }
            ReduceOp::Prod => {
                acc[out_idx] = acc[out_idx].acc_mul(val);
            }
            ReduceOp::Max => {
                if !initialized[out_idx] {
                    acc[out_idx] = val;
                    initialized[out_idx] = true;
                } else if val > acc[out_idx] {
                    acc[out_idx] = val;
                }
            }
            ReduceOp::Min => {
                if !initialized[out_idx] {
                    acc[out_idx] = val;
                    initialized[out_idx] = true;
                } else if val < acc[out_idx] {
                    acc[out_idx] = val;
                }
            }
            ReduceOp::All => {
                acc[out_idx] = if val != A::ZERO && acc[out_idx] != A::ZERO {
                    A::ONE
                } else {
                    A::ZERO
                };
            }
            ReduceOp::Any => {
                acc[out_idx] = if val != A::ZERO || acc[out_idx] != A::ZERO {
                    A::ONE
                } else {
                    A::ZERO
                };
            }
        }

        if linear + 1 < numel {
            advance_coord(&mut coord, shape);
        }
    }

    for i in 0..out_numel {
        let mut out_val = acc[i];
        if matches!(op, ReduceOp::Mean) {
            out_val = out_val.acc_div(reduce_count);
        }
        *output.add(i) = T::from_f64(out_val.into());
    }
}
