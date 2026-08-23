//! Precision-aware reduction helpers (FP32/FP64 accumulation)

use super::multi_dim::reduce_multi_dim_fused;
use super::single_dim::reduce_non_last_dim_outer;
use crate::dispatch_dtype;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{AccumulationPrecision, ReduceOp, reduce_output_shape};
use crate::runtime::cpu::kernels::{self, Accumulator};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::common::should_fuse_multi_dim_reduction;

/// Reduce implementation with explicit accumulation precision
pub fn reduce_impl_with_precision(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    precision: AccumulationPrecision,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    for &d in dims {
        if d >= ndim {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim,
            });
        }
    }

    if dims.len() == 1 && dims[0] == ndim - 1 && a.is_contiguous() {
        let reduce_size = shape[ndim - 1];
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = outer_size.max(1);

        let out_shape = reduce_output_shape(shape, dims, keepdim);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

        let a_ptr = a.ptr();
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::reduce_kernel_with_precision::<T>(
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                    precision,
                );
            }
        }, op_name);

        Ok(out)
    } else if dims.is_empty() {
        Ok(a.clone())
    } else if should_fuse_multi_dim_reduction(a, dims) {
        reduce_multi_dim_fused(client, op, a, dims, keepdim, precision, op_name)
    } else {
        let a_contig = ensure_contiguous(a)?;

        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse();

        let mut current = a_contig;
        for &dim in &sorted_dims {
            current = reduce_single_dim_with_precision(
                client, op, &current, dim, keepdim, precision, op_name,
            )?;
        }

        Ok(current)
    }
}

/// Reduce a single dimension with explicit accumulation precision.
fn reduce_single_dim_with_precision(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
    precision: AccumulationPrecision,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim + 1..].iter().product();
    let inner_size = inner_size.max(1);

    let out_shape = reduce_output_shape(shape, &[dim], keepdim);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let a_ptr = a.ptr();
    let out_ptr = out.ptr();

    if dim == ndim - 1 {
        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::reduce_kernel_with_precision::<T>(
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                    precision,
                );
            }
        }, op_name);
    } else {
        dispatch_dtype!(dtype, T => {
            unsafe {
                reduce_non_last_dim_with_precision::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    outer_size,
                    reduce_size,
                    inner_size,
                    precision,
                );
            }
        }, op_name);
    }

    Ok(out)
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim_with_precision<T: Element>(
    client: &CpuClient,
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    precision: AccumulationPrecision,
) {
    match precision {
        AccumulationPrecision::Native => {
            // Delegate to the native-precision non-last-dim reduction.
            // We inline the serial path here since we already dispatched precision.
            for outer in 0..outer_size {
                reduce_non_last_dim_outer(op, a, out, outer, reduce_size, inner_size);
            }
        }
        AccumulationPrecision::FP32 | AccumulationPrecision::BF16 => {
            reduce_non_last_dim_acc_runtime::<T, f32>(
                client,
                op,
                a,
                out,
                outer_size,
                reduce_size,
                inner_size,
            );
        }
        AccumulationPrecision::FP64 => {
            reduce_non_last_dim_acc_runtime::<T, f64>(
                client,
                op,
                a,
                out,
                outer_size,
                reduce_size,
                inner_size,
            );
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim_acc<T: Element, A: Accumulator>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        reduce_non_last_dim_acc_outer::<T, A>(op, a, out, outer, reduce_size, inner_size);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub(super) unsafe fn reduce_non_last_dim_acc_outer<T: Element, A: Accumulator>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for inner in 0..inner_size {
        let first_idx = outer * reduce_size * inner_size + inner;
        let first_val = A::acc_in((*a.add(first_idx)).to_f64());

        let mut acc: A = match op {
            ReduceOp::Sum | ReduceOp::Mean => A::ZERO,
            ReduceOp::Prod => A::ONE,
            ReduceOp::Max | ReduceOp::Min => first_val,
            ReduceOp::All => A::ONE,
            ReduceOp::Any => A::ZERO,
        };

        for r in 0..reduce_size {
            let idx = outer * reduce_size * inner_size + r * inner_size + inner;
            let val = A::acc_in((*a.add(idx)).to_f64());

            acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => acc.acc_add(val),
                ReduceOp::Prod => acc.acc_mul(val),
                ReduceOp::Max => {
                    if val > acc {
                        val
                    } else {
                        acc
                    }
                }
                ReduceOp::Min => {
                    if val < acc {
                        val
                    } else {
                        acc
                    }
                }
                ReduceOp::All => {
                    if val != A::ZERO && acc != A::ZERO {
                        A::ONE
                    } else {
                        A::ZERO
                    }
                }
                ReduceOp::Any => {
                    if val != A::ZERO || acc != A::ZERO {
                        A::ONE
                    } else {
                        A::ZERO
                    }
                }
            };
        }

        if matches!(op, ReduceOp::Mean) {
            acc = acc.acc_div(reduce_size);
        }

        let out_idx = outer * inner_size + inner;
        *out.add(out_idx) = T::from_f64(acc.into());
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn reduce_non_last_dim_acc_runtime<T: Element, A: Accumulator>(
    client: &CpuClient,
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    #[cfg(feature = "rayon")]
    {
        if outer_size > 1 {
            return reduce_non_last_dim_acc_parallel::<T, A>(
                client,
                op,
                a,
                out,
                outer_size,
                reduce_size,
                inner_size,
            );
        }
    }

    #[cfg(not(feature = "rayon"))]
    let _ = client;

    reduce_non_last_dim_acc::<T, A>(op, a, out, outer_size, reduce_size, inner_size)
}

#[cfg(feature = "rayon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim_acc_parallel<T: Element, A: Accumulator>(
    client: &CpuClient,
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    let min_len = client.rayon_min_len();
    let a_addr = a as usize;
    let out_addr = out as usize;
    client.install_parallelism(|| {
        (0..outer_size)
            .into_par_iter()
            .with_min_len(min_len)
            .for_each(|outer| unsafe {
                let a_ptr = a_addr as *const T;
                let out_ptr = out_addr as *mut T;
                reduce_non_last_dim_acc_outer::<T, A>(
                    op,
                    a_ptr,
                    out_ptr,
                    outer,
                    reduce_size,
                    inner_size,
                );
            });
    });
}
