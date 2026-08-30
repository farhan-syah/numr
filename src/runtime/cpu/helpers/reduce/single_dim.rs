//! Single-dimension reduction with native precision

use crate::dispatch_dtype;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{Kernel, ReduceOp, max_identity, min_identity, reduce_output_shape};
use crate::runtime::cpu::kernels::wide_acc::{WideAcc, int_mean_from_sum};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Reduce a single dimension of a tensor using native precision.
///
/// Uses chunked iteration for non-last dimensions to handle strided memory access.
pub(super) fn reduce_single_dim(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
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
    // No `.max(1)` here: `product()` of an empty slice is already 1, so the
    // clamp only ever fires when a dimension is genuinely 0 — and then it
    // fabricates a row the allocation does not have, so the kernel reads a
    // null pointer. An empty extent must stay 0.
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let out_shape = reduce_output_shape(shape, &[dim], keepdim);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

    if dim == ndim - 1 {
        let a_ptr = a.ptr();
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::reduce::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, op_name);
    } else {
        let a_ptr = a.ptr();
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                reduce_non_last_dim_runtime::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    outer_size,
                    reduce_size,
                    inner_size,
                );
            }
        }, op_name);
    }

    Ok(out)
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        reduce_non_last_dim_outer(op, a, out, outer, reduce_size, inner_size);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub(super) unsafe fn reduce_non_last_dim_outer<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    // A zero-length reduce dimension leaves `Max` and `Min` with no element to
    // seed from, and the loop below seeds them by reading index `inner` of an
    // empty allocation — a silent out-of-bounds read, since `CpuRuntime::allocate`
    // hands back a dangling, non-null address for zero bytes. Both are given the identity of their
    // own reduction; every other op already starts from its identity and
    // iterates zero times.
    if reduce_size == 0 && matches!(op, ReduceOp::Max | ReduceOp::Min) {
        // Floats fold to -/+inf, integers to the dtype's own extreme.
        let identity = if matches!(op, ReduceOp::Max) {
            T::from_f64(max_identity(T::DTYPE))
        } else {
            T::from_f64(min_identity(T::DTYPE))
        };
        for inner in 0..inner_size {
            *out.add(outer * inner_size + inner) = identity;
        }
        return;
    }

    // A float narrower than F32 must not accumulate in its own dtype: the
    // running sum saturates and stalls on a constant. Widen to f32 and narrow
    // only the final result. F32, F64, and integers keep the loop below
    // unchanged, so their results are bit-for-bit what they were.
    if T::DTYPE.is_narrow_float() {
        super::precision::reduce_non_last_dim_acc_outer::<T, f32>(
            op,
            a,
            out,
            outer,
            reduce_size,
            inner_size,
        );
        return;
    }

    // Integer `sum`, `prod` and `mean` build a running total that can leave the
    // dtype's range: the loop below would wrap (release) or panic (debug), and
    // for `mean` the divided result can still be representable. Accumulate in
    // i128 and saturate once at write-out. Every other integer reduction keeps
    // the loop below.
    if T::DTYPE.is_int() && matches!(op, ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Mean) {
        reduce_int_non_last_dim_outer::<T>(op, a, out, outer, reduce_size, inner_size);
        return;
    }

    for inner in 0..inner_size {
        let mut acc = match op {
            ReduceOp::Sum | ReduceOp::Mean => T::zero(),
            ReduceOp::Prod => T::one(),
            ReduceOp::Max => {
                let idx = outer * reduce_size * inner_size + inner;
                *a.add(idx)
            }
            ReduceOp::Min => {
                let idx = outer * reduce_size * inner_size + inner;
                *a.add(idx)
            }
            ReduceOp::All => T::one(),
            ReduceOp::Any => T::zero(),
        };

        for r in 0..reduce_size {
            let idx = outer * reduce_size * inner_size + r * inner_size + inner;
            let val = *a.add(idx);

            acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => acc + val,
                ReduceOp::Prod => acc * val,
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
                    if val.to_f64() != 0.0 && acc.to_f64() != 0.0 {
                        T::one()
                    } else {
                        T::zero()
                    }
                }
                ReduceOp::Any => {
                    if val.to_f64() != 0.0 || acc.to_f64() != 0.0 {
                        T::one()
                    } else {
                        T::zero()
                    }
                }
            };
        }

        if matches!(op, ReduceOp::Mean) {
            acc = T::from_f64(acc.to_f64() / reduce_size as f64);
        }

        let out_idx = outer * inner_size + inner;
        *out.add(out_idx) = acc;
    }
}

/// Integer `sum`, `prod` and `mean` over one non-last dimension, accumulating
/// in i128. `op` must be one of those three.
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn reduce_int_non_last_dim_outer<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    let is_prod = matches!(op, ReduceOp::Prod);
    let is_mean = matches!(op, ReduceOp::Mean);
    for inner in 0..inner_size {
        let mut acc = if is_prod { i128::ONE } else { i128::ZERO };
        for r in 0..reduce_size {
            let idx = outer * reduce_size * inner_size + r * inner_size + inner;
            let val = i128::from_elem(*a.add(idx));
            acc = if is_prod {
                acc.wide_mul(val)
            } else {
                acc.wide_add(val)
            };
        }
        *out.add(outer * inner_size + inner) = if is_mean {
            int_mean_from_sum::<T>(acc, reduce_size)
        } else {
            acc.to_elem::<T>()
        };
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim_runtime<T: Element>(
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
            return reduce_non_last_dim_parallel(
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

    reduce_non_last_dim(op, a, out, outer_size, reduce_size, inner_size);
}

#[cfg(feature = "rayon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim_parallel<T: Element>(
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
                reduce_non_last_dim_outer(op, a_ptr, out_ptr, outer, reduce_size, inner_size);
            });
    });
}
