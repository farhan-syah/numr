//! Binary operations kernels
//!
//! Provides element-wise binary operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.
//! On aarch64, f32 and f64 operations use NEON when available.

use super::binary_int::{binary_int_elem, binary_int_kernel};
use super::ipow::pow_elem;
use crate::dtype::Element;
use crate::ops::BinaryOp;

/// Execute a binary operation element-wise with automatic SIMD dispatch
///
/// On x86-64 and aarch64, dispatches to optimized SIMD implementations for f32/f64:
/// - x86-64: AVX-512 (16 f32s or 8 f64s) or AVX2 (8 f32s or 4 f64s) per iteration
/// - aarch64: NEON (4 f32s or 2 f64s) per iteration
///
/// Scalar fallback for other types or non-SIMD platforms
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
/// - `out` must not overlap with `a` or `b` unless they are the same pointer
#[inline]
pub unsafe fn binary_op_kernel<T: Element>(
    op: BinaryOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    len: usize,
) {
    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::binary;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                binary::binary_f32(op, a as *const f32, b as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                binary::binary_f64(op, a as *const f64, b as *const f64, out as *mut f64, len);
                return;
            }
            DType::I32 => {
                binary::binary_i32(op, a as *const i32, b as *const i32, out as *mut i32, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                binary::binary_f16(
                    op,
                    a as *const half::f16,
                    b as *const half::f16,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                binary::binary_bf16(
                    op,
                    a as *const half::bf16,
                    b as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback for non-SIMD types or non-x86 platforms
    binary_op_scalar(op, a, b, out, len);
}

/// Scalar binary operation for all Element types
#[inline]
unsafe fn binary_op_scalar<T: Element>(
    op: BinaryOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    len: usize,
) {
    // Integer dtypes wrap on overflow and define division by zero; the plain
    // operators below panic on both. See `binary_int`.
    if binary_int_kernel(op, a, b, out, len) {
        return;
    }

    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    match op {
        BinaryOp::Add => {
            for i in 0..len {
                out_slice[i] = a_slice[i] + b_slice[i];
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                out_slice[i] = a_slice[i] - b_slice[i];
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                out_slice[i] = a_slice[i] * b_slice[i];
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                out_slice[i] = a_slice[i] / b_slice[i];
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                out_slice[i] = pow_elem(a_slice[i], b_slice[i]);
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] > b_slice[i] {
                    a_slice[i]
                } else {
                    b_slice[i]
                };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] < b_slice[i] {
                    a_slice[i]
                } else {
                    b_slice[i]
                };
            }
        }
        BinaryOp::Atan2 => {
            // atan2(y, x) requires conversion to f64 and back
            for i in 0..len {
                let y = a_slice[i].to_f64();
                let x = b_slice[i].to_f64();
                out_slice[i] = T::from_f64(y.atan2(x));
            }
        }
    }
}

// ============================================================================
// f32/f64 specific scalar functions (used by SIMD modules for tail handling)
// ============================================================================

/// Scalar binary operation for f32 (used by SIMD for small arrays and tail)
#[inline]
pub unsafe fn binary_scalar_f32(
    op: BinaryOp,
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    len: usize,
) {
    match op {
        BinaryOp::Add => {
            for i in 0..len {
                *out.add(i) = *a.add(i) + *b.add(i);
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                *out.add(i) = *a.add(i) - *b.add(i);
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                *out.add(i) = *a.add(i) * *b.add(i);
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                *out.add(i) = *a.add(i) / *b.add(i);
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                let av = *a.add(i);
                let bv = *b.add(i);
                *out.add(i) = if av > bv { av } else { bv };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                let av = *a.add(i);
                let bv = *b.add(i);
                *out.add(i) = if av < bv { av } else { bv };
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).powf(*b.add(i));
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).atan2(*b.add(i));
            }
        }
    }
}

/// Scalar binary operation for f64 (used by SIMD for small arrays and tail)
#[inline]
pub unsafe fn binary_scalar_f64(
    op: BinaryOp,
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    len: usize,
) {
    match op {
        BinaryOp::Add => {
            for i in 0..len {
                *out.add(i) = *a.add(i) + *b.add(i);
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                *out.add(i) = *a.add(i) - *b.add(i);
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                *out.add(i) = *a.add(i) * *b.add(i);
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                *out.add(i) = *a.add(i) / *b.add(i);
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                let av = *a.add(i);
                let bv = *b.add(i);
                *out.add(i) = if av > bv { av } else { bv };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                let av = *a.add(i);
                let bv = *b.add(i);
                *out.add(i) = if av < bv { av } else { bv };
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).powf(*b.add(i));
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).atan2(*b.add(i));
            }
        }
    }
}

/// Scalar binary operation for i32 (used by SIMD for small arrays and tail)
#[inline]
pub unsafe fn binary_scalar_i32(
    op: BinaryOp,
    a: *const i32,
    b: *const i32,
    out: *mut i32,
    len: usize,
) {
    match op {
        BinaryOp::Add => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).wrapping_add(*b.add(i));
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).wrapping_sub(*b.add(i));
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).wrapping_mul(*b.add(i));
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                let bv = *b.add(i);
                *out.add(i) = if bv != 0 {
                    (*a.add(i)).wrapping_div(bv)
                } else {
                    0
                };
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                let av = *a.add(i);
                let bv = *b.add(i);
                *out.add(i) = if av > bv { av } else { bv };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                let av = *a.add(i);
                let bv = *b.add(i);
                *out.add(i) = if av < bv { av } else { bv };
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                *out.add(i) = pow_elem(*a.add(i), *b.add(i));
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                let y = *a.add(i) as f64;
                let x = *b.add(i) as f64;
                *out.add(i) = y.atan2(x) as i32;
            }
        }
    }
}

/// Execute a binary operation with broadcasting support
///
/// Uses strides to handle arbitrary broadcasting patterns. Stride of 0 means
/// the dimension is broadcast (all indices access the same element).
///
/// # Safety
/// - All pointers must be valid for the specified shapes and strides
/// - `out` must not overlap with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn binary_op_strided_kernel<T: Element>(
    op: BinaryOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    out_shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    a_offset: usize,
    b_offset: usize,
) {
    let ndim = out_shape.len();
    let total = out_shape.iter().product::<usize>();

    if total == 0 {
        return;
    }

    // Optimize for common case: both inputs are contiguous and same shape
    let is_simple = ndim > 0 && {
        let mut expected_stride = 1isize;
        let mut simple = true;
        for i in (0..ndim).rev() {
            if a_strides[i] != expected_stride || b_strides[i] != expected_stride {
                simple = false;
                break;
            }
            expected_stride *= out_shape[i] as isize;
        }
        simple && a_offset == 0 && b_offset == 0
    };

    if is_simple {
        binary_op_kernel(op, a, b, out, total);
        return;
    }

    // General strided iteration
    let mut indices = vec![0usize; ndim];
    let mut a_idx = a_offset as isize;
    let mut b_idx = b_offset as isize;

    for out_idx in 0..total {
        let a_val = *a.offset(a_idx);
        let b_val = *b.offset(b_idx);

        // Integer dtypes take the wrapping path; see `binary_int`.
        let result = match binary_int_elem(op, a_val, b_val) {
            Some(v) => v,
            None => match op {
                BinaryOp::Add => a_val + b_val,
                BinaryOp::Sub => a_val - b_val,
                BinaryOp::Mul => a_val * b_val,
                BinaryOp::Div => a_val / b_val,
                BinaryOp::Pow => pow_elem(a_val, b_val),
                BinaryOp::Max => {
                    if a_val > b_val {
                        a_val
                    } else {
                        b_val
                    }
                }
                BinaryOp::Min => {
                    if a_val < b_val {
                        a_val
                    } else {
                        b_val
                    }
                }
                BinaryOp::Atan2 => T::from_f64(a_val.to_f64().atan2(b_val.to_f64())),
            },
        };

        *out.add(out_idx) = result;

        // Increment multi-dimensional index
        for dim in (0..ndim).rev() {
            indices[dim] += 1;
            a_idx += a_strides[dim];
            b_idx += b_strides[dim];

            if indices[dim] < out_shape[dim] {
                break;
            }

            indices[dim] = 0;
            a_idx -= (out_shape[dim] as isize) * a_strides[dim];
            b_idx -= (out_shape[dim] as isize) * b_strides[dim];
        }
    }
}
