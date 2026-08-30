//! Special reduction operations
//!
//! Contains argmax, argmin, softmax, and variance kernels.

use crate::dtype::Element;

/// Argmax along a dimension - returns indices of maximum values
///
/// # Arguments
/// * `a` - Input pointer (outer_size * reduce_size * inner_size elements)
/// * `out` - Output pointer (outer_size * inner_size elements) for i64 indices
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` i64 elements
#[inline]
pub unsafe fn argmax_kernel<T: Element>(
    a: *const T,
    out: *mut i64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Base index for this (outer, inner) position
            let base_idx = outer * reduce_size * inner_size + inner;

            // Find index of maximum value
            let mut max_val = (*a.add(base_idx)).to_f64();
            let mut max_idx: i64 = 0;

            for r in 1..reduce_size {
                let idx = base_idx + r * inner_size;
                let val = (*a.add(idx)).to_f64();
                if val > max_val {
                    max_val = val;
                    max_idx = r as i64;
                }
            }

            *out.add(outer * inner_size + inner) = max_idx;
        }
    }
}

/// Argmin along a dimension - returns indices of minimum values
///
/// # Arguments
/// * `a` - Input pointer (outer_size * reduce_size * inner_size elements)
/// * `out` - Output pointer (outer_size * inner_size elements) for i64 indices
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` i64 elements
#[inline]
pub unsafe fn argmin_kernel<T: Element>(
    a: *const T,
    out: *mut i64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Base index for this (outer, inner) position
            let base_idx = outer * reduce_size * inner_size + inner;

            // Find index of minimum value
            let mut min_val = (*a.add(base_idx)).to_f64();
            let mut min_idx: i64 = 0;

            for r in 1..reduce_size {
                let idx = base_idx + r * inner_size;
                let val = (*a.add(idx)).to_f64();
                if val < min_val {
                    min_val = val;
                    min_idx = r as i64;
                }
            }

            *out.add(outer * inner_size + inner) = min_idx;
        }
    }
}

/// Softmax along the last dimension
///
/// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration with vectorized exp
/// - AVX2: 8 f32s or 4 f64s per iteration with vectorized exp
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `a` - Input pointer (outer_size * dim_size elements)
/// * `out` - Output pointer (outer_size * dim_size elements)
/// * `outer_size` - Number of independent softmax operations
/// * `dim_size` - Size of the softmax dimension
///
/// # Safety
/// - `a` and `out` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    dim_size: usize,
) {
    // A zero-length softmax dimension makes the output empty, so there is
    // nothing to write — but the SIMD paths below and the scalar fallback all
    // seed their max from `a[base]`, which on an empty allocation reads past the
    // end (`CpuRuntime::allocate` hands back a dangling, non-null address for
    // zero bytes). Answer before any of them run.
    if dim_size == 0 {
        return;
    }

    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use crate::dtype::DType;
        use crate::runtime::cpu::kernels::simd::softmax;

        match T::DTYPE {
            DType::F32 => {
                softmax::softmax_f32(a as *const f32, out as *mut f32, outer_size, dim_size);
                return;
            }
            DType::F64 => {
                softmax::softmax_f64(a as *const f64, out as *mut f64, outer_size, dim_size);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                softmax::softmax_f16(
                    a as *const half::f16,
                    out as *mut half::f16,
                    outer_size,
                    dim_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                softmax::softmax_bf16(
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    outer_size,
                    dim_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    softmax_kernel_scalar(a, out, outer_size, dim_size);
}

/// Scalar softmax for all Element types using online algorithm (2-pass).
#[inline]
unsafe fn softmax_kernel_scalar<T: Element>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: Online max + sum
        let mut max_val = (*a.add(base)).to_f64();
        let mut sum = 1.0f64;
        for d in 1..dim_size {
            let val = (*a.add(base + d)).to_f64();
            if val > max_val {
                sum = sum * (max_val - val).exp() + 1.0;
                max_val = val;
            } else {
                sum += (val - max_val).exp();
            }
        }

        // Pass 2: exp(x - max) / sum
        let inv_sum = 1.0 / sum;
        for d in 0..dim_size {
            let val = (*a.add(base + d)).to_f64();
            *out.add(base + d) = T::from_f64((val - max_val).exp() * inv_sum);
        }
    }
}

/// Softmax backward kernel: d_input = output * (grad - sum(grad * output))
///
/// Dispatches to SIMD for f32/f64, with f16/bf16 block-convert wrappers.
/// Falls back to scalar for other types.
///
/// # Safety
/// - `grad`, `output`, `d_input` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_bwd_kernel<T: Element>(
    grad: *const T,
    output: *const T,
    d_input: *mut T,
    outer_size: usize,
    dim_size: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use crate::dtype::DType;
        use crate::runtime::cpu::kernels::simd::softmax_bwd;

        match T::DTYPE {
            DType::F32 => {
                softmax_bwd::softmax_bwd_f32(
                    grad as *const f32,
                    output as *const f32,
                    d_input as *mut f32,
                    outer_size,
                    dim_size,
                );
                return;
            }
            DType::F64 => {
                softmax_bwd::softmax_bwd_f64(
                    grad as *const f64,
                    output as *const f64,
                    d_input as *mut f64,
                    outer_size,
                    dim_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                softmax_bwd::softmax_bwd_f16(
                    grad as *const half::f16,
                    output as *const half::f16,
                    d_input as *mut half::f16,
                    outer_size,
                    dim_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                softmax_bwd::softmax_bwd_bf16(
                    grad as *const half::bf16,
                    output as *const half::bf16,
                    d_input as *mut half::bf16,
                    outer_size,
                    dim_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::dtype::DType;
        use crate::runtime::cpu::kernels::simd::softmax_bwd;

        match T::DTYPE {
            DType::F32 => {
                softmax_bwd::softmax_bwd_f32(
                    grad as *const f32,
                    output as *const f32,
                    d_input as *mut f32,
                    outer_size,
                    dim_size,
                );
                return;
            }
            DType::F64 => {
                softmax_bwd::softmax_bwd_f64(
                    grad as *const f64,
                    output as *const f64,
                    d_input as *mut f64,
                    outer_size,
                    dim_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                softmax_bwd::softmax_bwd_f16(
                    grad as *const half::f16,
                    output as *const half::f16,
                    d_input as *mut half::f16,
                    outer_size,
                    dim_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                softmax_bwd::softmax_bwd_bf16(
                    grad as *const half::bf16,
                    output as *const half::bf16,
                    d_input as *mut half::bf16,
                    outer_size,
                    dim_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    softmax_bwd_kernel_scalar(grad, output, d_input, outer_size, dim_size);
}

/// Scalar softmax backward for all Element types
#[inline]
unsafe fn softmax_bwd_kernel_scalar<T: Element>(
    grad: *const T,
    output: *const T,
    d_input: *mut T,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: dot = sum(grad * output)
        let mut dot = 0.0f64;
        for d in 0..dim_size {
            dot += (*grad.add(base + d)).to_f64() * (*output.add(base + d)).to_f64();
        }

        // Pass 2: d_input = output * (grad - dot)
        for d in 0..dim_size {
            let idx = base + d;
            let g = (*grad.add(idx)).to_f64();
            let out = (*output.add(idx)).to_f64();
            *d_input.add(idx) = T::from_f64(out * (g - dot));
        }
    }
}

/// Compute variance along a dimension
///
/// variance = sum((x - mean)^2) / (N - correction)
///
/// # Arguments
/// * `a` - Input data pointer
/// * `out` - Output pointer (for variance values)
/// * `outer_size` - Product of dimensions before the reduce dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduce dimension
/// * `correction` - Degrees of freedom correction (0 for population, 1 for sample)
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` elements
#[inline]
pub unsafe fn variance_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    correction: usize,
) {
    let total_size = reduce_size;
    let divisor = (total_size.saturating_sub(correction)).max(1) as f64;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // First pass: compute mean
            let mut sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                sum += (*a.add(idx)).to_f64();
            }
            let mean = sum / (reduce_size as f64);

            // Second pass: compute variance
            let mut var_sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let diff = (*a.add(idx)).to_f64() - mean;
                var_sum += diff * diff;
            }

            let out_idx = outer * inner_size + inner;
            *out.add(out_idx) = T::from_f64(var_sum / divisor);
        }
    }
}
