//! Fused elementwise kernel entry points
//!
//! - fused_mul_add: out = a * b + c
//! - fused_add_mul: out = (a + b) * c
//! - fused_mul_add_scalar: out = a * scale + bias

use super::binary_int::binary_int_fused_elem;
use crate::dtype::Element;
use crate::ops::BinaryOp;

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`
///
/// # Safety
/// - `a`, `b`, `c`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn fused_mul_add_kernel<T: Element>(
    a: *const T,
    b: *const T,
    c: *const T,
    out: *mut T,
    len: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::fused_elementwise;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                fused_elementwise::fused_mul_add_f32(
                    a as *const f32,
                    b as *const f32,
                    c as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                fused_elementwise::fused_mul_add_f64(
                    a as *const f64,
                    b as *const f64,
                    c as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                fused_elementwise::fused_mul_add_f16(
                    a as *const half::f16,
                    b as *const half::f16,
                    c as *const half::f16,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                fused_elementwise::fused_mul_add_bf16(
                    a as *const half::bf16,
                    b as *const half::bf16,
                    c as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {}
        }
    }

    fused_ternary_scalar(
        a,
        b,
        c,
        out,
        len,
        BinaryOp::Mul,
        BinaryOp::Add,
        |x, y, z| x * y + z,
    );
}

/// Fused add-multiply: `out[i] = (a[i] + b[i]) * c[i]`
///
/// # Safety
/// - `a`, `b`, `c`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn fused_add_mul_kernel<T: Element>(
    a: *const T,
    b: *const T,
    c: *const T,
    out: *mut T,
    len: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::fused_elementwise;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                fused_elementwise::fused_add_mul_f32(
                    a as *const f32,
                    b as *const f32,
                    c as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                fused_elementwise::fused_add_mul_f64(
                    a as *const f64,
                    b as *const f64,
                    c as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                fused_elementwise::fused_add_mul_f16(
                    a as *const half::f16,
                    b as *const half::f16,
                    c as *const half::f16,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                fused_elementwise::fused_add_mul_bf16(
                    a as *const half::bf16,
                    b as *const half::bf16,
                    c as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {}
        }
    }

    fused_ternary_scalar(
        a,
        b,
        c,
        out,
        len,
        BinaryOp::Add,
        BinaryOp::Mul,
        |x, y, z| (x + y) * z,
    );
}

/// Fused multiply-add scalar: `out[i] = a[i] * scale + bias`
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn fused_mul_add_scalar_kernel<T: Element>(
    a: *const T,
    scale: f64,
    bias: f64,
    out: *mut T,
    len: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::fused_elementwise;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                fused_elementwise::fused_mul_add_scalar_f32_kernel(
                    a as *const f32,
                    scale as f32,
                    bias as f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                fused_elementwise::fused_mul_add_scalar_f64_kernel(
                    a as *const f64,
                    scale,
                    bias,
                    out as *mut f64,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                fused_elementwise::fused_mul_add_scalar_f32_f16(
                    a as *const half::f16,
                    scale as f32,
                    bias as f32,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                fused_elementwise::fused_mul_add_scalar_f32_bf16(
                    a as *const half::bf16,
                    scale as f32,
                    bias as f32,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {}
        }
    }

    // Scalar fallback
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    // An integer row must equal `add_scalar(mul_scalar(a, scale), bias)`, which
    // wraps at each step. Computing in f64 and converting once saturates
    // instead, so the fused and unfused paths would disagree exactly where the
    // product leaves the dtype.
    if T::DTYPE.is_int() {
        let scale_t = T::from_f64(scale);
        let bias_t = T::from_f64(bias);
        for i in 0..len {
            // `None` only for a non-integer dtype, which the guard excluded.
            if let Some(v) =
                binary_int_fused_elem(BinaryOp::Mul, BinaryOp::Add, a_slice[i], scale_t, bias_t)
            {
                out_slice[i] = v;
            }
        }
        return;
    }

    for i in 0..len {
        let val = a_slice[i].to_f64();
        out_slice[i] = T::from_f64(val * scale + bias);
    }
}

/// Generic scalar fallback for ternary fused ops.
///
/// `op1` and `op2` name the same computation as `op` for the integer dtypes,
/// which take the wrapping element path instead of `op`'s f64 one: a fused
/// integer op must answer exactly what the unfused sequence answers, and that
/// sequence wraps at each step rather than saturating once at the end.
#[inline]
unsafe fn fused_ternary_scalar<T: Element, F: Fn(f64, f64, f64) -> f64>(
    a: *const T,
    b: *const T,
    c: *const T,
    out: *mut T,
    len: usize,
    op1: BinaryOp,
    op2: BinaryOp,
    op: F,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let c_slice = std::slice::from_raw_parts(c, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    if T::DTYPE.is_int() {
        for i in 0..len {
            // `None` only for a non-integer dtype, which the guard excluded.
            if let Some(v) = binary_int_fused_elem(op1, op2, a_slice[i], b_slice[i], c_slice[i]) {
                out_slice[i] = v;
            }
        }
        return;
    }

    for i in 0..len {
        let x = a_slice[i].to_f64();
        let y = b_slice[i].to_f64();
        let z = c_slice[i].to_f64();
        out_slice[i] = T::from_f64(op(x, y, z));
    }
}
