//! Mask-driven kernels: count, compact, and fill by a `u8` mask.
//!
//! Each dispatches to the SIMD mask routines on x86-64 and aarch64, with a
//! scalar fallback elsewhere.

use crate::dtype::Element;

/// Count elements where mask is true.
///
/// Returns the count of non-zero elements in the mask.
///
/// # Safety
/// - `mask` must be valid pointer to `numel` u8 elements
#[inline]
// Only the `not(target_arch = "x86_64")` arm of `masked_select` calls this; the
// x86_64 arm gets its count from the SIMD kernel. So it IS dead code on x86_64,
// and a missing re-export here went unnoticed until an aarch64 build failed.
#[allow(dead_code)]
pub unsafe fn masked_count_kernel(mask: *const u8, numel: usize) -> usize {
    // Use SIMD on x86_64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::index;
        return index::masked_count(mask, numel);
    }

    // Scalar fallback for other architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mask_slice = std::slice::from_raw_parts(mask, numel);
        mask_slice.iter().filter(|&&m| m != 0).count()
    }
}

/// Select elements where mask is true, returning a flattened result.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `mask` - Mask tensor pointer (u8: 0=false, non-zero=true)
/// * `out` - Output pointer (must be sized for count of true elements)
/// * `numel` - Number of elements in input/mask
///
/// # Safety
/// - All pointers must be valid for the specified size
/// - `out` must have enough space for all selected elements
#[inline]
pub unsafe fn masked_select_kernel<T: Element>(
    a: *const T,
    mask: *const u8,
    out: *mut T,
    numel: usize,
) {
    // Use SIMD for f32/f64 types on x86_64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::index;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let _ = index::masked_select_f32(a as *const f32, mask, out as *mut f32, numel);
            return;
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let _ = index::masked_select_f64(a as *const f64, mask, out as *mut f64, numel);
            return;
        }
    }

    // Scalar fallback for other types
    let a_slice = std::slice::from_raw_parts(a, numel);
    let mask_slice = std::slice::from_raw_parts(mask, numel);

    let mut out_idx = 0;
    for i in 0..numel {
        if mask_slice[i] != 0 {
            *out.add(out_idx) = a_slice[i];
            out_idx += 1;
        }
    }
}

/// Fill elements where mask is true with a scalar value.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `mask` - Mask tensor pointer (u8: 0=false, non-zero=true)
/// * `out` - Output pointer
/// * `numel` - Number of elements
/// * `value` - Value to fill where mask is true
///
/// # Safety
/// - All pointers must be valid for the specified size
#[inline]
pub unsafe fn masked_fill_kernel<T: Element>(
    a: *const T,
    mask: *const u8,
    out: *mut T,
    numel: usize,
    value: f64,
) {
    // Use SIMD for f32/f64 types on x86_64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::index;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            index::masked_fill_f32(a as *const f32, mask, out as *mut f32, numel, value as f32);
            return;
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            index::masked_fill_f64(a as *const f64, mask, out as *mut f64, numel, value);
            return;
        }
    }

    // Scalar fallback for other types
    let a_slice = std::slice::from_raw_parts(a, numel);
    let mask_slice = std::slice::from_raw_parts(mask, numel);
    let out_slice = std::slice::from_raw_parts_mut(out, numel);

    let fill_val = T::from_f64(value);

    for i in 0..numel {
        out_slice[i] = if mask_slice[i] != 0 {
            fill_val
        } else {
            a_slice[i]
        };
    }
}
