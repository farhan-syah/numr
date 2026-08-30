//! Histogram kernel for integer tensors, and the range scan that sizes it.

use crate::dtype::Element;

/// Count occurrences of each value in an integer tensor.
///
/// # Arguments
/// * `input` - Input integer tensor pointer (i64 values)
/// * `weights` - Optional weights pointer (same length as input)
/// * `out` - Output pointer (histogram)
/// * `numel` - Number of elements in input
/// * `output_len` - Length of output histogram
/// * `reject_negative` - When true, stop and report a negative input value.
///   When false, ignore every value outside `[0, output_len)`.
///
/// # Safety
/// - All pointers must be valid for the specified sizes
///
/// # Returns
/// * `true` on success, `false` if `reject_negative` is set and a negative value was found
#[inline]
pub unsafe fn bincount_kernel<T: Element>(
    input: *const i64,
    weights: *const T,
    out: *mut T,
    numel: usize,
    output_len: usize,
    reject_negative: bool,
) -> bool {
    // Initialize output to zero
    let out_slice = std::slice::from_raw_parts_mut(out, output_len);
    for elem in out_slice.iter_mut() {
        *elem = T::zero();
    }

    let input_slice = std::slice::from_raw_parts(input, numel);
    let has_weights = !weights.is_null();

    for i in 0..numel {
        let val = input_slice[i];
        if val < 0 {
            if reject_negative {
                return false; // Negative value found
            }
            continue; // Caller-sized path ignores out-of-range values
        }
        let idx = val as usize;
        if idx < output_len {
            if has_weights {
                let w = *weights.add(i);
                out_slice[idx] = out_slice[idx] + w;
            } else {
                out_slice[idx] = out_slice[idx] + T::one();
            }
        }
    }

    true
}

/// Find the maximum value in an i64 tensor.
///
/// # Safety
/// - `input` must be valid for `numel` elements
#[inline]
pub unsafe fn max_i64_kernel(input: *const i64, numel: usize) -> i64 {
    if numel == 0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts(input, numel);
    *slice.iter().max().unwrap_or(&-1)
}
