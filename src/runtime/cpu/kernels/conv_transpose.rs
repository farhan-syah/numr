//! CPU transposed 1D convolution kernel.
//!
//! # Gather formulation
//!
//! Transposed convolution is naturally written as a *scatter*
//! (`out[l*stride - pad_left + k*dilation] += x[l] * w[k]`), but scatter needs a
//! zero-init pass and read-modify-writes the same output element from many
//! input positions. Every backend in numr therefore uses the *gather* form
//! instead, so each output element is computed once and written once:
//!
//! ```text
//! out[ot] = sum over taps k, input channels ic of x[l] * w[k]
//!   where  num = ot + pad_left - k * dilation
//!          num >= 0                (checked BEFORE the modulo: Rust's `%` on a
//!                                   negative isize truncates, it does not floor)
//!          num % stride == 0
//!          l = num / stride  <  length
//! ```
//!
//! This is exactly the formulation used by `conv.cu`'s `conv_transpose1d_*`
//! kernel and `conv_transpose1d_f32.wgsl`, including the accumulation order
//! (taps outer, input channels inner, bias added last), so all three backends
//! agree numerically.

use crate::dtype::Element;
use crate::ops::conv_transpose_common::ConvTranspose1dParams;

/// Transposed 1D convolution, gather form.
///
/// Layouts: input `(batch, c_in, length)`, weight `(c_in, c_out/groups, kernel)`
/// — input channels lead, unlike conv1d — output `(batch, c_out, output_length)`.
/// Bias, when present, is added once per output channel.
///
/// This is the dtype-generic path used for F16/BF16/FP8 (which have no SIMD
/// kernel) and as the scalar fallback for F32/F64.
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid, aligned, and non-overlapping
/// - Arrays have sufficient size for the shapes in `params`
/// - `params` came from `validate_conv_transpose1d` (so `stride >= 1`,
///   `groups >= 1`, and the channel counts divide evenly)
pub unsafe fn conv_transpose1d_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    bias: Option<*const T>,
    output: *mut T,
    params: ConvTranspose1dParams,
) {
    let ConvTranspose1dParams {
        batch,
        c_in,
        length,
        c_out,
        kernel_size,
        stride,
        dilation,
        groups,
        pad_left,
        output_length,
        ..
    } = params;

    let c_in_per_group = c_in / groups;
    let c_out_per_group = c_out / groups;
    // `c_out == 0` implies `c_out_per_group == 0`; the `oc` loop below is then
    // empty, so the division by `c_out_per_group` is never reached.
    if c_out == 0 || output_length == 0 {
        return;
    }

    let pad_left_i = pad_left as isize;

    for b in 0..batch {
        for oc in 0..c_out {
            let g = oc / c_out_per_group;
            let oc_local = oc % c_out_per_group;
            let c_in_start = g * c_in_per_group;
            let out_row = (b * c_out + oc) * output_length;

            for ot in 0..output_length {
                let mut sum = T::zero();

                for k in 0..kernel_size {
                    let num = ot as isize + pad_left_i - (k * dilation) as isize;
                    // Must precede the modulo: `-1 % 2 == -1` in Rust, not `1`.
                    if num < 0 {
                        continue;
                    }
                    let num = num as usize;
                    if !num.is_multiple_of(stride) {
                        continue;
                    }
                    let l = num / stride;
                    if l >= length {
                        continue;
                    }

                    for ic in 0..c_in_per_group {
                        let c_in_abs = c_in_start + ic;
                        let x = *input.add((b * c_in + c_in_abs) * length + l);
                        let w =
                            *weight.add((c_in_abs * c_out_per_group + oc_local) * kernel_size + k);
                        sum = sum + x * w;
                    }
                }

                if let Some(bias_ptr) = bias {
                    sum = sum + *bias_ptr.add(oc);
                }

                *output.add(out_row + ot) = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::ops::PaddingMode;
    use crate::ops::conv_transpose_common::validate_conv_transpose1d;

    fn params(
        input_shape: &[usize],
        weight_shape: &[usize],
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
    ) -> ConvTranspose1dParams {
        validate_conv_transpose1d(
            input_shape,
            weight_shape,
            None,
            stride,
            padding,
            0,
            dilation,
            groups,
            DType::F32,
            DType::F32,
            None,
        )
        .expect("test shapes are valid")
    }

    #[test]
    fn gather_matches_known_stride1_reference() {
        // input [1,2,3] * kernel [1,1,1] -> full convolution [1, 3, 6, 5, 3]
        let input = [1.0f32, 2.0, 3.0];
        let weight = [1.0f32, 1.0, 1.0];
        let p = params(&[1, 1, 3], &[1, 1, 3], 1, PaddingMode::Valid, 1, 1);
        let mut out = vec![0.0f32; p.output_length];
        unsafe {
            conv_transpose1d_kernel(input.as_ptr(), weight.as_ptr(), None, out.as_mut_ptr(), p);
        }
        assert_eq!(out, vec![1.0, 3.0, 6.0, 5.0, 3.0]);
    }

    #[test]
    fn gather_matches_known_stride2_reference() {
        let input = [1.0f32, 2.0];
        let weight = [1.0f32, 1.0, 1.0];
        let p = params(&[1, 1, 2], &[1, 1, 3], 2, PaddingMode::Valid, 1, 1);
        let mut out = vec![0.0f32; p.output_length];
        unsafe {
            conv_transpose1d_kernel(input.as_ptr(), weight.as_ptr(), None, out.as_mut_ptr(), p);
        }
        assert_eq!(out, vec![1.0, 1.0, 3.0, 2.0, 2.0]);
    }

    /// `pad_left` makes `ot + pad_left - k*dilation` negative for the leading
    /// taps; a `%` before the sign check would wrongly admit them.
    #[test]
    fn negative_numerator_is_rejected_before_modulo() {
        let input = [1.0f32, 2.0, 3.0];
        let weight = [1.0f32, 10.0, 100.0];
        let p = params(
            &[1, 1, 3],
            &[1, 1, 3],
            2,
            PaddingMode::Custom(1, 1, 0, 0),
            1,
            1,
        );
        let mut out = vec![0.0f32; p.output_length];
        unsafe {
            conv_transpose1d_kernel(input.as_ptr(), weight.as_ptr(), None, out.as_mut_ptr(), p);
        }
        // Raw (unpadded) scatter output for stride 2 is
        // [1, 10, 100+2, 20, 200+3, 30, 300]; cropping one frame off each side
        // leaves [10, 102, 20, 203, 30].
        assert_eq!(out, vec![10.0, 102.0, 20.0, 203.0, 30.0]);
    }
}
