//! Shared validation and shape maths for transposed 1D convolution.
//!
//! Transposed convolution is the gradient of a convolution with respect to its
//! input, run as a forward op. It is what upsampling vocoder/GAN decoders and
//! alias-free resamplers are built from.
//!
//! Two things differ from [`conv1d`](super::conv_common::validate_conv1d) and
//! are easy to get wrong:
//!
//! * **Weight layout is `[c_in, c_out / groups, kernel]`** — the INPUT channel
//!   count leads, the opposite of conv1d's `[c_out, c_in / groups, kernel]`.
//! * **Padding shrinks the output** rather than growing it, and
//!   `output_padding` adds to one side only, to resolve the ambiguity where
//!   several input lengths map to the same output length when `stride > 1`.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::PaddingMode;
use crate::ops::conv_common::{
    validate_1d_tensor, validate_3d_tensor, validate_float_dtype, validate_positive,
    validate_same_dtype,
};

/// Parameters for `conv_transpose1d` after validation.
#[derive(Debug, Clone, Copy)]
pub struct ConvTranspose1dParams {
    pub batch: usize,
    pub c_in: usize,
    pub length: usize,
    pub c_out: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub output_padding: usize,
    pub output_length: usize,
}

/// Effective span of the (dilated) kernel: `dilation * (kernel_size - 1) + 1`.
#[inline]
pub fn effective_kernel_span(kernel_size: usize, dilation: usize) -> usize {
    dilation * kernel_size.saturating_sub(1) + 1
}

/// Output length of a transposed convolution.
///
/// `(length - 1) * stride - (pad_left + pad_right) + dilation * (kernel_size - 1)
///  + output_padding + 1`, saturating at zero.
#[inline]
pub fn compute_transpose_output_size(
    length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
    output_padding: usize,
) -> usize {
    if length == 0 {
        return 0;
    }
    let grown =
        (length - 1) * stride + effective_kernel_span(kernel_size, dilation) + output_padding;
    grown.saturating_sub(pad_left + pad_right)
}

/// Resolves a [`PaddingMode`] for a transposed convolution.
///
/// `Same` is defined here as "output length equals `length * stride`", which is
/// the upsampling convention these layers exist for. The total padding that
/// achieves it is `span + output_padding - stride`, split evenly with the extra
/// sample (if any) going right.
fn resolve_transpose_padding_1d(
    padding: PaddingMode,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
) -> (usize, usize) {
    match padding {
        PaddingMode::Valid => (0, 0),
        PaddingMode::Same => {
            let span = effective_kernel_span(kernel_size, dilation);
            let total = (span + output_padding).saturating_sub(stride);
            let left = total / 2;
            (left, total - left)
        }
        PaddingMode::Custom(left, right, _, _) => (left, right),
    }
}

/// Validates and extracts parameters for `conv_transpose1d`.
#[allow(clippy::too_many_arguments)]
pub fn validate_conv_transpose1d(
    input_shape: &[usize],
    weight_shape: &[usize],
    bias_shape: Option<&[usize]>,
    stride: usize,
    padding: PaddingMode,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    input_dtype: DType,
    weight_dtype: DType,
    bias_dtype: Option<DType>,
) -> Result<ConvTranspose1dParams> {
    const OP: &str = "conv_transpose1d";

    validate_3d_tensor(input_shape, "input", OP)?;
    validate_3d_tensor(weight_shape, "weight", OP)?;

    validate_float_dtype(input_dtype, OP)?;
    validate_same_dtype(input_dtype, weight_dtype, OP)?;
    if let Some(b_dtype) = bias_dtype {
        validate_same_dtype(input_dtype, b_dtype, OP)?;
    }

    validate_positive(stride, "stride", OP)?;
    validate_positive(dilation, "dilation", OP)?;
    validate_positive(groups, "groups", OP)?;

    let batch = input_shape[0];
    let c_in = input_shape[1];
    let length = input_shape[2];
    let kernel_size = weight_shape[2];

    // Weight is [c_in, c_out / groups, kernel] — the transpose of conv1d's layout.
    if weight_shape[0] != c_in {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!(
                "{OP}: weight dim 0 must equal input channels ({c_in}), got {}. \
                 Transposed convolution takes [c_in, c_out/groups, kernel], not \
                 conv1d's [c_out, c_in/groups, kernel]",
                weight_shape[0]
            ),
        });
    }
    if !c_in.is_multiple_of(groups) {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: format!("{OP}: input channels ({c_in}) must be divisible by groups ({groups})"),
        });
    }
    let c_out = weight_shape[1] * groups;

    if output_padding >= stride.max(dilation) {
        return Err(Error::InvalidArgument {
            arg: "output_padding",
            reason: format!(
                "{OP}: output_padding ({output_padding}) must be smaller than \
                 max(stride, dilation) = {}",
                stride.max(dilation)
            ),
        });
    }

    if let Some(b_shape) = bias_shape {
        validate_1d_tensor(b_shape, "bias", OP)?;
        if b_shape[0] != c_out {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!(
                    "{OP}: bias length {} must equal c_out ({c_out})",
                    b_shape[0]
                ),
            });
        }
    }

    let (pad_left, pad_right) =
        resolve_transpose_padding_1d(padding, kernel_size, stride, dilation, output_padding);

    let output_length = compute_transpose_output_size(
        length,
        kernel_size,
        stride,
        dilation,
        pad_left,
        pad_right,
        output_padding,
    );

    Ok(ConvTranspose1dParams {
        batch,
        c_in,
        length,
        c_out,
        kernel_size,
        stride,
        dilation,
        groups,
        pad_left,
        pad_right,
        output_padding,
        output_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_size_matches_pytorch_formula() {
        // (L-1)*s - (pl+pr) + d*(k-1) + op + 1
        assert_eq!(
            compute_transpose_output_size(10, 12, 2, 1, 0, 0, 0),
            9 * 2 + 12
        );
        assert_eq!(
            compute_transpose_output_size(4, 3, 1, 1, 1, 1, 0),
            3 + 3 - 2
        );
        // stride 2, k 4, no padding: doubles then adds the kernel tail.
        assert_eq!(
            compute_transpose_output_size(5, 4, 2, 1, 0, 0, 0),
            4 * 2 + 4
        );
    }

    #[test]
    fn same_padding_upsamples_exactly_by_stride() {
        let p = validate_conv_transpose1d(
            &[1, 4, 16],
            &[4, 4, 12],
            None,
            2,
            PaddingMode::Same,
            0,
            1,
            1,
            DType::F32,
            DType::F32,
            None,
        )
        .unwrap();
        assert_eq!(p.output_length, 32, "Same must give length * stride");
    }

    /// The weight layout is the classic trap: conv1d's `[c_out, c_in/g, k]`
    /// silently produces a wrongly-shaped result if accepted here.
    #[test]
    fn rejects_conv1d_style_weight_layout() {
        let err = validate_conv_transpose1d(
            &[1, 8, 16],
            &[16, 8, 3], // c_out-first: wrong for transposed conv
            None,
            1,
            PaddingMode::Valid,
            0,
            1,
            1,
            DType::F32,
            DType::F32,
            None,
        );
        assert!(err.is_err());
    }

    #[test]
    fn c_out_accounts_for_groups() {
        let p = validate_conv_transpose1d(
            &[2, 8, 5],
            &[8, 3, 4], // groups=4 -> c_out = 3 * 4 = 12
            None,
            1,
            PaddingMode::Valid,
            0,
            1,
            4,
            DType::F32,
            DType::F32,
            None,
        )
        .unwrap();
        assert_eq!(p.c_out, 12);
    }

    #[test]
    fn rejects_output_padding_at_or_above_stride() {
        let err = validate_conv_transpose1d(
            &[1, 2, 4],
            &[2, 2, 3],
            None,
            2,
            PaddingMode::Valid,
            2, // == stride
            1,
            1,
            DType::F32,
            DType::F32,
            None,
        );
        assert!(err.is_err());
    }
}
