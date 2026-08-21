//! Shared loop nest for the SIMD `conv1d` kernels.
//!
//! # Why output positions, and not input channels
//!
//! The tensor layout is `(batch, channels, length)`, so two neighbouring input
//! CHANNELS sit `length` elements apart. Vectorising the channel axis therefore
//! cannot use a vector load at all — it has to pack the lanes one scalar at a
//! time (a manual gather of BOTH input and weight), reduce into a single
//! accumulator, and finish with a horizontal sum. It also degenerates to
//! nothing at all for a depthwise convolution, where `c_in_per_group == 1`.
//!
//! Neighbouring OUTPUT POSITIONS, by contrast, read neighbouring input
//! elements. Fixing `(b, g, oc)` and reducing over `(kx, ic)` while vectorising
//! `ox` turns the weight into a scalar broadcast (`set1`) and the input into a
//! contiguous vector load whenever `stride == 1`. Depthwise vectorises fully.
//!
//! # The interior/boundary split
//!
//! The tap index is
//!
//! ```text
//! ix(ox, kx) = ox * stride + kx * dilation - pad_left
//! ```
//!
//! which is monotonically non-decreasing in `kx`. So "every tap of this `ox` is
//! in bounds" needs checking only at the two ends, `kx = 0` and `kx = K - 1`:
//!
//! ```text
//! ix(ox, 0)     >= 0        <=>  ox * stride >= pad_left
//!                           <=>  ox >= ceil(pad_left / stride)              = interior_lo
//! ix(ox, K - 1) <= length-1 <=>  ox * stride <= length - 1 + pad_left - (K-1)*dilation
//!                           <=>  ox <  ceil((length + pad_left - (K-1)*dilation) / stride)
//!                                                                           = interior_hi
//! ```
//!
//! (The second equivalence uses `floor((M - 1) / s) + 1 == ceil(M / s)` for
//! `M >= 1`. When `M = length + pad_left - (K-1)*dilation` is zero or negative —
//! a kernel whose dilated span exceeds the padded input — NO output position is
//! fully interior, so `interior_hi` is clamped to 0. That subtraction is done in
//! `isize` precisely so `usize` cannot wrap here.)
//!
//! `ox` in `[0, interior_lo)` and `[interior_hi, output_length)` runs scalar,
//! keeping the per-tap `0 <= ix < length` check. Those two edges are together at
//! most `(K-1)*dilation / stride + 1` positions wide, so vectorising them would
//! buy nothing. `ox` in `[interior_lo, interior_hi)` runs vectorised with NO
//! per-tap bounds check at all.
//!
//! # Accumulation order
//!
//! Within a lane the reduction is `ic` outer, `kx` inner — the same order as the
//! scalar kernel — and the bias is added once, after the full reduction. The
//! only numerical difference from scalar is FMA contraction of the multiply-add.

/// Expands the `conv1d` loop nest for one dtype.
///
/// The ISA-specific block receives the fully interior output run for one
/// `(batch, group, out-channel)` triple and must compute all `n` of its outputs:
///
/// - `$op`: `*mut $ty`, output for `ox = interior_lo`, `n` contiguous elements
/// - `$ip`: `*const $ty`, input at `(b, c_in_start, ix(interior_lo, 0))`; the
///   element for `(j, ic, kx)` is `$ip[ic * length + j * stride + kx * dilation]`
/// - `$wp`: `*const $ty`, weight at `(c_out_idx, 0, 0)`; the element for
///   `(ic, kx)` is `$wp[ic * kernel_size + kx]`
/// - `$n`: number of interior output positions (always `>= 1`)
/// - `$nic`: `c_in_per_group`
/// - `$bv`: the bias value for this output channel (zero when there is no bias)
///
/// Every index above is in bounds by construction, so the block must not
/// bounds-check taps. `stride`, `dilation`, `kernel_size` and `length` are read
/// by the block from its own `params` argument (macro hygiene keeps the locals
/// expanded here invisible to it).
macro_rules! conv1d_body {
    (
        $ty:ty,
        $input:expr, $weight:expr, $bias:expr, $output:expr, $params:expr,
        |$op:ident, $ip:ident, $wp:ident, $n:ident, $nic:ident, $bv:ident| $interior:block
    ) => {{
        let input: *const $ty = $input;
        let weight: *const $ty = $weight;
        let bias: Option<*const $ty> = $bias;
        let output: *mut $ty = $output;

        let crate::ops::conv_common::Conv1dParams {
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
        } = $params;

        let c_in_per_group = c_in / groups;
        let c_out_per_group = c_out / groups;

        // Interior window, derived once for the whole call (see module docs).
        // `stride == 0` is not a valid convolution; treat the whole output as
        // boundary rather than dividing by zero.
        let (interior_lo, interior_hi) = if stride == 0 {
            (0usize, 0usize)
        } else {
            let span = kernel_size.saturating_sub(1) * dilation;
            let reach = (length + pad_left) as isize - span as isize;
            let hi = if reach <= 0 {
                0
            } else {
                (reach as usize).div_ceil(stride).min(output_length)
            };
            (pad_left.div_ceil(stride).min(hi), hi)
        };

        for b in 0..batch {
            for g in 0..groups {
                let c_in_start = g * c_in_per_group;
                let c_out_start = g * c_out_per_group;
                let in_base = (b * c_in + c_in_start) * length;

                for oc in 0..c_out_per_group {
                    let c_out_idx = c_out_start + oc;
                    let w_base = c_out_idx * c_in_per_group * kernel_size;
                    let out_base = (b * c_out + c_out_idx) * output_length;
                    let bias_val = match bias {
                        Some(p) => *p.add(c_out_idx),
                        None => <$ty>::default(),
                    };

                    // Boundary positions: some tap may fall outside the input,
                    // so every tap keeps its own bounds check.
                    for ox in (0..interior_lo).chain(interior_hi..output_length) {
                        let mut sum = <$ty>::default();
                        for ic in 0..c_in_per_group {
                            let x_row = in_base + ic * length;
                            let w_row = w_base + ic * kernel_size;
                            for kx in 0..kernel_size {
                                let ix = (ox * stride) as isize + (kx * dilation) as isize
                                    - pad_left as isize;
                                if ix >= 0 && (ix as usize) < length {
                                    sum +=
                                        *input.add(x_row + ix as usize) * *weight.add(w_row + kx);
                                }
                            }
                        }
                        *output.add(out_base + ox) = sum + bias_val;
                    }

                    if interior_hi > interior_lo {
                        // `interior_lo * stride >= pad_left` by construction, so
                        // this offset is non-negative.
                        let ix0 = interior_lo * stride - pad_left;
                        let $op = output.add(out_base + interior_lo);
                        let $ip = input.add(in_base + ix0);
                        let $wp = weight.add(w_base);
                        let $n = interior_hi - interior_lo;
                        let $nic = c_in_per_group;
                        let $bv = bias_val;
                        $interior
                    }
                }
            }
        }
    }};
}

pub(super) use conv1d_body;
