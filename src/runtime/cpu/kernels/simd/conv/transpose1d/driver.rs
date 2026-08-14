//! Shared polyphase loop nest for the SIMD `conv_transpose1d` kernels.
//!
//! # Why polyphase, and not channel vectorisation
//!
//! The primary caller is a **depthwise** transposed convolution (`groups ==
//! c_in`, so `c_in_per_group == 1`). Vectorising over input channels — what
//! `conv1d`/`conv2d` do — degenerates to scalar there and buys nothing. So we
//! vectorise over **output positions** instead.
//!
//! # The index derivation
//!
//! Start from the gather condition (identical to `conv.cu` / the WGSL shader):
//! output `ot` is fed by input `l` through tap `k` when
//!
//! ```text
//! ot = l * stride - pad_left + k * dilation
//! ```
//!
//! Fix a tap `k` and let `off_k = k * dilation - pad_left` (signed). Then the
//! set of outputs that tap `k` touches is exactly
//!
//! ```text
//! ot = off_k + l * stride,   l in [0, length)
//! ```
//!
//! an arithmetic progression of step `stride`. Its residue class is fixed:
//! every `ot` reached by tap `k` has `ot % stride == r_k` where
//! `r_k = off_k.rem_euclid(stride)`. That is the polyphase decomposition — tap
//! `k` contributes to phase `r_k` and to no other phase.
//!
//! Re-index phase `r` by `p`, via `ot = r + p * stride`. Substituting:
//!
//! ```text
//! p = base_k + l,   base_k = (off_k - r_k) / stride   (exact division, signed)
//! ```
//!
//! So within a phase, `p` advances by exactly 1 as `l` advances by 1: the input
//! reads *and* the phase-accumulator writes are both unit-stride, which is what
//! makes the inner loop a plain contiguous `acc[j] += w * x[j]` AXPY.
//!
//! Valid `l` range, from `0 <= p < n_r` and `0 <= l < length`:
//!
//! ```text
//! n_r  = number of ot in phase r  = ceil((output_length - r) / stride)
//! l_lo = max(0, -base_k)
//! l_hi = min(length, n_r - base_k)
//! ```
//!
//! The phase accumulator `acc` holds `n_r <= ceil(output_length / stride)`
//! elements. It is zeroed per `(batch, c_out, phase)`, accumulated over taps
//! (outer) then input channels (inner) — the same order as the CUDA/WGSL
//! kernels — and scattered back to `output[.. + r + p * stride]` once, adding
//! bias at that point so the bias lands last exactly as it does on GPU.
//!
//! Each output element is written exactly once: the phases `r in 0..stride`
//! partition `0..output_length`, so no zero-init pass and no read-modify-write
//! hazard exists.

/// Expands the polyphase gather loop nest for one dtype.
///
/// `$axpy` supplies the ISA-specific inner loop and is handed
/// `(acc_ptr, x_ptr, w, n)`, meaning `acc[j] += w * x[j]` for `j in 0..n` with
/// both pointers unit-stride.
macro_rules! conv_transpose1d_body {
    (
        $ty:ty,
        $input:expr, $weight:expr, $bias:expr, $output:expr, $params:expr, $acc:expr,
        |$ap:ident, $xp:ident, $w:ident, $n:ident| $axpy:block
    ) => {{
        let input: *const $ty = $input;
        let weight: *const $ty = $weight;
        let bias: Option<*const $ty> = $bias;
        let output: *mut $ty = $output;
        let acc: &mut [$ty] = $acc;

        let crate::ops::conv_transpose_common::ConvTranspose1dParams {
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

        if c_out != 0 && output_length != 0 {
            let c_in_per_group = c_in / groups;
            let c_out_per_group = c_out / groups;
            let stride_i = stride as isize;
            let pad_left_i = pad_left as isize;

            for b in 0..batch {
                for oc in 0..c_out {
                    let g = oc / c_out_per_group;
                    let oc_local = oc % c_out_per_group;
                    let c_in_start = g * c_in_per_group;
                    let out_row = (b * c_out + oc) * output_length;
                    let bias_val = match bias {
                        Some(p) => *p.add(oc),
                        None => <$ty>::default(),
                    };

                    for r in 0..stride {
                        if r >= output_length {
                            break;
                        }
                        // Count of output positions in this phase.
                        let n_r = (output_length - r).div_ceil(stride);
                        let acc_ptr = acc.as_mut_ptr();
                        for p in 0..n_r {
                            *acc_ptr.add(p) = <$ty>::default();
                        }

                        for k in 0..kernel_size {
                            let off_k = (k * dilation) as isize - pad_left_i;
                            let r_k = off_k.rem_euclid(stride_i);
                            if r_k as usize != r {
                                continue;
                            }
                            // Exact division: off_k - r_k is a multiple of stride.
                            let base_k = (off_k - r_k) / stride_i;

                            let l_lo = if base_k < 0 { (-base_k) as usize } else { 0 };
                            let l_hi_i = n_r as isize - base_k;
                            if l_hi_i <= l_lo as isize {
                                continue;
                            }
                            let l_hi = (l_hi_i as usize).min(length);
                            if l_hi <= l_lo {
                                continue;
                            }
                            let run = l_hi - l_lo;
                            let p0 = (base_k + l_lo as isize) as usize;

                            for ic in 0..c_in_per_group {
                                let c_in_abs = c_in_start + ic;
                                let $w = *weight
                                    .add((c_in_abs * c_out_per_group + oc_local) * kernel_size + k);
                                let $xp = input.add((b * c_in + c_in_abs) * length + l_lo);
                                let $ap = acc_ptr.add(p0);
                                let $n = run;
                                $axpy
                            }
                        }

                        // Single strided write-back; bias added last, matching
                        // the CUDA/WGSL accumulation order.
                        for p in 0..n_r {
                            *output.add(out_row + r + p * stride) = *acc_ptr.add(p) + bias_val;
                        }
                    }
                }
            }
        }
    }};
}

pub(super) use conv_transpose1d_body;
