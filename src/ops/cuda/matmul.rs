//! Matrix multiplication operations for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    MatmulOps, ShapeOps, matmul_bias_output_shape, matmul_output_shape, validate_matmul_bias_dtypes,
};
use crate::runtime::cuda::kernels::int_matmul_has_kernel;
use crate::runtime::cuda::ops::helpers::{
    matmul_batched_native, matmul_bias_batched_native, matmul_bias_native, matmul_native,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::validate_binary_dtypes;
use crate::tensor::Tensor;

impl MatmulOps<CudaRuntime> for CudaClient {
    fn matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_binary_dtypes(a, b)?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let k_b = if b_shape.len() >= 2 {
            b_shape[b_shape.len() - 2]
        } else {
            b_shape[b_shape.len() - 1]
        };
        if k != k_b {
            return Err(Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }

        let out_shape = matmul_output_shape(a_shape, b_shape).ok_or(Error::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        })?;

        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Native tiled CUDA kernel. The integer dtypes are gated by
        // `int_matmul_has_kernel`, which is the same predicate the launcher
        // uses, so this match and `matmul_int.cu`'s instantiation list cannot
        // drift apart. FP8 has its own kernels in `kernels/matmul_fp8.cu` and
        // accumulates in F32, matching CPU.
        match dtype {
            DType::F32
            | DType::F64
            | DType::F16
            | DType::BF16
            | DType::FP8E4M3
            | DType::FP8E5M2 => {
                if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    // Pad unaligned F16/BF16 (m>16) up to 16-multiples so the WMMA
                    // tensor-core kernel fires. Critical for the varlen-embedding path:
                    // M = total_tokens is rarely a multiple of 16, so without this F16
                    // dropped to the ~150x-slower generic kernel (57 vs 8500 GFLOP/s).
                    // Zero-padding is exact (extra K contributes 0; extra M rows / N
                    // cols are sliced off); the WMMA kernel only ever sees aligned dims.
                    let pad_for_wmma = matches!(dtype, DType::F16 | DType::BF16)
                        && m > 16
                        && (!m.is_multiple_of(16)
                            || !k.is_multiple_of(16)
                            || !n.is_multiple_of(16));

                    if pad_for_wmma {
                        let m_pad = m.next_multiple_of(16);
                        let k_pad = k.next_multiple_of(16);
                        let n_pad = n.next_multiple_of(16);
                        // pad(t, [last_before, last_after, 2nd_last_before, 2nd_last_after])
                        // — only the last two dims (M=2nd-last of A, K=last of A; N=last
                        // of B, K=2nd-last of B) are padded; any leading batch dims are
                        // untouched.
                        let a_pad = self.pad(a, &[0, k_pad - k, 0, m_pad - m], 0.0)?;
                        let b_pad = self.pad(b, &[0, n_pad - n, 0, k_pad - k], 0.0)?;
                        let out_pad =
                            matmul_native(self, &a_pad, &b_pad, dtype, m_pad, k_pad, n_pad)?;
                        // Slice the M (2nd-last) and N (last) dims back via negative
                        // indexing — NOT dims 0/1, since the output may carry leading
                        // batch dims (e.g. a 3D [1, m, n] from the padded encoder forward,
                        // where narrowing dim 0 — the size-1 batch — gave a [0, …] tensor).
                        out_pad.narrow(-2, 0, m)?.narrow(-1, 0, n)?.contiguous()
                    } else {
                        matmul_native(self, a, b, dtype, m, k, n)
                    }
                }
            }
            // Integers never take the WMMA padding branch above, so they only
            // need the two native entry points. I8 is included and returns an
            // I32 tensor: the helpers allocate through `int_matmul_output_dtype`,
            // which mirrors CPU's quantized-accumulation branch.
            d if int_matmul_has_kernel(d) => {
                if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    matmul_native(self, a, b, dtype, m, k, n)
                }
            }
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "matmul",
            }),
        }
    }

    fn matmul_bias(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate dtypes using unified helper (ensures consistent error handling across backends)
        let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

        // Validate bias is 1D
        if bias.shape().len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("bias must be 1D tensor, got shape {:?}", bias.shape()),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let bias_shape = bias.shape();

        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Validate inner dimensions
        let k_b = if b_shape.len() >= 2 {
            b_shape[b_shape.len() - 2]
        } else {
            b_shape[b_shape.len() - 1]
        };
        if k != k_b {
            return Err(Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }

        // Validate bias length matches N
        if bias_shape[0] != n {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!(
                    "bias length {} must match output columns {}",
                    bias_shape[0], n
                ),
            });
        }

        let out_shape =
            matmul_bias_output_shape(a_shape, b_shape, bias_shape).ok_or(Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            })?;

        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Native tiled CUDA kernel with fused bias. FP8 and the integers are
        // included: CPU seeds its wide accumulator with the bias, so composing
        // matmul with a separate add would narrow twice and report a different
        // number.
        match dtype {
            DType::F32
            | DType::F64
            | DType::F16
            | DType::BF16
            | DType::FP8E4M3
            | DType::FP8E5M2 => {
                if batch_size > 1 {
                    matmul_bias_batched_native(self, a, b, bias, dtype, batch_size, m, k, n)
                } else {
                    // Pad unaligned F16/BF16 (m>16) up to 16-multiples so WMMA fires
                    // (see matmul() for rationale). bias is [n] → pad to [n_pad].
                    let pad_for_wmma = matches!(dtype, DType::F16 | DType::BF16)
                        && m > 16
                        && (!m.is_multiple_of(16)
                            || !k.is_multiple_of(16)
                            || !n.is_multiple_of(16));

                    if pad_for_wmma {
                        let m_pad = m.next_multiple_of(16);
                        let k_pad = k.next_multiple_of(16);
                        let n_pad = n.next_multiple_of(16);
                        let a_pad = self.pad(a, &[0, k_pad - k, 0, m_pad - m], 0.0)?;
                        let b_pad = self.pad(b, &[0, n_pad - n, 0, k_pad - k], 0.0)?;
                        let bias_pad = self.pad(bias, &[0, n_pad - n], 0.0)?;
                        let out_pad = matmul_bias_native(
                            self, &a_pad, &b_pad, &bias_pad, dtype, m_pad, k_pad, n_pad,
                        )?;
                        // Slice M (2nd-last) and N (last) via negative indexing — see matmul().
                        out_pad.narrow(-2, 0, m)?.narrow(-1, 0, n)?.contiguous()
                    } else {
                        matmul_bias_native(self, a, b, bias, dtype, m, k, n)
                    }
                }
            }
            // Integers have their own fused-bias kernels in `matmul_int.cu`, and
            // fused is the only correct form: the bias seeds the 128-bit
            // accumulator, so composing a matmul with an elementwise add would
            // saturate the product and then wrap the bias into the element type.
            // I8 keeps its element type here — CPU `matmul_bias` has no I8
            // branch, so the bias is I8 and so is the result. Only the plain
            // form widens.
            d if int_matmul_has_kernel(d) => {
                if batch_size > 1 {
                    matmul_bias_batched_native(self, a, b, bias, dtype, batch_size, m, k, n)
                } else {
                    matmul_bias_native(self, a, b, bias, dtype, m, k, n)
                }
            }
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "matmul_bias",
            }),
        }
    }
}
