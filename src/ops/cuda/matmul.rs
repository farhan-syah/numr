//! Matrix multiplication operations for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::BinaryOps;
use crate::ops::{
    MatmulOps, ShapeOps, matmul_bias_output_shape, matmul_output_shape, validate_matmul_bias_dtypes,
};
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

        // Native tiled CUDA kernel. I32 and I64 are the only integer dtypes with
        // a CUDA arithmetic pipeline (see `kernels/binary.cu`), so they are the
        // only ones with an integer matmul kernel.
        match dtype {
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 | DType::I32 | DType::I64 => {
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

        // Native tiled CUDA kernel with fused bias
        match dtype {
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
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
            _ => {
                // FP8 and other dtypes: fall back to matmul + add
                let mm = self.matmul(a, b)?;
                self.add(&mm, &bias.reshape(&[1, n])?)
            }
        }
    }
}
