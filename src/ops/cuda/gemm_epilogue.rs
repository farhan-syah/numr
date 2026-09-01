//! CUDA implementation of GEMM epilogue operations.

#[cfg(feature = "fp8")]
use crate::dtype::DType;
use crate::error::{Error, Result};
#[cfg(feature = "fp8")]
use crate::ops::TypeConversionOps;
use crate::ops::{
    GemmActivation, GemmEpilogueOps, ShapeOps, matmul_bias_output_shape,
    validate_gemm_epilogue_dtypes,
};
use crate::runtime::cuda::kernels::{
    launch_gemm_bias_act_bwd_batched_kernel, launch_gemm_bias_act_bwd_kernel,
    use_wmma_after_padding,
};
use crate::runtime::cuda::ops::helpers::{
    gemm_bias_act_batched_native, gemm_bias_act_native, gemm_bias_residual_batched_native,
    gemm_bias_residual_native,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Device, ensure_contiguous};
use crate::tensor::Tensor;

impl GemmEpilogueOps<CudaRuntime> for CudaClient {
    fn matmul_bias_activation(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        activation: GemmActivation,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_activation",
        )?;

        // FP8: compute in F32 (tiled GEMM with shared memory needs native arithmetic)
        #[cfg(feature = "fp8")]
        if dtype == DType::FP8E4M3 || dtype == DType::FP8E5M2 {
            let a_f32 = self.cast(a, DType::F32)?;
            let b_f32 = self.cast(b, DType::F32)?;
            let bias_f32 = self.cast(bias, DType::F32)?;
            let result = self.matmul_bias_activation(&a_f32, &b_f32, &bias_f32, activation)?;
            return self.cast(&result, dtype);
        }

        if bias.shape().len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("bias must be 1D tensor, got shape {:?}", bias.shape()),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let out_shape = matmul_bias_output_shape(a_shape, b_shape, bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            },
        )?;

        // Unclamped: an unbatched matmul takes 0 dims and already products to 1, so
        // a clamp would only fabricate a batch for a genuinely zero batch dim — and
        // then pick the single-matmul branch, writing one m*n tile into an empty
        // allocation.
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();

        // A zero-element output has nothing to compute. The launchers derive their
        // grid from `m`, `n` and the batch count without flooring them, and a grid
        // extent of 0 is a launch error, so return before any launch.
        if out_shape.iter().product::<usize>() == 0 {
            return Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);
        }

        if batch_size > 1 {
            return gemm_bias_act_batched_native(
                self, a, b, bias, dtype, &out_shape, batch_size, m, n, k, activation,
            );
        }

        // Pad unaligned F16/BF16 (m>16) up to 16-multiples so the WMMA tensor-core
        // kernel fires, the same rule plain matmul and matmul_bias apply
        // (src/ops/cuda/matmul.rs). Without it any M that is not a multiple of 16
        // silently keeps the generic kernel. `use_wmma_after_padding` is derived
        // from the launcher's own `use_wmma`, so the padding decision cannot
        // disagree with the dispatch decision.
        //
        // Zero-padding is exact here: the extra K contributes 0 to the accumulator,
        // and the extra M rows / N cols — where the bias and the activation still
        // apply — are sliced off before the result is returned.
        let caps = self.device.profile().caps;
        if use_wmma_after_padding(dtype, caps, m, n, k) {
            let m_pad = m.next_multiple_of(16);
            let k_pad = k.next_multiple_of(16);
            let n_pad = n.next_multiple_of(16);
            let a_pad = self.pad(a, &[0, k_pad - k, 0, m_pad - m], 0.0)?;
            let b_pad = self.pad(b, &[0, n_pad - n, 0, k_pad - k], 0.0)?;
            // bias is 1-D [n], so it takes a two-element padding spec.
            let bias_pad = self.pad(bias, &[0, n_pad - n], 0.0)?;
            let out_pad_shape =
                matmul_bias_output_shape(a_pad.shape(), b_pad.shape(), bias_pad.shape()).ok_or(
                    Error::ShapeMismatch {
                        expected: a_pad.shape().to_vec(),
                        got: b_pad.shape().to_vec(),
                    },
                )?;
            let out_pad = gemm_bias_act_native(
                self,
                &a_pad,
                &b_pad,
                &bias_pad,
                dtype,
                &out_pad_shape,
                m_pad,
                n_pad,
                k_pad,
                activation,
            )?;
            // Slice the M (2nd-last) and N (last) dims back via negative indexing —
            // NOT dims 0/1, since the output may carry leading batch dims.
            return out_pad.narrow(-2, 0, m)?.narrow(-1, 0, n)?.contiguous();
        }

        gemm_bias_act_native(self, a, b, bias, dtype, &out_shape, m, n, k, activation)
    }

    fn matmul_bias_residual(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        residual: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_residual",
        )?;

        // FP8: compute in F32
        #[cfg(feature = "fp8")]
        if dtype == DType::FP8E4M3 || dtype == DType::FP8E5M2 {
            let a_f32 = self.cast(a, DType::F32)?;
            let b_f32 = self.cast(b, DType::F32)?;
            let bias_f32 = self.cast(bias, DType::F32)?;
            let res_f32 = self.cast(residual, DType::F32)?;
            let result = self.matmul_bias_residual(&a_f32, &b_f32, &bias_f32, &res_f32)?;
            return self.cast(&result, dtype);
        }

        if residual.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: residual.dtype(),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        let out_shape = matmul_bias_output_shape(a_shape, b_shape, bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            },
        )?;

        if residual.shape() != out_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: out_shape.clone(),
                got: residual.shape().to_vec(),
            });
        }

        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Unclamped: an unbatched matmul takes 0 dims and already products to 1, so
        // a clamp would only fabricate a batch for a genuinely zero batch dim — and
        // then pick the single-matmul branch, writing one m*n tile into an empty
        // allocation.
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();

        // A zero-element output has nothing to compute. The launchers derive their
        // grid from `m`, `n` and the batch count without flooring them, and a grid
        // extent of 0 is a launch error, so return before any launch.
        if out_shape.iter().product::<usize>() == 0 {
            return Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);
        }

        if batch_size > 1 {
            return gemm_bias_residual_batched_native(
                self, a, b, bias, residual, dtype, &out_shape, batch_size, m, n, k,
            );
        }

        // Same WMMA padding rule as matmul_bias_activation above. The residual is
        // [M,N]-shaped, so it takes the 2-D padding spec A and B take, NOT the 1-D
        // one the bias takes: padding it as a vector would shift every row and
        // corrupt the interior, not just the edge.
        let caps = self.device.profile().caps;
        if use_wmma_after_padding(dtype, caps, m, n, k) {
            let m_pad = m.next_multiple_of(16);
            let k_pad = k.next_multiple_of(16);
            let n_pad = n.next_multiple_of(16);
            let a_pad = self.pad(a, &[0, k_pad - k, 0, m_pad - m], 0.0)?;
            let b_pad = self.pad(b, &[0, n_pad - n, 0, k_pad - k], 0.0)?;
            let bias_pad = self.pad(bias, &[0, n_pad - n], 0.0)?;
            let res_pad = self.pad(residual, &[0, n_pad - n, 0, m_pad - m], 0.0)?;
            let out_pad_shape =
                matmul_bias_output_shape(a_pad.shape(), b_pad.shape(), bias_pad.shape()).ok_or(
                    Error::ShapeMismatch {
                        expected: a_pad.shape().to_vec(),
                        got: b_pad.shape().to_vec(),
                    },
                )?;
            let out_pad = gemm_bias_residual_native(
                self,
                &a_pad,
                &b_pad,
                &bias_pad,
                &res_pad,
                dtype,
                &out_pad_shape,
                m_pad,
                n_pad,
                k_pad,
            )?;
            return out_pad.narrow(-2, 0, m)?.narrow(-1, 0, n)?.contiguous();
        }

        gemm_bias_residual_native(self, a, b, bias, residual, dtype, &out_shape, m, n, k)
    }

    fn matmul_bias_activation_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        activation: GemmActivation,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_activation_bwd",
        )?;
        if grad.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: grad.dtype(),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Unclamped: an unbatched matmul takes 0 dims and already products to 1, so
        // a clamp would only fabricate a batch for a genuinely zero batch dim — and
        // then pick the single-matmul branch, writing one m*n tile into an empty
        // allocation.
        let batch_size: usize = a_shape
            .iter()
            .take(a_shape.len().saturating_sub(2))
            .product();

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;
        let grad_contig = ensure_contiguous(grad)?;

        let d_a = Tensor::<CudaRuntime>::empty(a_shape, dtype, &self.device)?;
        let d_b = Tensor::<CudaRuntime>::zeros(b_shape, dtype, &self.device)?;
        let d_bias = Tensor::<CudaRuntime>::zeros(&[n], dtype, &self.device)?;

        // No batch contributes, so every gradient sums over nothing and stays at the
        // additive identity `d_b` and `d_bias` were seeded with. A zero `m`, `n` or
        // `k` still reaches the launcher, which skips only the individual kernels
        // whose own output is empty — `d_bias` is a real sum even when `k == 0`.
        if batch_size == 0 {
            return Ok((d_a, d_b, d_bias));
        }

        // Temporary buffer for grad_pre (M * N elements, reused per batch)
        let grad_pre = Tensor::<CudaRuntime>::empty(&[m, n], dtype, &self.device)?;

        unsafe {
            if batch_size > 1 {
                launch_gemm_bias_act_bwd_batched_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    grad_contig.ptr(),
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    grad_pre.ptr(),
                    d_a.ptr(),
                    d_b.ptr(),
                    d_bias.ptr(),
                    batch_size,
                    m,
                    n,
                    k,
                    activation,
                )?;
            } else {
                launch_gemm_bias_act_bwd_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    grad_contig.ptr(),
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    grad_pre.ptr(),
                    d_a.ptr(),
                    d_b.ptr(),
                    d_bias.ptr(),
                    m,
                    n,
                    k,
                    activation,
                )?;
            }
        }

        Ok((d_a, d_b, d_bias))
    }
}
