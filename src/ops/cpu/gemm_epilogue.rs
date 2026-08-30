//! CPU implementation of GEMM epilogue operations.

use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{GemmActivation, GemmEpilogueOps};
use crate::ops::{matmul_bias_output_shape, validate_gemm_epilogue_dtypes};
use crate::runtime::cpu::helpers::{dispatch_dtype, ensure_contiguous};
use crate::runtime::cpu::kernels::{
    matmul_bias_activation_bwd_kernel, matmul_bias_activation_kernel, matmul_bias_residual_kernel,
};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl GemmEpilogueOps<CpuRuntime> for CpuClient {
    fn matmul_bias_activation(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        activation: GemmActivation,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_activation",
        )?;

        let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            },
        )?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;

        // No `.max(1)`: an unbatched matmul takes 0 dims and already products to 1,
        // so a clamp would only fabricate a batch for a genuinely zero batch dim.
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();

        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;

        // A zero-element output has nothing to compute. The dispatch below takes the
        // single-matmul branch for any `batch_size <= 1`, and a batch of 0 lands
        // there too — it would then write one full m*n tile into an empty
        // allocation. Return before that.
        if out.numel() == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let out_ptr = out.ptr();

        let lda = k;
        let ldb = n;
        let ldc = n;

        dispatch_dtype!(dtype, T => {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;

                if batch_size > 1 {
                    let min_len = self.rayon_min_len();
                    self.install_parallelism(|| {
                        (0..batch_size)
                            .into_par_iter()
                            .with_min_len(min_len)
                            .for_each(|batch| unsafe {
                            matmul_bias_activation_kernel::<T>(
                                (a_ptr as *const T).add(batch * m * k),
                                (b_ptr as *const T).add(batch * k * n),
                                bias_ptr as *const T,
                                (out_ptr as *mut T).add(batch * m * n),
                                m, n, k, lda, ldb, ldc,
                                activation,
                            );
                        });
                    });
                } else {
                    unsafe {
                        matmul_bias_activation_kernel::<T>(
                            a_ptr as *const T,
                            b_ptr as *const T,
                            bias_ptr as *const T,
                            out_ptr as *mut T,
                            m, n, k, lda, ldb, ldc,
                            activation,
                        );
                    }
                }
            }

            #[cfg(not(feature = "rayon"))]
            unsafe {
                for batch in 0..batch_size {
                    matmul_bias_activation_kernel::<T>(
                        (a_ptr as *const T).add(batch * m * k),
                        (b_ptr as *const T).add(batch * k * n),
                        bias_ptr as *const T,
                        (out_ptr as *mut T).add(batch * m * n),
                        m, n, k, lda, ldb, ldc,
                        activation,
                    );
                }
            }
        }, "matmul_bias_activation");

        Ok(out)
    }

    fn matmul_bias_residual(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        residual: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_residual",
        )?;
        if residual.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: residual.dtype(),
            });
        }

        let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            },
        )?;

        // Validate residual shape matches output shape
        if residual.shape() != out_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: out_shape.clone(),
                got: residual.shape().to_vec(),
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

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;
        let residual_contig = ensure_contiguous(residual)?;

        // No `.max(1)`: an unbatched matmul takes 0 dims and already products to 1,
        // so a clamp would only fabricate a batch for a genuinely zero batch dim.
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();

        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;

        // A zero-element output has nothing to compute. The dispatch below takes the
        // single-matmul branch for any `batch_size <= 1`, and a batch of 0 lands
        // there too — it would then write one full m*n tile into an empty
        // allocation. Return before that.
        if out.numel() == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let res_ptr = residual_contig.ptr();
        let out_ptr = out.ptr();

        let lda = k;
        let ldb = n;
        let ldc = n;

        dispatch_dtype!(dtype, T => {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;

                if batch_size > 1 {
                    let min_len = self.rayon_min_len();
                    self.install_parallelism(|| {
                        (0..batch_size)
                            .into_par_iter()
                            .with_min_len(min_len)
                            .for_each(|batch| unsafe {
                            matmul_bias_residual_kernel::<T>(
                                (a_ptr as *const T).add(batch * m * k),
                                (b_ptr as *const T).add(batch * k * n),
                                bias_ptr as *const T,
                                (res_ptr as *const T).add(batch * m * n),
                                (out_ptr as *mut T).add(batch * m * n),
                                m, n, k, lda, ldb, ldc,
                            );
                        });
                    });
                } else {
                    unsafe {
                        matmul_bias_residual_kernel::<T>(
                            a_ptr as *const T,
                            b_ptr as *const T,
                            bias_ptr as *const T,
                            res_ptr as *const T,
                            out_ptr as *mut T,
                            m, n, k, lda, ldb, ldc,
                        );
                    }
                }
            }

            #[cfg(not(feature = "rayon"))]
            unsafe {
                for batch in 0..batch_size {
                    matmul_bias_residual_kernel::<T>(
                        (a_ptr as *const T).add(batch * m * k),
                        (b_ptr as *const T).add(batch * k * n),
                        bias_ptr as *const T,
                        (res_ptr as *const T).add(batch * m * n),
                        (out_ptr as *mut T).add(batch * m * n),
                        m, n, k, lda, ldb, ldc,
                    );
                }
            }
        }, "matmul_bias_residual");

        Ok(out)
    }

    fn matmul_bias_activation_bwd(
        &self,
        grad: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        activation: GemmActivation,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
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

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;
        let grad_contig = ensure_contiguous(grad)?;

        // No `.max(1)`: an unbatched matmul takes 0 dims and already products to 1,
        // so a clamp would only fabricate a batch for a genuinely zero batch dim.
        let batch_size: usize = a_shape
            .iter()
            .take(a_shape.len().saturating_sub(2))
            .product();

        // Output gradients
        let d_a = Tensor::<CpuRuntime>::empty(a_shape, dtype, &self.device)?;
        let d_b = Tensor::<CpuRuntime>::zeros(b_shape, dtype, &self.device)?;

        // No batch contributes. The batched branch below zeroes `d_b` over k*n
        // elements before accumulating, which overruns `d_b` when its own batch dim
        // is 0. Both gradients sum over nothing here, so both are the additive
        // identity `d_b` was already seeded with.
        if batch_size == 0 {
            let d_bias_full = Tensor::<CpuRuntime>::zeros(&[n], dtype, &self.device)?;
            return Ok((d_a, d_b, d_bias_full));
        }

        // d_bias is always [N] — we need to sum across batches
        let d_bias_full = Tensor::<CpuRuntime>::empty(&[n], dtype, &self.device)?;

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let grad_ptr = grad_contig.ptr();
        let d_a_ptr = d_a.ptr();
        let d_b_ptr = d_b.ptr();
        let d_bias_ptr = d_bias_full.ptr();

        let lda = k;
        let ldb = n;
        let ld_grad = n;

        dispatch_dtype!(dtype, T => {
            if batch_size == 1 {
                unsafe {
                    matmul_bias_activation_bwd_kernel::<T>(
                        grad_ptr as *const T,
                        a_ptr as *const T,
                        b_ptr as *const T,
                        bias_ptr as *const T,
                        d_a_ptr as *mut T,
                        d_b_ptr as *mut T,
                        d_bias_ptr as *mut T,
                        m, n, k, lda, ldb, ld_grad,
                        activation,
                    );
                }
            } else {
                // For batched: compute per-batch, accumulate d_b and d_bias
                // Zero out d_b and d_bias first
                unsafe {
                    for i in 0..k * n {
                        *(d_b_ptr as *mut T).add(i) = T::zero();
                    }
                    for j in 0..n {
                        *(d_bias_ptr as *mut T).add(j) = T::zero();
                    }
                }

                let mut temp_d_b = vec![T::zero(); k * n];
                let mut temp_d_bias = vec![T::zero(); n];

                for batch in 0..batch_size {
                    unsafe {
                        matmul_bias_activation_bwd_kernel::<T>(
                            (grad_ptr as *const T).add(batch * m * n),
                            (a_ptr as *const T).add(batch * m * k),
                            (b_ptr as *const T).add(batch * k * n),
                            bias_ptr as *const T,
                            (d_a_ptr as *mut T).add(batch * m * k),
                            temp_d_b.as_mut_ptr(),
                            temp_d_bias.as_mut_ptr(),
                            m, n, k, lda, ldb, ld_grad,
                            activation,
                        );

                        // Accumulate d_b
                        for i in 0..k * n {
                            let ptr = (d_b_ptr as *mut T).add(i);
                            *ptr += temp_d_b[i];
                        }
                        // Accumulate d_bias
                        for j in 0..n {
                            let ptr = (d_bias_ptr as *mut T).add(j);
                            *ptr += temp_d_bias[j];
                        }
                    }
                }
            }
        }, "matmul_bias_activation_bwd");

        Ok((d_a, d_b, d_bias_full))
    }
}
