//! CPU implementation of fused matrix multiplication with bias.
//!
//! Same tiled kernel as plain matmul, with the bias added in the epilogue so
//! the result is not read back for a separate broadcast add.
//!
//! `#[path]`-included into `runtime::cpu::ops`, so `super` here is that module.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
};
use crate::tensor::Tensor;

impl CpuClient {
    /// Fused matmul+bias: `C = A @ B + bias`, broadcast over the batch.
    ///
    /// Carries the body of [`crate::ops::MatmulOps::matmul_bias`] for the CPU
    /// runtime; the trait method delegates here.
    pub(super) fn matmul_bias_impl(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::{matmul_bias_output_shape, validate_matmul_bias_dtypes};
        use crate::runtime::cpu::kernels::matmul_bias_kernel;

        // Validate dtypes using unified helper (ensures consistent error handling across backends)
        let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

        // Compute output shape (also validates bias shape)
        let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            },
        )?;

        // Get matrix dimensions (last two dims)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Require row-major contiguous tensors for SIMD-optimized packing
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;

        // Calculate batch size from output shape, and per-operand batch sizes for broadcasting
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Batch dims broadcast per dimension, so each output batch needs its own
        // source index per operand rather than a single batch count.
        let (a_batch_idx, b_batch_idx) =
            crate::ops::matmul::matmul_batch_indices(a_shape, b_shape, &out_shape);

        // I8 widens to I32 exactly as the plain form does, and the validator
        // above has already required the I32 bias that seeds the accumulator.
        if dtype == DType::I8 {
            return super::matmul_i8::matmul_i8_i32(
                self,
                &a_contig,
                &b_contig,
                Some(&bias_contig),
                &out_shape,
                &a_batch_idx,
                &b_batch_idx,
                m,
                n,
                k,
            );
        }

        // Create output tensor
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let out_ptr = out.ptr();

        // Leading dimensions for contiguous row-major matrices
        let lda = k;
        let ldb = n;
        let ldc = n;

        // Dispatch based on dtype
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
                            let a_offset = a_batch_idx[batch] * m * k;
                            let b_offset = b_batch_idx[batch] * k * n;
                            let out_offset = batch * m * n;

                            matmul_bias_kernel::<T>(
                                (a_ptr as *const T).add(a_offset),
                                (b_ptr as *const T).add(b_offset),
                                bias_ptr as *const T, // bias is 1D, same for all batches
                                (out_ptr as *mut T).add(out_offset),
                                m,
                                n,
                                k,
                                lda,
                                ldb,
                                ldc,
                            );
                        });
                    });
                } else {
                    unsafe {
                        let a_offset = 0;
                        let b_offset = 0;
                        let out_offset = 0;

                        matmul_bias_kernel::<T>(
                            (a_ptr as *const T).add(a_offset),
                            (b_ptr as *const T).add(b_offset),
                            bias_ptr as *const T,
                            (out_ptr as *mut T).add(out_offset),
                            m,
                            n,
                            k,
                            lda,
                            ldb,
                            ldc,
                        );
                    }
                }
            }

            #[cfg(not(feature = "rayon"))]
            unsafe {
                for batch in 0..batch_size {
                    let a_offset = a_batch_idx[batch] * m * k;
                    let b_offset = b_batch_idx[batch] * k * n;
                    let out_offset = batch * m * n;

                    matmul_bias_kernel::<T>(
                        (a_ptr as *const T).add(a_offset),
                        (b_ptr as *const T).add(b_offset),
                        bias_ptr as *const T, // bias is 1D, same for all batches
                        (out_ptr as *mut T).add(out_offset),
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                }
            }
        }, "matmul_bias");

        Ok(out)
    }
}
