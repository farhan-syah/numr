//! CPU implementation of matrix multiplication operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Kernel, MatmulOps};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
};
use crate::tensor::Tensor;

/// MatmulOps implementation for CPU runtime.
impl MatmulOps<CpuRuntime> for CpuClient {
    fn matmul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::matmul_output_shape;

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        let dtype = a.dtype();

        // Compute output shape
        let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        })?;

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

        // GEMV-BT fast path: detect transposed B and use dot-product kernel
        // When B has shape [K,N] with strides [1,K], it's a transpose of contiguous [N,K].
        // For small M (decode), we can dot A rows against B's original [N,K] rows directly,
        // avoiding the costly contiguous copy (e.g. 500MB for lm_head weights).
        if m <= 16 && b_shape.len() >= 2 && dtype != DType::I8 {
            let b_strides = b.strides();
            let ndim = b_shape.len();
            let stride_row = b_strides[ndim - 2]; // stride for K dimension
            let stride_col = b_strides[ndim - 1]; // stride for N dimension

            // Check if B is a simple transpose: shape [K,N], strides [1, K]
            // meaning the underlying data is contiguous [N,K]
            if stride_row == 1 && stride_col == k as isize {
                let a_contig = ensure_contiguous(a)?;
                let a_ptr = a_contig.ptr();
                let b_ptr = b.ptr(); // Use original ptr - data is contiguous [N,K]

                // Create output tensor
                let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);
                let out_ptr = out.ptr();
                let ldc = n;

                dispatch_dtype!(dtype, T => {
                    for batch in 0..batch_size {
                        let a_offset = a_batch_idx[batch] * m * k;
                        let b_offset = b_batch_idx[batch] * n * k;
                        let out_offset = batch * m * n;

                        #[cfg(feature = "rayon")]
                        {
                            use rayon::prelude::*;

                            // Parallelize over output columns for large N
                            // Each thread computes a chunk of columns independently
                            let min_cols_per_thread = 64usize;
                            let num_threads = rayon::current_num_threads();
                            let chunk_size = ((n + num_threads - 1) / num_threads).max(min_cols_per_thread);

                            if n > min_cols_per_thread && num_threads > 1 {
                                // Convert to usize for Send safety - each thread
                                // accesses disjoint memory regions
                                let a_send = (a_ptr as usize) + a_offset * std::mem::size_of::<T>();
                                let b_send = (b_ptr as usize) + b_offset * std::mem::size_of::<T>();
                                let out_send = (out_ptr as usize) + out_offset * std::mem::size_of::<T>();
                                let elem_size = std::mem::size_of::<T>();

                                self.install_parallelism(|| {
                                    (0..n).into_par_iter().step_by(chunk_size).for_each(|col_start| {
                                        let col_end = (col_start + chunk_size).min(n);
                                        let chunk_n = col_end - col_start;
                                        unsafe {
                                            let a_base = a_send as *const T;
                                            let b_chunk = (b_send + col_start * k * elem_size) as *const T;
                                            let out_chunk = (out_send + col_start * elem_size) as *mut T;

                                            crate::runtime::cpu::kernels::gemv_bt_kernel::<T>(
                                                a_base,
                                                b_chunk,
                                                out_chunk,
                                                m, chunk_n, k, n,
                                            );
                                        }
                                    });
                                });
                            } else {
                                unsafe {
                                    crate::runtime::cpu::kernels::gemv_bt_kernel::<T>(
                                        (a_ptr as *const T).add(a_offset),
                                        (b_ptr as *const T).add(b_offset),
                                        (out_ptr as *mut T).add(out_offset),
                                        m, n, k, ldc,
                                    );
                                }
                            }
                        }

                        #[cfg(not(feature = "rayon"))]
                        unsafe {
                            crate::runtime::cpu::kernels::gemv_bt_kernel::<T>(
                                (a_ptr as *const T).add(a_offset),
                                (b_ptr as *const T).add(b_offset),
                                (out_ptr as *mut T).add(out_offset),
                                m, n, k, ldc,
                            );
                        }
                    }
                }, "matmul_gemv_bt");

                return Ok(out);
            }
        }

        // Require row-major contiguous tensors for SIMD-optimized packing
        // Non-contiguous tensors (transposed, views) are copied to contiguous layout
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();

        // Leading dimensions for contiguous row-major matrices
        let lda = k;
        let ldb = n;
        let ldc = n;

        // Special case: i8 × i8 → i32 matmul (quantized accumulation)
        if dtype == DType::I8 {
            use crate::runtime::cpu::kernels::matmul_i8_to_i32_kernel;

            let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::I32, &self.device);
            let out_ptr = out.ptr();

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

                                matmul_i8_to_i32_kernel(
                                    (a_ptr as *const i8).add(a_offset),
                                    (b_ptr as *const i8).add(b_offset),
                                    (out_ptr as *mut i32).add(out_offset),
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
                        matmul_i8_to_i32_kernel(
                            a_ptr as *const i8,
                            b_ptr as *const i8,
                            out_ptr as *mut i32,
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

                    matmul_i8_to_i32_kernel(
                        (a_ptr as *const i8).add(a_offset),
                        (b_ptr as *const i8).add(b_offset),
                        (out_ptr as *mut i32).add(out_offset),
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                }
            }

            return Ok(out);
        }

        // Create output tensor
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);
        let out_ptr = out.ptr();

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

                            <Self as Kernel<CpuRuntime>>::matmul::<T>(
                                self,
                                (a_ptr as *const T).add(a_offset),
                                (b_ptr as *const T).add(b_offset),
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
                        <Self as Kernel<CpuRuntime>>::matmul::<T>(
                            self,
                            (a_ptr as *const T).add(a_offset),
                            (b_ptr as *const T).add(b_offset),
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

                    <Self as Kernel<CpuRuntime>>::matmul::<T>(
                        self,
                        (a_ptr as *const T).add(a_offset),
                        (b_ptr as *const T).add(b_offset),
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
        }, "matmul");

        Ok(out)
    }

    fn matmul_bias(
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

        // Create output tensor
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);

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
