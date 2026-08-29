//! CPU implementation of matrix multiplication operations.

#[cfg(feature = "rayon")]
use super::matmul_columns::{column_chunk_count, matmul_bt_columns, matmul_columns};
/// Fixed column-chunk width for the GEMV-BT (`m <= 16`) parallel path.
///
/// Deliberately a constant rather than `n / num_threads`: the chunk boundaries
/// must be a pure function of the shape so results do not depend on how many
/// cores the machine has. 64 matches the historical minimum this path used and
/// still yields 64 chunks at `n = 4096`, more units than any common core count.
const GEMV_COLUMN_CHUNK_WIDTH: usize = 64;

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

        // B is the transposed view of a contiguous [N,K] buffer — the layout every
        // Linear weight has. Both paths below read that buffer directly instead of
        // materializing the [K,N] view, which otherwise copies the whole weight
        // matrix on every call (a profiled VoxCPM2 decode moved ~50 GB through
        // copy_strided over four generated patches, 41% of all instructions).
        let b_transposed = crate::ops::matmul::is_transposed_b(b_shape, b.strides(), k, n);

        // GEMV-BT fast path: for small M (decode), dot A rows against B's original
        // [N,K] rows directly. A different kernel from the tiled path below, and the
        // faster one at this shape.
        if m <= 16 && b_transposed && dtype != DType::I8 {
            let a_contig = ensure_contiguous(a)?;
            let a_ptr = a_contig.ptr();
            let b_ptr = b.ptr(); // Use original ptr - data is contiguous [N,K]

            // Create output tensor
            let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;
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

                        // Parallelize over output columns for large N.
                        // Each thread computes a chunk of columns independently.
                        //
                        // The chunk WIDTH is fixed, never derived from the
                        // thread count, for the same reason the tiled path's
                        // split is (see `matmul_columns`): sizing chunks by
                        // `n / num_threads` moves the block boundaries with the
                        // machine, so float rounding at a boundary differs
                        // between a 4-core laptop and a 24-core workstation and
                        // the same input yields different output. Measured on a
                        // VoxCPM2 decode: 1 thread and 24 threads produced
                        // different speech. Only the SCHEDULING of this fixed
                        // chunk list may vary with the pool.
                        let chunk_size = GEMV_COLUMN_CHUNK_WIDTH;

                        if n > chunk_size {
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

        // Larger M with the same transposed weight: the tiled kernel packs its B
        // panels straight out of the [N,K] buffer. Packing is a strided gather
        // either way, so the packed panels — and therefore the accumulation order
        // and the result — are identical to a materialized B, at no copy.
        //
        // The predicate is what keeps that guarantee: it holds only where both
        // sides run the tiled kernel (f32/f64, tiled-sized shape, SIMD hardware).
        // Every other dtype and shape reaches the transposed layout through a
        // different kernel, which agrees within tolerance but not bit for bit, so
        // those keep materializing B below.
        if b_transposed
            && crate::runtime::cpu::kernels::matmul_bt_matches_contiguous(dtype, m, n, k)
        {
            let a_contig = ensure_contiguous(a)?;
            let a_ptr = a_contig.ptr();
            let b_ptr = b.ptr(); // Use original ptr - data is contiguous [N,K]

            let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;
            let out_ptr = out.ptr();
            let ldc = n;

            // A transposed B's batch stride is N*K, the same element count a
            // contiguous [K,N] batch spans, so the batch index arithmetic is
            // unchanged. `is_transposed_b` is what makes that hold: it accepts the
            // layout only when the underlying buffer is densely packed [.., N, K].
            dispatch_dtype!(dtype, T => {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;

                    // Column split when the columns offer more units than the
                    // batches — every decode shape on this path is
                    // single-batch. Never both, and never a thread count in the
                    // test: see `matmul_columns` for the axis rule and for why
                    // the boundaries must not depend on the machine.
                    if let Some(chunks) = column_chunk_count(batch_size, m, n, k) {
                        for batch in 0..batch_size {
                            unsafe {
                                matmul_bt_columns::<T>(
                                    self,
                                    (a_ptr as *const T).add(a_batch_idx[batch] * m * k),
                                    (b_ptr as *const T).add(b_batch_idx[batch] * n * k),
                                    (out_ptr as *mut T).add(batch * m * n),
                                    m, n, k, ldc, chunks,
                                );
                            }
                        }
                    } else if batch_size > 1 {
                        let min_len = self.rayon_min_len();
                        self.install_parallelism(|| {
                            (0..batch_size)
                                .into_par_iter()
                                .with_min_len(min_len)
                                .for_each(|batch| unsafe {
                                    crate::runtime::cpu::kernels::matmul_bt_kernel::<T>(
                                        (a_ptr as *const T).add(a_batch_idx[batch] * m * k),
                                        (b_ptr as *const T).add(b_batch_idx[batch] * n * k),
                                        (out_ptr as *mut T).add(batch * m * n),
                                        m, n, k, ldc,
                                    );
                                });
                        });
                    } else {
                        unsafe {
                            crate::runtime::cpu::kernels::matmul_bt_kernel::<T>(
                                a_ptr as *const T,
                                b_ptr as *const T,
                                out_ptr as *mut T,
                                m, n, k, ldc,
                            );
                        }
                    }
                }

                #[cfg(not(feature = "rayon"))]
                unsafe {
                    for batch in 0..batch_size {
                        crate::runtime::cpu::kernels::matmul_bt_kernel::<T>(
                            (a_ptr as *const T).add(a_batch_idx[batch] * m * k),
                            (b_ptr as *const T).add(b_batch_idx[batch] * n * k),
                            (out_ptr as *mut T).add(batch * m * n),
                            m, n, k, ldc,
                        );
                    }
                }
            }, "matmul_bt");

            return Ok(out);
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

        // I8 widens to I32 (quantized accumulation). Both matmul forms share
        // that path, so it lives in `matmul_i8.rs`.
        if dtype == DType::I8 {
            return super::matmul_i8::matmul_i8_i32(
                self,
                &a_contig,
                &b_contig,
                None,
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
        let out_ptr = out.ptr();

        // Dispatch based on dtype
        dispatch_dtype!(dtype, T => {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;

                // Same axis rule as the transposed-B path above: columns when
                // they offer more units than the batch axis, batches otherwise,
                // never both.
                if let Some(chunks) = column_chunk_count(batch_size, m, n, k) {
                    for batch in 0..batch_size {
                        unsafe {
                            matmul_columns::<T>(
                                self,
                                (a_ptr as *const T).add(a_batch_idx[batch] * m * k),
                                (b_ptr as *const T).add(b_batch_idx[batch] * k * n),
                                (out_ptr as *mut T).add(batch * m * n),
                                m, n, k, lda, ldb, ldc, chunks,
                            );
                        }
                    }
                } else if batch_size > 1 {
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
