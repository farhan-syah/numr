//! CPU implementation of semiring matrix multiplication.

use crate::error::{Error, Result};
use crate::ops::SemiringMatmulOps;
use crate::ops::matmul_output_shape;
use crate::ops::semiring::SemiringOp;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels::semiring_matmul::semiring_matmul_kernel,
};
use crate::tensor::Tensor;

impl SemiringMatmulOps<CpuRuntime> for CpuClient {
    fn semiring_matmul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        op: SemiringOp,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        let dtype = a.dtype();

        // Validate dtype is supported for this semiring
        if !op.validate_dtype(dtype) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "semiring_matmul",
            });
        }

        // Compute output shape (reuses standard matmul shape logic)
        let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        })?;

        // Get matrix dimensions
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Ensure contiguous layout
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;

        // Calculate batch size from output shape and per-input batch counts
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Batch dims broadcast per dimension, so each output batch needs its own
        // source index per operand rather than a wrapping batch count.
        let (a_batch_idx, b_batch_idx) =
            crate::ops::matmul::matmul_batch_indices(a_shape, b_shape, &out_shape);

        // Create output tensor
        let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &self.device)?;

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let out_ptr = out.ptr();

        let lda = k;
        let ldb = n;
        let ldc = n;

        // Handle OrAnd separately for Bool dtype
        if op == SemiringOp::OrAnd {
            // Bool is stored as u8 internally
            unsafe {
                for batch in 0..batch_size {
                    let a_offset = a_batch_idx[batch] * m * k;
                    let b_offset = b_batch_idx[batch] * k * n;
                    let out_offset = batch * m * n;

                    or_and_kernel(
                        (a_ptr as *const u8).add(a_offset),
                        (b_ptr as *const u8).add(b_offset),
                        (out_ptr as *mut u8).add(out_offset),
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

        // Dispatch based on dtype for numeric types
        dispatch_dtype!(dtype, T => {
            unsafe {
                for batch in 0..batch_size {
                    let a_offset = a_batch_idx[batch] * m * k;
                    let b_offset = b_batch_idx[batch] * k * n;
                    let out_offset = batch * m * n;

                    semiring_matmul_kernel::<T>(
                        (a_ptr as *const T).add(a_offset),
                        (b_ptr as *const T).add(b_offset),
                        (out_ptr as *mut T).add(out_offset),
                        m, n, k, lda, ldb, ldc,
                        op,
                    );
                }
            }
        }, "semiring_matmul");

        Ok(out)
    }
}

/// Boolean OR-AND kernel for transitive closure.
///
/// # Safety
/// Pointers must be valid for the specified dimensions.
#[allow(clippy::too_many_arguments)]
unsafe fn or_and_kernel(
    a: *const u8,
    b: *const u8,
    out: *mut u8,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut any = false;
            for kk in 0..k {
                let av = unsafe { *a.add(i * lda + kk) != 0 };
                let bv = unsafe { *b.add(kk * ldb + j) != 0 };
                if av && bv {
                    any = true;
                    break; // Short-circuit: OR found a true
                }
            }
            unsafe { *out.add(i * ldc + j) = if any { 1 } else { 0 } };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    fn make_client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    #[test]
    fn test_semiring_min_plus_2x2() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 3.0, 7.0, 1.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 2.0, 5.0, 0.0], &[2, 2], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::MinPlus).unwrap();
        assert_eq!(c.to_vec::<f32>(), vec![0.0, 2.0, 6.0, 1.0]);
    }

    #[test]
    fn test_semiring_max_plus_2x2() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 3.0, 7.0, 1.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 2.0, 5.0, 0.0], &[2, 2], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::MaxPlus).unwrap();
        assert_eq!(c.to_vec::<f32>(), vec![8.0, 3.0, 7.0, 9.0]);
    }

    #[test]
    fn test_semiring_max_min_2x2() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 3.0, 2.0, 8.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 1.0, 6.0, 7.0], &[2, 2], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::MaxMin).unwrap();
        assert_eq!(c.to_vec::<f32>(), vec![4.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn test_semiring_or_and() {
        let (client, device) = make_client();
        // Adjacency: A = [[T, F], [F, T]], B = [[F, T], [T, F]]
        // C[0,0] = (T&F)|(F&T) = F|F = F
        // C[0,1] = (T&T)|(F&F) = T|F = T
        // C[1,0] = (F&F)|(T&T) = F|T = T
        // C[1,1] = (F&T)|(T&F) = F|F = F
        let a = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 0, 1], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[0u8, 1, 1, 0], &[2, 2], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::OrAnd).unwrap();
        assert_eq!(c.to_vec::<u8>(), vec![0, 1, 1, 0]);
    }

    #[test]
    fn test_semiring_batched() {
        let (client, device) = make_client();
        // Batch of 2, each 2x2
        let a = Tensor::<CpuRuntime>::from_slice(
            &[0.0f32, 3.0, 7.0, 1.0, 1.0, 2.0, 3.0, 4.0],
            &[2, 2, 2],
            &device,
        );
        let b = Tensor::<CpuRuntime>::from_slice(
            &[0.0f32, 2.0, 5.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            &[2, 2, 2],
            &device,
        );
        let c = client.semiring_matmul(&a, &b, SemiringOp::MinPlus).unwrap();
        let result = c.to_vec::<f32>();
        // Batch 0: same as test above: [0, 2, 6, 1]
        // Batch 1: A=[[1,2],[3,4]], B=[[1,0],[0,1]]
        // C[0,0]=min(1+1, 2+0)=min(2,2)=2
        // C[0,1]=min(1+0, 2+1)=min(1,3)=1
        // C[1,0]=min(3+1, 4+0)=min(4,4)=4
        // C[1,1]=min(3+0, 4+1)=min(3,5)=3
        assert_eq!(result, vec![0.0, 2.0, 6.0, 1.0, 2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_semiring_non_square() {
        let (client, device) = make_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(
            &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 2],
            &device,
        );
        let c = client.semiring_matmul(&a, &b, SemiringOp::MinPlus).unwrap();
        assert_eq!(c.to_vec::<f32>(), vec![8.0, 9.0, 11.0, 12.0]);
    }

    #[test]
    fn test_semiring_dtype_mismatch() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        assert!(client.semiring_matmul(&a, &b, SemiringOp::MinPlus).is_err());
    }

    #[test]
    fn test_semiring_invalid_dtype_for_or_and() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[1, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2, 1], &device);
        assert!(client.semiring_matmul(&a, &b, SemiringOp::OrAnd).is_err());
    }

    #[test]
    fn test_semiring_f64() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 3.0, 7.0, 1.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 2.0, 5.0, 0.0], &[2, 2], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::MinPlus).unwrap();
        assert_eq!(c.to_vec::<f64>(), vec![0.0, 2.0, 6.0, 1.0]);
    }

    #[test]
    fn test_semiring_i32() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[0i32, 3, 7, 1], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[0i32, 2, 5, 0], &[2, 2], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::MinPlus).unwrap();
        assert_eq!(c.to_vec::<i32>(), vec![0, 2, 6, 1]);
    }

    #[test]
    fn test_semiring_1x1() {
        let (client, device) = make_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1, 1], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::MinPlus).unwrap();
        assert_eq!(c.to_vec::<f32>(), vec![8.0]);
    }

    #[test]
    fn test_semiring_plus_max() {
        let (client, device) = make_client();
        // PlusMax = (+, max): C[i,j] = Σ_k max(A[i,k], B[k,j])
        // A = [[1, 3], [2, 4]], B = [[5, 2], [1, 6]]
        // C[0,0] = max(1,5) + max(3,1) = 5 + 3 = 8
        // C[0,1] = max(1,2) + max(3,6) = 2 + 6 = 8
        // C[1,0] = max(2,5) + max(4,1) = 5 + 4 = 9
        // C[1,1] = max(2,2) + max(4,6) = 2 + 6 = 8
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 2.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 2.0, 1.0, 6.0], &[2, 2], &device);
        let c = client.semiring_matmul(&a, &b, SemiringOp::PlusMax).unwrap();
        assert_eq!(c.to_vec::<f32>(), vec![8.0, 8.0, 9.0, 8.0]);
    }
}
