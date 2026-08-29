//! Semiring matrix multiplication for CUDA runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::SemiringMatmulOps;
use crate::ops::matmul_output_shape;
use crate::ops::semiring::SemiringOp;
use crate::runtime::cuda::ops::helpers::{semiring_matmul_batched_native, semiring_matmul_native};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::validate_binary_dtypes;
use crate::tensor::Tensor;

/// Map SemiringOp to the u32 op code used by the CUDA kernel.
fn semiring_op_code(op: SemiringOp) -> u32 {
    match op {
        SemiringOp::MinPlus => 0,
        SemiringOp::MaxPlus => 1,
        SemiringOp::MaxMin => 2,
        SemiringOp::MinMax => 3,
        SemiringOp::OrAnd => 4,
        SemiringOp::PlusMax => 5,
    }
}

impl SemiringMatmulOps<CudaRuntime> for CudaClient {
    fn semiring_matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        op: SemiringOp,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_binary_dtypes(a, b)?;

        if !op.validate_dtype(dtype) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "semiring_matmul",
            });
        }

        // Supported CUDA kernel dtypes
        match dtype {
            DType::F32 | DType::F64 | DType::I32 | DType::Bool | DType::U8 => {}
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {}
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 | DType::FP8E5M2 => {}
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "CUDA semiring_matmul",
                });
            }
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

        let op_code = semiring_op_code(op);

        // Bool uses the u8 kernel (same underlying type)
        let kernel_dtype = if dtype == DType::Bool {
            DType::U8
        } else {
            dtype
        };

        if batch_size > 1 {
            semiring_matmul_batched_native(self, a, b, kernel_dtype, batch_size, m, k, n, op_code)
        } else {
            semiring_matmul_native(self, a, b, kernel_dtype, m, k, n, op_code)
        }
    }
}
