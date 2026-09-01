//! CUDA-specific helper functions for kernel launching and tensor operations.
//!
//! One module per operation family. Each prepares operands (contiguity,
//! broadcasting, output dtype and shape) and launches the matching kernel.

mod binary;
mod compare;
mod gemm_epilogue;
mod matmul;
mod matmul_bias;
mod reduce;
mod scalar;
mod semiring_matmul;
mod unary;

pub(crate) use binary::{native_binary_op, native_binary_op_into};
pub(crate) use compare::native_compare_op;
pub(crate) use gemm_epilogue::{
    gemm_bias_act_batched_native, gemm_bias_act_native, gemm_bias_residual_batched_native,
    gemm_bias_residual_native,
};
pub(crate) use matmul::{matmul_batched_native, matmul_native};
pub(crate) use matmul_bias::{matmul_bias_batched_native, matmul_bias_native};
pub(crate) use reduce::native_reduce_op;
pub(crate) use scalar::native_scalar_op;
pub(crate) use semiring_matmul::{semiring_matmul_batched_native, semiring_matmul_native};
pub(crate) use unary::native_unary_op;
