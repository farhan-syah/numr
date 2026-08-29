//! Native GPU operation implementations for WebGPU.
//!
//! This module contains native_* helper functions that launch WGSL compute shaders.
//! Each submodule focuses on a specific category of operations.

mod helpers;

mod activation;
mod binary;
mod cast;
mod compare;
mod conditional;
mod cumulative;
mod fused_elementwise;
mod gemm_epilogue;
mod indexing;
pub(crate) mod logical;
mod masking;
mod matmul;
mod normalization;
mod reduce;
mod scatter_wide;
mod semiring_matmul;
mod unary;

// Re-export all native functions for use by ops/wgpu/ implementations
pub(crate) use activation::{
    native_fused_activation_mul_bwd, native_fused_activation_mul_fwd, native_parametric_activation,
};
pub(crate) use binary::{native_binary_op, native_binary_op_into, native_scalar_op};
pub(crate) use cast::native_cast_op;
pub(crate) use compare::native_compare_op;
pub(crate) use conditional::{native_clamp, native_where_cond};
pub(crate) use cumulative::{native_cumprod, native_cumsum, native_logsumexp};
pub(crate) use fused_elementwise::{
    native_fused_add_mul, native_fused_mul_add, native_fused_mul_add_scalar,
};
pub(crate) use gemm_epilogue::{
    native_gemm_bias_activation, native_gemm_bias_activation_bwd, native_gemm_bias_residual,
};
pub(crate) use indexing::{
    native_gather, native_index_put, native_index_select, native_scatter, native_slice_assign,
};
pub(crate) use masking::{native_embedding_lookup, native_masked_fill, native_masked_select};
pub(crate) use scatter_wide::native_scatter_reduce_int_wide;
pub(crate) mod matmul_broadcast;
pub(crate) use matmul::{native_matmul, native_matmul_bias};
pub(crate) use normalization::{
    native_fused_add_layer_norm, native_fused_add_layer_norm_bwd, native_fused_add_rms_norm,
    native_fused_add_rms_norm_bwd, native_group_norm, native_layer_norm, native_rms_norm,
};
pub(crate) use reduce::{
    native_argreduce_op, native_reduce_op, native_softmax, native_softmax_bwd,
};
pub(crate) use semiring_matmul::native_semiring_matmul;
pub(crate) use unary::native_unary_op;
