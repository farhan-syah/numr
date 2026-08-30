//! Helper functions for CPU tensor operations
//!
//! This module contains shared helper functions used by operation implementations,
//! organized by functional category.

pub mod activation;
pub mod binary;
pub mod bincount;
pub mod compare;
pub mod cumulative;
pub mod fused_elementwise;
pub mod indexing;
pub mod reduce;
pub mod scalar;
pub mod shape;
pub mod unary;

// Re-export all helper functions
pub use activation::{
    ActivationOp, FusedActivationMulOp, activation_op_impl, elu_impl, fused_activation_mul_impl,
    leaky_relu_impl,
};
pub use binary::{binary_op_impl, binary_op_into_impl};
pub use bincount::{bincount_impl, bincount_with_len_impl};
pub use compare::compare_op_impl;
pub use cumulative::{cumprod_impl, cumsum_impl, logsumexp_impl};
pub use fused_elementwise::{fused_add_mul_impl, fused_mul_add_impl, fused_mul_add_scalar_impl};
pub use indexing::{
    embedding_lookup_impl, gather_2d_impl, gather_impl, gather_nd_impl, index_put_impl,
    index_select_impl, masked_fill_impl, masked_select_impl, scatter_impl, scatter_reduce_impl,
    slice_assign_impl,
};
pub use reduce::{reduce_impl, reduce_impl_with_precision};
pub use scalar::scalar_op_impl;
pub use shape::{cat_impl, chunk_impl, pad_impl, repeat_impl, roll_impl, split_impl, stack_impl};
pub use unary::unary_op_impl;

// Re-export operation types used by callers
pub use crate::ops::{BinaryOp, UnaryOp};

// Re-export dispatch macro and shared helpers
pub use crate::dispatch_dtype;
pub(crate) use crate::runtime::ensure_contiguous;
