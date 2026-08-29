//! Reduction operation implementations for WebGPU.

mod argreduce;
mod reduce_op;
mod softmax;

pub(crate) use argreduce::native_argreduce_op;
pub(crate) use reduce_op::native_reduce_op;
pub(crate) use softmax::{native_softmax, native_softmax_bwd};
