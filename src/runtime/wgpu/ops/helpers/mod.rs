//! Helper functions and parameter structs for WebGPU operations.
//!
//! Buffer plumbing lives in `buffer`, `broadcast` and `indices`; every other
//! module holds the params structs for one kernel family, laid out to match the
//! matching WGSL struct byte for byte.

mod broadcast;
mod buffer;
mod creation_params;
mod cumulative_params;
mod distribution_params;
mod elementwise_params;
mod indexing_params;
mod indices;
mod matmul_params;
mod norm_params;
mod quasirandom_params;
mod reduce_params;
mod rng;
mod shape_params;
mod sort_params;

pub(crate) use broadcast::*;
pub(crate) use buffer::*;
pub(crate) use creation_params::*;
pub(crate) use cumulative_params::*;
pub(crate) use distribution_params::*;
pub(crate) use elementwise_params::*;
pub(crate) use indexing_params::*;
pub(crate) use indices::*;
pub(crate) use matmul_params::*;
pub(crate) use norm_params::*;
pub(crate) use quasirandom_params::*;
pub(crate) use reduce_params::*;
pub(crate) use rng::*;
pub(crate) use shape_params::*;
pub(crate) use sort_params::*;
