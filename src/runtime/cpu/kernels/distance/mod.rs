//! CPU kernels for distance computation.

pub mod acc;
pub mod metrics;
pub mod pairwise;

pub use pairwise::{cdist_kernel, pdist_kernel, squareform_inverse_kernel, squareform_kernel};
