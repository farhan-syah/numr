//! Index operation kernels (gather, scatter, masked operations).

pub mod bincount;
pub mod embedding;
pub mod gather;
pub mod masked;
pub mod scatter;
pub mod select;

pub use bincount::{bincount_kernel, max_i64_kernel};
pub use embedding::embedding_lookup_kernel;
pub use gather::{gather_2d_kernel, gather_kernel, gather_nd_kernel};
pub use masked::{masked_fill_kernel, masked_select_kernel};
// Only `masked_select`'s non-x86_64 arm calls this; the x86_64 arm takes its
// count from the SIMD kernel. Gated to match that caller, so neither target
// carries an unused import.
#[cfg(not(target_arch = "x86_64"))]
pub use masked::masked_count_kernel;
pub use scatter::{scatter_kernel, scatter_reduce_kernel};
pub use select::{index_put_kernel, index_select_kernel, slice_assign_kernel};
