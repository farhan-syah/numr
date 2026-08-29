//! Index operation WGSL kernel launchers.
//!
//! All operations run entirely on GPU with no CPU fallback.

mod bincount;
mod gather;
mod masked;
mod scatter;
mod scatter_reduce;
mod shader_registry;
mod validate;

pub use bincount::launch_bincount;
pub use gather::{
    launch_embedding_lookup, launch_gather, launch_gather_2d, launch_gather_nd, launch_index_select,
};
pub use masked::{
    launch_masked_count, launch_masked_fill, launch_masked_prefix_sum, launch_masked_select,
};
pub use scatter::{launch_copy, launch_index_put, launch_scatter, launch_slice_assign};
pub use scatter_reduce::{
    launch_scatter_reduce, launch_scatter_reduce_count, launch_scatter_reduce_mean_div,
    launch_scatter_reduce_prod,
};
pub use validate::launch_validate_indices;
