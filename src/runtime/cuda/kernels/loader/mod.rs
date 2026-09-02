//! CUDA kernel loading, caching, and launching infrastructure.
//!
//! Multi-arch fatbins are compiled by `build.rs`, loaded on first use, and
//! cached per-device. The launch helpers below are grouped by kernel family,
//! each with the module selectors and tile constants that family needs.

mod col_transpose1d;
mod dtype_modules;
mod elementwise;
mod gemm_epilogue_wmma;
mod gemv;
mod grouped_matmul;
mod grouped_matmul_tile;
mod im2col;
mod im2col2d;
mod launch_dims;
mod matmul;
mod matmul_bias;
mod matmul_bias_f32;
mod matmul_config;
mod matmul_f32;
mod matmul_fp8;
mod matmul_int;
mod matmul_wmma;
mod matmul_wmma_tile;
mod module_cache;
mod names;
mod reduce_split;
mod semiring_matmul;

pub use col_transpose1d::{col_transpose1d_has_kernel, launch_col_transpose1d};
pub(crate) use dtype_modules::{cumulative_module, reduce_module, unary_module};
pub use elementwise::{launch_binary_kernel, launch_unary_kernel};
pub(crate) use gemm_epilogue_wmma::{
    launch_gemm_bias_act_wmma_batched_kernel, launch_gemm_bias_act_wmma_kernel,
    launch_gemm_bias_residual_wmma_batched_kernel, launch_gemm_bias_residual_wmma_kernel,
};
pub use gemv::{launch_gemv_kernel_bt, launch_gemv_kernel_bt_mr};
pub use grouped_matmul::launch_grouped_matmul;
pub use im2col::{im2col_has_kernel, launch_im2col1d};
pub use im2col2d::{im2col2d_has_kernel, launch_im2col2d};
pub use launch_dims::{
    BLOCK_SIZE, LaunchConfig, MAX_GRID_DIM_YZ, check_shared_mem_fits, elementwise_launch_config,
    launch_config, reduce_dim_launch_config, reduce_launch_config, softmax_launch_config,
};
pub use matmul::{launch_matmul_batched_kernel, launch_matmul_kernel};
pub use matmul_bias::{launch_matmul_bias_batched_kernel, launch_matmul_bias_kernel};
pub use matmul_config::{
    default_tile_config, f32_batched_tile_config, f32_tiled_launch_config, f32_tiled_suffix,
    matmul_batched_launch_config, matmul_launch_config,
};
pub use matmul_int::{int_matmul_has_kernel, int_matmul_output_dtype};
pub(crate) use matmul_wmma::{use_wmma, use_wmma_after_padding};
pub use module_cache::{get_kernel_function, get_or_load_module, preload_modules};
pub use names::{dtype_suffix, kernel_name, kernel_names};
pub(crate) use reduce_split::reduce_split_count;
pub use semiring_matmul::{launch_semiring_matmul_batched_kernel, launch_semiring_matmul_kernel};
