//! CUDA kernel loading, caching, and launching infrastructure.
//!
//! PTX files are compiled by `build.rs`, loaded on first use, and cached
//! per-device. The launch helpers below are grouped by kernel family, each with
//! the module selectors and tile constants that family needs.

mod dtype_modules;
mod elementwise;
mod gemv;
mod launch_dims;
mod matmul;
mod matmul_bias;
mod matmul_config;
mod matmul_f32;
mod matmul_fp8;
mod matmul_int;
mod matmul_wmma;
mod module_cache;
mod names;
mod semiring_matmul;

pub(crate) use dtype_modules::{cumulative_module, reduce_module, unary_module};
pub use elementwise::{launch_binary_kernel, launch_unary_kernel};
pub use gemv::{launch_gemv_kernel_bt, launch_gemv_kernel_bt_mr};
pub use launch_dims::{
    BLOCK_SIZE, LaunchConfig, elementwise_launch_config, launch_config, reduce_dim_launch_config,
    reduce_launch_config, softmax_launch_config,
};
pub use matmul::{launch_matmul_batched_kernel, launch_matmul_kernel};
pub use matmul_bias::{launch_matmul_bias_batched_kernel, launch_matmul_bias_kernel};
pub use matmul_config::{matmul_batched_launch_config, matmul_launch_config};
pub use matmul_int::{int_matmul_has_kernel, int_matmul_output_dtype};
pub use module_cache::{get_kernel_function, get_or_load_module, preload_modules};
pub use names::{dtype_suffix, kernel_name, kernel_names};
pub use semiring_matmul::{launch_semiring_matmul_batched_kernel, launch_semiring_matmul_kernel};
