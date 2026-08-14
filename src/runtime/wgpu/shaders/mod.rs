//! WGSL compute shader infrastructure for WebGPU operations
//!
//! This module provides native WGSL compute shaders for tensor operations.
//! All operations run entirely on the GPU without CPU fallback.
//! Shaders are static `.wgsl` files embedded at compile time via `include_str!()`.
//! WebGPU supports F32, I32, U32 only (no F64/F16/BF16).

pub mod advanced_random;
pub mod complex;
pub mod conv;
pub mod copy;
pub mod cumulative;
pub mod distance;
pub mod distributions;
pub mod dtype_support;
pub mod fft;
pub mod index;
pub mod linalg;
pub mod logical;
pub mod quasirandom;
pub mod shape;
pub mod sort;
pub mod sort_cmp;
pub(crate) mod sort_global;
pub mod special;
pub mod statistics;

// Operation launchers
pub mod activation_launcher;
pub mod elementwise;
pub mod fused_add_norm;
pub mod gemm_epilogue;
pub mod gemm_epilogue_bwd;
pub mod gemv_bt;
pub mod matmul;
pub mod matrix_funcs_launcher;
pub mod norm;
pub mod reduce;
pub mod semiring_matmul;
#[cfg(feature = "sparse")]
pub mod sparse_24;
#[cfg(feature = "sparse")]
pub mod sparse_algorithms_launcher;
#[cfg(feature = "sparse")]
pub mod sparse_conversions_launcher;
#[cfg(feature = "sparse")]
pub mod sparse_level_compute_launcher;
#[cfg(feature = "sparse")]
pub mod sparse_linalg_launcher;
#[cfg(feature = "sparse")]
pub mod sparse_merge_launcher;
#[cfg(feature = "sparse")]
pub mod sparse_spmv_launcher;
pub mod where_launcher;

mod linalg_launchers;
mod linalg_shaders;
mod pipeline;

#[cfg(feature = "sparse")]
/// GPU-native level computation kernels for sparse factorization
pub mod sparse_level_compute {
    pub use crate::runtime::wgpu::shaders::sparse_level_compute_launcher::{
        launch_cast_i64_to_i32, launch_compute_levels_ilu_iter, launch_compute_levels_lower_iter,
        launch_compute_levels_upper_iter, launch_scatter_by_level,
    };
}

pub use activation_launcher::{launch_clamp_op, launch_elu, launch_leaky_relu};
pub mod fused_activation_mul;
pub mod fused_elementwise;
pub use advanced_random::{
    launch_pcg64_randn, launch_pcg64_uniform, launch_philox_randn, launch_philox_uniform,
    launch_threefry_randn, launch_threefry_uniform, launch_xoshiro256_randn,
    launch_xoshiro256_uniform,
};
pub use complex::{
    launch_angle_real, launch_complex_div_real, launch_complex_mul_real, launch_complex_op,
    launch_from_real_imag,
};
pub use conv::{launch_conv1d, launch_conv2d, launch_depthwise_conv2d};
pub use cumulative::{
    launch_cumprod, launch_cumprod_strided, launch_cumsum, launch_cumsum_strided, launch_logsumexp,
    launch_logsumexp_strided,
};
pub use distance::{
    distance_metric_p_value, distance_metric_to_index, launch_cdist, launch_pdist,
    launch_squareform, launch_squareform_inverse,
};
pub use distributions::{
    MultinomialCountParams, launch_bernoulli, launch_beta_dist, launch_binomial,
    launch_chi_squared, launch_exponential, launch_f_distribution, launch_gamma_dist,
    launch_laplace, launch_multinomial_count, launch_poisson, launch_student_t,
};
pub use fused_activation_mul::{
    launch_gelu_mul, launch_gelu_mul_bwd, launch_relu_mul, launch_relu_mul_bwd, launch_sigmoid_mul,
    launch_sigmoid_mul_bwd, launch_silu_mul, launch_silu_mul_bwd,
};
pub use fused_add_norm::{
    launch_fused_add_layer_norm, launch_fused_add_layer_norm_bwd, launch_fused_add_rms_norm,
    launch_fused_add_rms_norm_bwd, launch_reduce_sum_rows,
};
pub use fused_elementwise::{
    launch_fused_add_mul, launch_fused_mul_add, launch_fused_mul_add_scalar,
};
pub use index::{
    launch_bincount, launch_gather_2d, launch_gather_nd, launch_scatter_reduce,
    launch_scatter_reduce_count, launch_scatter_reduce_mean_div, launch_scatter_reduce_prod,
    launch_slice_assign,
};
pub use logical::{launch_logical_and, launch_logical_not, launch_logical_or, launch_logical_xor};
pub use matrix_funcs_launcher::{
    compute_schur_func_gpu, launch_diagonal_func, launch_parlett_column,
    launch_validate_eigenvalues,
};
pub use pipeline::{LayoutKey, PipelineCache, WORKGROUP_SIZE, workgroup_count};
pub use quasirandom::{launch_halton, launch_latin_hypercube, launch_sobol};
#[cfg(feature = "sparse")]
pub use sparse_24::{Sparse24Params, launch_sparse_24_decompress, launch_sparse_24_prune};
#[cfg(feature = "sparse")]
pub use sparse_algorithms_launcher::{
    launch_dsmm_csc, launch_spgemm_accumulate, launch_spgemm_scatter, launch_spgemm_symbolic,
};
#[cfg(feature = "sparse")]
pub use sparse_conversions_launcher::{
    launch_coo_to_csc_scatter, launch_coo_to_csr_scatter, launch_copy_ptrs, launch_count_nonzeros,
    launch_csc_to_csr_scatter, launch_csr_to_csc_scatter, launch_csr_to_dense,
    launch_dense_to_coo_scatter, launch_expand_col_ptrs, launch_expand_row_ptrs, launch_histogram,
};
#[cfg(feature = "sparse")]
pub use sparse_level_compute_launcher::{
    launch_cast_i64_to_i32, launch_compute_levels_ilu_iter, launch_compute_levels_lower_iter,
    launch_compute_levels_upper_iter, launch_scatter_by_level,
};
#[cfg(feature = "sparse")]
pub use sparse_linalg_launcher::{
    launch_extract_lower_count, launch_extract_lower_scatter, launch_split_lu_count,
    launch_split_lu_scatter_l, launch_split_lu_scatter_u,
};
#[cfg(feature = "sparse")]
pub use sparse_merge_launcher::{
    launch_csc_add_compute, launch_csc_div_compute, launch_csc_merge_count, launch_csc_mul_compute,
    launch_csc_mul_count, launch_csc_sub_compute, launch_csr_add_compute, launch_csr_div_compute,
    launch_csr_merge_count, launch_csr_mul_compute, launch_csr_mul_count, launch_csr_sub_compute,
    launch_exclusive_scan_i32,
};
#[cfg(feature = "sparse")]
pub use sparse_spmv_launcher::{launch_csr_extract_diagonal, launch_csr_spmm, launch_csr_spmv};
pub use special::{
    launch_special_binary, launch_special_binary_with_two_ints, launch_special_ternary,
    launch_special_unary, launch_special_unary_with_2f32, launch_special_unary_with_3f32,
    launch_special_unary_with_int, launch_special_unary_with_two_ints,
};
pub use statistics::{launch_mode_dim, launch_mode_full};
pub use where_launcher::{launch_where_broadcast_op, launch_where_generic_op, launch_where_op};
