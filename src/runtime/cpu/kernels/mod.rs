//! CPU kernel implementations
//!
//! This module provides low-level compute kernels for CPU operations.
//! Kernels are generic over `T: Element` and dispatch based on operation type.

#![allow(unsafe_op_in_unsafe_fn)] // Kernels are already marked unsafe, inner unsafe is redundant

pub mod advanced_random;
pub mod binary;
pub mod compare;
pub mod complex;
pub mod conv;
pub mod conv_transpose;
pub mod cumulative;
pub mod distance;
pub mod distributions;
pub mod fft;
pub mod fused_add_norm;
pub mod fused_elementwise;
pub mod gemm_epilogue;
pub mod index;
pub mod ipow;
pub mod logical;
pub mod matmul;
pub mod matmul_i8;
pub mod memory;
pub mod norm;
pub mod quasirandom;
pub mod reduce;
pub(crate) mod rng;
pub mod scalar;
pub mod semiring_matmul;
pub mod simd;
pub mod sobol_data;
pub mod sort;
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "sparse")]
pub mod sparse_24;
pub mod unary;
pub mod where_select;
pub mod wide_acc;

// Re-export all kernel functions for convenient access
pub use advanced_random::{
    pcg64_randn_kernel, pcg64_uniform_kernel, philox_randn_kernel, philox_uniform_kernel,
    threefry_randn_kernel, threefry_uniform_kernel, xoshiro256_randn_kernel,
    xoshiro256_uniform_kernel,
};
pub use binary::{binary_op_kernel, binary_op_strided_kernel};
pub use compare::{compare_op_kernel, compare_op_strided_kernel};
pub use complex::{
    angle_complex64, angle_complex128, angle_real_f32, angle_real_f64, complex64_div_real,
    complex64_mul_real, complex128_div_real, complex128_mul_real, conj_complex64, conj_complex128,
    from_real_imag_f32, from_real_imag_f64, imag_complex64, imag_complex128, real_complex64,
    real_complex128,
};
pub use conv::{conv1d_kernel, conv2d_kernel, depthwise_conv2d_kernel};
pub use conv_transpose::conv_transpose1d_kernel;
pub use cumulative::{
    cumprod_kernel, cumprod_strided_kernel, cumsum_kernel, cumsum_strided_kernel, logsumexp_kernel,
    logsumexp_strided_kernel,
};
pub use distance::{cdist_kernel, pdist_kernel, squareform_inverse_kernel, squareform_kernel};
pub use distributions::{
    bernoulli_kernel, beta_kernel, binomial_kernel, chi_squared_kernel, exponential_kernel,
    f_distribution_kernel, gamma_kernel, laplace_kernel, poisson_kernel, student_t_kernel,
};
pub use fft::{
    fftshift_c64, fftshift_c128, ifftshift_c64, ifftshift_c128, irfft_c64, irfft_c128, rfft_c64,
    rfft_c128, stockham_fft_batched_c64, stockham_fft_batched_c128,
};
pub use fused_add_norm::{
    fused_add_layer_norm_bwd_kernel, fused_add_layer_norm_kernel, fused_add_rms_norm_bwd_kernel,
    fused_add_rms_norm_kernel,
};
pub use fused_elementwise::{
    fused_add_mul_kernel, fused_mul_add_kernel, fused_mul_add_scalar_kernel,
};
pub use gemm_epilogue::{
    matmul_bias_activation_bwd_kernel, matmul_bias_activation_kernel, matmul_bias_residual_kernel,
};
pub use index::{
    bincount_kernel, embedding_lookup_kernel, gather_2d_kernel, gather_kernel, gather_nd_kernel,
    index_put_kernel, index_select_kernel, masked_fill_kernel, masked_select_kernel,
    max_i64_kernel, scatter_kernel, scatter_reduce_kernel, slice_assign_kernel,
};
pub use logical::{logical_and_kernel, logical_not_kernel, logical_or_kernel, logical_xor_kernel};
pub use matmul::{
    gemv_bt_kernel, matmul_bias_kernel, matmul_bt_kernel, matmul_bt_matches_contiguous,
    matmul_kernel,
};
pub use matmul_i8::matmul_i8_to_i32_kernel;
pub use memory::{
    arange_kernel, cast_kernel, copy_kernel, eye_kernel, fill_kernel, linspace_kernel,
    multinomial_kernel_with_replacement, multinomial_kernel_without_replacement, one_hot_kernel,
    rand_normal_kernel, rand_uniform_kernel, randint_kernel, randperm_kernel,
};
pub use norm::{group_norm_kernel, layer_norm_kernel, rms_norm_kernel};
pub use quasirandom::{
    halton_f32, halton_f64, latin_hypercube_f32, latin_hypercube_f64, sobol_f32, sobol_f64,
};
pub use reduce::{
    Accumulator, argmax_kernel, argmin_kernel, reduce_kernel, reduce_kernel_with_precision,
    softmax_bwd_kernel, softmax_kernel, variance_kernel,
};
pub use scalar::{rsub_scalar_kernel, scalar_op_kernel};
pub use sort::{
    argsort_kernel, count_nonzero_kernel, count_unique_kernel, extract_unique_kernel,
    flat_to_multi_index_kernel, nonzero_flat_kernel, searchsorted_kernel, sort_kernel,
    sort_values_kernel, topk_kernel, unique_with_counts_kernel,
};
pub use unary::{
    clamp_kernel, elu_kernel, gelu_kernel, gelu_mul_kernel, isinf_kernel, isnan_kernel,
    leaky_relu_kernel, relu_kernel, relu_mul_kernel, sigmoid_kernel, sigmoid_mul_kernel,
    silu_kernel, silu_mul_kernel, unary_op_kernel,
};
pub use where_select::{
    where_kernel, where_kernel_generic, where_strided_kernel, where_strided_kernel_generic,
};

// Re-export SIMD dot product kernels for downstream crates (e.g., boostr quantized ops)
#[allow(unused_imports)]
pub use simd::dot::{i8xi8_dot_f32, i8xi8_dot_i32};

// Re-export sparse kernel functions for external use
#[cfg(feature = "sparse")]
#[allow(unused_imports)]
pub use sparse::{
    divide_by_pivot, find_pivot, find_pivot_range, gather_and_clear, gather_and_clear_i32,
    scatter_column, scatter_column_i32, sparse_axpy, sparse_axpy_i32, swap_rows,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{BinaryOp, ReduceOp, UnaryOp};

    #[test]
    fn test_binary_add() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];

        unsafe {
            binary_op_kernel(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_binary_mul() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 4];

        unsafe {
            binary_op_kernel(BinaryOp::Mul, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_unary_neg() {
        let a = [1.0f32, -2.0, 3.0, -4.0];
        let mut out = [0.0f32; 4];

        unsafe {
            unary_op_kernel(UnaryOp::Neg, a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_unary_sqrt() {
        let a = [1.0f32, 4.0, 9.0, 16.0];
        let mut out = [0.0f32; 4];

        unsafe {
            unary_op_kernel(UnaryOp::Sqrt, a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu() {
        let a = [-1.0f32, 0.0, 1.0, -2.0];
        let mut out = [0.0f32; 4];

        unsafe {
            relu_kernel(a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_sigmoid() {
        let a = [0.0f32];
        let mut out = [0.0f32; 1];

        unsafe {
            sigmoid_kernel(a.as_ptr(), out.as_mut_ptr(), 1);
        }

        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_2x2() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        unsafe {
            matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2);
        }

        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_3x2_2x4() {
        // A = [[1, 2], [3, 4], [5, 6]] (3x2)
        // B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
        // C = A @ B (3x4)
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 12];

        unsafe {
            matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 4, 2, 2, 4, 4);
        }

        // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
        // Row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
        // Row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
        assert_eq!(
            c,
            [
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_reduce_sum() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 2];

        unsafe {
            // Reduce 3 elements per output, 2 outputs
            reduce_kernel(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_reduce_mean() {
        let a = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut out = [0.0f32; 2];

        unsafe {
            reduce_kernel(ReduceOp::Mean, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [2.0, 20.0]); // [6/3, 60/3]
    }

    #[test]
    fn test_reduce_max() {
        let a = [1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0];
        let mut out = [0.0f32; 2];

        unsafe {
            reduce_kernel(ReduceOp::Max, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [5.0, 8.0]);
    }

    #[test]
    fn test_softmax() {
        let a = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];

        unsafe {
            softmax_kernel(a.as_ptr(), out.as_mut_ptr(), 1, 3);
        }

        // Check that outputs sum to 1
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check monotonicity: out[0] < out[1] < out[2]
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_fill() {
        let mut out = [0.0f32; 4];

        unsafe {
            fill_kernel(out.as_mut_ptr(), 7.5f32, 4);
        }

        assert_eq!(out, [7.5, 7.5, 7.5, 7.5]);
    }

    #[test]
    fn test_copy() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];

        unsafe {
            copy_kernel(src.as_ptr(), dst.as_mut_ptr(), 4);
        }

        assert_eq!(dst, src);
    }

    #[test]
    fn test_i32_binary_add() {
        let a = [1i32, 2, 3, 4];
        let b = [5i32, 6, 7, 8];
        let mut out = [0i32; 4];

        unsafe {
            binary_op_kernel(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [6, 8, 10, 12]);
    }
}
