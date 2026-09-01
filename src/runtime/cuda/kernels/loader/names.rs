//! Kernel and PTX module names.
//!
//! `kernel_names` holds the PTX module name constants; `dtype_suffix` and
//! `kernel_name` build the `{op}_{dtype}` names the `.cu` files instantiate.

use crate::dtype::DType;

/// Kernel operation categories for consistent naming.
pub mod kernel_names {
    /// Binary operations (two tensor inputs)
    pub const BINARY_MODULE: &str = "binary";
    /// Unary operations (one tensor input)
    pub const UNARY_MODULE: &str = "unary";
    /// Integer unary operations, which wrap in the dtype instead of computing
    /// in a float register
    pub const UNARY_INT_MODULE: &str = "unary_int";
    /// Scalar operations (tensor + scalar input)
    pub const SCALAR_MODULE: &str = "scalar";
    /// Reduction operations (sum, max, min)
    pub const REDUCE_MODULE: &str = "reduce";
    /// Integer reduction operations, which accumulate in `Numr128`
    pub const REDUCE_INT_MODULE: &str = "reduce_int";
    /// Comparison operations (eq, ne, lt, le, gt, ge)
    pub const COMPARE_MODULE: &str = "compare";
    /// Element-wise activation functions (relu, sigmoid, silu, gelu, leaky_relu, elu)
    pub const ACTIVATION_MODULE: &str = "activation";
    /// Softmax forward + backward kernels
    pub const SOFTMAX_MODULE: &str = "softmax";
    /// Normalization operations (rms_norm, layer_norm)
    pub const NORM_MODULE: &str = "norm";
    /// Fused add + normalization operations
    pub const FUSED_ADD_NORM_MODULE: &str = "fused_add_norm";
    /// Type casting operations (cast between dtypes)
    pub const CAST_MODULE: &str = "cast";
    /// Utility operations (fill, arange, linspace, eye)
    pub const UTILITY_MODULE: &str = "utility";
    /// Random sampling operations (rand, randn, randint, multinomial)
    pub const UTILITY_RANDOM_MODULE: &str = "utility_random";
    /// Coordinate-addressed indexing (gather_nd, gather_2d, slice_assign)
    pub const INDEX_ND_MODULE: &str = "index_nd";
    /// Scatter with reduction (sum, prod, max, min, mean)
    pub const SCATTER_REDUCE_MODULE: &str = "scatter_reduce";
    /// Ternary operations (where)
    pub const TERNARY_MODULE: &str = "ternary";
    /// Prefix sum operations (exclusive scan)
    #[cfg(feature = "sparse")]
    pub const SCAN_MODULE: &str = "scan";
    /// Sparse matrix operations (SpMV, SpMM)
    #[cfg(feature = "sparse")]
    pub const SPARSE_SPMV_MODULE: &str = "sparse_spmv";
    /// Sparse matrix element-wise operations (add, sub, mul)
    #[cfg(feature = "sparse")]
    pub const SPARSE_MERGE_MODULE: &str = "sparse_merge";
    /// Sparse format conversion operations (COO↔CSR↔CSC)
    #[cfg(feature = "sparse")]
    pub const SPARSE_CONVERT_MODULE: &str = "sparse_convert";
    /// COO sparse element-wise operations with CUB sort
    #[cfg(feature = "sparse")]
    pub const SPARSE_COO_MODULE: &str = "sparse_coo";
    /// Dense × Sparse matrix multiplication (DSMM / SpMM)
    #[cfg(feature = "sparse")]
    pub const DSMM_MODULE: &str = "dsmm";
    /// Linear algebra basic operations (trace, diag, diagflat, identity, transpose)
    pub const LINALG_BASIC_MODULE: &str = "linalg_basic";
    /// Banded linear system solvers (Thomas, banded LU)
    pub const LINALG_BANDED_MODULE: &str = "linalg_banded";
    /// Linear algebra solvers (forward_sub, backward_sub, det_from_lu, apply_permutation)
    pub const LINALG_SOLVERS_MODULE: &str = "linalg_solvers";
    /// Matrix decompositions (LU, Cholesky, QR)
    pub const LINALG_DECOMP_MODULE: &str = "linalg_decomp";
    /// SVD decomposition (Jacobi algorithm)
    pub const LINALG_SVD_MODULE: &str = "linalg_svd";
    /// Symmetric eigenvalue decomposition (Jacobi algorithm)
    pub const LINALG_EIGEN_MODULE: &str = "linalg_eigen";
    /// Schur decomposition (Hessenberg + QR iteration)
    pub const LINALG_SCHUR_MODULE: &str = "linalg_schur";
    /// General eigenvalue decomposition
    pub const LINALG_EIGEN_GENERAL_MODULE: &str = "linalg_eigen_general";
    /// Advanced decompositions (rsf2csf)
    pub const LINALG_ADVANCED_MODULE: &str = "linalg_advanced";
    /// QZ decomposition (generalized Schur - double-shift algorithm)
    pub const LINALG_QZ_MODULE: &str = "linalg_qz";
    /// Matrix functions (exp, log, sqrt on quasi-triangular matrices)
    pub const LINALG_MATRIX_FUNCS_MODULE: &str = "linalg_matrix_funcs";
    /// Matrix multiplication operations (native tiled GEMM)
    pub const MATMUL_MODULE: &str = "matmul";
    /// Tensor-core WMMA GEMM for F16/BF16 (sm_70+)
    pub const MATMUL_WMMA_MODULE: &str = "matmul_wmma";
    /// GEMV operations (matrix-vector multiply for small M)
    pub const GEMV_MODULE: &str = "gemv";
    /// Integer GEMV operations, which accumulate in `Numr128`
    pub const GEMV_INT_MODULE: &str = "gemv_int";
    /// Compile-time-tiled integer GEMM
    pub const MATMUL_INT_MODULE: &str = "matmul_int";
    /// Compile-time-tiled FP8 GEMM (FP8E4M3, FP8E5M2), F32 accumulation
    pub const MATMUL_FP8_MODULE: &str = "matmul_fp8";
    /// Cumulative operations (cumsum, cumprod, logsumexp)
    pub const CUMULATIVE_MODULE: &str = "cumulative";
    /// Integer cumulative operations, which accumulate in `Numr128`
    pub const CUMULATIVE_INT_MODULE: &str = "cumulative_int";
    /// Distribution sampling operations (bernoulli, beta, gamma, etc.)
    pub const DISTRIBUTIONS_MODULE: &str = "distributions";
    /// Quasi-random sequence generation (sobol, halton, latin_hypercube)
    pub const QUASIRANDOM_MODULE: &str = "quasirandom";
    /// Advanced PRNGs (philox, threefry, pcg64, xoshiro256)
    pub const ADVANCED_RANDOM_MODULE: &str = "advanced_random";
    /// Statistics operations (mode)
    pub const STATISTICS_MODULE: &str = "statistics";
    /// Semiring matrix multiplication operations
    pub const SEMIRING_MATMUL_MODULE: &str = "semiring_matmul";
    /// conv1d im2col gather (packs receptive fields into a GEMM operand)
    pub const IM2COL_MODULE: &str = "im2col";
    /// conv_transpose1d column gather (packs contributing samples for a GEMM)
    pub const COL_TRANSPOSE1D_MODULE: &str = "col_transpose1d";

    /// Generate kernel name for reduction operations.
    #[inline]
    pub fn reduce_kernel(op: &str) -> String {
        format!("reduce_{}", op)
    }

    /// Generate kernel name for dimension-wise reduction operations.
    #[inline]
    pub fn reduce_dim_kernel(op: &str) -> String {
        format!("reduce_{}_dim", op)
    }
}

/// Get the kernel name suffix for a given dtype.
pub fn dtype_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::FP8E4M3 => "fp8_e4m3",
        DType::FP8E5M2 => "fp8_e5m2",
        DType::I64 => "i64",
        DType::I32 => "i32",
        DType::I16 => "i16",
        DType::I8 => "i8",
        DType::U64 => "u64",
        DType::U32 => "u32",
        DType::U16 => "u16",
        DType::U8 => "u8",
        DType::Bool => "bool",
        DType::Complex64 => "c64",
        DType::Complex128 => "c128",
    }
}

/// Generate a kernel name with dtype suffix.
///
/// # Example
///
/// ```ignore
/// let name = kernel_name("add", DType::F32); // "add_f32"
/// ```
#[inline]
pub fn kernel_name(base: &str, dtype: DType) -> String {
    format!("{}_{}", base, dtype_suffix(dtype))
}
