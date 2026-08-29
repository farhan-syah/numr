// Backend parity tests for MatmulOps trait.
//
// `float` holds the dtype-parameterized coverage, now over the whole numeric
// domain. The CPU backend scope stops at 32 bits, so `integer_cuda` holds the
// hand-built CUDA-vs-CPU I64 coverage that reaches past it, and
// `integer_gemv_cuda` covers the small-M GEMV kernels those shapes route to.

pub mod float;
pub mod integer_cuda;
pub mod integer_gemv_cuda;
