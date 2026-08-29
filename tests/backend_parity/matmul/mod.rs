// Backend parity tests for MatmulOps trait.
//
// `float` holds the dtype-parameterized coverage, now over the whole numeric
// domain. The CPU backend scope stops at 32 bits, so `integer_cuda` holds the
// hand-built CUDA-vs-CPU signed coverage that reaches past it, and
// `integer_gemv_cuda` covers the small-M GEMV kernels those shapes route to.
// `integer_dtypes_cuda` holds the unsigned widths, where the widening into the
// 128-bit accumulator must zero-extend rather than sign-extend.
//
// `i8_cuda` stands apart from both: I8 is the one width whose plain `matmul`
// widens its output to I32 while its fused-bias form stays I8, so its parity
// tests assert the output dtype as well as the values.
//
// `integer_wgpu` pins the WebGPU I32/U32 kernels at the accumulator boundary:
// their operands stay 32-bit, but WGSL has no 64-bit integer, so the accumulator
// they build out of 32-bit limbs needs coverage the small operands in `float`
// cannot give.

pub mod float;
pub mod i8_cuda;
pub mod integer_cuda;
pub mod integer_dtypes_cuda;
pub mod integer_gemv_cuda;
pub mod integer_wgpu;
