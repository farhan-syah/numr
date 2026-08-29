// Floating-point cumulative CUDA kernels: cumsum, cumprod, logsumexp.
//
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2.
//
// Integer cumsum/cumprod live in `cumulative_int.cu` (PTX module
// "cumulative_int", see `cumulative_module` in loader.rs): they accumulate in
// `Numr128` rather than in a float register, and there is no integer
// logsumexp. This file is PTX module "cumulative"
// (kernel_names::CUMULATIVE_MODULE).
//
// Kernel naming, matching the names the Rust launchers build in
// src/runtime/cuda/kernels/cumulative.rs from dtype_suffix() in loader.rs:
//   cumsum_{suffix}              scan along the last dimension
//   cumsum_strided_{suffix}      scan along a non-last dimension
//   cumprod_{suffix}             /  cumprod_strided_{suffix}
//   logsumexp_{suffix}           /  logsumexp_strided_{suffix}
//
// The storage policies, the accumulator choice, and the three loop bodies all
// live in cumulative_ops.cuh.

#include "cumulative_ops.cuh"

// One dtype's six kernels. `P` is the storage policy, `S` the element type in
// the signature, `SUF` the kernel-name suffix.
#define NUMR_CUMULATIVE_ROW(P, S, SUF)                                          \
    __global__ void cumsum_##SUF(                                               \
        const S* in, S* out, unsigned int scan_size, unsigned int outer_size    \
    ) { cum_simple_impl<P, false>(in, out, scan_size, outer_size); }            \
    __global__ void cumsum_strided_##SUF(                                       \
        const S* in, S* out, unsigned int scan_size, unsigned int outer_size,   \
        unsigned int inner_size                                                 \
    ) { cum_strided_impl<P, false>(in, out, scan_size, outer_size, inner_size); } \
    __global__ void cumprod_##SUF(                                              \
        const S* in, S* out, unsigned int scan_size, unsigned int outer_size    \
    ) { cum_simple_impl<P, true>(in, out, scan_size, outer_size); }             \
    __global__ void cumprod_strided_##SUF(                                      \
        const S* in, S* out, unsigned int scan_size, unsigned int outer_size,   \
        unsigned int inner_size                                                 \
    ) { cum_strided_impl<P, true>(in, out, scan_size, outer_size, inner_size); } \
    __global__ void logsumexp_##SUF(                                            \
        const S* in, S* out, unsigned int reduce_size, unsigned int outer_size  \
    ) { logsumexp_simple_impl<P>(in, out, reduce_size, outer_size); }           \
    __global__ void logsumexp_strided_##SUF(                                    \
        const S* in, S* out, unsigned int reduce_size, unsigned int outer_size, \
        unsigned int inner_size                                                 \
    ) { logsumexp_strided_impl<P>(in, out, reduce_size, outer_size, inner_size); }

extern "C" {

NUMR_CUMULATIVE_ROW(CumF32, float, f32)
NUMR_CUMULATIVE_ROW(CumF64, double, f64)
NUMR_CUMULATIVE_ROW(CumF16, __half, f16)
NUMR_CUMULATIVE_ROW(CumBF16, __nv_bfloat16, bf16)
NUMR_CUMULATIVE_ROW(CumFp8E4M3, unsigned char, fp8_e4m3)
NUMR_CUMULATIVE_ROW(CumFp8E5M2, unsigned char, fp8_e5m2)

} // extern "C"
