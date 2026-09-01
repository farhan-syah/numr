// col_transpose1d CUDA kernels - materialise conv_transpose1d's contributing
// input samples as a GEMM operand.
// Supports: f32, f64, f16, bf16
//
// Input:  (N, C_in, L)
// Output: (N, C_in*K, L_out)
//
// GATHER, NOT SCATTER. Transposed convolution is usually written as
// `out[l*stride - pad + k*dilation] += x[l] * w[k]`, whose column form is a
// col2im scatter-add with overlapping writes and therefore needs atomics.
// Reading the same relation backwards removes them: for a fixed output
// position `ox`, tap `k` can only be fed by ONE input sample, so every column
// element is written exactly once by exactly one thread.
//
//   col[n, ic*K + k, ox] = input[n, ic, l]
//     where  num = ox + pad_left - k * dilation
//            num >= 0                 (checked BEFORE the division: the taps
//                                      that overhang the left edge give a
//                                      negative numerator)
//            num % stride == 0        (otherwise this tap lands between input
//                                      samples and contributes nothing)
//            l = num / stride  <  L
//
// This matches `conv_transpose.rs` and `conv.cu`'s direct kernel tap for tap.
//
// The contraction axis (input channel x tap) is the ROW axis and the spatial
// axis is the COLUMN axis, so
//   [C_out, C_in*K] @ [C_in*K, L_out]
// batched over N stacks directly into [N, C_out, L_out]. The left operand is a
// permute+reshape of the weight, done once on the host.
//
// GROUPS DO NOT APPEAR HERE. The caller restricts this path to `groups == 1`.
//
// THREAD MAP. threadIdx.x walks consecutive output positions, so both the
// gathered input span and the written column row are contiguous and coalesce.
// blockIdx.y walks the contraction rows and blockIdx.z the batch, each with a
// grid-stride loop so neither axis can exceed the 65535 grid limit.
//
// ZERO FILL. Taps with no contributing input sample are written as zero, so the
// GEMM contracts over them without changing the result. `output_padding` needs
// no special case: it only lengthens L_out, and the extra positions gather
// nothing because their `l` runs past L.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

#define DEFINE_COL_TRANSPOSE1D_KERNEL(suffix, dtype) \
__global__ void col_transpose1d_##suffix( \
    const dtype* __restrict__ input, \
    dtype* __restrict__ col, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int length, \
    unsigned int kernel_size, \
    unsigned int output_length, \
    unsigned int stride, \
    unsigned int pad_left, \
    unsigned int dilation \
) { \
    unsigned int ox = blockIdx.x * blockDim.x + threadIdx.x; \
    if (ox >= output_length) return; \
    \
    unsigned int rows = c_in * kernel_size; \
    int num_base = (int)ox + (int)pad_left; \
    \
    for (unsigned int r = blockIdx.y; r < rows; r += gridDim.y) { \
        unsigned int ic = r / kernel_size; \
        unsigned int kx = r - ic * kernel_size; \
        int num = num_base - (int)(kx * dilation); \
        /* Sign first: a negative numerator has no valid input sample at all. */ \
        bool valid = (num >= 0); \
        unsigned int l = 0; \
        if (valid) { \
            unsigned int unum = (unsigned int)num; \
            valid = (unum % stride == 0); \
            l = unum / stride; \
            valid = valid && (l < length); \
        } \
        \
        for (unsigned int b = blockIdx.z; b < batch; b += gridDim.z) { \
            dtype v = (dtype)0; \
            if (valid) { \
                v = input[(size_t)b * c_in * length \
                        + (size_t)ic * length \
                        + l]; \
            } \
            col[((size_t)b * rows + r) * output_length + ox] = v; \
        } \
    } \
}

// Instantiations must stay inside `extern "C"` so the launcher can look the
// kernels up by their unmangled `col_transpose1d_<dtype>` names.
extern "C" {

DEFINE_COL_TRANSPOSE1D_KERNEL(f32, float)
DEFINE_COL_TRANSPOSE1D_KERNEL(f64, double)
DEFINE_COL_TRANSPOSE1D_KERNEL(f16, __half)
DEFINE_COL_TRANSPOSE1D_KERNEL(bf16, __nv_bfloat16)

} // extern "C"
