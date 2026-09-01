// im2col2d CUDA kernels - materialise the conv2d receptive fields as a GEMM operand.
// Supports: f32, f64, f16, bf16
//
// Input:  (N, C_in, H, W)
// Output: (N, C_in*Kh*Kw, H_out*W_out)
//
// The contraction axis (input channel x kh x kw) is the ROW axis and the
// flattened spatial axis (oh*W_out + ow) is the COLUMN axis. A conv2d weight
// (C_out, C_in, Kh, Kw) is already row-major in that order, so it reshapes
// with no copy to (C_out, C_in*Kh*Kw). Then
//   [C_out, C_in*Kh*Kw] @ [C_in*Kh*Kw, H_out*W_out]
// batched over N stacks directly into [N, C_out, H_out*W_out], the conv2d
// output layout reshaped flat. No transpose and no final permute.
//
// This path is restricted to groups == 1 by the caller, so groups do not
// appear here.
//
// THREAD MAP. threadIdx.x walks consecutive flattened output positions
// (oh*W_out + ow), so the written column row is contiguous and coalesces.
// blockIdx.y walks the contraction rows and blockIdx.z the batch, each with a
// grid-stride loop so neither axis can exceed the 65535 grid limit.
//
// PADDING. Taps whose input coordinate falls outside [0, H) x [0, W) are
// written as zero, so the GEMM contracts over them without changing the
// result.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

#define DEFINE_IM2COL2D_KERNEL(suffix, dtype) \
__global__ void im2col2d_##suffix( \
    const dtype* __restrict__ input, \
    dtype* __restrict__ col, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int height, \
    unsigned int width, \
    unsigned int kernel_h, \
    unsigned int kernel_w, \
    unsigned int output_h, \
    unsigned int output_w, \
    unsigned int stride_h, \
    unsigned int stride_w, \
    unsigned int pad_top, \
    unsigned int pad_left, \
    unsigned int dilation_h, \
    unsigned int dilation_w \
) { \
    unsigned int spatial = output_h * output_w; \
    unsigned int op = blockIdx.x * blockDim.x + threadIdx.x; \
    if (op >= spatial) return; \
    \
    unsigned int oh = op / output_w; \
    unsigned int ow = op - oh * output_w; \
    \
    unsigned int taps = kernel_h * kernel_w; \
    unsigned int rows = c_in * taps; \
    int ih_base = (int)(oh * stride_h) - (int)pad_top; \
    int iw_base = (int)(ow * stride_w) - (int)pad_left; \
    \
    for (unsigned int r = blockIdx.y; r < rows; r += gridDim.y) { \
        unsigned int ic = r / taps; \
        unsigned int tap = r - ic * taps; \
        unsigned int kh = tap / kernel_w; \
        unsigned int kw = tap - kh * kernel_w; \
        int ih = ih_base + (int)(kh * dilation_h); \
        int iw = iw_base + (int)(kw * dilation_w); \
        bool valid = (ih >= 0) && ((unsigned int)ih < height) \
                  && (iw >= 0) && ((unsigned int)iw < width); \
        \
        for (unsigned int b = blockIdx.z; b < batch; b += gridDim.z) { \
            dtype v = (dtype)0; \
            if (valid) { \
                v = input[(size_t)b * c_in * height * width \
                        + (size_t)ic * height * width \
                        + (size_t)(unsigned int)ih * width \
                        + (unsigned int)iw]; \
            } \
            col[((size_t)b * rows + r) * spatial + op] = v; \
        } \
    } \
}

// Instantiations must stay inside `extern "C"` so the launcher can look the
// kernels up by their unmangled `im2col2d_<dtype>` names.
extern "C" {

DEFINE_IM2COL2D_KERNEL(f32, float)
DEFINE_IM2COL2D_KERNEL(f64, double)
DEFINE_IM2COL2D_KERNEL(f16, __half)
DEFINE_IM2COL2D_KERNEL(bf16, __nv_bfloat16)

} // extern "C"
