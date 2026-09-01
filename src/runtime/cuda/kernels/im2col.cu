// im2col CUDA kernels - materialise the conv1d receptive fields as a GEMM operand.
// Supports: f32, f64, f16, bf16
//
// Input:  (N, C_in, L)
// Output: (N, C_in*K, L_out)
//
// The contraction axis (input channel x tap) is the ROW axis and the spatial
// axis is the COLUMN axis. A conv1d weight (C_out, C_in/groups, K) reshapes
// with no copy to (groups, C_out/groups, (C_in/groups)*K), so
//   [C_out/groups, (C_in/groups)*K] @ [(C_in/groups)*K, L_out]
// batched over (N, groups) stacks directly into [N, C_out, L_out]. That is the
// conv1d output layout, so the GEMM needs no transpose and no final permute.
//
// GROUPS DO NOT APPEAR HERE. Row r of the column buffer is
// (input_channel * K + tap), and the row-major split of that axis into
// (groups, (C_in/groups)*K) is exactly the group blocking, because output
// channels are ordered group-major. The grouping is a reshape on the host side.
//
// THREAD MAP. threadIdx.x walks consecutive output positions, so both the
// gathered input span and the written column row are contiguous and coalesce.
// blockIdx.y walks the contraction rows and blockIdx.z the batch, each with a
// grid-stride loop so neither axis can exceed the 65535 grid limit.
//
// PADDING. Taps whose input index falls outside [0, L) are written as zero, so
// the GEMM contracts over them without changing the result.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

#define DEFINE_IM2COL1D_KERNEL(suffix, dtype) \
__global__ void im2col1d_##suffix( \
    const dtype* __restrict__ input, \
    dtype* __restrict__ col, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int length, \
    unsigned int kernel_size, \
    unsigned int output_length, \
    unsigned int stride, \
    unsigned int padding, \
    unsigned int dilation \
) { \
    unsigned int ox = blockIdx.x * blockDim.x + threadIdx.x; \
    if (ox >= output_length) return; \
    \
    unsigned int rows = c_in * kernel_size; \
    int ix_base = (int)(ox * stride) - (int)padding; \
    \
    for (unsigned int r = blockIdx.y; r < rows; r += gridDim.y) { \
        unsigned int ic = r / kernel_size; \
        unsigned int kx = r - ic * kernel_size; \
        int ix = ix_base + (int)(kx * dilation); \
        bool valid = (ix >= 0) && ((unsigned int)ix < length); \
        \
        for (unsigned int b = blockIdx.z; b < batch; b += gridDim.z) { \
            dtype v = (dtype)0; \
            if (valid) { \
                v = input[(size_t)b * c_in * length \
                        + (size_t)ic * length \
                        + (unsigned int)ix]; \
            } \
            col[((size_t)b * rows + r) * output_length + ox] = v; \
        } \
    } \
}

// Instantiations must stay inside `extern "C"` so the launcher can look the
// kernels up by their unmangled `im2col1d_<dtype>` names.
extern "C" {

DEFINE_IM2COL1D_KERNEL(f32, float)
DEFINE_IM2COL1D_KERNEL(f64, double)
DEFINE_IM2COL1D_KERNEL(f16, __half)
DEFINE_IM2COL1D_KERNEL(bf16, __nv_bfloat16)

} // extern "C"
