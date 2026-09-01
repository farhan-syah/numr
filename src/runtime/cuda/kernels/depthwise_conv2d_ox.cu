// Depthwise conv2d variant: DEPTHWISE_CONV2D_OX_BLOCK consecutive output
// COLUMNS per thread, one (batch, channel, oy) row. Two-dimensional
// restatement of conv1d_ox.cu — read that file's header comment first.
//
// TARGET SHAPE. The flat depthwise_conv2d kernel in conv.cu gives one thread
// one output element, so it re-derives the (b, c, oy, ox) coordinates, the
// base pointers and every tap bound per element, and reuses nothing between
// adjacent columns. Its loads scale as kernel_h * kernel_w per output, which
// is why the gap to the memory-bandwidth floor widens with kernel size. This
// kernel amortizes the prologue over DEPTHWISE_CONV2D_OX_BLOCK columns and,
// at stride_w == 1 with dilation_w == 1, reuses the overlapping taps:
// kernel_h * (OX_BLOCK + kernel_w - 1) loads instead of
// kernel_h * kernel_w * OX_BLOCK.
//
// TWO PATHS, ONE KERNEL. stride_w and dilation_w are launch-uniform, so
// branching on them costs one uniformly-taken branch per thread, never
// divergence:
//
//   FAST PATH (stride_w == 1, dilation_w == 1): sliding-window reuse along
//   the width axis. For output column p in [0, OX_BLOCK) and tap kx in
//   [0, kernel_w), the absolute input column is t = ix_base0 + p + kx, where
//   ix_base0 is the input column of p = 0 at kx = 0. Walking t ascending from
//   max(0, ix_base0) to min(width, ix_base0 + active - 1 + kernel_w) loads
//   each distinct column once and distributes it to every accumulator p whose
//   kx = t - ix_base0 - p lands in [0, kernel_w). t never leaves [0, width),
//   so an out-of-range tap is skipped rather than dereferenced or substituted
//   with an implicit zero. The row bound (iy in [0, height)) depends only on
//   ky, not on t, so it is tested once per kernel row and the whole run is
//   skipped when it fails.
//
//   GENERAL PATH (stride_w > 1 or dilation_w > 1): once stride_w > 1 the
//   columns each output reads are disjoint, so there is nothing to share.
//   This path runs the per-position bounds-checked accumulation OX_BLOCK
//   times, sharing only the base-pointer and channel prologue. Correct for
//   arbitrary stride/padding/dilation on both axes.
//
// PARTIAL BLOCK. `active = min(OX_BLOCK, output_w - ox_base)` tracks how many
// of the block's columns are real outputs. Inactive lanes run the same loops
// but never store, so control flow stays uniform, and the fast path's upper
// bound on t uses `active` so no lane pulls the run past what it needs.
//
// SUMMATION ORDER. Both paths sum ky-major with kx ascending inside, adding
// bias last — the same order the flat kernel uses, so no reassociation is
// introduced against it.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// Must match DEPTHWISE_CONV2D_OX_BLOCK in src/runtime/cuda/kernels/conv.rs.
#define DEPTHWISE_CONV2D_OX_BLOCK 4u

#define DEPTHWISE_CONV2D_OX_PARAMS(dtype) \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int channels, \
    unsigned int height, \
    unsigned int width, \
    unsigned int kernel_h, \
    unsigned int kernel_w, \
    unsigned int output_h, \
    unsigned int output_w, \
    unsigned int stride_h, \
    unsigned int stride_w, \
    unsigned int pad_h, \
    unsigned int pad_w, \
    unsigned int dilation_h, \
    unsigned int dilation_w, \
    unsigned int has_bias

#define DEFINE_DEPTHWISE_CONV2D_OX_KERNEL(suffix, dtype) \
__global__ void depthwise_conv2d_ox_##suffix(DEPTHWISE_CONV2D_OX_PARAMS(dtype)) { \
    unsigned int ox_base = (blockIdx.x * blockDim.x + threadIdx.x) * DEPTHWISE_CONV2D_OX_BLOCK; \
    /* The y axis carries (channel, oy) folded together: output_h is often too \
       small to fill a grid dimension on its own, and folding keeps channels \
       off the z axis, which the batch already uses. */ \
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; \
    unsigned int b = blockIdx.z; \
    if (ox_base >= output_w || row >= channels * output_h) return; \
    \
    unsigned int oy = row % output_h; \
    unsigned int c = row / output_h; \
    \
    unsigned int active = output_w - ox_base; \
    if (active > DEPTHWISE_CONV2D_OX_BLOCK) { active = DEPTHWISE_CONV2D_OX_BLOCK; } \
    \
    const dtype* in_base = input \
        + (size_t)b * channels * height * width \
        + (size_t)c * height * width; \
    const dtype* w_base = weight + (size_t)c * kernel_h * kernel_w; \
    \
    dtype acc0 = (dtype)0; \
    dtype acc1 = (dtype)0; \
    dtype acc2 = (dtype)0; \
    dtype acc3 = (dtype)0; \
    \
    if (stride_w == 1u && dilation_w == 1u) { \
        int ix_base0 = (int)ox_base - (int)pad_w; \
        int t_lo = ix_base0; \
        if (t_lo < 0) { t_lo = 0; } \
        int t_hi = ix_base0 + (int)(active - 1u) + (int)kernel_w; \
        if (t_hi > (int)width) { t_hi = (int)width; } \
        \
        for (unsigned int ky = 0; ky < kernel_h; ky++) { \
            int iy = (int)(oy * stride_h + ky * dilation_h) - (int)pad_h; \
            /* Loop-invariant over the whole run: test once, not per column. */ \
            if (iy < 0 || iy >= (int)height) { continue; } \
            const dtype* r = in_base + (size_t)iy * width; \
            const dtype* w = w_base + (size_t)ky * kernel_w; \
            for (int t = t_lo; t < t_hi; t++) { \
                /* One load feeds up to DEPTHWISE_CONV2D_OX_BLOCK accumulators. */ \
                dtype x = r[t]; \
                int kx0 = t - ix_base0; \
                int kx = kx0; \
                if (kx >= 0 && kx < (int)kernel_w) { acc0 = acc0 + x * w[kx]; } \
                kx = kx0 - 1; \
                if (kx >= 0 && kx < (int)kernel_w) { acc1 = acc1 + x * w[kx]; } \
                kx = kx0 - 2; \
                if (kx >= 0 && kx < (int)kernel_w) { acc2 = acc2 + x * w[kx]; } \
                kx = kx0 - 3; \
                if (kx >= 0 && kx < (int)kernel_w) { acc3 = acc3 + x * w[kx]; } \
            } \
        } \
    } else { \
        for (unsigned int p = 0; p < DEPTHWISE_CONV2D_OX_BLOCK && p < active; p++) { \
            unsigned int ox = ox_base + p; \
            dtype acc = (dtype)0; \
            for (unsigned int ky = 0; ky < kernel_h; ky++) { \
                int iy = (int)(oy * stride_h + ky * dilation_h) - (int)pad_h; \
                if (iy < 0 || iy >= (int)height) { continue; } \
                const dtype* r = in_base + (size_t)iy * width; \
                const dtype* w = w_base + (size_t)ky * kernel_w; \
                for (unsigned int kx = 0; kx < kernel_w; kx++) { \
                    int ix = (int)(ox * stride_w + kx * dilation_w) - (int)pad_w; \
                    if (ix >= 0 && ix < (int)width) { acc = acc + r[ix] * w[kx]; } \
                } \
            } \
            if (p == 0u) { acc0 = acc; } \
            else if (p == 1u) { acc1 = acc; } \
            else if (p == 2u) { acc2 = acc; } \
            else { acc3 = acc; } \
        } \
    } \
    \
    dtype* out_base = output \
        + (size_t)b * channels * output_h * output_w \
        + (size_t)c * output_h * output_w \
        + (size_t)oy * output_w \
        + ox_base; \
    unsigned int has_b = (has_bias != 0u && bias != nullptr) ? 1u : 0u; \
    dtype bv = has_b != 0u ? bias[c] : (dtype)0; \
    \
    if (has_b != 0u) { acc0 = acc0 + bv; } \
    out_base[0] = acc0; \
    if (active > 1u) { \
        if (has_b != 0u) { acc1 = acc1 + bv; } \
        out_base[1] = acc1; \
    } \
    if (active > 2u) { \
        if (has_b != 0u) { acc2 = acc2 + bv; } \
        out_base[2] = acc2; \
    } \
    if (active > 3u) { \
        if (has_b != 0u) { acc3 = acc3 + bv; } \
        out_base[3] = acc3; \
    } \
}

extern "C" {

DEFINE_DEPTHWISE_CONV2D_OX_KERNEL(f32, float)
DEFINE_DEPTHWISE_CONV2D_OX_KERNEL(f64, double)
DEFINE_DEPTHWISE_CONV2D_OX_KERNEL(f16, __half)
DEFINE_DEPTHWISE_CONV2D_OX_KERNEL(bf16, __nv_bfloat16)

} // extern "C"
