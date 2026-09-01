// Conv1d kernel variant: CONV1D_OX_BLOCK consecutive output POSITIONS per
// thread, same output channel. Position-axis analogue of conv1d_oc4's
// channel-axis blocking in conv.cu — read that file's header comment first.
//
// TARGET SHAPE. Depthwise conv1d (c_out_per_group == 1) cannot use conv1d_oc4
// (it needs c_out_per_group >= 4). With one output position per thread the
// scalar kernel pays its whole prologue — group-index derivation, base-pointer
// math, tap-range derivation — once per output element, and reuses none of the
// overlapping input taps between adjacent positions. This kernel amortizes
// that prologue over CONV1D_OX_BLOCK positions and, when stride == 1 and
// dilation == 1, also reuses overlapping taps: 4 consecutive outputs at
// kernel_size == 4 need 7 distinct input loads instead of 16.
//
// TWO PATHS, ONE KERNEL. stride and dilation are launch-uniform (identical
// for every thread in the grid), so branching on `stride == 1 && dilation ==
// 1` costs one uniformly-taken branch per thread, not divergence:
//
//   FAST PATH (stride == 1, dilation == 1): sliding-window reuse. For output
//   p in [0, CONV1D_OX_BLOCK) and tap kx in [0, kernel_size), the absolute
//   input position is t = ix_base0 + p + kx, where ix_base0 is the input
//   coordinate of output p=0 at kx=0. Looping t ascending from
//   max(0, ix_base0) to min(length, ix_base0 + active - 1 + kernel_size)
//   loads each distinct input position exactly once and, for each loaded x,
//   distributes it to every accumulator p whose kx = t - ix_base0 - p falls
//   in [0, kernel_size). Because t only ever ranges over [0, length), a tap
//   outside the input is never dereferenced — same guarantee the scalar
//   kernel gets from CONV1D_TAP_RANGE, just re-expressed as a loop bound
//   instead of a per-thread kx range. Because t increases monotonically, kx
//   = t - ix_base0 - p increases monotonically for each fixed p, so taps
//   still accumulate ascending in kx, matching the cross-backend order
//   contract. No product ever multiplies a possibly non-finite weight by an
//   implicit zero: an out-of-range tap is skipped, never substituted with 0.
//
//   GENERAL PATH (stride > 1 or dilation > 1): taps don't overlap between
//   outputs once stride > 1 (the inputs each output reads are disjoint), so
//   there is nothing to share. This path runs CONV1D_TAP_RANGE and the
//   scalar accumulation loop once per output position, sharing only the
//   group-index and base-pointer prologue across the block. Still correct
//   for arbitrary stride/dilation/padding/groups; the tap reuse above is a
//   bonus specific to stride == 1, dilation == 1.
//
// PARTIAL BLOCK. `active = min(CONV1D_OX_BLOCK, output_length - ox_base)`
// tracks how many of the block's positions are real outputs. Inactive lanes
// (last block of a row, when output_length % CONV1D_OX_BLOCK != 0) still run
// through the same loops — their acc values are computed but never stored —
// so every thread takes the same control flow and no lane reads past a valid
// input index (t's upper bound already uses `active`, not the full block
// width, so an inactive lane cannot pull the loop bound past what it needs).
//
// REASSOCIATION. Same contract as conv.cu: neither path sums in the flat
// kernel's original order, but each preserves ascending-kx summation with
// bias added last, matching the CPU reference within the existing tolerance.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "conv1d_common.cuh"

// Must match CONV1D_OX_BLOCK in src/runtime/cuda/kernels/conv.rs.
#define CONV1D_OX_BLOCK 4u

#define DEFINE_CONV1D_OX_KERNEL(suffix, dtype) \
__global__ void conv1d_ox_##suffix(CONV1D_PARAMS(dtype)) { \
    unsigned int ox_base = (blockIdx.x * blockDim.x + threadIdx.x) * CONV1D_OX_BLOCK; \
    unsigned int oc = blockIdx.y * blockDim.y + threadIdx.y; \
    unsigned int b = blockIdx.z; \
    if (ox_base >= output_length || oc >= c_out) return; \
    \
    unsigned int active = output_length - ox_base; \
    if (active > CONV1D_OX_BLOCK) { active = CONV1D_OX_BLOCK; } \
    \
    /* Same group-resolution rule as the scalar kernel in conv.cu: uniform \
       across the block, never a source of warp divergence. */ \
    unsigned int c_in_start; \
    if (groups == 1u) { \
        c_in_start = 0u; \
    } else if (c_out_per_group == 1u) { \
        c_in_start = oc * c_in_per_group; \
    } else { \
        c_in_start = (oc / c_out_per_group) * c_in_per_group; \
    } \
    \
    const dtype* in_base = input \
        + (size_t)b * c_in * length \
        + (size_t)c_in_start * length; \
    const dtype* w_base = weight + (size_t)oc * c_in_per_group * kernel_size; \
    \
    dtype acc0 = (dtype)0; \
    dtype acc1 = (dtype)0; \
    dtype acc2 = (dtype)0; \
    dtype acc3 = (dtype)0; \
    \
    if (stride == 1u && dilation == 1u) { \
        int ix_base0 = (int)ox_base - (int)padding; \
        int t_lo = ix_base0; \
        if (t_lo < 0) { t_lo = 0; } \
        int t_hi = ix_base0 + (int)(active - 1u) + (int)kernel_size; \
        if (t_hi > (int)length) { t_hi = (int)length; } \
        \
        for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
            const dtype* r = in_base + (size_t)ic * length; \
            const dtype* w = w_base + (size_t)ic * kernel_size; \
            for (int t = t_lo; t < t_hi; t++) { \
                /* One load feeds up to CONV1D_OX_BLOCK accumulators. */ \
                dtype x = r[t]; \
                int kx0 = t - ix_base0; \
                int kx = kx0; \
                if (kx >= 0 && kx < (int)kernel_size) { acc0 = acc0 + x * w[kx]; } \
                kx = kx0 - 1; \
                if (kx >= 0 && kx < (int)kernel_size) { acc1 = acc1 + x * w[kx]; } \
                kx = kx0 - 2; \
                if (kx >= 0 && kx < (int)kernel_size) { acc2 = acc2 + x * w[kx]; } \
                kx = kx0 - 3; \
                if (kx >= 0 && kx < (int)kernel_size) { acc3 = acc3 + x * w[kx]; } \
            } \
        } \
    } else { \
        for (unsigned int p = 0; p < CONV1D_OX_BLOCK && p < active; p++) { \
            unsigned int ox = ox_base + p; \
            CONV1D_TAP_RANGE() \
            dtype acc = (dtype)0; \
            for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
                const dtype* r = in_base + (size_t)ic * length; \
                const dtype* w = w_base + (size_t)ic * kernel_size; \
                for (unsigned int kx = kx_lo; kx < kx_hi; kx++) { \
                    acc = acc + r[ix_base + (int)(kx * dilation)] * w[kx]; \
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
        + (size_t)b * c_out * output_length \
        + (size_t)oc * output_length \
        + ox_base; \
    unsigned int has_b = (has_bias != 0u && bias != nullptr) ? 1u : 0u; \
    dtype bv = has_b != 0u ? bias[oc] : (dtype)0; \
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

DEFINE_CONV1D_OX_KERNEL(f32, float)
DEFINE_CONV1D_OX_KERNEL(f64, double)
DEFINE_CONV1D_OX_KERNEL(f16, __half)
DEFINE_CONV1D_OX_KERNEL(bf16, __nv_bfloat16)

} // extern "C"
