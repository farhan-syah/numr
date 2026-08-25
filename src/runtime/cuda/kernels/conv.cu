// Convolution CUDA kernels - conv1d, conv2d, depthwise_conv2d
// Supports: f32, f64, f16, bf16
//
// Direct convolution approach - each thread computes one output element.
// Input layout: NCHW (batch, channels, height, width)
// Weight layout: (C_out, C_in/groups, K_h, K_w)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Conv1d Kernel Templates
// Input:  (N, C_in, L)
// Weight: (C_out, C_in/groups, K)
// Output: (N, C_out, L_out)
//
// Two variants per dtype:
//   conv1d_<dt>      - one output channel per thread (grouped/depthwise/tail)
//   conv1d_oc4_<dt>  - four output channels per thread (register-blocked)
//
// THREAD MAP (both variants). threadIdx.x walks CONSECUTIVE output positions
// because L is the fastest-varying axis of [N, C, L], so the output row a warp
// writes and the input span it reads are contiguous and coalesce. threadIdx.y
// and blockIdx.y pin the output-channel slot, blockIdx.z pins the batch item.
// blockDim.x is always >= 32, so every warp owns ONE slot and the weight loads
// stay warp-uniform broadcasts. Packing slots into threadIdx.y keeps the block
// at 128 threads even when output_length is tiny (26 at the hot shape), which
// keeps warps-per-SM off the 16-resident-blocks limit.
//
// WHY REGISTER-BLOCK OVER oc. At the hot shape (batch 1, c_in = c_out = 1536,
// K = 7, L_out = 26) the arithmetic is 429 M MACs - ~0.07 ms of FMA issue - yet
// the kernel measures ~7 ms. It is not compute-bound. With one output channel
// per thread the inner step is 1 input load + 1 weight load + 1 FMA, so every
// MAC drags ~160 B of L1/L2 traffic (a 104 B input row plus a 32 B weight
// sector) behind it. Each of the 1536 output channels re-reads the WHOLE input
// independently: ~1.7 GB of L2 traffic per launch.
//
// Holding OC_BLOCK accumulators for OC_BLOCK output channels at the same ox
// loads the input value ONCE and multiplies it into all of them, so input
// traffic and input-load instructions both drop by OC_BLOCK. Weight traffic is
// unchanged - each output channel needs its own filter either way.
//
// OC_BLOCK = 4 is the knee. Per MAC the traffic goes 128 B input + 32 B weight
// -> 32 B + 32 B, a 2.5x cut; OC_BLOCK = 8 would only reach 16 B + 32 B (a
// further 1.33x) while halving the thread count again. Thread count is the
// scarce resource here: the whole problem is 39936 output elements, so every
// doubling of OC_BLOCK halves the warps available to hide memory latency.
//
// The OC_BLOCK accumulators are independent by construction, so the ~10752-long
// FP-add dependency chain of the scalar version breaks for free. No separate
// strip-mining over ic is needed, and register pressure stays low (4 accumulators
// plus 4 weight pointers and the index math, ~40 registers, no spills).
//
// NO SHARED MEMORY. Staging the input in shared memory was measured 26% slower:
// it adds two __syncthreads() per ic (~3072 per block) without removing a single
// global load, because the reuse it captures is the intra-warp reuse L1 already
// serves. The reuse that actually matters is ACROSS oc, and that lives in
// registers, not in shared memory.
//
// PADDING. The valid tap range [kx_lo, kx_hi) is derived once per thread from
// the monotonicity of ix = ox*stride - padding + kx*dilation, so the innermost
// loop carries no range test. Taps outside the input are never touched, so a
// non-finite weight in the padded region cannot leak in through a 0 * w product.
//
// REASSOCIATION. Neither variant sums in the original flat kernel's order. The
// shift is the same order as the existing CPU-vs-CUDA gap (~8e-5), below what
// flips a downstream FSQ code index.
// ============================================================================

// Shared parameter list for both conv1d variants.
#define CONV1D_PARAMS(dtype) \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int length, \
    unsigned int c_out, \
    unsigned int kernel_size, \
    unsigned int output_length, \
    unsigned int stride, \
    unsigned int padding, \
    unsigned int dilation, \
    unsigned int groups, \
    unsigned int has_bias

/* Valid tap range for this thread: ix = ix_base + kx*dilation must land in
   [0, length). Both bounds are loop-invariant, so the inner loop is branch-free. */
#define CONV1D_TAP_RANGE() \
    int ix_base = (int)(ox * stride) - (int)padding; \
    int dil = (int)dilation; \
    unsigned int kx_lo = 0u; \
    unsigned int kx_hi = kernel_size; \
    if (ix_base < 0) { \
        kx_lo = (unsigned int)(((-ix_base) + dil - 1) / dil); \
    } \
    { \
        int room = (int)length - ix_base; \
        if (room <= 0) { \
            kx_hi = 0u; \
        } else { \
            unsigned int hi = (unsigned int)((room + dil - 1) / dil); \
            if (hi < kx_hi) { kx_hi = hi; } \
        } \
    } \
    if (kx_lo > kx_hi) { kx_lo = kx_hi; }

// ----------------------------------------------------------------------------
// Scalar variant: one output channel per thread.
// Used for grouped and depthwise convolutions and whenever c_out/groups is too
// small to register-block. At depthwise (groups == c_in) the ic loop runs once
// and the chain is kernel_size long - same work as the untiled kernel.
// ----------------------------------------------------------------------------
#define DEFINE_CONV1D_KERNEL(suffix, dtype) \
__global__ void conv1d_##suffix(CONV1D_PARAMS(dtype)) { \
    unsigned int ox = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int oc = blockIdx.y * blockDim.y + threadIdx.y; \
    unsigned int b = blockIdx.z; \
    if (ox >= output_length || oc >= c_out) return; \
    \
    unsigned int c_in_per_group = c_in / groups; \
    unsigned int c_out_per_group = c_out / groups; \
    unsigned int c_in_start = (oc / c_out_per_group) * c_in_per_group; \
    \
    const dtype* in_base = input \
        + (size_t)b * c_in * length \
        + (size_t)c_in_start * length; \
    const dtype* w_base = weight + (size_t)oc * c_in_per_group * kernel_size; \
    \
    CONV1D_TAP_RANGE() \
    \
    dtype acc = (dtype)0; \
    for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
        const dtype* r = in_base + (size_t)ic * length; \
        const dtype* w = w_base + (size_t)ic * kernel_size; \
        for (unsigned int kx = kx_lo; kx < kx_hi; kx++) { \
            acc = acc + r[ix_base + (int)(kx * dilation)] * w[kx]; \
        } \
    } \
    \
    if (has_bias != 0u && bias != nullptr) { \
        acc = acc + bias[oc]; \
    } \
    \
    output[(size_t)b * c_out * output_length + (size_t)oc * output_length + ox] = acc; \
}

// ----------------------------------------------------------------------------
// Register-blocked variant: four output channels per thread, same ox.
//
// A slot is one (group, chunk-of-four) pair, so the four channels a thread owns
// always sit in the SAME group and share one c_in range - the blocking stays
// valid under `groups`. The last chunk of a group can be partial: inactive lanes
// of the block point at channel oc_base so their loads stay in bounds, and only
// the active channels are stored.
// ----------------------------------------------------------------------------
#define DEFINE_CONV1D_OC4_KERNEL(suffix, dtype) \
__global__ void conv1d_oc4_##suffix(CONV1D_PARAMS(dtype)) { \
    unsigned int ox = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int slot = blockIdx.y * blockDim.y + threadIdx.y; \
    unsigned int b = blockIdx.z; \
    \
    unsigned int c_in_per_group = c_in / groups; \
    unsigned int c_out_per_group = c_out / groups; \
    unsigned int chunks_per_group = (c_out_per_group + 3u) / 4u; \
    if (ox >= output_length || slot >= groups * chunks_per_group) return; \
    \
    unsigned int g = slot / chunks_per_group; \
    unsigned int chunk = slot - g * chunks_per_group; \
    unsigned int oc_base = g * c_out_per_group + chunk * 4u; \
    unsigned int remaining = c_out_per_group - chunk * 4u; \
    unsigned int active = remaining < 4u ? remaining : 4u; \
    \
    const dtype* in_base = input \
        + (size_t)b * c_in * length \
        + (size_t)(g * c_in_per_group) * length; \
    \
    size_t w_stride = (size_t)c_in_per_group * kernel_size; \
    const dtype* w0 = weight + (size_t)oc_base * w_stride; \
    const dtype* w1 = active > 1u ? w0 + w_stride : w0; \
    const dtype* w2 = active > 2u ? w0 + 2u * w_stride : w0; \
    const dtype* w3 = active > 3u ? w0 + 3u * w_stride : w0; \
    \
    CONV1D_TAP_RANGE() \
    \
    dtype acc0 = (dtype)0; \
    dtype acc1 = (dtype)0; \
    dtype acc2 = (dtype)0; \
    dtype acc3 = (dtype)0; \
    \
    for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
        const dtype* r = in_base + (size_t)ic * length; \
        size_t woff = (size_t)ic * kernel_size; \
        for (unsigned int kx = kx_lo; kx < kx_hi; kx++) { \
            /* One input load feeds four MACs - the whole point of the blocking. */ \
            dtype x = r[ix_base + (int)(kx * dilation)]; \
            acc0 = acc0 + x * w0[woff + kx]; \
            acc1 = acc1 + x * w1[woff + kx]; \
            acc2 = acc2 + x * w2[woff + kx]; \
            acc3 = acc3 + x * w3[woff + kx]; \
        } \
    } \
    \
    dtype* out_base = output \
        + (size_t)b * c_out * output_length \
        + (size_t)oc_base * output_length \
        + ox; \
    unsigned int has_b = (has_bias != 0u && bias != nullptr) ? 1u : 0u; \
    \
    if (has_b != 0u) { acc0 = acc0 + bias[oc_base]; } \
    out_base[0] = acc0; \
    if (active > 1u) { \
        if (has_b != 0u) { acc1 = acc1 + bias[oc_base + 1u]; } \
        out_base[(size_t)output_length] = acc1; \
    } \
    if (active > 2u) { \
        if (has_b != 0u) { acc2 = acc2 + bias[oc_base + 2u]; } \
        out_base[(size_t)2 * output_length] = acc2; \
    } \
    if (active > 3u) { \
        if (has_b != 0u) { acc3 = acc3 + bias[oc_base + 3u]; } \
        out_base[(size_t)3 * output_length] = acc3; \
    } \
}

// ============================================================================
// ConvTranspose1d Kernel Template
// Input:  (N, C_in, L)
// Weight: (C_in, C_out/groups, K)   <-- input channels lead, unlike conv1d
// Output: (N, C_out, L_out)
//
// Written in GATHER form: each thread owns one output element and searches for
// the input samples that scatter into it. The scatter form would need atomics
// (many inputs hit the same output when stride < kernel), which would cost
// both speed and bit-reproducibility.
//
// An input index j contributes to output t through tap k when
//   t = j*stride - padding + k*dilation
// so j = (t + padding - k*dilation) / stride, and only when that divides evenly.
// ============================================================================

#define DEFINE_CONV_TRANSPOSE1D_KERNEL(suffix, dtype) \
__global__ void conv_transpose1d_##suffix( \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int length, \
    unsigned int c_out, \
    unsigned int kernel_size, \
    unsigned int output_length, \
    unsigned int stride, \
    unsigned int padding, \
    unsigned int dilation, \
    unsigned int groups, \
    unsigned int has_bias \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = batch * c_out * output_length; \
    if (idx >= total) return; \
    \
    unsigned int ox = idx % output_length; \
    unsigned int oc = (idx / output_length) % c_out; \
    unsigned int b = idx / (c_out * output_length); \
    \
    unsigned int c_in_per_group = c_in / groups; \
    unsigned int c_out_per_group = c_out / groups; \
    unsigned int g = oc / c_out_per_group; \
    unsigned int oc_local = oc % c_out_per_group; \
    unsigned int c_in_start = g * c_in_per_group; \
    \
    dtype sum = (dtype)0; \
    \
    for (unsigned int kx = 0; kx < kernel_size; kx++) { \
        int shifted = (int)(ox + padding) - (int)(kx * dilation); \
        if (shifted < 0) continue; \
        if ((unsigned int)shifted % stride != 0u) continue; \
        unsigned int j = (unsigned int)shifted / stride; \
        if (j >= length) continue; \
        \
        for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
            unsigned int c_in_idx = c_in_start + ic; \
            unsigned int input_idx = b * c_in * length + c_in_idx * length + j; \
            unsigned int weight_idx = c_in_idx * c_out_per_group * kernel_size \
                                    + oc_local * kernel_size + kx; \
            sum = sum + input[input_idx] * weight[weight_idx]; \
        } \
    } \
    \
    if (has_bias != 0u && bias != nullptr) { \
        sum = sum + bias[oc]; \
    } \
    \
    output[idx] = sum; \
}

// ============================================================================
// Conv2d Kernel Template
// Input: (N, C_in, H, W)
// Weight: (C_out, C_in/groups, K_h, K_w)
// Output: (N, C_out, H_out, W_out)
// ============================================================================

#define DEFINE_CONV2D_KERNEL(suffix, dtype) \
__global__ void conv2d_##suffix( \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int height, \
    unsigned int width, \
    unsigned int c_out, \
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
    unsigned int groups, \
    unsigned int has_bias \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = batch * c_out * output_h * output_w; \
    if (idx >= total) return; \
    \
    unsigned int ox = idx % output_w; \
    unsigned int oy = (idx / output_w) % output_h; \
    unsigned int oc = (idx / (output_w * output_h)) % c_out; \
    unsigned int b = idx / (c_out * output_h * output_w); \
    \
    unsigned int c_in_per_group = c_in / groups; \
    unsigned int c_out_per_group = c_out / groups; \
    unsigned int g = oc / c_out_per_group; \
    unsigned int c_in_start = g * c_in_per_group; \
    \
    dtype sum = (dtype)0; \
    \
    for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
        unsigned int c_in_idx = c_in_start + ic; \
        \
        for (unsigned int ky = 0; ky < kernel_h; ky++) { \
            for (unsigned int kx = 0; kx < kernel_w; kx++) { \
                int iy = (int)(oy * stride_h + ky * dilation_h) - (int)pad_h; \
                int ix = (int)(ox * stride_w + kx * dilation_w) - (int)pad_w; \
                \
                if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) { \
                    unsigned int input_idx = b * c_in * height * width \
                        + c_in_idx * height * width \
                        + (unsigned int)iy * width \
                        + (unsigned int)ix; \
                    unsigned int weight_idx = oc * c_in_per_group * kernel_h * kernel_w \
                        + ic * kernel_h * kernel_w \
                        + ky * kernel_w \
                        + kx; \
                    sum = sum + input[input_idx] * weight[weight_idx]; \
                } \
            } \
        } \
    } \
    \
    if (has_bias != 0u && bias != nullptr) { \
        sum = sum + bias[oc]; \
    } \
    \
    output[idx] = sum; \
}

// ============================================================================
// Depthwise Conv2d Kernel Template
// Input: (N, C, H, W)
// Weight: (C, 1, K_h, K_w)
// Output: (N, C, H_out, W_out)
// Each channel has its own independent filter
// ============================================================================

#define DEFINE_DEPTHWISE_CONV2D_KERNEL(suffix, dtype) \
__global__ void depthwise_conv2d_##suffix( \
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
    unsigned int has_bias \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = batch * channels * output_h * output_w; \
    if (idx >= total) return; \
    \
    unsigned int ox = idx % output_w; \
    unsigned int oy = (idx / output_w) % output_h; \
    unsigned int c = (idx / (output_w * output_h)) % channels; \
    unsigned int b = idx / (channels * output_h * output_w); \
    \
    dtype sum = (dtype)0; \
    \
    for (unsigned int ky = 0; ky < kernel_h; ky++) { \
        for (unsigned int kx = 0; kx < kernel_w; kx++) { \
            int iy = (int)(oy * stride_h + ky * dilation_h) - (int)pad_h; \
            int ix = (int)(ox * stride_w + kx * dilation_w) - (int)pad_w; \
            \
            if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) { \
                unsigned int input_idx = b * channels * height * width \
                    + c * height * width \
                    + (unsigned int)iy * width \
                    + (unsigned int)ix; \
                unsigned int weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx; \
                sum = sum + input[input_idx] * weight[weight_idx]; \
            } \
        } \
    } \
    \
    if (has_bias != 0u && bias != nullptr) { \
        sum = sum + bias[c]; \
    } \
    \
    output[idx] = sum; \
}

// ============================================================================
// Instantiate kernels for all supported dtypes
// ============================================================================

extern "C" {

// F32 kernels
DEFINE_CONV1D_KERNEL(f32, float)
DEFINE_CONV1D_OC4_KERNEL(f32, float)
DEFINE_CONV_TRANSPOSE1D_KERNEL(f32, float)
DEFINE_CONV2D_KERNEL(f32, float)
DEFINE_DEPTHWISE_CONV2D_KERNEL(f32, float)

// F64 kernels
DEFINE_CONV1D_KERNEL(f64, double)
DEFINE_CONV1D_OC4_KERNEL(f64, double)
DEFINE_CONV_TRANSPOSE1D_KERNEL(f64, double)
DEFINE_CONV2D_KERNEL(f64, double)
DEFINE_DEPTHWISE_CONV2D_KERNEL(f64, double)

// F16 kernels (half precision)
DEFINE_CONV1D_KERNEL(f16, __half)
DEFINE_CONV1D_OC4_KERNEL(f16, __half)
DEFINE_CONV_TRANSPOSE1D_KERNEL(f16, __half)
DEFINE_CONV2D_KERNEL(f16, __half)
DEFINE_DEPTHWISE_CONV2D_KERNEL(f16, __half)

// BF16 kernels (bfloat16)
DEFINE_CONV1D_KERNEL(bf16, __nv_bfloat16)
DEFINE_CONV1D_OC4_KERNEL(bf16, __nv_bfloat16)
DEFINE_CONV_TRANSPOSE1D_KERNEL(bf16, __nv_bfloat16)
DEFINE_CONV2D_KERNEL(bf16, __nv_bfloat16)
DEFINE_DEPTHWISE_CONV2D_KERNEL(bf16, __nv_bfloat16)

// FP8 E4M3 kernels (compute in float, load/store as FP8)
__global__ void conv1d_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ input,
    const numr_fp8_e4m3* __restrict__ weight,
    const numr_fp8_e4m3* __restrict__ bias,
    numr_fp8_e4m3* __restrict__ output,
    unsigned int batch,
    unsigned int c_in,
    unsigned int length,
    unsigned int c_out,
    unsigned int kernel_size,
    unsigned int output_length,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int groups,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * c_out * output_length;
    if (idx >= total) return;

    unsigned int ox = idx % output_length;
    unsigned int oc = (idx / output_length) % c_out;
    unsigned int b = idx / (c_out * output_length);

    unsigned int c_in_per_group = c_in / groups;
    unsigned int c_out_per_group = c_out / groups;
    unsigned int g = oc / c_out_per_group;
    unsigned int c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (unsigned int ic = 0; ic < c_in_per_group; ic++) {
        unsigned int c_in_idx = c_in_start + ic;
        for (unsigned int kx = 0; kx < kernel_size; kx++) {
            int ix = (int)(ox * stride + kx * dilation) - (int)padding;
            if (ix >= 0 && ix < (int)length) {
                unsigned int input_idx = b * c_in * length + c_in_idx * length + (unsigned int)ix;
                unsigned int weight_idx = oc * c_in_per_group * kernel_size + ic * kernel_size + kx;
                sum += fp8_e4m3_to_f32(input[input_idx].data) * fp8_e4m3_to_f32(weight[weight_idx].data);
            }
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e4m3_to_f32(bias[oc].data);
    }

    output[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sum));
}

__global__ void conv2d_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ input,
    const numr_fp8_e4m3* __restrict__ weight,
    const numr_fp8_e4m3* __restrict__ bias,
    numr_fp8_e4m3* __restrict__ output,
    unsigned int batch,
    unsigned int c_in,
    unsigned int height,
    unsigned int width,
    unsigned int c_out,
    unsigned int kh,
    unsigned int kw,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int pad_h,
    unsigned int pad_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int groups,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * c_out * out_h * out_w;
    if (idx >= total) return;

    unsigned int ow = idx % out_w;
    unsigned int oh = (idx / out_w) % out_h;
    unsigned int oc = (idx / (out_w * out_h)) % c_out;
    unsigned int b = idx / (c_out * out_h * out_w);

    unsigned int c_in_per_group = c_in / groups;
    unsigned int c_out_per_group = c_out / groups;
    unsigned int g = oc / c_out_per_group;
    unsigned int c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (unsigned int ic = 0; ic < c_in_per_group; ic++) {
        unsigned int c_in_idx = c_in_start + ic;
        for (unsigned int ky = 0; ky < kh; ky++) {
            for (unsigned int kx = 0; kx < kw; kx++) {
                int iy = (int)(oh * stride_h + ky * dilation_h) - (int)pad_h;
                int ix = (int)(ow * stride_w + kx * dilation_w) - (int)pad_w;
                if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) {
                    unsigned int input_idx = b * c_in * height * width + c_in_idx * height * width + (unsigned int)iy * width + (unsigned int)ix;
                    unsigned int weight_idx = oc * c_in_per_group * kh * kw + ic * kh * kw + ky * kw + kx;
                    sum += fp8_e4m3_to_f32(input[input_idx].data) * fp8_e4m3_to_f32(weight[weight_idx].data);
                }
            }
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e4m3_to_f32(bias[oc].data);
    }

    output[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sum));
}

__global__ void depthwise_conv2d_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ input,
    const numr_fp8_e4m3* __restrict__ weight,
    const numr_fp8_e4m3* __restrict__ bias,
    numr_fp8_e4m3* __restrict__ output,
    unsigned int batch,
    unsigned int channels,
    unsigned int height,
    unsigned int width,
    unsigned int kh,
    unsigned int kw,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int pad_h,
    unsigned int pad_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    unsigned int ow = idx % out_w;
    unsigned int oh = (idx / out_w) % out_h;
    unsigned int c = (idx / (out_w * out_h)) % channels;
    unsigned int b = idx / (channels * out_h * out_w);

    float sum = 0.0f;

    for (unsigned int ky = 0; ky < kh; ky++) {
        for (unsigned int kx = 0; kx < kw; kx++) {
            int iy = (int)(oh * stride_h + ky * dilation_h) - (int)pad_h;
            int ix = (int)(ow * stride_w + kx * dilation_w) - (int)pad_w;
            if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) {
                unsigned int input_idx = b * channels * height * width + c * height * width + (unsigned int)iy * width + (unsigned int)ix;
                unsigned int weight_idx = c * kh * kw + ky * kw + kx;
                sum += fp8_e4m3_to_f32(input[input_idx].data) * fp8_e4m3_to_f32(weight[weight_idx].data);
            }
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e4m3_to_f32(bias[c].data);
    }

    output[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sum));
}

// ConvTranspose1d: gather form, see DEFINE_CONV_TRANSPOSE1D_KERNEL comment above
// for the scatter/gather index relation (t = j*stride - padding + k*dilation).
__global__ void conv_transpose1d_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ input,
    const numr_fp8_e4m3* __restrict__ weight,
    const numr_fp8_e4m3* __restrict__ bias,
    numr_fp8_e4m3* __restrict__ output,
    unsigned int batch,
    unsigned int c_in,
    unsigned int length,
    unsigned int c_out,
    unsigned int kernel_size,
    unsigned int output_length,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int groups,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * c_out * output_length;
    if (idx >= total) return;

    unsigned int ox = idx % output_length;
    unsigned int oc = (idx / output_length) % c_out;
    unsigned int b = idx / (c_out * output_length);

    unsigned int c_in_per_group = c_in / groups;
    unsigned int c_out_per_group = c_out / groups;
    unsigned int g = oc / c_out_per_group;
    unsigned int oc_local = oc % c_out_per_group;
    unsigned int c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (unsigned int kx = 0; kx < kernel_size; kx++) {
        int shifted = (int)(ox + padding) - (int)(kx * dilation);
        if (shifted < 0) continue;
        if ((unsigned int)shifted % stride != 0u) continue;
        unsigned int j = (unsigned int)shifted / stride;
        if (j >= length) continue;

        for (unsigned int ic = 0; ic < c_in_per_group; ic++) {
            unsigned int c_in_idx = c_in_start + ic;
            unsigned int input_idx = b * c_in * length + c_in_idx * length + j;
            unsigned int weight_idx = c_in_idx * c_out_per_group * kernel_size
                                    + oc_local * kernel_size + kx;
            sum += fp8_e4m3_to_f32(input[input_idx].data) * fp8_e4m3_to_f32(weight[weight_idx].data);
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e4m3_to_f32(bias[oc].data);
    }

    output[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sum));
}

// FP8 E5M2 kernels (compute in float, load/store as FP8)
__global__ void conv1d_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ input,
    const numr_fp8_e5m2* __restrict__ weight,
    const numr_fp8_e5m2* __restrict__ bias,
    numr_fp8_e5m2* __restrict__ output,
    unsigned int batch,
    unsigned int c_in,
    unsigned int length,
    unsigned int c_out,
    unsigned int kernel_size,
    unsigned int output_length,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int groups,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * c_out * output_length;
    if (idx >= total) return;

    unsigned int ox = idx % output_length;
    unsigned int oc = (idx / output_length) % c_out;
    unsigned int b = idx / (c_out * output_length);

    unsigned int c_in_per_group = c_in / groups;
    unsigned int c_out_per_group = c_out / groups;
    unsigned int g = oc / c_out_per_group;
    unsigned int c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (unsigned int ic = 0; ic < c_in_per_group; ic++) {
        unsigned int c_in_idx = c_in_start + ic;
        for (unsigned int kx = 0; kx < kernel_size; kx++) {
            int ix = (int)(ox * stride + kx * dilation) - (int)padding;
            if (ix >= 0 && ix < (int)length) {
                unsigned int input_idx = b * c_in * length + c_in_idx * length + (unsigned int)ix;
                unsigned int weight_idx = oc * c_in_per_group * kernel_size + ic * kernel_size + kx;
                sum += fp8_e5m2_to_f32(input[input_idx].data) * fp8_e5m2_to_f32(weight[weight_idx].data);
            }
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e5m2_to_f32(bias[oc].data);
    }

    output[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sum));
}

__global__ void conv2d_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ input,
    const numr_fp8_e5m2* __restrict__ weight,
    const numr_fp8_e5m2* __restrict__ bias,
    numr_fp8_e5m2* __restrict__ output,
    unsigned int batch,
    unsigned int c_in,
    unsigned int height,
    unsigned int width,
    unsigned int c_out,
    unsigned int kh,
    unsigned int kw,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int pad_h,
    unsigned int pad_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int groups,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * c_out * out_h * out_w;
    if (idx >= total) return;

    unsigned int ow = idx % out_w;
    unsigned int oh = (idx / out_w) % out_h;
    unsigned int oc = (idx / (out_w * out_h)) % c_out;
    unsigned int b = idx / (c_out * out_h * out_w);

    unsigned int c_in_per_group = c_in / groups;
    unsigned int c_out_per_group = c_out / groups;
    unsigned int g = oc / c_out_per_group;
    unsigned int c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (unsigned int ic = 0; ic < c_in_per_group; ic++) {
        unsigned int c_in_idx = c_in_start + ic;
        for (unsigned int ky = 0; ky < kh; ky++) {
            for (unsigned int kx = 0; kx < kw; kx++) {
                int iy = (int)(oh * stride_h + ky * dilation_h) - (int)pad_h;
                int ix = (int)(ow * stride_w + kx * dilation_w) - (int)pad_w;
                if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) {
                    unsigned int input_idx = b * c_in * height * width + c_in_idx * height * width + (unsigned int)iy * width + (unsigned int)ix;
                    unsigned int weight_idx = oc * c_in_per_group * kh * kw + ic * kh * kw + ky * kw + kx;
                    sum += fp8_e5m2_to_f32(input[input_idx].data) * fp8_e5m2_to_f32(weight[weight_idx].data);
                }
            }
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e5m2_to_f32(bias[oc].data);
    }

    output[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sum));
}

__global__ void depthwise_conv2d_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ input,
    const numr_fp8_e5m2* __restrict__ weight,
    const numr_fp8_e5m2* __restrict__ bias,
    numr_fp8_e5m2* __restrict__ output,
    unsigned int batch,
    unsigned int channels,
    unsigned int height,
    unsigned int width,
    unsigned int kh,
    unsigned int kw,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int pad_h,
    unsigned int pad_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    unsigned int ow = idx % out_w;
    unsigned int oh = (idx / out_w) % out_h;
    unsigned int c = (idx / (out_w * out_h)) % channels;
    unsigned int b = idx / (channels * out_h * out_w);

    float sum = 0.0f;

    for (unsigned int ky = 0; ky < kh; ky++) {
        for (unsigned int kx = 0; kx < kw; kx++) {
            int iy = (int)(oh * stride_h + ky * dilation_h) - (int)pad_h;
            int ix = (int)(ow * stride_w + kx * dilation_w) - (int)pad_w;
            if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) {
                unsigned int input_idx = b * channels * height * width + c * height * width + (unsigned int)iy * width + (unsigned int)ix;
                unsigned int weight_idx = c * kh * kw + ky * kw + kx;
                sum += fp8_e5m2_to_f32(input[input_idx].data) * fp8_e5m2_to_f32(weight[weight_idx].data);
            }
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e5m2_to_f32(bias[c].data);
    }

    output[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sum));
}

// ConvTranspose1d: gather form, see DEFINE_CONV_TRANSPOSE1D_KERNEL comment above
// for the scatter/gather index relation (t = j*stride - padding + k*dilation).
__global__ void conv_transpose1d_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ input,
    const numr_fp8_e5m2* __restrict__ weight,
    const numr_fp8_e5m2* __restrict__ bias,
    numr_fp8_e5m2* __restrict__ output,
    unsigned int batch,
    unsigned int c_in,
    unsigned int length,
    unsigned int c_out,
    unsigned int kernel_size,
    unsigned int output_length,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int groups,
    unsigned int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * c_out * output_length;
    if (idx >= total) return;

    unsigned int ox = idx % output_length;
    unsigned int oc = (idx / output_length) % c_out;
    unsigned int b = idx / (c_out * output_length);

    unsigned int c_in_per_group = c_in / groups;
    unsigned int c_out_per_group = c_out / groups;
    unsigned int g = oc / c_out_per_group;
    unsigned int oc_local = oc % c_out_per_group;
    unsigned int c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (unsigned int kx = 0; kx < kernel_size; kx++) {
        int shifted = (int)(ox + padding) - (int)(kx * dilation);
        if (shifted < 0) continue;
        if ((unsigned int)shifted % stride != 0u) continue;
        unsigned int j = (unsigned int)shifted / stride;
        if (j >= length) continue;

        for (unsigned int ic = 0; ic < c_in_per_group; ic++) {
            unsigned int c_in_idx = c_in_start + ic;
            unsigned int input_idx = b * c_in * length + c_in_idx * length + j;
            unsigned int weight_idx = c_in_idx * c_out_per_group * kernel_size
                                    + oc_local * kernel_size + kx;
            sum += fp8_e5m2_to_f32(input[input_idx].data) * fp8_e5m2_to_f32(weight[weight_idx].data);
        }
    }

    if (has_bias != 0u && bias != nullptr) {
        sum += fp8_e5m2_to_f32(bias[oc].data);
    }

    output[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sum));
}

} // extern "C"
