// Bluestein (chirp-z) FFT stages for arbitrary transform sizes.
//
// The radix-2 Stockham kernels in fft.cu only accept power-of-two N. Bluestein
// rewrites an N-point DFT as a cyclic convolution of length
// M = next_power_of_two(2N - 1), which those kernels DO accept. These three
// kernels are the pre/post work around that convolution:
//
//   premultiply  : a[k] = x[k] * chirp[k] for k < N, zero for N <= k < M
//   pointwise_mul: spectrum[k] *= kernel_spectrum[k]   (kernel shared by batch)
//   postmultiply : out[k] = chirp[k] * conv[k] * scale for k < N
//
// The chirp and kernel_spectrum tables are built on the host in f64 (see
// numr::algorithm::fft_bluestein) and uploaded, so a CUDA transform cannot
// disagree with a CPU transform about the chirp. Only the convolution runs in
// the caller's own dtype.

#include <cuda_runtime.h>
#include <math.h>
#include "dtype_traits.cuh"

__device__ __forceinline__ float2 bs_cmul_f32(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ double2 bs_cmul_f64(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

extern "C" {

// ============================================================================
// Stage 1: chirp premultiply into the zero-padded convolution buffer
// ============================================================================
//
// `out` has batch_size * m elements and is fully written here (the tail beyond
// N is zeroed rather than left undefined), so the caller need not pre-clear it.

__global__ void bluestein_premultiply_c64(
    const float2* __restrict__ input,
    const float2* __restrict__ chirp,
    float2* __restrict__ out,
    int n,
    int m,
    int batch_size
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * m;
    if (gid >= total) return;

    int k = (int)(gid % m);
    long long b = gid / m;

    if (k < n) {
        out[gid] = bs_cmul_f32(input[b * n + k], chirp[k]);
    } else {
        out[gid] = make_float2(0.0f, 0.0f);
    }
}

__global__ void bluestein_premultiply_c128(
    const double2* __restrict__ input,
    const double2* __restrict__ chirp,
    double2* __restrict__ out,
    int n,
    int m,
    int batch_size
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * m;
    if (gid >= total) return;

    int k = (int)(gid % m);
    long long b = gid / m;

    if (k < n) {
        out[gid] = bs_cmul_f64(input[b * n + k], chirp[k]);
    } else {
        out[gid] = make_double2(0.0, 0.0);
    }
}

// ============================================================================
// Stage 1b: real-input chirp premultiply
// ============================================================================
//
// rfft feeds real samples; treating them as complex with zero imaginary part
// would need a separate widening pass. This does the widen and the multiply in
// one read.

__global__ void bluestein_premultiply_real_c64(
    const float* __restrict__ input,
    const float2* __restrict__ chirp,
    float2* __restrict__ out,
    int n,
    int m,
    int batch_size
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * m;
    if (gid >= total) return;

    int k = (int)(gid % m);
    long long b = gid / m;

    if (k < n) {
        float x = input[b * n + k];
        float2 w = chirp[k];
        out[gid] = make_float2(x * w.x, x * w.y);
    } else {
        out[gid] = make_float2(0.0f, 0.0f);
    }
}

__global__ void bluestein_premultiply_real_c128(
    const double* __restrict__ input,
    const double2* __restrict__ chirp,
    double2* __restrict__ out,
    int n,
    int m,
    int batch_size
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * m;
    if (gid >= total) return;

    int k = (int)(gid % m);
    long long b = gid / m;

    if (k < n) {
        double x = input[b * n + k];
        double2 w = chirp[k];
        out[gid] = make_double2(x * w.x, x * w.y);
    } else {
        out[gid] = make_double2(0.0, 0.0);
    }
}

// ============================================================================
// Stage 2: pointwise multiply by the kernel spectrum
// ============================================================================
//
// The kernel spectrum depends only on (N, direction), so one length-M table is
// shared across the whole batch and indexed modulo m.

__global__ void bluestein_pointwise_mul_c64(
    float2* __restrict__ spectrum,
    const float2* __restrict__ kernel_spectrum,
    int m,
    int batch_size
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * m;
    if (gid >= total) return;

    int k = (int)(gid % m);
    spectrum[gid] = bs_cmul_f32(spectrum[gid], kernel_spectrum[k]);
}

__global__ void bluestein_pointwise_mul_c128(
    double2* __restrict__ spectrum,
    const double2* __restrict__ kernel_spectrum,
    int m,
    int batch_size
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * m;
    if (gid >= total) return;

    int k = (int)(gid % m);
    spectrum[gid] = bs_cmul_f64(spectrum[gid], kernel_spectrum[k]);
}

// ============================================================================
// Stage 3: chirp postmultiply, crop back to N, apply normalization
// ============================================================================
//
// `out_n` lets rfft keep only the first N/2 + 1 bins without a second pass;
// full transforms pass out_n == n.

__global__ void bluestein_postmultiply_c64(
    const float2* __restrict__ conv,
    const float2* __restrict__ chirp,
    float2* __restrict__ out,
    int m,
    int out_n,
    int batch_size,
    float scale
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * out_n;
    if (gid >= total) return;

    int k = (int)(gid % out_n);
    long long b = gid / out_n;

    float2 v = bs_cmul_f32(chirp[k], conv[b * (long long)m + k]);
    out[gid] = make_float2(v.x * scale, v.y * scale);
}

__global__ void bluestein_postmultiply_c128(
    const double2* __restrict__ conv,
    const double2* __restrict__ chirp,
    double2* __restrict__ out,
    int m,
    int out_n,
    int batch_size,
    double scale
) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)batch_size * out_n;
    if (gid >= total) return;

    int k = (int)(gid % out_n);
    long long b = gid / out_n;

    double2 v = bs_cmul_f64(chirp[k], conv[b * (long long)m + k]);
    out[gid] = make_double2(v.x * scale, v.y * scale);
}

} // extern "C"
