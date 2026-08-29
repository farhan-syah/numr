// Deterministic tensor-creation CUDA kernels: fill, arange, linspace, eye.
//
// Random sampling lives in utility_random.cu (PTX module "utility_random");
// this file is PTX module "utility" (kernel_names::UTILITY_MODULE).
//
// Kernel naming matches the names the Rust launchers build in
// src/runtime/cuda/kernels/utility.rs from dtype_suffix() in loader.rs.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// Fill - initialise a tensor with a constant value
// ============================================================================
// One kernel per dtype, because the value arrives as a kernel argument of that
// dtype rather than as a widened scalar. Every dtype FillValue can name has a
// kernel here: a dtype missing from this list would fall back to a wider fill
// kernel and write past the end of each element.

#define NUMR_FILL(T, S)                                                         \
    __global__ void fill_##S(T* out, T value, unsigned int n) {                 \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = value;                                                   \
        }                                                                       \
    }

NUMR_FILL(float, f32)
NUMR_FILL(double, f64)
NUMR_FILL(__half, f16)
NUMR_FILL(__nv_bfloat16, bf16)
NUMR_FILL(numr_fp8_e4m3, fp8_e4m3)
NUMR_FILL(numr_fp8_e5m2, fp8_e5m2)
NUMR_FILL(long long, i64)
NUMR_FILL(int, i32)
NUMR_FILL(short, i16)
NUMR_FILL(signed char, i8)
NUMR_FILL(unsigned long long, u64)
NUMR_FILL(unsigned int, u32)
NUMR_FILL(unsigned short, u16)
NUMR_FILL(unsigned char, u8)

// ============================================================================
// Arange, linspace, eye
// ============================================================================
// These keep one hand-written kernel per dtype: the scalar parameters differ in
// type from row to row (integer linspace interpolates in double, unsigned
// arange offsets in signed 64-bit), and those types are part of the launch ABI.


__global__ void arange_f32(float* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + step * (float)idx;
    }
}

__global__ void arange_f64(double* out, double start, double step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + step * (double)idx;
    }
}

__global__ void arange_f16(__half* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(start + step * (float)idx);
    }
}

__global__ void arange_bf16(__nv_bfloat16* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(start + step * (float)idx);
    }
}

__global__ void arange_i32(int* out, int start, int step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + step * (int)idx;
    }
}

__global__ void arange_i64(long long* out, long long start, long long step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + step * (long long)idx;
    }
}

__global__ void arange_u32(unsigned int* out, unsigned int start, int step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use signed arithmetic to avoid overflow when step is negative
        // Compute offset as signed, then add to start
        long long offset = (long long)step * (long long)idx;
        out[idx] = (unsigned int)((long long)start + offset);
    }
}

__global__ void arange_u64(unsigned long long* out, unsigned long long start, long long step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use signed arithmetic to avoid overflow when step is negative
        // Cast to signed for computation, then back to unsigned
        long long signed_start = (long long)start;
        long long offset = step * (long long)idx;
        out[idx] = (unsigned long long)(signed_start + offset);
    }
}

// ============================================================================
// Linspace - Generate evenly spaced values from start to stop (inclusive)
// ============================================================================

__global__ void linspace_f32(float* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx] = start + (stop - start) * t;
    }
}

__global__ void linspace_f64(double* out, double start, double stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        double t = (double)idx / (double)(steps - 1);
        out[idx] = start + (stop - start) * t;
    }
}

__global__ void linspace_f16(__half* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx] = __float2half(start + (stop - start) * t);
    }
}

__global__ void linspace_bf16(__nv_bfloat16* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx] = __float2bfloat16(start + (stop - start) * t);
    }
}

// Integer linspace - computation in double, then convert to integer
// This matches NumPy behavior and allows linspace to work with all dtypes
__global__ void linspace_i32(int* out, double start, double stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        double t = (double)idx / (double)(steps - 1);
        out[idx] = (int)(start + (stop - start) * t);
    }
}

__global__ void linspace_i64(long long* out, double start, double stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        double t = (double)idx / (double)(steps - 1);
        out[idx] = (long long)(start + (stop - start) * t);
    }
}

__global__ void linspace_u32(unsigned int* out, double start, double stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        double t = (double)idx / (double)(steps - 1);
        out[idx] = (unsigned int)(start + (stop - start) * t);
    }
}

__global__ void linspace_u64(unsigned long long* out, double start, double stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        double t = (double)idx / (double)(steps - 1);
        out[idx] = (unsigned long long)(start + (stop - start) * t);
    }
}

// ============================================================================
// Eye - Generate identity matrix
// ============================================================================

__global__ void eye_f32(float* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void eye_f64(double* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? 1.0 : 0.0;
    }
}

__global__ void eye_f16(__half* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void eye_bf16(__nv_bfloat16* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    }
}

__global__ void eye_i32(int* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? 1 : 0;
    }
}

__global__ void eye_i64(long long* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? 1LL : 0LL;
    }
}

__global__ void eye_u32(unsigned int* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? 1u : 0u;
    }
}

__global__ void eye_u64(unsigned long long* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx] = (row == col) ? 1ULL : 0ULL;
    }
}

// ============================================================================
// FP8 Arange
// ============================================================================

__global__ void arange_fp8_e4m3(numr_fp8_e4m3* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].data = f32_to_fp8_e4m3(start + step * (float)idx);
    }
}

__global__ void arange_fp8_e5m2(numr_fp8_e5m2* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].data = f32_to_fp8_e5m2(start + step * (float)idx);
    }
}

// ============================================================================
// FP8 Linspace
// ============================================================================

__global__ void linspace_fp8_e4m3(numr_fp8_e4m3* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx].data = f32_to_fp8_e4m3(start + (stop - start) * t);
    }
}

__global__ void linspace_fp8_e5m2(numr_fp8_e5m2* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx].data = f32_to_fp8_e5m2(start + (stop - start) * t);
    }
}

// ============================================================================
// FP8 Eye
// ============================================================================

__global__ void eye_fp8_e4m3(numr_fp8_e4m3* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx].data = (row == col) ? f32_to_fp8_e4m3(1.0f) : f32_to_fp8_e4m3(0.0f);
    }
}

__global__ void eye_fp8_e5m2(numr_fp8_e5m2* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;
    if (idx < total) {
        unsigned int row = idx / m;
        unsigned int col = idx % m;
        out[idx].data = (row == col) ? f32_to_fp8_e5m2(1.0f) : f32_to_fp8_e5m2(0.0f);
    }
}

} // extern "C"
