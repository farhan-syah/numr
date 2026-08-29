// Utility CUDA kernels
// Supports: fill (initialize tensor with constant value)
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64, u8 (bool)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// Fill Operations - Initialize tensor with constant value
// ============================================================================

__global__ void fill_f32(float* out, float value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_f64(double* out, double value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_f16(__half* out, __half value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_bf16(__nv_bfloat16* out, __nv_bfloat16 value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_i32(int* out, int value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_i64(long long* out, long long value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_u8(unsigned char* out, unsigned char value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_fp8_e4m3(numr_fp8_e4m3* out, numr_fp8_e4m3 value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_fp8_e5m2(numr_fp8_e5m2* out, numr_fp8_e5m2 value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

} // extern "C"

// ============================================================================
// Random Number Generation - Device Functions (outside extern "C")
// ============================================================================

// xorshift128+ state per thread
struct XorShift128PlusState {
    unsigned long long s0;
    unsigned long long s1;
};

// Initialize state from seed and thread index
__device__ __forceinline__ void xorshift128plus_init(XorShift128PlusState* state, unsigned long long seed, unsigned int idx) {
    // Use splitmix64 to initialize both state values from seed + idx
    unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s0 = z ^ (z >> 31);

    z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL + 1;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s1 = z ^ (z >> 31);

    // Ensure non-zero state
    if (state->s0 == 0) state->s0 = 1;
    if (state->s1 == 0) state->s1 = 1;
}

// Generate next random 64-bit value
__device__ __forceinline__ unsigned long long xorshift128plus_next(XorShift128PlusState* state) {
    unsigned long long s1 = state->s0;
    unsigned long long s0 = state->s1;
    unsigned long long result = s0 + s1;
    state->s0 = s0;
    s1 ^= s1 << 23;
    state->s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return result;
}

// Convert to uniform [0, 1)
__device__ __forceinline__ double xorshift128plus_uniform(XorShift128PlusState* state) {
    // Use upper 53 bits for double precision
    return (double)(xorshift128plus_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

// Box-Muller transform for normal distribution
__device__ __forceinline__ void box_muller(XorShift128PlusState* state, float* z0, float* z1) {
    double u1 = xorshift128plus_uniform(state);
    double u2 = xorshift128plus_uniform(state);

    // Avoid log(0)
    if (u1 < 1e-12) u1 = 1e-12;

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    *z0 = (float)(r * cos(theta));
    *z1 = (float)(r * sin(theta));
}

__device__ __forceinline__ void box_muller_f64(XorShift128PlusState* state, double* z0, double* z1) {
    double u1 = xorshift128plus_uniform(state);
    double u2 = xorshift128plus_uniform(state);

    if (u1 < 1e-15) u1 = 1e-15;

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    *z0 = r * cos(theta);
    *z1 = r * sin(theta);
}

// ============================================================================
// Uniform Random [0, 1) - Native CUDA kernels
// ============================================================================

extern "C" {

__global__ void rand_f32(float* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (float)xorshift128plus_uniform(&state);
    }
}

__global__ void rand_f64(double* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = xorshift128plus_uniform(&state);
    }
}

// ============================================================================
// Normal Random (mean=0, std=1) - Native CUDA kernels using Box-Muller
// ============================================================================

__global__ void randn_f32(float* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Box-Muller generates pairs, so we handle two elements per thread when possible
    unsigned int pair_idx = idx * 2;

    if (pair_idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);

        float z0, z1;
        box_muller(&state, &z0, &z1);

        out[pair_idx] = z0;
        if (pair_idx + 1 < n) {
            out[pair_idx + 1] = z1;
        }
    }
}

__global__ void randn_f64(double* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pair_idx = idx * 2;

    if (pair_idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);

        double z0, z1;
        box_muller_f64(&state, &z0, &z1);

        out[pair_idx] = z0;
        if (pair_idx + 1 < n) {
            out[pair_idx + 1] = z1;
        }
    }
}

// F16 variants
__global__ void rand_f16(__half* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        __half val = __float2half((float)xorshift128plus_uniform(&state));
        // rand must stay in [0, 1). Narrowing to half can round a value near
        // 1.0 up to exactly 1.0, so clamp to F16's largest value below 1.0:
        // `largest_value_below_one` in `src/dtype/dtype_enum.rs` is the
        // authority for this bound (2047/2048, 10 mantissa bits).
        if (__hge(val, __float2half(1.0f))) {
            val = __float2half(0.99951171875f);
        }
        out[idx] = val;
    }
}

__global__ void randn_f16(__half* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pair_idx = idx * 2;

    if (pair_idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);

        float z0, z1;
        box_muller(&state, &z0, &z1);

        out[pair_idx] = __float2half(z0);
        if (pair_idx + 1 < n) {
            out[pair_idx + 1] = __float2half(z1);
        }
    }
}

// BF16 variants
__global__ void rand_bf16(__nv_bfloat16* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        float fval = (float)xorshift128plus_uniform(&state);
        __nv_bfloat16 val = __float2bfloat16(fval);
        // rand must stay in [0, 1). Narrowing to bf16 can round a value near
        // 1.0 up to exactly 1.0, so clamp to BF16's largest value below 1.0:
        // `largest_value_below_one` in `src/dtype/dtype_enum.rs` is the
        // authority for this bound (255/256, 7 mantissa bits).
        if (__bfloat162float(val) >= 1.0f) {
            val = __float2bfloat16(0.99609375f);
        }
        out[idx] = val;
    }
}

__global__ void randn_bf16(__nv_bfloat16* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pair_idx = idx * 2;

    if (pair_idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);

        float z0, z1;
        box_muller(&state, &z0, &z1);

        out[pair_idx] = __float2bfloat16(z0);
        if (pair_idx + 1 < n) {
            out[pair_idx + 1] = __float2bfloat16(z1);
        }
    }
}

// ============================================================================
// Random Integer [low, high) - Native CUDA kernels
// ============================================================================

__global__ void randint_i8(signed char* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = (signed char)(low + (long long)(r % (unsigned long long)range));
    }
}

__global__ void randint_i16(short* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = (short)(low + (long long)(r % (unsigned long long)range));
    }
}

__global__ void randint_i32(int* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = (int)(low + (long long)(r % (unsigned long long)range));
    }
}

__global__ void randint_i64(long long* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = low + (long long)(r % (unsigned long long)range);
    }
}

__global__ void randint_u8(unsigned char* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = (unsigned char)(low + (long long)(r % (unsigned long long)range));
    }
}

__global__ void randint_u16(unsigned short* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = (unsigned short)(low + (long long)(r % (unsigned long long)range));
    }
}

__global__ void randint_u32(unsigned int* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = (unsigned int)(low + (long long)(r % (unsigned long long)range));
    }
}

__global__ void randint_u64(unsigned long long* out, long long low, long long range, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        unsigned long long r = xorshift128plus_next(&state);
        out[idx] = (unsigned long long)(low + (long long)(r % (unsigned long long)range));
    }
}

// ============================================================================
// Arange - Generate evenly spaced values in [start, stop)
// ============================================================================

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

} // extern "C" - close before template functions

// ============================================================================
// Multinomial Sampling - Template device functions (outside extern "C")
// ============================================================================

// Multinomial with replacement: each thread samples one index for one distribution
// Uses prefix sum (CDF) + binary search for inverse transform sampling
// Note: This is a device function that contains the kernel logic, called from typed __global__ wrappers
template<typename T>
__device__ void multinomial_with_replacement_impl(
    const T* probs,           // [num_distributions, num_categories]
    long long* out,           // [num_distributions, num_samples]
    unsigned long long seed,
    unsigned int num_distributions,
    unsigned int num_categories,
    unsigned int num_samples
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_distributions * num_samples;
    if (idx >= total) return;

    unsigned int dist = idx / num_samples;
    unsigned int sample = idx % num_samples;

    // Initialize RNG for this thread
    XorShift128PlusState state;
    xorshift128plus_init(&state, seed, idx);

    // Get pointer to this distribution's probabilities
    const T* prob_row = probs + dist * num_categories;

    // Compute sum of probabilities for normalization
    double sum = 0.0;
    for (unsigned int i = 0; i < num_categories; i++) {
        sum += (double)prob_row[i];
    }

    // Generate uniform random value
    double u = xorshift128plus_uniform(&state);

    // Binary search using CDF (on-the-fly computation)
    // Find smallest index where cumsum/sum >= u
    double cumsum = 0.0;
    unsigned int result = num_categories - 1;  // Default to last category
    for (unsigned int i = 0; i < num_categories; i++) {
        cumsum += (double)prob_row[i];
        if (cumsum / sum >= u) {
            result = i;
            break;
        }
    }

    out[dist * num_samples + sample] = (long long)result;
}

// Multinomial without replacement: requires sequential sampling within each distribution
// Each thread block handles one distribution
// Note: This is a device function that contains the kernel logic, called from typed __global__ wrappers
template<typename T>
__device__ void multinomial_without_replacement_impl(
    const T* probs,           // [num_distributions, num_categories]
    long long* out,           // [num_distributions, num_samples]
    unsigned long long seed,
    unsigned int num_distributions,
    unsigned int num_categories,
    unsigned int num_samples,
    double* shared_probs      // Shared memory passed from kernel
) {
    unsigned int dist = blockIdx.x;
    if (dist >= num_distributions) return;

    // Only thread 0 does the work (sequential sampling requirement)
    if (threadIdx.x != 0) return;

    // Initialize RNG
    XorShift128PlusState state;
    xorshift128plus_init(&state, seed, dist);

    // Get pointers
    const T* prob_row = probs + dist * num_categories;
    long long* out_row = out + dist * num_samples;

    // Copy probabilities to shared memory (so we can zero them out)
    for (unsigned int i = 0; i < num_categories; i++) {
        shared_probs[i] = (double)prob_row[i];
    }

    // Sample without replacement
    for (unsigned int s = 0; s < num_samples; s++) {
        // Compute sum of remaining probabilities
        double sum = 0.0;
        for (unsigned int i = 0; i < num_categories; i++) {
            sum += shared_probs[i];
        }

        // Generate uniform random value
        double u = xorshift128plus_uniform(&state);

        // Binary search using CDF
        double cumsum = 0.0;
        unsigned int result = num_categories - 1;
        for (unsigned int i = 0; i < num_categories; i++) {
            cumsum += shared_probs[i];
            if (cumsum / sum >= u) {
                result = i;
                break;
            }
        }

        out_row[s] = (long long)result;

        // Zero out selected category
        shared_probs[result] = 0.0;
    }
}

// ============================================================================
// Multinomial Sampling - Typed kernel wrappers (inside extern "C")
// ============================================================================

extern "C" {

// Instantiate for F32
__global__ void multinomial_with_replacement_f32(
    const float* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    multinomial_with_replacement_impl<float>(probs, out, seed, num_distributions, num_categories, num_samples);
}

__global__ void multinomial_without_replacement_f32(
    const float* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    extern __shared__ double shared_probs[];
    multinomial_without_replacement_impl<float>(probs, out, seed, num_distributions, num_categories, num_samples, shared_probs);
}

// Instantiate for F64
__global__ void multinomial_with_replacement_f64(
    const double* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    multinomial_with_replacement_impl<double>(probs, out, seed, num_distributions, num_categories, num_samples);
}

__global__ void multinomial_without_replacement_f64(
    const double* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    extern __shared__ double shared_probs[];
    multinomial_without_replacement_impl<double>(probs, out, seed, num_distributions, num_categories, num_samples, shared_probs);
}

// Instantiate for F16
__global__ void multinomial_with_replacement_f16(
    const __half* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    multinomial_with_replacement_impl<__half>(probs, out, seed, num_distributions, num_categories, num_samples);
}

__global__ void multinomial_without_replacement_f16(
    const __half* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    extern __shared__ double shared_probs[];
    multinomial_without_replacement_impl<__half>(probs, out, seed, num_distributions, num_categories, num_samples, shared_probs);
}

// Instantiate for BF16
__global__ void multinomial_with_replacement_bf16(
    const __nv_bfloat16* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    multinomial_with_replacement_impl<__nv_bfloat16>(probs, out, seed, num_distributions, num_categories, num_samples);
}

__global__ void multinomial_without_replacement_bf16(
    const __nv_bfloat16* probs, long long* out, unsigned long long seed,
    unsigned int num_distributions, unsigned int num_categories, unsigned int num_samples
) {
    extern __shared__ double shared_probs[];
    multinomial_without_replacement_impl<__nv_bfloat16>(probs, out, seed, num_distributions, num_categories, num_samples, shared_probs);
}

} // extern "C"
