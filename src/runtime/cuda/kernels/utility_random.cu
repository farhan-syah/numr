// Random-sampling CUDA kernels: rand, randn, randint, multinomial.
//
// Split out of utility.cu, which now holds only the deterministic creation
// kernels. This is PTX module "utility_random"
// (kernel_names::UTILITY_RANDOM_MODULE); the generator itself lives in
// rng_xorshift.cuh.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "rng_xorshift.cuh"

extern "C" {

// ============================================================================
// Uniform [0, 1) and standard normal
// ============================================================================

__global__ void rand_f32(float* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        float val = (float)xorshift128plus_uniform(&state);
        // rand must stay in [0, 1). Narrowing the f64 sample to f32 can round
        // a value near 1.0 up to exactly 1.0, so clamp to F32's largest value
        // below 1.0: `largest_value_below_one` in `src/dtype/dtype_enum.rs`
        // is the authority for this bound (1 - 2^-24, 23 mantissa bits).
        if (val >= 1.0f) {
            val = 0.99999994039535522f;
        }
        out[idx] = val;
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
// Random integer [low, low + range)
// ============================================================================
// One draw per element, reduced modulo `range`. The modulo is computed in
// unsigned 64-bit and the offset applied in signed 64-bit, so a negative `low`
// stays correct for the unsigned output types too.

#define NUMR_RANDINT(T, S)                                                      \
    __global__ void randint_##S(T* out, long long low, long long range,         \
                                unsigned long long seed, unsigned int n) {      \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            XorShift128PlusState state;                                         \
            xorshift128plus_init(&state, seed, idx);                            \
            unsigned long long r = xorshift128plus_next(&state);                \
            out[idx] = (T)(low + (long long)(r % (unsigned long long)range));   \
        }                                                                       \
    }

NUMR_RANDINT(signed char, i8)
NUMR_RANDINT(short, i16)
NUMR_RANDINT(int, i32)
NUMR_RANDINT(long long, i64)
NUMR_RANDINT(unsigned char, u8)
NUMR_RANDINT(unsigned short, u16)
NUMR_RANDINT(unsigned int, u32)
NUMR_RANDINT(unsigned long long, u64)

} // extern "C"

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
