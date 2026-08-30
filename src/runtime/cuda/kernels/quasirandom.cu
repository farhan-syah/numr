// Quasi-random sequence generation CUDA kernels
// Implements: Sobol, Halton, Latin Hypercube Sampling
// Types: f32, f64

#include <math.h>

// ============================================================================
// Sobol Sequence - Direction Vectors
// ============================================================================

// Direction vectors are passed from host via global memory.
// This allows support for all 21,201 dimensions from Joe & Kuo (2008).
// Each dimension has 32 direction vectors (one per bit).

// ============================================================================
// Halton Sequence - Prime Numbers
// ============================================================================

// First 100 prime numbers for Halton sequence
__constant__ unsigned int primes[100] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541
};

// ============================================================================
// Random Number Generation for LHS
// ============================================================================

struct XorShift128PlusState {
    unsigned long long s0;
    unsigned long long s1;
};

__device__ __forceinline__ void xorshift128plus_init(XorShift128PlusState* state, unsigned long long seed, unsigned int idx) {
    unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s0 = z ^ (z >> 31);

    z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL + 1;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s1 = z ^ (z >> 31);

    if (state->s0 == 0) state->s0 = 1;
    if (state->s1 == 0) state->s1 = 1;
}

__device__ __forceinline__ unsigned long long xorshift128plus_next(XorShift128PlusState* state) {
    unsigned long long s1 = state->s0;
    unsigned long long s0 = state->s1;
    unsigned long long result = s0 + s1;
    state->s0 = s0;
    s1 ^= s1 << 23;
    state->s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return result;
}

__device__ __forceinline__ double xorshift128plus_uniform(XorShift128PlusState* state) {
    return (double)(xorshift128plus_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

// ============================================================================
// van der Corput Sequence (used for Halton)
// ============================================================================

__device__ __forceinline__ float van_der_corput_f32(unsigned int index, unsigned int base) {
    float result = 0.0f;
    float f = 1.0f / (float)base;
    unsigned int i = index;
    while (i > 0) {
        result += f * (float)(i % base);
        i /= base;
        f /= (float)base;
    }
    return result;
}

__device__ __forceinline__ double van_der_corput_f64(unsigned int index, unsigned int base) {
    double result = 0.0;
    double f = 1.0 / (double)base;
    unsigned int i = index;
    while (i > 0) {
        result += f * (double)(i % base);
        i /= base;
        f /= (double)base;
    }
    return result;
}

// ============================================================================
// Sobol Sequence Kernels
// ============================================================================

extern "C" {

// Each thread generates one point across all dimensions
// direction_vectors: [dimension][32] pre-computed direction vectors from host
__global__ void sobol_f32(
    float* out,
    const unsigned int* direction_vectors,
    unsigned int n_points,
    unsigned int dimension,
    unsigned int skip
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        unsigned int point_index = skip + idx;

        // Gray code
        unsigned int gray = point_index ^ (point_index >> 1);

        for (unsigned int d = 0; d < dimension; d++) {
            // Get direction vectors for this dimension
            const unsigned int* v = direction_vectors + d * 32;

            // Compute Sobol point using direction vectors
            unsigned int x = 0;
            for (int bit = 0; bit < 32; bit++) {
                if ((gray & (1u << bit)) != 0) {
                    x ^= v[bit];
                }
            }

            // Convert to float in [0, 1)
            out[idx * dimension + d] = (float)x / (float)(1ULL << 32);
        }
    }
}

__global__ void sobol_f64(
    double* out,
    const unsigned int* direction_vectors,
    unsigned int n_points,
    unsigned int dimension,
    unsigned int skip
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        unsigned int point_index = skip + idx;

        // Gray code
        unsigned int gray = point_index ^ (point_index >> 1);

        for (unsigned int d = 0; d < dimension; d++) {
            // Get direction vectors for this dimension
            const unsigned int* v = direction_vectors + d * 32;

            // Compute Sobol point using direction vectors
            unsigned int x = 0;
            for (int bit = 0; bit < 32; bit++) {
                if ((gray & (1u << bit)) != 0) {
                    x ^= v[bit];
                }
            }

            // Convert to double in [0, 1)
            out[idx * dimension + d] = (double)x / (double)(1ULL << 32);
        }
    }
}

// ============================================================================
// Halton Sequence Kernels
// ============================================================================

__global__ void halton_f32(float* out, unsigned int n_points, unsigned int dimension, unsigned int skip) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        unsigned int point_index = skip + idx;

        for (unsigned int d = 0; d < dimension; d++) {
            unsigned int base = primes[d];
            out[idx * dimension + d] = van_der_corput_f32(point_index, base);
        }
    }
}

__global__ void halton_f64(double* out, unsigned int n_points, unsigned int dimension, unsigned int skip) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        unsigned int point_index = skip + idx;

        for (unsigned int d = 0; d < dimension; d++) {
            unsigned int base = primes[d];
            out[idx * dimension + d] = van_der_corput_f64(point_index, base);
        }
    }
}

// ============================================================================
// Latin Hypercube Sampling Kernels
// ============================================================================

// Fisher-Yates shuffle in shared memory
__device__ void fisher_yates_shuffle(unsigned int* array, unsigned int n, XorShift128PlusState* state) {
    for (unsigned int i = n - 1; i > 0; i--) {
        unsigned int j = (unsigned int)(xorshift128plus_uniform(state) * (i + 1));
        if (j > i) j = i; // Clamp to avoid overflow
        unsigned int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

__global__ void latin_hypercube_f32(float* out, unsigned int n_samples, unsigned int dimension, unsigned long long seed) {
    unsigned int dim = blockIdx.x;
    if (dim < dimension) {
        // Each block handles one dimension
        // Allocate shared memory for permutation
        extern __shared__ unsigned int intervals[];

        // Initialize intervals [0, 1, 2, ..., n_samples-1]
        for (unsigned int i = threadIdx.x; i < n_samples; i += blockDim.x) {
            intervals[i] = i;
        }
        __syncthreads();

        // Shuffle intervals using first thread
        if (threadIdx.x == 0) {
            XorShift128PlusState state;
            xorshift128plus_init(&state, seed, dim);
            fisher_yates_shuffle(intervals, n_samples, &state);
        }
        __syncthreads();

        // Generate random point within each interval
        for (unsigned int i = threadIdx.x; i < n_samples; i += blockDim.x) {
            XorShift128PlusState state;
            xorshift128plus_init(&state, seed, dim * n_samples + i);

            unsigned int interval = intervals[i];
            float lower = (float)interval / (float)n_samples;
            float upper = (float)(interval + 1) / (float)n_samples;
            float random_offset = (float)xorshift128plus_uniform(&state);
            // random_offset must stay in [0, 1). Narrowing the f64 sample to
            // f32 can round a value near 1.0 up to exactly 1.0, so clamp to
            // F32's largest value below 1.0: `largest_value_below_one` in
            // `src/dtype/dtype_enum.rs` is the authority for this bound
            // (1 - 2^-24, 23 mantissa bits).
            if (random_offset >= 1.0f) {
                random_offset = 0.99999994039535522f;
            }

            out[i * dimension + dim] = lower + random_offset * (upper - lower);
        }
    }
}

__global__ void latin_hypercube_f64(double* out, unsigned int n_samples, unsigned int dimension, unsigned long long seed) {
    unsigned int dim = blockIdx.x;
    if (dim < dimension) {
        // Each block handles one dimension
        // Allocate shared memory for permutation
        extern __shared__ unsigned int intervals[];

        // Initialize intervals [0, 1, 2, ..., n_samples-1]
        for (unsigned int i = threadIdx.x; i < n_samples; i += blockDim.x) {
            intervals[i] = i;
        }
        __syncthreads();

        // Shuffle intervals using first thread
        if (threadIdx.x == 0) {
            XorShift128PlusState state;
            xorshift128plus_init(&state, seed, dim);
            fisher_yates_shuffle(intervals, n_samples, &state);
        }
        __syncthreads();

        // Generate random point within each interval
        for (unsigned int i = threadIdx.x; i < n_samples; i += blockDim.x) {
            XorShift128PlusState state;
            xorshift128plus_init(&state, seed, dim * n_samples + i);

            unsigned int interval = intervals[i];
            double lower = (double)interval / (double)n_samples;
            double upper = (double)(interval + 1) / (double)n_samples;
            double random_offset = xorshift128plus_uniform(&state);

            out[i * dimension + dim] = lower + random_offset * (upper - lower);
        }
    }
}

} // extern "C"
