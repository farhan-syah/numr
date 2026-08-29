// xorshift128+ pseudo-random generator, shared by the random-sampling kernels
// in utility_random.cu.
//
// Each thread seeds its own state from (seed, element index) through splitmix64,
// so a given seed reproduces the same tensor regardless of launch geometry.

#ifndef NUMR_RNG_XORSHIFT_CUH
#define NUMR_RNG_XORSHIFT_CUH


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

#endif // NUMR_RNG_XORSHIFT_CUH
