// Shared device templates for semiring matrix multiplication.
//
//   C[i,j] = reduce_k( combine(A[i,k], B[k,j]) )
//
// Semiring codes, matching `SemiringOp` in src/ops/semiring.rs (the `op`
// kernel argument is that enum's discriminant):
//
//   0 = MinPlus:  reduce=min, combine=+
//   1 = MaxPlus:  reduce=max, combine=+
//   2 = MaxMin:   reduce=max, combine=min
//   3 = MinMax:   reduce=min, combine=max
//   4 = OrAnd:    reduce=OR,  combine=AND
//   5 = PlusMax:  reduce=+,   combine=max
//
// Every dtype runs the same triple loop; only the storage type and the
// accumulator differ, so a policy struct supplies those and `semiring_matmul.cu`
// holds nothing but one `extern "C"` row per dtype.
//
// Simple non-tiled kernel: one thread per output element. The inner loop is not
// amenable to the standard tiled GEMM shared-memory approach, because the
// combine/reduce pair does not distribute the way (+, *) does.

#ifndef NUMR_SEMIRING_MATMUL_OPS_CUH
#define NUMR_SEMIRING_MATMUL_OPS_CUH

#include <climits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Accumulator arithmetic
// ============================================================================
// `min`, `max` and the reduce identities, one overload set per accumulator
// type. The identities are the values `SemiringOp::reduce_identity_f64` names
// (+inf for a min-reduce, -inf for a max-reduce, 0 for OR and +), narrowed the
// way `Element::from_f64` narrows them: an integer type gets its own bound.

__device__ __forceinline__ float numr_sr_min(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float numr_sr_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double numr_sr_min(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ double numr_sr_max(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ int numr_sr_min(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int numr_sr_max(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ long long numr_sr_min(long long a, long long b) { return a < b ? a : b; }
__device__ __forceinline__ long long numr_sr_max(long long a, long long b) { return a > b ? a : b; }
__device__ __forceinline__ unsigned char numr_sr_min(unsigned char a, unsigned char b) { return a < b ? a : b; }
__device__ __forceinline__ unsigned char numr_sr_max(unsigned char a, unsigned char b) { return a > b ? a : b; }

// `combine` for MinPlus/MaxPlus and `reduce` for PlusMax are one elementwise
// addition each, so they WRAP rather than saturate - the CPU reference is
// `SemiringOp::combine`, which adds through `wrapping_add` on I32 and I64.
// The integer overloads add in the unsigned type of the same width, where
// wrapping is defined, and cast back.
__device__ __forceinline__ float numr_sr_add(float a, float b) { return a + b; }
__device__ __forceinline__ double numr_sr_add(double a, double b) { return a + b; }
__device__ __forceinline__ int numr_sr_add(int a, int b) {
    return (int)((unsigned int)a + (unsigned int)b);
}
__device__ __forceinline__ long long numr_sr_add(long long a, long long b) {
    return (long long)((unsigned long long)a + (unsigned long long)b);
}
__device__ __forceinline__ unsigned char numr_sr_add(unsigned char a, unsigned char b) {
    return (unsigned char)((unsigned int)a + (unsigned int)b);
}

template<typename A> struct SrIdentity;

#define NUMR_SR_IDENTITY(A, POS, NEG)                                           \
    template<> struct SrIdentity<A> {                                           \
        static __device__ __forceinline__ A min_reduce() { return (POS); }      \
        static __device__ __forceinline__ A max_reduce() { return (NEG); }      \
    };

NUMR_SR_IDENTITY(float, __int_as_float(0x7f800000), __int_as_float(0xff800000))
NUMR_SR_IDENTITY(double,
                 __longlong_as_double(0x7FF0000000000000LL),
                 __longlong_as_double(0xFFF0000000000000LL))
NUMR_SR_IDENTITY(int, INT_MAX, INT_MIN)
NUMR_SR_IDENTITY(long long, LLONG_MAX, LLONG_MIN)
NUMR_SR_IDENTITY(unsigned char, UCHAR_MAX, 0)

#undef NUMR_SR_IDENTITY

// ============================================================================
// Storage policies
// ============================================================================
// `S` is the element type in the kernel signature, `A` the accumulator. Narrow
// floats accumulate in float, so a min or max over F16 does not round twice.

struct SrF32 {
    typedef float S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct SrF64 {
    typedef double S;
    typedef double A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct SrF16 {
    typedef __half S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return __half2float(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = __float2half(v); }
};

struct SrBF16 {
    typedef __nv_bfloat16 S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return __bfloat162float(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = __float2bfloat16(v); }
};

struct SrFp8E4M3 {
    typedef numr_fp8_e4m3 S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return fp8_e4m3_to_f32(p[i].data); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i].data = f32_to_fp8_e4m3(v); }
};

struct SrFp8E5M2 {
    typedef numr_fp8_e5m2 S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return fp8_e5m2_to_f32(p[i].data); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i].data = f32_to_fp8_e5m2(v); }
};

struct SrI32 {
    typedef int S;
    typedef int A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct SrI64 {
    typedef long long S;
    typedef long long A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct SrU8 {
    typedef unsigned char S;
    typedef unsigned char A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

// ============================================================================
// Semiring operations
// ============================================================================

template<typename A>
__device__ __forceinline__ A sr_identity(unsigned int op) {
    switch (op) {
        case 0: case 3: return SrIdentity<A>::min_reduce();
        case 1: case 2: return SrIdentity<A>::max_reduce();
        // OrAnd and PlusMax both reduce into 0.
        default: return (A)0;
    }
}

template<typename A>
__device__ __forceinline__ A sr_combine(unsigned int op, A a, A b) {
    switch (op) {
        case 0: case 1: return numr_sr_add(a, b);
        case 2: return numr_sr_min(a, b);
        case 3: case 5: return numr_sr_max(a, b);
        case 4: return (a != (A)0 && b != (A)0) ? (A)1 : (A)0;
        default: return numr_sr_add(a, b);
    }
}

template<typename A>
__device__ __forceinline__ A sr_reduce(unsigned int op, A acc, A combined) {
    switch (op) {
        case 0: case 3: return numr_sr_min(acc, combined);
        case 1: case 2: return numr_sr_max(acc, combined);
        case 4: return (combined != (A)0) ? (A)1 : acc;
        case 5: return numr_sr_add(acc, combined);
        default: return numr_sr_min(acc, combined);
    }
}

// One output element: reduce over K, reading A row-major from `a_off` and B
// row-major from `b_off`.
template<typename P>
__device__ __forceinline__ typename P::A sr_dot(
    const typename P::S* __restrict__ A_,
    const typename P::S* __restrict__ B_,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int row,
    unsigned int col,
    unsigned int N,
    unsigned int K,
    unsigned int op
) {
    typedef typename P::A Acc;
    Acc acc = sr_identity<Acc>(op);
    for (unsigned int kk = 0; kk < K; kk++) {
        Acc a_val = P::load(A_, a_off + row * K + kk);
        Acc b_val = P::load(B_, b_off + kk * N + col);
        acc = sr_reduce<Acc>(op, acc, sr_combine<Acc>(op, a_val, b_val));
    }
    return acc;
}

template<typename P>
__device__ void semiring_matmul_impl(
    const typename P::S* __restrict__ A_,
    const typename P::S* __restrict__ B_,
    typename P::S* __restrict__ C_,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    P::store(C_, row * N + col, sr_dot<P>(A_, B_, 0u, 0u, row, col, N, K, op));
}

// Batched: `blockIdx.z` selects the batch. An operand with fewer batches than
// the output is reused cyclically, which is how a broadcast batch dimension
// reaches this kernel.
template<typename P>
__device__ void semiring_matmul_batched_impl(
    const typename P::S* __restrict__ A_,
    const typename P::S* __restrict__ B_,
    typename P::S* __restrict__ C_,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op,
    unsigned int batch_size,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    unsigned int batch = blockIdx.z;
    if (batch >= batch_size) return;

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int a_off = (batch % a_batch_count) * M * K;
    unsigned int b_off = (batch % b_batch_count) * K * N;
    unsigned int c_off = batch * M * N;

    P::store(C_, c_off + row * N + col, sr_dot<P>(A_, B_, a_off, b_off, row, col, N, K, op));
}

#endif // NUMR_SEMIRING_MATMUL_OPS_CUH
