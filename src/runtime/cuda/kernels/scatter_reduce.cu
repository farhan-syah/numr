// Scatter-with-reduction CUDA kernels.
//
// Two families, because floats and integers cannot use the same strategy:
//
//  * FLOAT (f32, f64): one thread per SOURCE element, combining into the
//    destination with an atomic. `mean` is three passes — atomic sum, atomic
//    count, then an element-wise divide — which is exact enough in a float
//    accumulator and keeps the fast atomic path.
//
//  * INTEGER (i64, i32, i16, i8, u64, u32, u16, u8): one thread per
//    DESTINATION element, scanning the source positions that map to it. There
//    is no 128-bit atomic, and an atomic in the element type would wrap: the
//    running total is an ACCUMULATOR, and this project's convention is that
//    accumulators saturate while elementwise ops wrap (see
//    src/runtime/cpu/kernels/wide_acc.rs). Owning the destination element lets
//    the thread keep a `Numr128` accumulator across the whole reduction and
//    narrow with saturation exactly once, at the store.
//
//    `mean` therefore divides ONCE, at the end, inside the 128-bit
//    accumulator: sum in Numr128, then numr128_div_u64_trunc by the count.
//    Summing in the element type and dividing the saturated total would report
//    INT_MAX/2 for a set whose true mean is representable, and a running
//    per-element mean would round every step.
//
// Both families read the destination already initialised by the caller — a
// copy of `dst` when include_self is set, otherwise the reduction's identity —
// so this file never seeds it.
//
// Semantics match the CPU reference `scatter_reduce_kernel` in
// src/runtime/cpu/kernels/index.rs: the index tensor is element-wise with the
// source, an out-of-range index is skipped, and `mean` divides by the number
// of contributions including the destination's own when include_self is set.
//
// Kernel naming matches the names launch_scatter_reduce and
// launch_scatter_reduce_int build in src/runtime/cuda/kernels/index/scatter.rs.
// This is PTX module "scatter_reduce" (kernel_names::SCATTER_REDUCE_MODULE).

#include "dtype_traits.cuh"
#include "numr128.cuh"

// ============================================================================
// Float atomics that CUDA does not provide natively
// ============================================================================
// Compare-and-swap loops on the integer view of the float. The retry condition
// is `assumed != old`, so a lane that lost the race recomputes against the
// value that won.

#define NUMR_ATOMIC_CAS_F32(NAME, EXPR)                                         \
    __device__ __forceinline__ float NAME(float* address, float val) {          \
        int* address_as_int = (int*)address;                                    \
        int old = *address_as_int;                                              \
        int assumed;                                                            \
        do {                                                                    \
            assumed = old;                                                      \
            float old_val = __int_as_float(assumed);                            \
            float new_val = (EXPR);                                             \
            old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));  \
        } while (assumed != old);                                               \
        return __int_as_float(old);                                             \
    }

#define NUMR_ATOMIC_CAS_F64(NAME, EXPR)                                         \
    __device__ __forceinline__ double NAME(double* address, double val) {       \
        unsigned long long* address_as_ull = (unsigned long long*)address;      \
        unsigned long long old = *address_as_ull;                               \
        unsigned long long assumed;                                             \
        do {                                                                    \
            assumed = old;                                                      \
            double old_val = __longlong_as_double(assumed);                     \
            double new_val = (EXPR);                                            \
            old = atomicCAS(address_as_ull, assumed,                            \
                            __double_as_longlong(new_val));                     \
        } while (assumed != old);                                               \
        return __longlong_as_double(old);                                       \
    }

NUMR_ATOMIC_CAS_F32(atomicMaxFloat, fmaxf(old_val, val))
NUMR_ATOMIC_CAS_F32(atomicMinFloat, fminf(old_val, val))
NUMR_ATOMIC_CAS_F32(atomicMulFloat, old_val * val)
NUMR_ATOMIC_CAS_F64(atomicMaxDouble, fmax(old_val, val))
NUMR_ATOMIC_CAS_F64(atomicMinDouble, fmin(old_val, val))
NUMR_ATOMIC_CAS_F64(atomicMulDouble, old_val * val)

#undef NUMR_ATOMIC_CAS_F32
#undef NUMR_ATOMIC_CAS_F64

// ============================================================================
// Float family: one thread per source element
// ============================================================================

// Resolve a source element's flat position and its destination position.
// Returns false when the index is out of range, which the caller skips.
__device__ __forceinline__ bool scatter_reduce_slots(
    const long long* __restrict__ indices, unsigned int idx,
    unsigned int dim_size, unsigned int inner_size, unsigned int src_dim_size,
    unsigned int* src_idx, unsigned int* dst_idx
) {
    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    *src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise with src, so the index is read at the
    // source element's own position, not at its coordinate along dim.
    long long index_val = indices[*src_idx];
    if (index_val < 0 || (unsigned long long)index_val >= dim_size) {
        return false;
    }

    *dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    return true;
}

#define NUMR_SCATTER_REDUCE_ATOMIC(OP, T, S, ATOMIC)                            \
    __global__ void scatter_reduce_##OP##_##S(                                  \
        const T* __restrict__ src, const long long* __restrict__ indices,       \
        T* __restrict__ dst, unsigned int, unsigned int outer_size,             \
        unsigned int dim_size, unsigned int inner_size,                         \
        unsigned int src_dim_size) {                                            \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx >= outer_size * src_dim_size * inner_size) return;              \
        unsigned int src_idx, dst_idx;                                          \
        if (!scatter_reduce_slots(indices, idx, dim_size, inner_size,           \
                                  src_dim_size, &src_idx, &dst_idx)) return;    \
        ATOMIC(&dst[dst_idx], src[src_idx]);                                    \
    }

// Counts contributions per destination element, the denominator of float mean.
#define NUMR_SCATTER_REDUCE_COUNT(T, S)                                         \
    __global__ void scatter_reduce_count_##S(                                   \
        const long long* __restrict__ indices, T* __restrict__ count,           \
        unsigned int, unsigned int outer_size, unsigned int dim_size,           \
        unsigned int inner_size, unsigned int src_dim_size) {                   \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx >= outer_size * src_dim_size * inner_size) return;              \
        unsigned int src_idx, dst_idx;                                          \
        if (!scatter_reduce_slots(indices, idx, dim_size, inner_size,           \
                                  src_dim_size, &src_idx, &dst_idx)) return;    \
        atomicAdd(&count[dst_idx], (T)1);                                       \
    }

// A destination element nobody scattered into has count 0 and keeps 0, which is
// what the CPU epilogue's `if count > 0` guard leaves behind.
#define NUMR_SCATTER_REDUCE_MEAN_DIV(T, S)                                      \
    __global__ void scatter_reduce_mean_div_##S(                                \
        const T* __restrict__ sum_buf, const T* __restrict__ count_buf,         \
        T* __restrict__ output, unsigned int n) {                               \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx >= n) return;                                                   \
        T c = count_buf[idx];                                                   \
        output[idx] = (c > (T)0) ? (sum_buf[idx] / c) : (T)0;                   \
    }

#define NUMR_SCATTER_REDUCE_FLOAT_ROW(T, S, MAXFN, MINFN, MULFN)                \
    NUMR_SCATTER_REDUCE_ATOMIC(sum, T, S, atomicAdd)                            \
    NUMR_SCATTER_REDUCE_ATOMIC(max, T, S, MAXFN)                                \
    NUMR_SCATTER_REDUCE_ATOMIC(min, T, S, MINFN)                                \
    NUMR_SCATTER_REDUCE_ATOMIC(prod, T, S, MULFN)                               \
    NUMR_SCATTER_REDUCE_COUNT(T, S)                                             \
    NUMR_SCATTER_REDUCE_MEAN_DIV(T, S)

// ============================================================================
// Integer family: one thread per destination element
// ============================================================================

#define NUMR_SR_SUM  0
#define NUMR_SR_PROD 1
#define NUMR_SR_MAX  2
#define NUMR_SR_MIN  3
#define NUMR_SR_MEAN 4

template<typename T, int OP>
__device__ __forceinline__ void scatter_reduce_int_impl(
    const T* __restrict__ src, const long long* __restrict__ indices,
    T* __restrict__ dst, unsigned int outer_size, unsigned int dim_size,
    unsigned int inner_size, unsigned int src_dim_size, unsigned int include_self
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * dim_size * inner_size) return;

    unsigned int inner = idx % inner_size;
    unsigned int d = (idx / inner_size) % dim_size;
    unsigned int outer = idx / (dim_size * inner_size);
    unsigned int dst_idx = outer * dim_size * inner_size + d * inner_size + inner;

    // Only source elements in this destination's own (outer, inner) lane can
    // land here, so the scan is over src_dim_size, not the whole source.
    unsigned int lane_base = outer * src_dim_size * inner_size + inner;

    if (OP == NUMR_SR_MAX || OP == NUMR_SR_MIN) {
        // Comparison needs no accumulator: the result is always one of the
        // inputs, so it is exact in the element type.
        T best = dst[dst_idx];
        for (unsigned int s = 0; s < src_dim_size; s++) {
            unsigned int src_idx = lane_base + s * inner_size;
            long long iv = indices[src_idx];
            if (iv < 0 || (unsigned long long)iv >= dim_size) continue;
            if ((unsigned int)iv != d) continue;
            T v = src[src_idx];
            if (OP == NUMR_SR_MAX ? (v > best) : (v < best)) {
                best = v;
            }
        }
        dst[dst_idx] = best;
        return;
    }

    Numr128 acc = Numr128From<T>::apply(dst[dst_idx]);
    // include_self makes the destination's own value one of the averaged
    // contributions, matching the CPU kernel's counts[] seed of 1.
    unsigned long long count = include_self ? 1ULL : 0ULL;

    for (unsigned int s = 0; s < src_dim_size; s++) {
        unsigned int src_idx = lane_base + s * inner_size;
        long long iv = indices[src_idx];
        if (iv < 0 || (unsigned long long)iv >= dim_size) continue;
        if ((unsigned int)iv != d) continue;
        Numr128 v = Numr128From<T>::apply(src[src_idx]);
        acc = (OP == NUMR_SR_PROD) ? numr128_mul_sat(acc, v) : numr128_add_sat(acc, v);
        count++;
    }

    if (OP == NUMR_SR_MEAN) {
        acc = numr128_div_u64_trunc(acc, count);
    }
    dst[dst_idx] = Numr128Narrow<T>::apply(acc);
}

#define NUMR_SCATTER_REDUCE_INT(OP_NAME, OP, T, S)                              \
    __global__ void scatter_reduce_int_##OP_NAME##_##S(                         \
        const T* __restrict__ src, const long long* __restrict__ indices,       \
        T* __restrict__ dst, unsigned int outer_size, unsigned int dim_size,    \
        unsigned int inner_size, unsigned int src_dim_size,                     \
        unsigned int include_self) {                                            \
        scatter_reduce_int_impl<T, OP>(src, indices, dst, outer_size, dim_size, \
                                       inner_size, src_dim_size, include_self); \
    }

#define NUMR_SCATTER_REDUCE_INT_ROW(T, S)                                       \
    NUMR_SCATTER_REDUCE_INT(sum, NUMR_SR_SUM, T, S)                             \
    NUMR_SCATTER_REDUCE_INT(prod, NUMR_SR_PROD, T, S)                           \
    NUMR_SCATTER_REDUCE_INT(max, NUMR_SR_MAX, T, S)                             \
    NUMR_SCATTER_REDUCE_INT(min, NUMR_SR_MIN, T, S)                             \
    NUMR_SCATTER_REDUCE_INT(mean, NUMR_SR_MEAN, T, S)

extern "C" {

NUMR_SCATTER_REDUCE_FLOAT_ROW(float, f32, atomicMaxFloat, atomicMinFloat, atomicMulFloat)
NUMR_SCATTER_REDUCE_FLOAT_ROW(double, f64, atomicMaxDouble, atomicMinDouble, atomicMulDouble)

// The element types are spelled `long long` rather than `int64_t` because
// Numr128From and Numr128Narrow are specialised on the built-in names, and on
// LP64 `int64_t` is `long`, a third distinct type with no specialisation.
NUMR_SCATTER_REDUCE_INT_ROW(long long, i64)
NUMR_SCATTER_REDUCE_INT_ROW(int, i32)
NUMR_SCATTER_REDUCE_INT_ROW(short, i16)
NUMR_SCATTER_REDUCE_INT_ROW(signed char, i8)
NUMR_SCATTER_REDUCE_INT_ROW(unsigned long long, u64)
NUMR_SCATTER_REDUCE_INT_ROW(unsigned int, u32)
NUMR_SCATTER_REDUCE_INT_ROW(unsigned short, u16)
NUMR_SCATTER_REDUCE_INT_ROW(unsigned char, u8)

} // extern "C"
