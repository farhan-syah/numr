// Integer reduction CUDA kernels (I64, I32, I16, I8, U64, U32, U16, U8)
//
// These are the integer counterparts of the float kernels in `reduce.cu`, split
// into their own translation unit for the same reason `gemv_int.cu` and
// `matmul_int.cu` were split out: they accumulate in `Numr128` rather than in a
// float register, and `reduce.cu` is already at its size limit.
//
// The grid geometry, the block-level tree reduction, and the [outer, reduce,
// inner] view of the tensor are identical to `reduce.cu` - only the accumulator
// changes. The Rust launcher computes one grid for both modules.
//
// Accumulator per op, matching `runtime/cpu/kernels/reduce/int_acc.rs`:
//
//   sum, prod   Numr128, saturating, narrowed and saturated once at the store.
//   mean        Numr128 sum, then a truncating divide by the reduce count,
//               then the same single narrow.
//   min, max    the element type itself - a comparison never leaves the range.
//   any, all    an int bitmask, exactly as the float path does.
//
// min/max seed from the element type's own bounds rather than from +/-INFINITY,
// which `reduce.cu` uses and which has no integer equivalent. The bound is a
// true identity here: `max(x, T_MIN) == x` for every representable x, including
// x == T_MIN.

#include "numr128.cuh"

// The launcher fixes the block at BLOCK_SIZE (256) for every reduction, so the
// shared arrays below are sized to it exactly, as `reduce.cu`'s are.
#define REDUCE_INT_BLOCK 256

// ============================================================================
// Element-type bounds, used as the identity for min/max/argmax/argmin.
// ============================================================================

template<typename T> struct IntLimits;

#define NUMR_INT_LIMITS(T, LO, HI) \
template<> struct IntLimits<T> { \
    static __device__ __forceinline__ T lowest() { return (T)(LO); } \
    static __device__ __forceinline__ T highest() { return (T)(HI); } \
};

NUMR_INT_LIMITS(long long, LLONG_MIN, LLONG_MAX)
NUMR_INT_LIMITS(int, INT_MIN, INT_MAX)
NUMR_INT_LIMITS(short, SHRT_MIN, SHRT_MAX)
NUMR_INT_LIMITS(signed char, SCHAR_MIN, SCHAR_MAX)
NUMR_INT_LIMITS(unsigned long long, 0, ULLONG_MAX)
NUMR_INT_LIMITS(unsigned int, 0, UINT_MAX)
NUMR_INT_LIMITS(unsigned short, 0, USHRT_MAX)
NUMR_INT_LIMITS(unsigned char, 0, UCHAR_MAX)

#undef NUMR_INT_LIMITS

// ============================================================================
// Global reductions - one partial result per block
// ============================================================================

// Sum or product across every element. `IsProd` picks the identity and the
// combining operation at compile time, so one body covers both.
template<typename T, bool IsProd>
__device__ void reduce_acc_int_impl(const T* input, T* output, unsigned int n) {
    __shared__ Numr128 shared[REDUCE_INT_BLOCK];
    const unsigned int tid = threadIdx.x;

    Numr128 acc = IsProd ? numr128_from_i64(1) : numr128_from_i64(0);
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        Numr128 v = Numr128From<T>::apply(input[i]);
        acc = IsProd ? numr128_mul_sat(acc, v) : numr128_add_sat(acc, v);
    }

    shared[tid] = acc;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = IsProd ? numr128_mul_sat(shared[tid], shared[tid + s])
                                 : numr128_add_sat(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    // Each block writes ONE partial result, so this narrow is per block rather
    // than per final answer: a caller that chains blocks re-enters in the
    // element type. That matches `reduce.cu`, whose partials are stored in T too.
    if (tid == 0) {
        output[blockIdx.x] = Numr128Narrow<T>::apply(shared[0]);
    }
}

// Min or max across every element, accumulated in the element type.
template<typename T, bool IsMax>
__device__ void reduce_cmp_int_impl(const T* input, T* output, unsigned int n) {
    __shared__ T shared[REDUCE_INT_BLOCK];
    const unsigned int tid = threadIdx.x;

    T best = IsMax ? IntLimits<T>::lowest() : IntLimits<T>::highest();
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        const T v = input[i];
        best = IsMax ? (v > best ? v : best) : (v < best ? v : best);
    }

    shared[tid] = best;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const T other = shared[tid + s];
            const T cur = shared[tid];
            shared[tid] = IsMax ? (other > cur ? other : cur) : (other < cur ? other : cur);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// ============================================================================
// Dimension-wise reductions
//
// The tensor is viewed as [outer_size, reduce_size, inner_size] and reduced
// along the middle axis. Both grid axes stride, for the reason documented at
// the same point in `reduce.cu`: `inner_size` routinely exceeds the 65535 cap
// on grid.y.
// ============================================================================

// Sum, product, or mean along a dimension. `IsProd` selects the operation and
// `IsMean` divides the finished sum by `reduce_size`; the two are never both
// set.
template<typename T, bool IsProd, bool IsMean>
__device__ void reduce_acc_dim_int_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    __shared__ Numr128 shared[REDUCE_INT_BLOCK];
    const unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            Numr128 acc = IsProd ? numr128_from_i64(1) : numr128_from_i64(0);
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                const unsigned int idx =
                    outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                Numr128 v = Numr128From<T>::apply(input[idx]);
                acc = IsProd ? numr128_mul_sat(acc, v) : numr128_add_sat(acc, v);
            }

            shared[tid] = acc;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = IsProd ? numr128_mul_sat(shared[tid], shared[tid + s])
                                         : numr128_add_sat(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                // The divide happens inside the wide accumulator, before the
                // single narrow: a sum that overflows the element type but whose
                // mean does not still reports the true mean.
                Numr128 total = IsMean ? numr128_div_u64_trunc(shared[0], reduce_size)
                                       : shared[0];
                output[outer_idx * inner_size + inner_idx] = Numr128Narrow<T>::apply(total);
            }
            __syncthreads();
        }
    }
}

// Min or max along a dimension, accumulated in the element type.
template<typename T, bool IsMax>
__device__ void reduce_cmp_dim_int_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    __shared__ T shared[REDUCE_INT_BLOCK];
    const unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            T best = IsMax ? IntLimits<T>::lowest() : IntLimits<T>::highest();
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                const unsigned int idx =
                    outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                const T v = input[idx];
                best = IsMax ? (v > best ? v : best) : (v < best ? v : best);
            }

            shared[tid] = best;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    const T other = shared[tid + s];
                    const T cur = shared[tid];
                    shared[tid] = IsMax ? (other > cur ? other : cur)
                                        : (other < cur ? other : cur);
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[outer_idx * inner_size + inner_idx] = shared[0];
            }
            __syncthreads();
        }
    }
}

// Any or all along a dimension. Non-zero is true, and the result is stored as
// 0 or 1 in the element type - the same contract as the float path.
template<typename T, bool IsAll>
__device__ void reduce_logical_dim_int_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    __shared__ int shared[REDUCE_INT_BLOCK];
    const unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            // A thread with no elements holds the identity of its operation, so
            // a reduce_size below blockDim.x cannot perturb the answer.
            int acc = IsAll ? 1 : 0;
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                const unsigned int idx =
                    outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                const bool nonzero = (input[idx] != (T)0);
                if (IsAll) {
                    if (!nonzero) acc = 0;
                } else {
                    if (nonzero) acc = 1;
                }
            }

            shared[tid] = acc;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = IsAll ? (shared[tid] & shared[tid + s])
                                        : (shared[tid] | shared[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[outer_idx * inner_size + inner_idx] = shared[0] ? (T)1 : (T)0;
            }
            __syncthreads();
        }
    }
}

// Argmax or argmin along a dimension, comparing in the element type.
//
// Seeding the value with the type bound and the index with 0 keeps the
// first-occurrence tie-break: a seeded lane only ever loses a strict comparison,
// and the one case where the bound equals the true extremum means every element
// equals it, whose first occurrence is index 0.
template<typename T, bool IsMax>
__device__ void arg_dim_int_impl(
    const T* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    __shared__ T shared_val[REDUCE_INT_BLOCK];
    __shared__ long long shared_idx[REDUCE_INT_BLOCK];
    const unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            T best = IsMax ? IntLimits<T>::lowest() : IntLimits<T>::highest();
            long long best_idx = 0;

            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                const unsigned int idx =
                    outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                const T v = input[idx];
                if (IsMax ? (v > best) : (v < best)) {
                    best = v;
                    best_idx = i;
                }
            }

            shared_val[tid] = best;
            shared_idx[tid] = best_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    const bool take = IsMax ? (shared_val[tid + s] > shared_val[tid])
                                            : (shared_val[tid + s] < shared_val[tid]);
                    if (take) {
                        shared_val[tid] = shared_val[tid + s];
                        shared_idx[tid] = shared_idx[tid + s];
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[outer_idx * inner_size + inner_idx] = shared_idx[0];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// extern "C" wrappers for Rust FFI
//
// Integer dtypes have exactly one accumulation variant, so the names carry no
// `_acc` suffix - `reduce_kernel_name` in `kernels/reduce.rs` already routes
// every integer dtype to the plain suffix.
// ============================================================================

#define INSTANTIATE_REDUCE_INT(T, SUFFIX) \
extern "C" __global__ void reduce_sum_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_acc_int_impl<T, false>(input, output, n); \
} \
extern "C" __global__ void reduce_prod_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_acc_int_impl<T, true>(input, output, n); \
} \
extern "C" __global__ void reduce_max_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_cmp_int_impl<T, true>(input, output, n); \
} \
extern "C" __global__ void reduce_min_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_cmp_int_impl<T, false>(input, output, n); \
} \
extern "C" __global__ void reduce_sum_dim_##SUFFIX( \
    const T* input, T* output, unsigned int o, unsigned int r, unsigned int i \
) { reduce_acc_dim_int_impl<T, false, false>(input, output, o, r, i); } \
extern "C" __global__ void reduce_prod_dim_##SUFFIX( \
    const T* input, T* output, unsigned int o, unsigned int r, unsigned int i \
) { reduce_acc_dim_int_impl<T, true, false>(input, output, o, r, i); } \
extern "C" __global__ void reduce_mean_dim_##SUFFIX( \
    const T* input, T* output, unsigned int o, unsigned int r, unsigned int i \
) { reduce_acc_dim_int_impl<T, false, true>(input, output, o, r, i); } \
extern "C" __global__ void reduce_max_dim_##SUFFIX( \
    const T* input, T* output, unsigned int o, unsigned int r, unsigned int i \
) { reduce_cmp_dim_int_impl<T, true>(input, output, o, r, i); } \
extern "C" __global__ void reduce_min_dim_##SUFFIX( \
    const T* input, T* output, unsigned int o, unsigned int r, unsigned int i \
) { reduce_cmp_dim_int_impl<T, false>(input, output, o, r, i); } \
extern "C" __global__ void reduce_any_dim_##SUFFIX( \
    const T* input, T* output, unsigned int o, unsigned int r, unsigned int i \
) { reduce_logical_dim_int_impl<T, false>(input, output, o, r, i); } \
extern "C" __global__ void reduce_all_dim_##SUFFIX( \
    const T* input, T* output, unsigned int o, unsigned int r, unsigned int i \
) { reduce_logical_dim_int_impl<T, true>(input, output, o, r, i); } \
extern "C" __global__ void argmax_dim_##SUFFIX( \
    const T* input, long long* output, unsigned int o, unsigned int r, unsigned int i \
) { arg_dim_int_impl<T, true>(input, output, o, r, i); } \
extern "C" __global__ void argmin_dim_##SUFFIX( \
    const T* input, long long* output, unsigned int o, unsigned int r, unsigned int i \
) { arg_dim_int_impl<T, false>(input, output, o, r, i); }

INSTANTIATE_REDUCE_INT(long long, i64)
INSTANTIATE_REDUCE_INT(int, i32)
INSTANTIATE_REDUCE_INT(short, i16)
INSTANTIATE_REDUCE_INT(signed char, i8)
INSTANTIATE_REDUCE_INT(unsigned long long, u64)
INSTANTIATE_REDUCE_INT(unsigned int, u32)
INSTANTIATE_REDUCE_INT(unsigned short, u16)
INSTANTIATE_REDUCE_INT(unsigned char, u8)
