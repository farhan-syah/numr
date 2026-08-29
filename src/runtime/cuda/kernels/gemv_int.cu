// Integer GEMV (I32, I64) CUDA Kernels
// C[M,N] = A[M,K] @ B[K,N] for small M (M <= 16, typically M=1 for LLM decode)
//
// These are the integer counterparts of the float kernels in `gemv.cu`, split
// into their own translation unit because they accumulate in `Numr128` rather
// than in a float register, and because `gemv.cu` is already at its size limit.
// The three families and the launch geometry match `gemv.cu` exactly:
//
// 1. gemv_*        : B is [K,N] row-major. One thread per output column.
// 2. gemv_bt_*     : B is [N,K] row-major. One warp per output column.
// 3. gemv_bt_mr_*  : B is [N,K] row-major. One warp per ROWS_PER_WARP columns,
//                    with vectorised 16-byte loads.
//
// Accumulation is the same rule as the tiled integer GEMM in `matmul_int.cu`:
// a 128-bit accumulator that never overflows, narrowed and saturated exactly
// once at the store. Integer addition in 128 bits is exact and associative, so
// the warp-reduction order here reaches the same total as the tiled kernel's
// sequential order - GEMV and tiled matmul agree bit for bit on any operands.
//
// The constants below must stay equal to `gemv.cu`'s, because the Rust launcher
// (`launch_gemv_kernel*` in `kernels/loader.rs`) computes one grid for both.

#include "numr128.cuh"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define ROWS_PER_WARP 2
#define FULL_WARP_MASK 0xFFFFFFFFu

// Check that a pointer is aligned to n bytes.
#define IS_ALIGNED(ptr, n) (((unsigned long long)(ptr)) % (n) == 0)

// ============================================================================
// Non-transposed B: one thread per output, iterate K
// B layout: [K, N] row-major - B[k,n] = B_data[k*N + n]
// ============================================================================

template<typename T>
__device__ __forceinline__ void gemv_int_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const T* a_row = A + (batch % a_batch_count) * M * K + m * K;
    const T* b_base = B + (batch % b_batch_count) * K * N;

    Numr128 acc = numr128_from_i64(0);
    for (unsigned int k = 0; k < K; k++) {
        acc = numr128_add(acc, numr128_mul_i64((long long)a_row[k], (long long)b_base[k * N + col]));
    }

    C[batch * M * N + m * N + col] = Numr128Narrow<T>::apply(acc);
}

extern "C" __global__ void gemv_i32(
    const int* __restrict__ A,
    const int* __restrict__ B,
    int* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    gemv_int_impl<int>(A, B, C, M, N, K, a_batch_count, b_batch_count);
}

extern "C" __global__ void gemv_i64(
    const long long* __restrict__ A,
    const long long* __restrict__ B,
    long long* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    gemv_int_impl<long long>(A, B, C, M, N, K, a_batch_count, b_batch_count);
}

// ============================================================================
// Transposed B: warp-cooperative K-reduction
// B layout: [N, K] row-major (weight matrix) - B_logical[k,n] = B_data[n*K + k]
// ============================================================================

template<typename T>
__device__ __forceinline__ void gemv_bt_int_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    // `col` is warp-uniform, so every lane of a surviving warp reaches the
    // shuffle reduction below.
    if (col >= N) return;

    const T* a_row = A + (batch % a_batch_count) * M * K + m * K;
    const T* b_row = B + (batch % b_batch_count) * N * K + col * K;

    Numr128 acc = numr128_from_i64(0);
    for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
        acc = numr128_add(acc, numr128_mul_i64((long long)a_row[k], (long long)b_row[k]));
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc = numr128_add(acc, numr128_shfl_down(FULL_WARP_MASK, acc, offset));
    }

    if (lane_id == 0) {
        C[batch * M * N + m * N + col] = Numr128Narrow<T>::apply(acc);
    }
}

extern "C" __global__ void gemv_bt_i32(
    const int* __restrict__ A,
    const int* __restrict__ B,
    int* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    gemv_bt_int_impl<int>(A, B, C, M, N, K, a_batch_count, b_batch_count);
}

extern "C" __global__ void gemv_bt_i64(
    const long long* __restrict__ A,
    const long long* __restrict__ B,
    long long* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    gemv_bt_int_impl<long long>(A, B, C, M, N, K, a_batch_count, b_batch_count);
}

// ============================================================================
// Multi-Row Transposed B with Vectorized Loads
//
// Each warp computes ROWS_PER_WARP output columns and loads the activation
// vector once for both, halving activation bandwidth. A 16-byte vector load
// carries 4 I32 or 2 I64 elements, matching `gemv.cu`'s float4 / double2 paths.
// ============================================================================

// 16-byte vector type per element type, and how many elements it carries.
template<typename T> struct gemv_int_vec;

template<> struct gemv_int_vec<int> {
    typedef int4 vec_t;
    static const unsigned int LANES = 4;
};

template<> struct gemv_int_vec<long long> {
    typedef longlong2 vec_t;
    static const unsigned int LANES = 2;
};

template<typename T>
__device__ __forceinline__ void gemv_bt_mr_int_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    typedef typename gemv_int_vec<T>::vec_t vec_t;
    constexpr unsigned int VEC = gemv_int_vec<T>::LANES;

    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * ROWS_PER_WARP;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int a_batch = batch % a_batch_count;
    const unsigned int b_batch = batch % b_batch_count;

    const T* a_row = A + a_batch * M * K + m * K;

    Numr128 acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        acc[r] = numr128_from_i64(0);
    }

    // A K that is a multiple of VEC keeps every B row start 16-byte aligned too,
    // because the rows are VEC-element multiples apart from an aligned base.
    const bool can_vec = (K % VEC == 0) && IS_ALIGNED(a_row, 16);

    if (can_vec) {
        const unsigned int K_vec = K / VEC;
        const vec_t* a_vec = reinterpret_cast<const vec_t*>(a_row);

        for (unsigned int vi = lane_id; vi < K_vec; vi += WARP_SIZE) {
            vec_t av = a_vec[vi];
            const T* a_elems = reinterpret_cast<const T*>(&av);

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    const vec_t* b_vec = reinterpret_cast<const vec_t*>(
                        B + b_batch * N * K + (col_base + r) * K);
                    vec_t bv = b_vec[vi];
                    const T* b_elems = reinterpret_cast<const T*>(&bv);

                    #pragma unroll
                    for (unsigned int j = 0; j < VEC; j++) {
                        acc[r] = numr128_add(
                            acc[r],
                            numr128_mul_i64((long long)a_elems[j], (long long)b_elems[j]));
                    }
                }
            }
        }
    } else {
        for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
            const long long a_val = (long long)a_row[k];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    acc[r] = numr128_add(
                        acc[r],
                        numr128_mul_i64(
                            a_val,
                            (long long)B[b_batch * N * K + (col_base + r) * K + k]));
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            acc[r] = numr128_add(acc[r], numr128_shfl_down(FULL_WARP_MASK, acc[r], off));
        }
        if (lane_id == 0 && col_base + r < N) {
            C[batch * M * N + m * N + col_base + r] = Numr128Narrow<T>::apply(acc[r]);
        }
    }
}

extern "C" __global__ void gemv_bt_mr_i32(
    const int* __restrict__ A,
    const int* __restrict__ B,
    int* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    gemv_bt_mr_int_impl<int>(A, B, C, M, N, K, a_batch_count, b_batch_count);
}

extern "C" __global__ void gemv_bt_mr_i64(
    const long long* __restrict__ A,
    const long long* __restrict__ B,
    long long* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    gemv_bt_mr_int_impl<long long>(A, B, C, M, N, K, a_batch_count, b_batch_count);
}
