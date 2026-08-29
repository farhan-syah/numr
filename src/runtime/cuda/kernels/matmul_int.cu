// Compile-Time-Tiled Integer GEMM (I16, I32, I64, U8, U16, U32, U64)
//
// Integer matmul accumulates in `Numr128`, never in the element type. This is
// what CPU does (`matmul_scalar_acc::<T, i128>` in
// `runtime/cpu/kernels/matmul/kernel.rs`), and the two backends must agree on
// exactly the inputs that stress the accumulator: a partial sum that leaves the
// output dtype's range and comes back reports the true value, because the
// accumulator is wide enough never to overflow. Clamping per step would report
// a different number, so nothing here clamps until the final store.
//
// The tile dimensions are C++ template parameters, not kernel arguments. With
// runtime tile sizes NVCC cannot size `reg_c` at compile time, so it must
// reserve the maximum and place the whole array in local memory - and every
// accumulate becomes a pair of local-memory round trips. Compile-time tiles size
// `reg_c[TM][TN]` exactly, keep it in registers, and let the micro-kernel unroll.
// This mirrors `matmul_f32_tiled_impl` in `matmul.cu`.
//
// One tile shape is instantiated per dtype: BM=BN=64, BK=8, TM=TN=4, giving
// (64/4) x (64/4) = 256 threads per block. A `Numr128` is 16 bytes, so each
// accumulator slot costs four registers and a 4x4 thread tile already spends 64
// registers on `reg_c`. An 8x8 tile would need 256, past the 255-register
// per-thread limit, and would spill exactly what compile-time tiling exists to
// prevent. 64x64x8 is therefore the widest shape that keeps the accumulator in
// registers. Shared memory is a static 8 KB at the widest element, I64
// (2 x 64 x 8 x 8 bytes), and less for every narrower one - well inside the
// 48 KB per-block default.
//
// The fused-bias variants seed the accumulator with the bias instead of adding
// it afterwards, which is what CPU does (`matmul_bias_scalar_acc` in
// `runtime/cpu/kernels/matmul/kernel.rs`). Composing a plain matmul with an
// elementwise add would narrow and saturate the product first and then wrap the
// bias into the element type, reporting a different number at the boundary.
//
// Single-buffered, not double-buffered: a double-buffered variant was
// implemented and benchmarked on an RTX 3060 and lost - I32 1024x1024 was 26%
// SLOWER double-buffered (14.81ms vs 11.72ms single), every other case within
// 3% (noise). `Numr128` accumulation is a software 128-bit multiply-add, so
// this kernel is compute-bound: there is already enough arithmetic to hide the
// global loads, so prefetching buys nothing while the doubled shared-memory
// footprint (16 KB for I64, 8 KB for I32) costs occupancy. This is the
// opposite of `matmul_f32_tiled_impl` in `matmul.cu`, which double-buffers
// because F32 matmul is memory-bound. Do not re-add double buffering here
// without a new measurement that overturns this one.

#include "numr128.cuh"

template<typename T, int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_int_tiled_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ bias,
    T* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // Static shared memory - compile-time sizes make the index arithmetic
    // constant-folded and let the loads below unroll.
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    const unsigned int tx = threadIdx.x;      // [0, BN/TN)
    const unsigned int ty = threadIdx.y;      // [0, BM/TM)
    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;
    const unsigned int thread_row = ty * TM;
    const unsigned int thread_col = tx * TN;

    // `bias` is null for the plain kernels. Where it is given it seeds the
    // accumulator, so the bias is part of the wide sum rather than a narrowed
    // value added to a narrowed product. `Numr128From` picks the widening by
    // element type, so an unsigned bias zero-extends.
    Numr128 reg_c[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const unsigned int bias_col = block_col + thread_col + j;
            reg_c[i][j] = (bias != nullptr && bias_col < N)
                ? Numr128From<T>::apply(bias[bias_col])
                : numr128_from_i64(0);
        }
    }

    T reg_a[TM];
    T reg_b[TN];

    const unsigned int num_k_tiles = (K + BK - 1) / BK;
    const unsigned int thread_id = ty * (BN / TN) + tx;
    const unsigned int num_threads = (BM / TM) * (BN / TN);

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * BK;

        // Out-of-range elements load as zero, so the micro-kernel below runs the
        // full BK trip count on ragged tiles without changing the result.
        for (unsigned int load_idx = thread_id; load_idx < BM * BK; load_idx += num_threads) {
            const unsigned int load_row = load_idx / BK;
            const unsigned int load_col = load_idx % BK;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            T val = (T)0;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row][load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < BK * BN; load_idx += num_threads) {
            const unsigned int load_row = load_idx / BN;
            const unsigned int load_col = load_idx % BN;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            T val = (T)0;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row][load_col] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_a[i] = As[thread_row + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                reg_b[j] = Bs[k][thread_col + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    // Widen both operands before multiplying: the product of
                    // two elements does not fit the element type. The widening
                    // is type-directed, so an unsigned element zero-extends
                    // instead of entering the accumulator negative.
                    reg_c[i][j] = Numr128MulAdd<T>::apply(reg_c[i][j], reg_a[i], reg_b[j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = Numr128Narrow<T>::apply(reg_c[i][j]);
                }
            }
        }
    }
}

// Batched: one grid z slice per batch element. `a_batch_count` / `b_batch_count`
// carry broadcasting, so a shared operand is read by every slice.
template<typename T, int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_batched_int_tiled_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ bias,
    T* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    // The bias is [N], so every batch slice reads the same vector.
    matmul_int_tiled_impl<T, BM, BN, BK, TM, TN>(
        A + (b % a_batch_count) * M * K,
        B + (b % b_batch_count) * K * N,
        bias,
        C + b * M * N,
        M, N, K
    );
}

// ---------------------------------------------------------------------------
// Extern "C" entry points (four per dtype)
//
// Grid: (ceil(N/64), ceil(M/64), batch)   Block: (16, 16, 1)
// Shared memory is static, so these launch with zero dynamic shared memory.
// ---------------------------------------------------------------------------

// One macro per dtype emits all four entry points - plain and fused-bias, each
// batched and not - so a new element type is one row rather than four
// hand-copied bodies.
//
// I8 is deliberately absent: CPU `matmul` on I8 returns an I32 tensor
// (quantized accumulation, see `ops/cpu/matmul.rs`), so an I8-in/I8-out kernel
// here would disagree with the reference on both dtype and value.
#define NUMR_MATMUL_INT_ENTRIES(SUFFIX, TYPE) \
extern "C" __global__ void matmul_##SUFFIX##_tiled_64x64x8_4x4( \
    const TYPE* __restrict__ A, \
    const TYPE* __restrict__ B, \
    TYPE* __restrict__ C, \
    unsigned int M, \
    unsigned int N, \
    unsigned int K \
) { \
    matmul_int_tiled_impl<TYPE, 64, 64, 8, 4, 4>(A, B, nullptr, C, M, N, K); \
} \
extern "C" __global__ void matmul_batched_##SUFFIX##_tiled_64x64x8_4x4( \
    const TYPE* __restrict__ A, \
    const TYPE* __restrict__ B, \
    TYPE* __restrict__ C, \
    unsigned int batch, \
    unsigned int M, \
    unsigned int N, \
    unsigned int K, \
    unsigned int a_batch_count, \
    unsigned int b_batch_count \
) { \
    matmul_batched_int_tiled_impl<TYPE, 64, 64, 8, 4, 4>( \
        A, B, nullptr, C, batch, M, N, K, a_batch_count, b_batch_count \
    ); \
} \
extern "C" __global__ void matmul_bias_##SUFFIX##_tiled_64x64x8_4x4( \
    const TYPE* __restrict__ A, \
    const TYPE* __restrict__ B, \
    const TYPE* __restrict__ bias, \
    TYPE* __restrict__ C, \
    unsigned int M, \
    unsigned int N, \
    unsigned int K \
) { \
    matmul_int_tiled_impl<TYPE, 64, 64, 8, 4, 4>(A, B, bias, C, M, N, K); \
} \
extern "C" __global__ void matmul_bias_batched_##SUFFIX##_tiled_64x64x8_4x4( \
    const TYPE* __restrict__ A, \
    const TYPE* __restrict__ B, \
    const TYPE* __restrict__ bias, \
    TYPE* __restrict__ C, \
    unsigned int batch, \
    unsigned int M, \
    unsigned int N, \
    unsigned int K, \
    unsigned int a_batch_count, \
    unsigned int b_batch_count \
) { \
    matmul_batched_int_tiled_impl<TYPE, 64, 64, 8, 4, 4>( \
        A, B, bias, C, batch, M, N, K, a_batch_count, b_batch_count \
    ); \
}

NUMR_MATMUL_INT_ENTRIES(i16, short)
NUMR_MATMUL_INT_ENTRIES(i32, int)
NUMR_MATMUL_INT_ENTRIES(i64, long long)
NUMR_MATMUL_INT_ENTRIES(u8, unsigned char)
NUMR_MATMUL_INT_ENTRIES(u16, unsigned short)
NUMR_MATMUL_INT_ENTRIES(u32, unsigned int)
NUMR_MATMUL_INT_ENTRIES(u64, unsigned long long)

#undef NUMR_MATMUL_INT_ENTRIES
