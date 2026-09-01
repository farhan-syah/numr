// WMMA staging macros: the 128-bit vectorised load path and its scalar
// fallback, used to fill the shared-memory A/B tiles before each K-step.
// Included only from matmul_wmma.cuh, which declares the tile constants
// (SMEM_STRIDE_A, SMEM_STRIDE_B, BLOCK_K, ...) these macros expand against.

#ifndef NUMR_MATMUL_WMMA_STAGE_CUH
#define NUMR_MATMUL_WMMA_STAGE_CUH

/* Number of halves in a 128-bit access. Both staging loops move whole
   WMMA_VEC_HALVES-wide groups so the shared-memory destination lands on a
   16-byte boundary. */
#define WMMA_VEC_HALVES 8

/* Guard for the 128-bit staging path on a row-major global operand.
   A 128-bit load FAULTS on a non-16-byte-aligned address, so both tests are
   correctness conditions, not tuning:
     - the operand base must itself be 16-byte aligned (a batch slice, a view
       offset, or any caller-supplied sub-buffer can break this);
     - every row start is base + row * ROW_STRIDE, so ROW_STRIDE must be a
       multiple of WMMA_VEC_HALVES for rows past the first to stay aligned.
   The column offsets the loops generate are already multiples of
   WMMA_VEC_HALVES. Drop either test and the kernel takes a misaligned-address
   fault, or silently reads the wrong bytes. */
#define WMMA_VEC_OK(PTR, ROW_STRIDE)                                         \
    (((reinterpret_cast<unsigned long long>(PTR)) & 15ull) == 0ull &&        \
     (((ROW_STRIDE) % WMMA_VEC_HALVES) == 0u))

/* One 128-bit staging access, in-bounds copy or zero-pad, through a single
   store form (SRC may point past the source matrix when !IN_BOUNDS — it is
   then never dereferenced). One expression per access, rather than two
   differently shaped stores into the same destination, is what lets a future
   async staging mechanism replace this macro with a predicated async copy
   instead of reconciling an async store against a synchronous one in the
   same buffer — see the file header for why mixing those was the kernel's
   disabled cp.async bug. No async copy is added here. A macro, not a
   __device__ helper: even __forceinline__ perturbed ptxas's register
   scheduling on one instantiation in testing, and this is a hot loop. */
#define WMMA_VEC_STAGE(DST, SRC, IN_BOUNDS)                                  \
    do {                                                                     \
        int4 wmma_vec_v = make_int4(0, 0, 0, 0);                             \
        if (IN_BOUNDS) {                                                     \
            wmma_vec_v = *(SRC);                                             \
        }                                                                    \
        *(DST) = wmma_vec_v;                                                 \
    } while (0)

/* Stage one K-tile into the given A/B buffers. Bounds-checked: positions      \
   past the M/N/K edge are zero-padded. Reads the tile constants declared by   \
   WMMA_TILE_CONSTANTS in the enclosing scope.                                 \
                                                                              \
   Two paths per tile. The fast path moves WMMA_VEC_HALVES halves per access   \
   and carries one branch per vector instead of one per element; it runs when  \
   the global operand satisfies WMMA_VEC_OK and the tile does not overhang the \
   contraction edge (A) or the N edge (B). Every term of that condition is     \
   block-uniform, so the branch costs nothing and hoists out of the loops.     \
   Anything else — a misaligned operand, a row stride that is not a multiple   \
   of WMMA_VEC_HALVES, a ragged trailing tile — takes the scalar path, which   \
   is the original element-at-a-time code and defines the semantics the fast   \
   path must match.                                                            \
                                                                              \
   Zero-padding is identical on both paths: an all-zero 128-bit word is eight  \
   halves of +0.0 in both F16 and BF16, the same value ZERO_EXPR produces.     \
   The M/N-edge row test stays inside the fast path, so partial tiles still    \
   zero-fill without falling back. */
#define WMMA_STAGE_TILE(HALF_T, ZERO_EXPR, A_ptr, B_ptr, DST_A, DST_B, K_OFF) \
{                                                                            \
    const unsigned int k_off = (K_OFF);                                      \
    /* A tile [BLOCK_TILE_M x BLOCK_K]. */                                   \
    if (WMMA_VEC_OK(A_ptr, K) && k_off + BLOCK_K <= K) {                     \
        constexpr unsigned int A_VECS_PER_ROW = BLOCK_K / WMMA_VEC_HALVES;   \
        for (unsigned int vi = tid;                                          \
             vi < BLOCK_TILE_M * A_VECS_PER_ROW; vi += num_threads) {        \
            unsigned int r  = vi / A_VECS_PER_ROW;                           \
            unsigned int c  = (vi % A_VECS_PER_ROW) * WMMA_VEC_HALVES;       \
            unsigned int gr = block_row + r;                                 \
            WMMA_VEC_STAGE(                                                  \
                reinterpret_cast<int4*>((DST_A) + r * SMEM_STRIDE_A + c),    \
                reinterpret_cast<const int4*>(                               \
                    (A_ptr) + gr * K + k_off + c),                           \
                gr < M);                                                     \
        }                                                                    \
    } else {                                                                 \
        for (unsigned int idx = tid;                                         \
             idx < BLOCK_TILE_M * BLOCK_K; idx += num_threads) {             \
            unsigned int r  = idx / BLOCK_K;                                 \
            unsigned int c  = idx % BLOCK_K;                                 \
            unsigned int gr = block_row + r;                                 \
            unsigned int gc = k_off + c;                                     \
            HALF_T val = ZERO_EXPR;                                          \
            if (gr < M && gc < K) val = (A_ptr)[gr * K + gc];                \
            (DST_A)[r * SMEM_STRIDE_A + c] = val;                            \
        }                                                                    \
    }                                                                        \
    /* B tile [BLOCK_K x BLOCK_TILE_N]. */                                   \
    if (WMMA_VEC_OK(B_ptr, N) && block_col + BLOCK_TILE_N <= N) {            \
        constexpr unsigned int B_VECS_PER_ROW =                              \
            BLOCK_TILE_N / WMMA_VEC_HALVES;                                  \
        for (unsigned int vi = tid;                                          \
             vi < BLOCK_K * B_VECS_PER_ROW; vi += num_threads) {             \
            unsigned int r  = vi / B_VECS_PER_ROW;                           \
            unsigned int c  = (vi % B_VECS_PER_ROW) * WMMA_VEC_HALVES;       \
            unsigned int gr = k_off + r;                                     \
            WMMA_VEC_STAGE(                                                  \
                reinterpret_cast<int4*>((DST_B) + r * SMEM_STRIDE_B + c),    \
                reinterpret_cast<const int4*>(                               \
                    (B_ptr) + gr * N + block_col + c),                       \
                gr < K);                                                     \
        }                                                                    \
    } else {                                                                 \
        for (unsigned int idx = tid;                                         \
             idx < BLOCK_K * BLOCK_TILE_N; idx += num_threads) {             \
            unsigned int r  = idx / BLOCK_TILE_N;                            \
            unsigned int c  = idx % BLOCK_TILE_N;                            \
            unsigned int gr = k_off + r;                                     \
            unsigned int gc = block_col + c;                                 \
            HALF_T val = ZERO_EXPR;                                          \
            if (gr < K && gc < N) val = (B_ptr)[gr * N + gc];                \
            (DST_B)[r * SMEM_STRIDE_B + c] = val;                            \
        }                                                                    \
    }                                                                        \
}

#endif  // NUMR_MATMUL_WMMA_STAGE_CUH
