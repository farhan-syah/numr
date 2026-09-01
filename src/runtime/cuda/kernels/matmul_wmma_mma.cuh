// Tensor-core compute backends for the WMMA GEMM body in matmul_wmma.cuh.
//
// The body macro selects one by token paste on its MMA parameter:
//   WMMA -> nvcuda::wmma fragments (F16).
//   RAW  -> raw ldmatrix + mma.sync PTX (BF16, sm_80+).
//
// Why two backends. nvcuda::wmma's BF16 specialisation never emits ldmatrix.
// For the same dtype-generic source it substitutes generic shared loads plus a
// register-synthesised transpose, where F16 gets the LDSM pair. The raw path
// issues the ldmatrix the hardware already has, and reaches F16 parity. F16 is
// left on nvcuda::wmma because its wmma path already emits the optimal
// ldmatrix pair, so a rewrite there would only risk changing its codegen.
//
// Both backends must produce the same accumulator values and write the same
// 16x16 row-major float scratch, so the shared epilogue is untouched by the
// choice. wmma's m16n16k16 mma is two m16n8k16 mma.sync operations on the same
// operands, so the raw path's accumulation order is the same one.
//
// The RAW macros expand only inside `__CUDA_ARCH__ >= 800` (BF16 mma.sync and
// BF16 wmma fragments both start there); there is no sm_75 BF16 kernel to fall
// back to.

#ifndef NUMR_MATMUL_WMMA_MMA_CUH
#define NUMR_MATMUL_WMMA_MMA_CUH

// ---------------------------------------------------------------------------
// WMMA backend (F16). Expands to exactly the nvcuda::wmma code this kernel has
// always used; keep it byte-identical.
// ---------------------------------------------------------------------------

#define NUMR_WMMA_ACC_DECL_WMMA(WM, WN)                                      \
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[WM][WN];     \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        for (unsigned int wj = 0; wj < (WN); wj++) {                         \
            fill_fragment(frag_c[wi][wj], 0.0f);                             \
        }                                                                    \
    }

/* One K-step of WMMA_K: load WM A fragments and WN B fragments, then
   WM x WN mma_syncs. */
#define NUMR_WMMA_KSTEP_WMMA(WM, WN, HALF_T, SMEM_A, SMEM_B, K_SUB)          \
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major>            \
        frag_a[WM];                                                          \
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major>            \
        frag_b[WN];                                                          \
                                                                             \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        unsigned int row = (warp_row * (WM) + wi) * WMMA_M;                  \
        const HALF_T* a_ptr = (SMEM_A) + row * SMEM_STRIDE_A + (K_SUB);      \
        load_matrix_sync(frag_a[wi], a_ptr, SMEM_STRIDE_A);                  \
    }                                                                        \
    for (unsigned int wj = 0; wj < (WN); wj++) {                             \
        unsigned int col = (warp_col * (WN) + wj) * WMMA_N;                  \
        const HALF_T* b_ptr = (SMEM_B) + (K_SUB) * SMEM_STRIDE_B + col;      \
        load_matrix_sync(frag_b[wj], b_ptr, SMEM_STRIDE_B);                  \
    }                                                                        \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        for (unsigned int wj = 0; wj < (WN); wj++) {                         \
            mma_sync(frag_c[wi][wj], frag_a[wi], frag_b[wj],                 \
                     frag_c[wi][wj]);                                        \
        }                                                                    \
    }

#define NUMR_WMMA_SCRATCH_WMMA(DST, WI, WJ)                                  \
    store_matrix_sync((DST), frag_c[WI][WJ], WMMA_N, mem_row_major);

// ---------------------------------------------------------------------------
// RAW backend (BF16, sm_80+).
// ---------------------------------------------------------------------------

/* ldmatrix takes a shared-window address, not a generic one. */
__device__ __forceinline__ unsigned int numr_mma_smem_addr(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

/* ldmatrix.x4 reads four 8x8 b16 matrices; lane l supplies the start of row
   l%8 of matrix l/8, and the four destination registers come back in matrix
   order. Both operands use the same lane split, so one row/col pair serves
   A and B.
     A (no transpose): matrices are (rows 0-7, k 0-7), (rows 8-15, k 0-7),
     (rows 0-7, k 8-15), (rows 8-15, k 8-15) -- the m16n8k16 A operand order.
     B (.trans): numr stores B row-major [k][n], and mma.sync row.col wants B
     k-contiguous per lane. .trans does that transpose in the load itself, no
     smem layout change and no extra instruction. Matrices come back as
     (n 0-7) in regs 0,1 and (n 8-15) in regs 2,3, which is one m16n8k16 B
     operand each. */
#define NUMR_MMA_LDM_X4(D0, D1, D2, D3, PTR)                                 \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                 \
                 "{%0, %1, %2, %3}, [%4];"                                   \
                 : "=r"(D0), "=r"(D1), "=r"(D2), "=r"(D3)                    \
                 : "r"(numr_mma_smem_addr(PTR)))

#define NUMR_MMA_LDM_X4_TRANS(D0, D1, D2, D3, PTR)                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "           \
                 "{%0, %1, %2, %3}, [%4];"                                   \
                 : "=r"(D0), "=r"(D1), "=r"(D2), "=r"(D3)                    \
                 : "r"(numr_mma_smem_addr(PTR)))

#define NUMR_MMA_M16N8K16_BF16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)       \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "      \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "            \
                 "{%0, %1, %2, %3};"                                         \
                 : "+f"(C0), "+f"(C1), "+f"(C2), "+f"(C3)                    \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1))

/* Each 16x16 warp tile is two m16n8k16 accumulators: n-sub 0 in x[0..3],
   n-sub 1 in x[4..7]. That is the element order nvcuda::wmma's 16x16x16 F32
   accumulator uses as well. */
#define NUMR_WMMA_ACC_DECL_RAW(WM, WN)                                       \
    float frag_c[WM][WN][8];                                                 \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        for (unsigned int wj = 0; wj < (WN); wj++) {                         \
            for (unsigned int wl = 0; wl < 8; wl++) {                        \
                frag_c[wi][wj][wl] = 0.0f;                                   \
            }                                                                \
        }                                                                    \
    }                                                                        \
    /* ldmatrix lane split: matrix = lane/8, row within matrix = lane%8. */  \
    const unsigned int mma_lane = threadIdx.x % 32;                          \
    const unsigned int mma_row = ((mma_lane >> 3) & 1u) * 8 + (mma_lane & 7u); \
    const unsigned int mma_col = (mma_lane >> 4) * 8;

#define NUMR_WMMA_KSTEP_RAW(WM, WN, HALF_T, SMEM_A, SMEM_B, K_SUB)           \
    unsigned int frag_a[WM][4];                                              \
    unsigned int frag_b[WN][4];                                              \
                                                                             \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        unsigned int row = (warp_row * (WM) + wi) * WMMA_M + mma_row;        \
        const HALF_T* a_ptr =                                                \
            (SMEM_A) + row * SMEM_STRIDE_A + (K_SUB) + mma_col;              \
        NUMR_MMA_LDM_X4(frag_a[wi][0], frag_a[wi][1], frag_a[wi][2],         \
                        frag_a[wi][3], a_ptr);                               \
    }                                                                        \
    for (unsigned int wj = 0; wj < (WN); wj++) {                             \
        unsigned int col = (warp_col * (WN) + wj) * WMMA_N + mma_col;        \
        const HALF_T* b_ptr =                                                \
            (SMEM_B) + ((K_SUB) + mma_row) * SMEM_STRIDE_B + col;            \
        NUMR_MMA_LDM_X4_TRANS(frag_b[wj][0], frag_b[wj][1], frag_b[wj][2],   \
                              frag_b[wj][3], b_ptr);                         \
    }                                                                        \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        for (unsigned int wj = 0; wj < (WN); wj++) {                         \
            NUMR_MMA_M16N8K16_BF16(                                          \
                frag_c[wi][wj][0], frag_c[wi][wj][1],                        \
                frag_c[wi][wj][2], frag_c[wi][wj][3],                        \
                frag_a[wi][0], frag_a[wi][1], frag_a[wi][2], frag_a[wi][3],  \
                frag_b[wj][0], frag_b[wj][1]);                               \
            NUMR_MMA_M16N8K16_BF16(                                          \
                frag_c[wi][wj][4], frag_c[wi][wj][5],                        \
                frag_c[wi][wj][6], frag_c[wi][wj][7],                        \
                frag_a[wi][0], frag_a[wi][1], frag_a[wi][2], frag_a[wi][3],  \
                frag_b[wj][2], frag_b[wj][3]);                               \
        }                                                                    \
    }

/* m16n8k16 accumulator element l of the lane sits at
   row = (l/2)*8 + lane/4, col = (lane%4)*2 + l%2; the second mma adds 8 to
   the column. Writing that into the 16x16 row-major scratch reproduces
   store_matrix_sync's result, so every epilogue functor stays untouched. */
#define NUMR_WMMA_SCRATCH_RAW(DST, WI, WJ)                                   \
    for (unsigned int ns = 0; ns < 2; ns++) {                                \
        for (unsigned int h = 0; h < 2; h++) {                               \
            float* dst_pair = (DST)                                          \
                + ((mma_lane >> 2) + h * 8) * WMMA_N                         \
                + (mma_lane & 3u) * 2 + ns * 8;                              \
            *reinterpret_cast<float2*>(dst_pair) = make_float2(              \
                frag_c[WI][WJ][ns * 4 + h * 2],                              \
                frag_c[WI][WJ][ns * 4 + h * 2 + 1]);                         \
        }                                                                    \
    }

#endif  // NUMR_MATMUL_WMMA_MMA_CUH
