// Shared parameter list and tap-range derivation for conv1d kernel variants.
// Used by conv.cu (conv1d_*, conv1d_oc4_*) and conv1d_ox.cu (conv1d_ox_*).
#ifndef NUMR_CONV1D_COMMON_CUH
#define NUMR_CONV1D_COMMON_CUH

// c_in_per_group and c_out_per_group are loop-invariant per launch (they don't
// depend on the thread's oc/ox), so the host computes them once and passes them
// in rather than every thread repeating an integer division in its prologue.
#define CONV1D_PARAMS(dtype) \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int length, \
    unsigned int c_out, \
    unsigned int kernel_size, \
    unsigned int output_length, \
    unsigned int stride, \
    unsigned int padding, \
    unsigned int dilation, \
    unsigned int groups, \
    unsigned int c_in_per_group, \
    unsigned int c_out_per_group, \
    unsigned int has_bias

/* Valid tap range for this thread: ix = ix_base + kx*dilation must land in
   [0, length). Both bounds are loop-invariant, so the inner loop is branch-free. */
// At dilation == 1 the general ceil-division formulas reduce exactly (no
// rounding change): ((-ix_base) + dil - 1) / dil == -ix_base, and
// (room + dil - 1) / dil == room. That branch is taken free of runtime
// division; dilation > 1 keeps the general division-based formula.
#define CONV1D_TAP_RANGE() \
    int ix_base = (int)(ox * stride) - (int)padding; \
    int dil = (int)dilation; \
    unsigned int kx_lo = 0u; \
    unsigned int kx_hi = kernel_size; \
    if (dil == 1) { \
        if (ix_base < 0) { \
            kx_lo = (unsigned int)(-ix_base); \
        } \
        { \
            int room = (int)length - ix_base; \
            if (room <= 0) { \
                kx_hi = 0u; \
            } else { \
                unsigned int hi = (unsigned int)room; \
                if (hi < kx_hi) { kx_hi = hi; } \
            } \
        } \
    } else { \
        if (ix_base < 0) { \
            kx_lo = (unsigned int)(((-ix_base) + dil - 1) / dil); \
        } \
        { \
            int room = (int)length - ix_base; \
            if (room <= 0) { \
                kx_hi = 0u; \
            } else { \
                unsigned int hi = (unsigned int)((room + dil - 1) / dil); \
                if (hi < kx_hi) { kx_hi = hi; } \
            } \
        } \
    } \
    if (kx_lo > kx_hi) { kx_lo = kx_hi; }

#endif // NUMR_CONV1D_COMMON_CUH
