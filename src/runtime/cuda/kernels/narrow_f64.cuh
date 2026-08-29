// f64 -> F16/BF16 narrowing that matches numr's CPU reference exactly.
//
// numr's F16/BF16 narrowing semantics are NOT "round the f64 once to the
// target type". They are whatever the `half` crate does, because
// `Element::from_f64` for those two dtypes is `half::f16::from_f64` /
// `half::bf16::from_f64` (src/dtype/element.rs). CPU is the reference, so CUDA
// copies `half`, not IEEE.
//
// `half` 2.7.1 (the resolved version; 2.4.1 is byte-identical here) does two
// different things for the two types:
//
//   f16  - `binary16/arch.rs` dispatches on the host ISA. On x86-64 with F16C
//          (this project's build target) it evaluates `f32_to_f16_x86_f16c(f as
//          f32)`: the f64 is rounded to f32 first, then the F16C instruction
//          rounds f32 to f16. That is a DOUBLE rounding, and it differs from a
//          single rounding whenever the f64 carries bits below half an f32 ulp.
//          aarch64 with `fp16` single-rounds instead, and the software fallback
//          is a third algorithm - so this is deliberately platform-specific.
//
//   bf16 - `bfloat/convert.rs::f64_to_bf16` is pure software with no ISA
//          dispatch. It discards the low 32 mantissa bits of the f64 outright,
//          then rounds the remaining 20-bit mantissa to 7 bits, half-to-even.
//          Its sticky window is only f64 mantissa bits 43..32; bits 31..0 never
//          influence the result. That is neither an f32 stage nor a single
//          rounding, so it is ported here bit for bit.
//
// Do NOT "fix" these to `__double2half` / `__double2bfloat16`. Those are true
// single roundings; they disagree with CPU, and swapping them in is exactly the
// regression `tests/backend_parity/utility_float_precision.rs` and the f64->F16
// /BF16 rows of `tests/cuda_cast_dtype_coverage.rs` exist to catch.

#ifndef NUMR_NARROW_F64_CUH
#define NUMR_NARROW_F64_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// f64 -> F16 through f32, matching `half::f16::from_f64` on x86-64 with F16C.
__device__ __forceinline__ __half numr_f64_to_f16(double v) {
    return __float2half((float)v);
}

// f64 -> BF16, a port of `half`'s `f64_to_bf16`. Every branch, mask and
// rounding test below is the Rust function term for term.
__device__ __forceinline__ __nv_bfloat16 numr_f64_to_bf16(double v) {
    unsigned long long val = (unsigned long long)__double_as_longlong(v);
    // The low 32 mantissa bits are dropped here and never consulted again,
    // except to tell a NaN from an infinity.
    unsigned int x = (unsigned int)(val >> 32);

    unsigned int sign = x & 0x80000000u;
    unsigned int exp = x & 0x7FF00000u;
    unsigned int man = x & 0x000FFFFFu;

    unsigned short bits;
    if (exp == 0x7FF00000u) {
        // Infinity or NaN. A NaN keeps its shifted mantissa and gains the
        // quiet bit; the low 32 bits are checked so a NaN carried only there
        // is not mistaken for an infinity.
        unsigned int nan_bit = (man == 0u && (unsigned int)val == 0u) ? 0u : 0x0040u;
        bits = (unsigned short)((sign >> 16) | 0x7F80u | nan_bit | (man >> 13));
    } else {
        unsigned int half_sign = sign >> 16;
        long long half_exp = (long long)(exp >> 20) - 1023 + 127;

        if (half_exp >= 0xFF) {
            bits = (unsigned short)(half_sign | 0x7F80u); // overflow -> signed infinity
        } else if (half_exp <= 0) {
            if (7 - half_exp > 21) {
                bits = (unsigned short)half_sign; // full underflow -> signed zero
            } else {
                // Subnormal: restore the hidden bit, then shift and round.
                unsigned int sub_man = man | 0x00100000u;
                unsigned int half_man = sub_man >> (int)(14 - half_exp);
                unsigned int round_bit = 1u << (int)(13 - half_exp);
                if ((sub_man & round_bit) != 0u && (sub_man & (3u * round_bit - 1u)) != 0u) {
                    half_man += 1u;
                }
                bits = (unsigned short)(half_sign | half_man);
            }
        } else {
            unsigned int assembled = half_sign | (((unsigned int)half_exp) << 7) | (man >> 13);
            // Round half to even: the mask covers the kept LSB, the round bit
            // and the 12 sticky bits, so a bare tie leaves the LSB at 0.
            unsigned int round_bit = 0x00001000u;
            if ((man & round_bit) != 0u && (man & (3u * round_bit - 1u)) != 0u) {
                assembled += 1u;
            }
            bits = (unsigned short)assembled;
        }
    }

    __nv_bfloat16_raw raw;
    raw.x = bits;
    __nv_bfloat16 out;
    out = raw;
    return out;
}

#endif // NUMR_NARROW_F64_CUH
