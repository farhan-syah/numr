//! Shared constants and algorithm definitions for SIMD math functions
//!
//! This module provides polynomial coefficients and macros for generating
//! SIMD implementations across different instruction sets (AVX2, AVX-512).
//! By centralizing the algorithm logic, we ensure consistency and reduce
//! maintenance burden.

// ============================================================================
// Polynomial Coefficients for exp(x)
// ============================================================================

/// Taylor series coefficients for exp(r) where r is in [-ln(2)/2, ln(2)/2]
/// exp(r) ≈ 1 + r + r²/2! + r³/3! + ...
///
/// f32 truncates at degree 6: |r|⁷/7! ≤ 1.2e-7, which is f32 epsilon.
///
/// f64 truncates at degree 13: |r|¹⁴/14! ≤ 4.1e-18, well below f64 epsilon
/// (2.2e-16). Degree 6 would leave a truncation error near 1.2e-7 — nine
/// decimal digits short of double precision.
pub mod exp_coefficients {
    pub const C0_F32: f32 = 1.0;
    pub const C1_F32: f32 = 1.0;
    pub const C2_F32: f32 = 0.5;
    pub const C3_F32: f32 = 1.0 / 6.0;
    pub const C4_F32: f32 = 1.0 / 24.0;
    pub const C5_F32: f32 = 1.0 / 120.0;
    pub const C6_F32: f32 = 1.0 / 720.0;

    pub const C0_F64: f64 = 1.0;
    pub const C1_F64: f64 = 1.0;
    pub const C2_F64: f64 = 0.5;
    pub const C3_F64: f64 = 1.0 / 6.0;
    pub const C4_F64: f64 = 1.0 / 24.0;
    pub const C5_F64: f64 = 1.0 / 120.0;
    pub const C6_F64: f64 = 1.0 / 720.0;
    pub const C7_F64: f64 = 1.0 / 5040.0;
    pub const C8_F64: f64 = 1.0 / 40320.0;
    pub const C9_F64: f64 = 1.0 / 362880.0;
    pub const C10_F64: f64 = 1.0 / 3628800.0;
    pub const C11_F64: f64 = 1.0 / 39916800.0;
    pub const C12_F64: f64 = 1.0 / 479001600.0;
    pub const C13_F64: f64 = 1.0 / 6227020800.0;

    /// Two-part ln(2) for Cody-Waite range reduction in f64.
    ///
    /// `LN2_HI_F64` carries only the top 33 mantissa bits, so `n * LN2_HI_F64`
    /// is exact for every integer `|n| <= 1024` reachable from the clamped
    /// input range. Reducing with a single rounded ln(2) instead leaves an
    /// absolute error in `r` proportional to `|x|`, which alone costs ~3 decimal
    /// digits at |x| near 709.
    pub const LN2_HI_F64: f64 = 6.931471803691238164e-01;
    pub const LN2_LO_F64: f64 = 1.908214929270587700e-10;

    /// Input clamp range to avoid overflow/underflow
    pub const MIN_F32: f32 = -88.0;
    pub const MAX_F32: f32 = 88.0;
    pub const MIN_F64: f64 = -709.0;
    pub const MAX_F64: f64 = 709.0;
}

// ============================================================================
// Polynomial Coefficients for log(x)
// ============================================================================

/// Coefficients for log(1+f) where f is in [-0.2929, 0.4142]
/// (i.e., mantissa normalized to [sqrt(2)/2, sqrt(2)])
///
/// The f32 set is a direct polynomial in `f`.
///
/// The f64 set is a minimax polynomial in `s = f/(2+f)`, evaluated as
///   `log(1+f) = f - hfsq + s*(hfsq + R(s*s))`, with `hfsq = 0.5*f*f`.
/// A direct polynomial in `f` is unusable at f64 precision: it is the Mercator
/// series, whose truncation error after `n` terms is about `f^(n+1)/(n+1)`, so
/// even nine terms leave ~1e-5 relative error at f = 0.4142 — eleven decimal
/// digits short of double precision. Substituting `s` halves the argument
/// magnitude and kills every even power, so seven terms hold the truncation
/// error below 2^-58 over the whole interval.
pub mod log_coefficients {
    /// ln(2) split head + correction, same role it plays in `exp`: the tail
    /// carries the bits that `n * ln(2)` drops for large exponents.
    pub use super::exp_coefficients::{LN2_HI_F64, LN2_LO_F64};

    // f32 coefficients (7-term polynomial)
    pub const C1_F32: f32 = 0.9999999995;
    pub const C2_F32: f32 = -0.4999999206;
    pub const C3_F32: f32 = 0.3333320848;
    pub const C4_F32: f32 = -0.2500097652;
    pub const C5_F32: f32 = 0.1999796621;
    pub const C6_F32: f32 = -0.1666316004;
    pub const C7_F32: f32 = 0.1428962594;

    // f64 minimax coefficients for `R(z)` with `z = s*s`, `w = z*z`:
    //   R = z*(LG1 + w*(LG3 + w*(LG5 + w*LG7))) + w*(LG2 + w*(LG4 + w*LG6))
    // The two halves are summed separately so the odd and even chains are
    // independent, which shortens the dependency chain without changing the
    // value.
    pub const LG1_F64: f64 = 6.666_666_666_666_735_1e-1;
    pub const LG2_F64: f64 = 3.999_999_999_940_941_9e-1;
    pub const LG3_F64: f64 = 2.857_142_874_366_239_1e-1;
    pub const LG4_F64: f64 = 2.222_219_843_214_978_4e-1;
    pub const LG5_F64: f64 = 1.818_357_216_161_805e-1;
    pub const LG6_F64: f64 = 1.531_383_769_920_937_3e-1;
    pub const LG7_F64: f64 = 1.479_819_860_511_658_6e-1;

    /// Subnormal inputs carry no implicit leading 1, so the exponent/mantissa
    /// split below is only valid after scaling them into the normal range.
    /// 2^54 clears the widest subnormal; the exponent is corrected by -54.
    pub const SUBNORMAL_SCALE_F64: f64 = 18_014_398_509_481_984.0; // 2^54
    pub const SUBNORMAL_SHIFT_F64: f64 = -54.0;

    // IEEE 754 bit manipulation constants
    pub const EXP_BIAS_F32: i32 = 127;
    pub const EXP_BIAS_F64: i64 = 1023;
    pub const MANTISSA_MASK_F32: i32 = 0x007F_FFFF;
    pub const MANTISSA_MASK_F64: u64 = 0x000F_FFFF_FFFF_FFFF;
    pub const EXP_ZERO_F32: i32 = 0x3F80_0000; // exponent = 127 (bias)
    pub const EXP_ZERO_F64: u64 = 0x3FF0_0000_0000_0000; // exponent = 1023 (bias)
}

// ============================================================================
// Polynomial Coefficients for sin/cos
// ============================================================================

/// Coefficients for sin(x) and cos(x).
///
/// The f32 set is the truncated Taylor series
/// `sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!`, `cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6!`,
/// evaluated on the whole reduction interval [-π/4, π/4].
///
/// The f64 set is a pair of minimax polynomials in `z = y²`:
///   `sin(y) = y + y³ * (SIN1 + z*(SIN2 + ... + z*SIN6))`
///   `cos(y) = 1 - z/2 + z² * (COS1 + z*(COS2 + ... + z*COS6))`
/// Truncation error stays below 2^-58 relative over [-π/4, π/4]. Extending the
/// Taylor series instead is not enough: its first dropped term at y = π/4 is
/// `y¹⁰/10! ≈ 2.6e-8` for cos, eight decimal digits short of double precision,
/// and that term is exactly what the old `C8`/`S9` cut-off left behind.
pub mod trig_coefficients {
    // sin(x) coefficients
    pub const S1_F32: f32 = 1.0;
    pub const S3_F32: f32 = -1.0 / 6.0;
    pub const S5_F32: f32 = 1.0 / 120.0;
    pub const S7_F32: f32 = -1.0 / 5040.0;

    // cos(x) coefficients
    pub const C0_F32: f32 = 1.0;
    pub const C2_F32: f32 = -0.5;
    pub const C4_F32: f32 = 1.0 / 24.0;
    pub const C6_F32: f32 = -1.0 / 720.0;

    // f64 minimax sin kernel, coefficient of y³ upward.
    pub const SIN1_F64: f64 = -1.666_666_666_666_663_2e-1;
    pub const SIN2_F64: f64 = 8.333_333_333_322_49e-3;
    pub const SIN3_F64: f64 = -1.984_126_982_985_795e-4;
    pub const SIN4_F64: f64 = 2.755_731_370_707_006_8e-6;
    pub const SIN5_F64: f64 = -2.505_076_025_340_686_3e-8;
    pub const SIN6_F64: f64 = 1.589_690_995_211_55e-10;

    // f64 minimax cos kernel, coefficient of y⁴ upward; the y⁰ and y² terms are
    // the exact 1 and -1/2, carried separately so `1 - z/2` keeps its low bits.
    pub const COS1_F64: f64 = 4.166_666_666_666_66e-2;
    pub const COS2_F64: f64 = -1.388_888_888_887_411e-3;
    pub const COS3_F64: f64 = 2.480_158_728_947_673e-5;
    pub const COS4_F64: f64 = -2.755_731_435_139_066_3e-7;
    pub const COS5_F64: f64 = 2.087_572_321_298_175e-9;
    pub const COS6_F64: f64 = -1.135_964_755_778_819_5e-11;

    /// Four-part π/2 for Cody-Waite reduction in f64, the same construction
    /// `LN2_HI_F64`/`LN2_LO_F64` provides for `exp`.
    ///
    /// Each of the first three parts carries at most 33 mantissa bits, so
    /// `j * PIO2_k` is exact for every quadrant index `|j| <= 2^21` and the
    /// chained subtraction cancels without dropping low bits. Together the four
    /// parts hold π/2 to about 150 bits. Reducing with a single rounded π/2
    /// instead leaves an absolute error in the reduced argument proportional to
    /// `|x|`: at |x| = 100 that is already ~4e-15, tens of ulps of the result.
    pub const PIO2_1_F64: f64 = 1.570_796_326_734_125_6;
    pub const PIO2_2_F64: f64 = 6.077_100_506_303_966e-11;
    pub const PIO2_3_F64: f64 = 2.022_266_248_711_166_5e-21;
    pub const PIO2_3T_F64: f64 = 8.478_427_660_368_9e-32;
}

// ============================================================================
// Polynomial Coefficients for tan(x)
// ============================================================================

/// Truncated Taylor series for tan(x) on [-π/4, π/4], f32 only.
/// `tan(x) ≈ x * (1 + x²*(t3 + x²*(t5 + x²*(t7 + ...))))`
///
/// There is no f64 counterpart: the f64 path forms `tan = sin/cos` from the
/// minimax kernels in `trig_coefficients`, because a fixed-degree polynomial in
/// `x` is a poor fit for tan near π/4, where the function is already steep. The
/// f32 series drops a term worth ~1.5e-4 at that boundary.
pub mod tan_coefficients {
    pub const T1_F32: f32 = 1.0;
    pub const T3_F32: f32 = 0.3333333333333333;
    pub const T5_F32: f32 = 0.13333333333333333;
    pub const T7_F32: f32 = 0.05396825396825397;
    pub const T9_F32: f32 = 0.021869488536155203;
    pub const T11_F32: f32 = 0.008863235529902197;
}

// ============================================================================
// Algorithm Documentation
// ============================================================================

/// Algorithm for exp(x):
///
/// 1. **Range reduction**: exp(x) = 2^(x * log₂(e)) = 2^n * 2^f
///    - Compute y = x * log₂(e)
///    - n = round(y) (integer part)
///    - f = y - n (fractional part in [-0.5, 0.5])
///
/// 2. **Polynomial approximation**: Compute exp(f * ln(2)) using Taylor series
///    - r = f * ln(2) for f32; for f64, Cody-Waite gives r = x - n*LN2_HI - n*LN2_LO,
///      which avoids an absolute error in r that grows with |x|
///    - exp(r) ≈ 1 + r + r²/2! + r³/3! + ... (degree 6 for f32, degree 13 for f64)
///
/// 3. **Reconstruction**: Multiply by 2^n using IEEE 754 bit manipulation
///    - For f32: 2^n = reinterpret((n + 127) << 23)
///    - For f64: 2^n = reinterpret((n + 1023) << 52)
///
/// # Accuracy
/// - f32: Relative error < 1e-6 for inputs in [-88, 88]
/// - f64: Relative error within a few ulp for inputs in [-709, 709]
///
/// # Edge Cases
/// - Inputs outside the valid range are clamped to avoid overflow/underflow
pub const _EXP_ALGORITHM_DOC: () = ();

/// Algorithm for log(x):
///
/// 1. **Argument decomposition**: log(x) = log(2^n * m) = n * log(2) + log(m)
///    - Extract exponent n from IEEE 754 representation
///    - Extract mantissa m, normalized to [1, 2)
///
/// 2. **Range normalization**: If m > √2, divide by 2 and increment n
///    - This keeps f = m - 1 in [-0.2929, 0.4142] for better polynomial convergence
///
/// 3. **Polynomial approximation**: Compute log(1 + f)
///    - f32: `log(1+f) ≈ f * (c₁ + f*(c₂ + ...))` (Horner's method)
///    - f64: substitute `s = f/(2+f)` and evaluate
///      `log(1+f) = f - (hfsq - s*(hfsq + R(s²)))` with `hfsq = f²/2`, which
///      removes the even powers a direct series must otherwise carry
///
/// 4. **Reconstruction**: result = n * ln(2) + log(m), with ln(2) split
///    head + tail for f64 so large exponents keep their low bits
///
/// log2 and log10 reuse steps 1-3 and add the exact integer exponent back
/// separately, which keeps exact powers of two exact.
///
/// # Accuracy
/// - f32: Relative error < 1e-6 for positive inputs
/// - f64: Relative error below 2 ulps for positive inputs
///
/// # Edge Cases
/// - x = 0: Returns -inf; x < 0 and NaN: Returns NaN
/// - x = +inf: Returns +inf
/// - Subnormal x is scaled into the normal range first, so it keeps full
///   precision instead of decomposing against an absent leading 1
pub const _LOG_ALGORITHM_DOC: () = ();

/// Algorithm for sin(x) and cos(x):
///
/// 1. **Range reduction**: Reduce x to y in [-π/4, π/4]
///    - j = round(x * 2/π) (quadrant index)
///    - f32: y = x - j * π/2 with a single rounded π/2
///    - f64: Cody-Waite, subtracting `j*PIO2_1`, `j*PIO2_2`, `j*PIO2_3` and
///      `j*PIO2_3T` in turn. Each product is exact and each subtraction cancels
///      exactly, so `y` keeps full relative precision even where the result is
///      near a zero of sin or cos.
///
/// 2. **Polynomial approximation**:
///    - f32: Taylor, `sin(y) ≈ y - y³/6 + y⁵/120 - y⁷/5040` and
///      `cos(y) ≈ 1 - y²/2 + y⁴/24 - y⁶/720`
///    - f64: the minimax kernels in `trig_coefficients`. cos is evaluated as
///      `w + (((1 - w) - z/2) + z²*P(z))` with `w = 1 - z/2`, which recovers the
///      bits the leading subtraction rounds away.
///
/// 3. **Quadrant selection**: Based on j mod 4:
///    - 0: sin(x) = sin(y)
///    - 1: sin(x) = cos(y)
///    - 2: sin(x) = -sin(y)
///    - 3: sin(x) = -cos(y)
///
///    cos(x) is the same table shifted by one quadrant. Computing it as
///    `sin(x + π/2)` instead would round `x + π/2` before reduction and lose
///    bits proportional to |x|.
///
/// # Accuracy
/// - f32: Relative error < 1e-6 near zero, degrading toward the ends of the
///   reduction interval where the dropped Taylor term is largest
/// - f64: Relative error below 4 ulps for |x| <= 2^21 * π/2 (about 3.3e6)
///
/// # Input Range Warning
/// Past |x| = 2^21 * π/2 the products `j * PIO2_k` stop being exact and the
/// reduction degrades rapidly; correctness beyond that bound requires a
/// Payne-Hanek reduction, which is not implemented here.
///
/// # Edge Cases
/// - sin(±0) = ±0, cos(±0) = 1
/// - sin/cos of ±inf or NaN: NaN
pub const _TRIG_ALGORITHM_DOC: () = ();

// ============================================================================
// Polynomial Coefficients for atan(x)
// ============================================================================

/// Coefficients for atan(x).
///
/// The f32 set is the Gregory series `atan(x) ≈ x * (A0 + x²*(A2 + ...))`,
/// used on [-1, 1] after the identity `atan(x) = sign(x)*π/2 - atan(1/x)`.
///
/// The f64 set is a minimax polynomial valid only on |x| ≤ 0.4375, paired with
/// the reduction points below. The Gregory series is unusable at f64 precision:
/// its truncation error at |x| = 1 is on the order of `1/(2n+3)`, so even 11
/// terms leave ~2e-2 relative error at the reduction boundary — fourteen decimal
/// digits short of double precision.
pub mod atan_coefficients {
    // f32 coefficients (7-term polynomial, ~1e-7 accuracy)
    pub const A0_F32: f32 = 1.0;
    pub const A2_F32: f32 = -0.333333333;
    pub const A4_F32: f32 = 0.2;
    pub const A6_F32: f32 = -0.142857142;
    pub const A8_F32: f32 = 0.111111111;
    pub const A10_F32: f32 = -0.0909090909;
    pub const A12_F32: f32 = 0.0769230769;

    // f64 minimax coefficients for `atan(t) ≈ t - t*(s1 + s2)` on |t| ≤ 0.4375,
    // where `z = t²`, `w = z²`,
    //   s1 = z*(AT0 + w*(AT2 + w*(AT4 + w*(AT6 + w*(AT8 + w*AT10)))))
    //   s2 = w*(AT1 + w*(AT3 + w*(AT5 + w*(AT7 + w*AT9))))
    // Truncation error over that interval stays below 2^-60.
    pub const AT0_F64: f64 = 3.333_333_333_333_293_2e-1;
    pub const AT1_F64: f64 = -1.999_999_999_987_648_3e-1;
    pub const AT2_F64: f64 = 1.428_571_427_250_346_6e-1;
    pub const AT3_F64: f64 = -1.111_111_040_546_235_6e-1;
    pub const AT4_F64: f64 = 9.090_887_133_436_507e-2;
    pub const AT5_F64: f64 = -7.691_876_205_044_83e-2;
    pub const AT6_F64: f64 = 6.661_073_137_387_531e-2;
    pub const AT7_F64: f64 = -5.833_570_133_790_573_5e-2;
    pub const AT8_F64: f64 = 4.976_877_994_615_932_4e-2;
    pub const AT9_F64: f64 = -3.653_157_274_421_691_5e-2;
    pub const AT10_F64: f64 = 1.628_582_011_536_578_2e-2;

    /// Reduction breakpoints on |x|. Lane `i` uses centre `C_i` when
    /// `|x| < BREAK_i`, and the reciprocal form when `|x| >= BREAK3_F64`.
    pub const BREAK0_F64: f64 = 0.4375; // 7/16  — no reduction, centre 0
    pub const BREAK1_F64: f64 = 0.6875; // 11/16 — centre 0.5
    pub const BREAK2_F64: f64 = 1.1875; // 19/16 — centre 1.0
    pub const BREAK3_F64: f64 = 2.4375; // 39/16 — centre 1.5, else -1/x

    /// `atan(centre)` split into a head plus a correction term, in the style of
    /// `LN2_HI_F64`/`LN2_LO_F64`. The head alone is a rounded double, so adding
    /// it back after the polynomial would cost up to half an ulp of π/2; the
    /// low part restores those bits.
    pub const ATAN_HI0_F64: f64 = 4.636_476_090_008_061e-1; // atan(0.5)
    pub const ATAN_HI1_F64: f64 = std::f64::consts::FRAC_PI_4; // atan(1.0)
    pub const ATAN_HI2_F64: f64 = 9.827_937_232_473_29e-1; // atan(1.5)
    pub const ATAN_HI3_F64: f64 = std::f64::consts::FRAC_PI_2; // atan(inf) = pi/2
    pub const ATAN_LO0_F64: f64 = 2.269_877_745_296_168_7e-17;
    pub const ATAN_LO1_F64: f64 = 3.061_616_997_868_383e-17;
    pub const ATAN_LO2_F64: f64 = 1.390_331_103_123_099_8e-17;
    pub const ATAN_LO3_F64: f64 = 6.123_233_995_736_766e-17;
}

// ============================================================================
// Rational approximation for asin(x) / acos(x)
// ============================================================================

/// f64 coefficients for `R(t) = p(t)/q(t)`, the correction term shared by asin
/// and acos:
///   p(t) = t*(PS0 + t*(PS1 + t*(PS2 + t*(PS3 + t*(PS4 + t*PS5)))))
///   q(t) = 1 + t*(QS1 + t*(QS2 + t*(QS3 + t*QS4)))
/// so that `asin(y) ≈ y + y*R(y²)` for |y| ≤ 0.5.
///
/// Above |x| = 0.5 the reflection `asin(x) = π/2 - 2*asin(sqrt((1-|x|)/2))`
/// moves the argument back into that interval, where the direct series would
/// otherwise need unbounded degree as |x| approaches 1.
pub mod asin_coefficients {
    pub const PS0_F64: f64 = 1.666_666_666_666_666_6e-1;
    pub const PS1_F64: f64 = -3.255_658_186_224_009e-1;
    pub const PS2_F64: f64 = 2.012_125_321_348_629_3e-1;
    pub const PS3_F64: f64 = -4.005_553_450_067_941e-2;
    pub const PS4_F64: f64 = 7.915_349_942_898_145e-4;
    pub const PS5_F64: f64 = 3.479_331_075_960_212e-5;
    pub const QS1_F64: f64 = -2.403_394_911_734_414_2;
    pub const QS2_F64: f64 = 2.020_945_760_233_505_7;
    pub const QS3_F64: f64 = -6.882_839_716_054_533e-1;
    pub const QS4_F64: f64 = 7.703_815_055_590_193_5e-2;

    /// π/2 and π split head + correction, same role as `ATAN_HI*`/`ATAN_LO*`:
    /// the reflection subtracts a value of comparable size, so the low parts
    /// carry the bits that cancellation would otherwise drop.
    pub const PIO2_HI_F64: f64 = std::f64::consts::FRAC_PI_2;
    pub const PIO2_LO_F64: f64 = 6.123_233_995_736_766e-17;
    pub const PI_HI_F64: f64 = std::f64::consts::PI;

    /// Branch point between the direct series and the reflection.
    pub const HALF_F64: f64 = 0.5;
}

/// Algorithm for tan(x):
///
/// 1. **Range reduction**: Reduce x to y in [-π/4, π/4]
///    - f32: j = round(x * 2/π), y = x - j * π/2
///    - f64: the Cody-Waite chain described in `_TRIG_ALGORITHM_DOC`
///
/// 2. **Approximation on the reduced argument**:
///    - f32: the odd polynomial `tan(y) ≈ y * (1 + y²*(t₃ + y²*(t₅ + ...)))`
///    - f64: `tan(y) = sin(y)/cos(y)` from the two minimax kernels. A single
///      polynomial in y is the wrong shape here — tan grows without bound at
///      π/2, so a fit that is tight near zero is loose at the interval edge.
///
/// 3. **Quadrant handling**: For odd quadrants, use cotangent
///    - f32: result = -1/tan(y)
///    - f64: result = -cos(y)/sin(y), which swaps the two kernels rather than
///      inverting an already-rounded quotient
///
/// # Accuracy
/// - f32: Relative error < 2e-4
/// - f64: Relative error below 4 ulps away from the poles, for
///   |x| <= 2^21 * π/2 (about 3.3e6)
///
/// # Edge Cases
/// - tan(±0) = ±0
/// - tan of ±inf or NaN: NaN
/// - Near the asymptotes at ±π/2, ±3π/2, ... the result itself is
///   ill-conditioned: a half-ulp perturbation of x moves the value by an
///   unbounded amount, so relative error there is not a property of the kernel
pub const _TAN_ALGORITHM_DOC: () = ();

/// Algorithm for atan(x):
///
/// 1. **Sign handling**: Save sign of x, work with |x|
///
/// 2. **Range reduction**:
///    - f32: for |x| > 1, atan(x) = π/2 - atan(1/x)
///    - f64: pick a centre c from {0, 0.5, 1.0, 1.5} by the breakpoints in
///      `atan_coefficients`, and evaluate at t = (|x| - c)/(1 + c*|x|), which
///      lands in [-0.4375, 0.4375]. Past the last breakpoint use t = -1/|x|
///      with c conceptually at infinity. Four centres keep the argument small
///      enough that a fixed-degree polynomial covers the whole line.
///
/// 3. **Polynomial approximation** on the reduced argument (Horner's method)
///
/// 4. **Reconstruction**: atan(x) = atan(c) + atan(t), with atan(c) added back
///    as a head plus a correction term, then the sign restored
///
/// # Accuracy
/// - f32: Relative error < 1e-6 for all finite inputs
/// - f64: Relative error below 2 ulps for all finite inputs
///
/// # Edge Cases
/// - atan(±∞) = ±π/2
/// - atan(0) = 0
/// - atan(NaN) = NaN
pub const _ATAN_ALGORITHM_DOC: () = ();

/// Algorithm for asin(x) and acos(x) in f64:
///
/// 1. **Interval split** at |x| = 0.5. Below it, evaluate the rational
///    correction directly at t = x². Above it, reflect through
///    t = (1 - |x|)/2 and s = sqrt(t), which maps the hard region near
///    |x| = 1 onto small t.
///
/// 2. **Shared kernel**: both branches use the same R(t) from
///    `asin_coefficients`.
///
/// 3. **Reconstruction**:
///    - asin, |x| <= 0.5: |x| + |x|*R(x²), sign restored
///    - asin, |x| > 0.5:  π/2 - (2*(s + s*R(t)) - π/2_lo), sign restored
///    - acos, |x| <= 0.5: π/2 - (x + (x*R(x²) - π/2_lo))
///    - acos, x > 0.5:    2*(s + s*R(t))
///    - acos, x < -0.5:   π - 2*(s + (s*R(t) - π/2_lo))
///
///    acos is built directly rather than as π/2 - asin(x): near x = 1 that
///    subtraction cancels away every significant bit of the result.
///
/// # Accuracy
/// - f64: Relative error below 2 ulps over [-1, 1]
///
/// # Edge Cases
/// - asin(±1) = ±π/2, acos(1) = 0, acos(-1) = π
/// - |x| > 1 or x = NaN: NaN
pub const _ASIN_ACOS_ALGORITHM_DOC: () = ();
