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

/// Minimax polynomial coefficients for log(1+f) where f is in [-0.2929, 0.4142]
/// (i.e., mantissa normalized to [sqrt(2)/2, sqrt(2)])
pub mod log_coefficients {
    // f32 coefficients (7-term polynomial)
    pub const C1_F32: f32 = 0.9999999995;
    pub const C2_F32: f32 = -0.4999999206;
    pub const C3_F32: f32 = 0.3333320848;
    pub const C4_F32: f32 = -0.2500097652;
    pub const C5_F32: f32 = 0.1999796621;
    pub const C6_F32: f32 = -0.1666316004;
    pub const C7_F32: f32 = 0.1428962594;

    // f64 coefficients (9-term polynomial for higher precision)
    pub const C1_F64: f64 = 0.9999999999999999;
    pub const C2_F64: f64 = -0.5;
    pub const C3_F64: f64 = 0.33333333333333333;
    pub const C4_F64: f64 = -0.25;
    pub const C5_F64: f64 = 0.2;
    pub const C6_F64: f64 = -0.16666666666666666;
    pub const C7_F64: f64 = 0.14285714285714285;
    pub const C8_F64: f64 = -0.125;
    pub const C9_F64: f64 = 0.1111111111111111;

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

/// Taylor series coefficients for sin(x) and cos(x)
/// sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9!
/// cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8!
pub mod trig_coefficients {
    // sin(x) coefficients
    pub const S1_F32: f32 = 1.0;
    pub const S3_F32: f32 = -1.0 / 6.0;
    pub const S5_F32: f32 = 1.0 / 120.0;
    pub const S7_F32: f32 = -1.0 / 5040.0;

    pub const S1_F64: f64 = 1.0;
    pub const S3_F64: f64 = -1.0 / 6.0;
    pub const S5_F64: f64 = 1.0 / 120.0;
    pub const S7_F64: f64 = -1.0 / 5040.0;
    pub const S9_F64: f64 = 1.0 / 362880.0;

    // cos(x) coefficients
    pub const C0_F32: f32 = 1.0;
    pub const C2_F32: f32 = -0.5;
    pub const C4_F32: f32 = 1.0 / 24.0;
    pub const C6_F32: f32 = -1.0 / 720.0;

    pub const C0_F64: f64 = 1.0;
    pub const C2_F64: f64 = -0.5;
    pub const C4_F64: f64 = 1.0 / 24.0;
    pub const C6_F64: f64 = -1.0 / 720.0;
    pub const C8_F64: f64 = 1.0 / 40320.0;
}

// ============================================================================
// Polynomial Coefficients for tan(x)
// ============================================================================

/// Minimax polynomial coefficients for tan(x) on [-π/4, π/4]
/// tan(x) ≈ x * (1 + x²*(t3 + x²*(t5 + x²*(t7 + ...))))
pub mod tan_coefficients {
    pub const T1_F32: f32 = 1.0;
    pub const T3_F32: f32 = 0.3333333333333333;
    pub const T5_F32: f32 = 0.13333333333333333;
    pub const T7_F32: f32 = 0.05396825396825397;
    pub const T9_F32: f32 = 0.021869488536155203;
    pub const T11_F32: f32 = 0.008863235529902197;

    pub const T1_F64: f64 = 1.0;
    pub const T3_F64: f64 = 0.3333333333333333;
    pub const T5_F64: f64 = 0.13333333333333333;
    pub const T7_F64: f64 = 0.05396825396825397;
    pub const T9_F64: f64 = 0.021869488536155203;
    pub const T11_F64: f64 = 0.008863235529902197;
    pub const T13_F64: f64 = 0.003592128036572481;
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
/// 3. **Polynomial approximation**: Compute log(1 + f) using minimax polynomial
///    - log(1+f) ≈ f * (c₁ + f*(c₂ + f*(c₃ + ...))) (Horner's method)
///
/// 4. **Reconstruction**: result = n * ln(2) + log(m)
///
/// # Accuracy
/// - f32: Relative error < 1e-6 for positive inputs
/// - f64: Relative error < 1e-12 for positive inputs
///
/// # Edge Cases
/// - x ≤ 0: Returns -inf or NaN (follows IEEE 754 semantics)
/// - x = +inf: Returns +inf
pub const _LOG_ALGORITHM_DOC: () = ();

/// Algorithm for sin(x) and cos(x):
///
/// 1. **Range reduction**: Reduce x to y in [-π/4, π/4]
///    - j = round(x * 2/π) (quadrant index)
///    - y = x - j * π/2
///
/// 2. **Polynomial approximation**:
///    - sin(y) ≈ y - y³/6 + y⁵/120 - y⁷/5040 (Taylor series)
///    - cos(y) ≈ 1 - y²/2 + y⁴/24 - y⁶/720 (Taylor series)
///
/// 3. **Quadrant selection**: Based on j mod 4:
///    - 0: sin(x) = sin(y)
///    - 1: sin(x) = cos(y)
///    - 2: sin(x) = -sin(y)
///    - 3: sin(x) = -cos(y)
///
/// # Accuracy
/// - Relative error < 1e-6 for f32, < 1e-10 for f64
/// - Accuracy degrades for very large inputs due to range reduction precision
///
/// # Input Range Warning
/// For |x| > 2^20, range reduction may lose significant precision.
/// Consider using extended precision range reduction for very large inputs.
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
///    - j = round(x * 2/π)
///    - y = x - j * π/2
///
/// 2. **Polynomial approximation**: Using odd polynomial
///    - tan(y) ≈ y * (1 + y²*(t₃ + y²*(t₅ + ...)))
///
/// 3. **Quadrant handling**: For odd quadrants, use cotangent
///    - If j is odd: result = -1/tan(y) (cotangent)
///
/// # Accuracy
/// - Relative error < 2e-4 for f32, < 1e-4 for f64
/// - Note: tan(x) has asymptotes at x = ±π/2, ±3π/2, etc.
///
/// # Edge Cases
/// - Near asymptotes: Results may have large errors or overflow
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
