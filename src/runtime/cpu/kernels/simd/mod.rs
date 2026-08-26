//! SIMD detection and dispatch for CPU kernels
//!
//! This module provides runtime CPU feature detection. Each operation type
//! (matmul, binary, unary, reduce) has its own submodule with SIMD kernels.
//!
//! # Module Structure
//!
//! ```text
//! simd/
//! ├── mod.rs              # This file: detection only
//! ├── {operation}/        # Each operation type
//! │   ├── mod.rs          # Unified dispatch (handles all architectures)
//! │   ├── x86_64/         # x86-64 implementations
//! │   │   ├── mod.rs
//! │   │   ├── avx2.rs
//! │   │   └── avx512.rs
//! │   └── aarch64/        # ARM64 implementations
//! │       ├── mod.rs
//! │       └── neon.rs
//! ```
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Vector Width | Status    |
//! |--------------|-----------------|--------------|-----------|
//! | x86-64       | AVX-512F/VL/DQ/BW + FMA | 512 bits | Supported |
//! | x86-64       | AVX2 + FMA      | 256 bits     | Supported |
//! | ARM64        | NEON + FP16     | 128 bits     | Supported |
//! | ARM64        | NEON            | 128 bits     | Supported |
//! | Any          | Scalar          | N/A          | Fallback  |

// Shared f16/bf16 ↔ f32 SIMD conversion utilities
#[cfg(feature = "f16")]
pub mod half_convert_utils;

// Macros for generating f16/bf16 block-convert-compute wrappers (must come before users)
// Always compiled - macros internally gate generated code with #[cfg(feature = "f16")]
#[macro_use]
mod half_macros;

// Operation modules - available on all architectures
// Each operation's mod.rs handles internal architecture dispatch
pub mod activations;
pub mod binary;
pub mod clamp;
pub mod compare;
pub mod conv;
pub mod cumulative;
pub mod dot;
pub mod fused_activation_mul;
pub mod fused_elementwise;
pub mod index;
pub mod logsumexp;
pub mod math;
pub mod matmul;
pub mod norm;
pub mod reduce;
pub mod scalar;
pub mod softmax;
pub mod softmax_bwd;
pub mod special;
pub mod unary;
pub mod where_select;

// x86-64 only: streaming store utilities
#[cfg(target_arch = "x86_64")]
pub mod streaming;

use std::sync::OnceLock;

/// SIMD capability level detected at runtime
///
/// Supports multiple architectures with ordered capability levels.
/// Higher values indicate more capable SIMD instruction sets.
///
/// Note: All variants are defined on all platforms for API completeness,
/// but some are only constructed at runtime on their respective architectures.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)] // Variants may not be constructed on all architectures
pub enum SimdLevel {
    // x86-64 variants (highest capability)
    /// AVX-512F + VL + DQ + BW with FMA (512-bit vectors, 16 f32s or 8 f64s)
    Avx512 = 4,
    /// AVX2 with FMA support (256-bit vectors, 8 f32s or 4 f64s)
    Avx2Fma = 3,

    // ARM64 variants
    /// NEON with native FP16 support (128-bit vectors, 4 f32s or 2 f64s)
    NeonFp16 = 2,
    /// NEON baseline for AArch64 (128-bit vectors, 4 f32s or 2 f64s)
    Neon = 1,

    // Universal fallback
    /// Scalar fallback (no SIMD)
    Scalar = 0,
}

// These methods are part of the public API but may not be used within the library itself.
// They are provided for library consumers to query SIMD capabilities.
#[allow(dead_code)]
impl SimdLevel {
    /// Returns true if this is an x86-64 SIMD level
    #[inline]
    pub const fn is_x86(self) -> bool {
        matches!(self, Self::Avx512 | Self::Avx2Fma)
    }

    /// Returns true if this is an ARM64 SIMD level
    #[inline]
    pub const fn is_arm64(self) -> bool {
        matches!(self, Self::Neon | Self::NeonFp16)
    }

    /// Returns true if this level supports 512-bit operations
    #[inline]
    pub const fn has_avx512(self) -> bool {
        matches!(self, Self::Avx512)
    }

    /// Returns true if this level supports 256-bit operations
    #[inline]
    pub const fn has_avx2(self) -> bool {
        matches!(self, Self::Avx512 | Self::Avx2Fma)
    }

    /// Returns true if this level supports NEON operations
    #[inline]
    pub const fn has_neon(self) -> bool {
        matches!(self, Self::Neon | Self::NeonFp16)
    }

    /// Returns the number of f32 elements per vector register
    #[inline]
    pub const fn f32_lanes(self) -> usize {
        match self {
            Self::Avx512 => 16,
            Self::Avx2Fma => 8,
            Self::Neon | Self::NeonFp16 => 4,
            Self::Scalar => 1,
        }
    }

    /// Returns the number of f64 elements per vector register
    #[inline]
    pub const fn f64_lanes(self) -> usize {
        match self {
            Self::Avx512 => 8,
            Self::Avx2Fma => 4,
            Self::Neon | Self::NeonFp16 => 2,
            Self::Scalar => 1,
        }
    }

    /// Returns the name of this SIMD level as a string
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Avx512 => "AVX-512",
            Self::Avx2Fma => "AVX2+FMA",
            Self::NeonFp16 => "NEON+FP16",
            Self::Neon => "NEON",
            Self::Scalar => "Scalar",
        }
    }
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Cached SIMD level detection
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Detect the best available SIMD level for the current CPU
///
/// This function is cached - the first call performs detection,
/// subsequent calls return the cached result with ~1ns overhead.
#[inline]
pub fn detect_simd() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(detect_simd_uncached)
}

/// Perform actual CPU feature detection (called once)
#[cold]
fn detect_simd_uncached() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        // All five features are required by kernels dispatched on
        // `SimdLevel::Avx512`, so all five are checked here. Removing any one
        // check makes the kernels named below fault with #UD on a CPU that
        // lacks that feature:
        // - `avx512f`: the 512-bit core (loads, arithmetic, masks).
        // - `avx512vl`: the 128/256-bit AVX-512 forms used by tail handling.
        // - `avx512dq`: the packed 64-bit integer conversions
        //   `_mm512_cvtpd_epi64` and `_mm512_cvtepi64_pd` in the f64 math
        //   kernels (`exp_f64`, `log_f64`, `sin_f64`, `tan_f64`).
        // - `avx512bw`: the byte and word ops in the i8 dot kernel
        //   (`_mm512_cvtepi8_epi16`, `_mm512_madd_epi16`) and in the index
        //   kernel `masked_count` (`_mm512_cmpneq_epi8_mask`).
        // - `fma`: the fused multiply-add used by every polynomial kernel.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("fma")
        {
            return SimdLevel::Avx512;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdLevel::Avx2Fma;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory for AArch64 - always available
        // Check for optional FP16 support
        if std::arch::is_aarch64_feature_detected!("fp16") {
            return SimdLevel::NeonFp16;
        }
        return SimdLevel::Neon;
    }

    #[allow(unreachable_code)]
    SimdLevel::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection_is_cached() {
        let level1 = detect_simd();
        let level2 = detect_simd();
        assert_eq!(level1, level2);
    }

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2Fma);
        assert!(SimdLevel::Avx2Fma > SimdLevel::NeonFp16);
        assert!(SimdLevel::NeonFp16 > SimdLevel::Neon);
        assert!(SimdLevel::Neon > SimdLevel::Scalar);
    }

    #[test]
    fn test_simd_level_capabilities() {
        // x86 capabilities
        assert!(SimdLevel::Avx512.has_avx512());
        assert!(SimdLevel::Avx512.has_avx2());
        assert!(!SimdLevel::Avx2Fma.has_avx512());
        assert!(SimdLevel::Avx2Fma.has_avx2());
        assert!(!SimdLevel::Scalar.has_avx512());
        assert!(!SimdLevel::Scalar.has_avx2());

        // ARM capabilities
        assert!(SimdLevel::Neon.has_neon());
        assert!(SimdLevel::NeonFp16.has_neon());
        assert!(!SimdLevel::Avx512.has_neon());
        assert!(!SimdLevel::Scalar.has_neon());
    }

    #[test]
    fn test_architecture_detection() {
        // x86 architecture
        assert!(SimdLevel::Avx512.is_x86());
        assert!(SimdLevel::Avx2Fma.is_x86());
        assert!(!SimdLevel::Neon.is_x86());
        assert!(!SimdLevel::Scalar.is_x86());

        // ARM architecture
        assert!(SimdLevel::Neon.is_arm64());
        assert!(SimdLevel::NeonFp16.is_arm64());
        assert!(!SimdLevel::Avx512.is_arm64());
        assert!(!SimdLevel::Scalar.is_arm64());
    }

    #[test]
    fn test_lane_counts() {
        // x86 lane counts
        assert_eq!(SimdLevel::Avx512.f32_lanes(), 16);
        assert_eq!(SimdLevel::Avx2Fma.f32_lanes(), 8);
        assert_eq!(SimdLevel::Avx512.f64_lanes(), 8);
        assert_eq!(SimdLevel::Avx2Fma.f64_lanes(), 4);

        // ARM lane counts
        assert_eq!(SimdLevel::Neon.f32_lanes(), 4);
        assert_eq!(SimdLevel::NeonFp16.f32_lanes(), 4);
        assert_eq!(SimdLevel::Neon.f64_lanes(), 2);
        assert_eq!(SimdLevel::NeonFp16.f64_lanes(), 2);

        // Scalar
        assert_eq!(SimdLevel::Scalar.f32_lanes(), 1);
        assert_eq!(SimdLevel::Scalar.f64_lanes(), 1);
    }
}
