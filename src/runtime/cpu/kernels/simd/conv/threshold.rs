//! SIMD dispatch thresholds shared by the `conv2d` and `depthwise_conv2d` drivers.

/// Minimum work per output row to justify SIMD overhead for f32
pub(super) const SIMD_THRESHOLD_F32: usize = 8;

/// Minimum work per output row to justify SIMD overhead for f64
pub(super) const SIMD_THRESHOLD_F64: usize = 4;
