//! High-level TensorOps trait
//!
//! Aggregates all operation traits into a single convenience trait.

use crate::runtime::Runtime;

use super::{
    ActivationOps, BinaryOps, ComplexOps, ConditionalOps, CumulativeOps, DistanceOps, IndexingOps,
    LinalgOps, MatmulOps, NormalizationOps, ReduceOps, SemiringMatmulOps, ShapeOps, SortingOps,
    StatisticalOps, TypeConversionOps, UnaryOps, UtilityOps,
};

/// Core tensor operations trait
///
/// This trait aggregates all operation traits into a single convenience trait.
/// It is implemented by `RuntimeClient` types, giving operations access to
/// the device and allocator for creating output tensors.
///
/// # Example
///
/// ```ignore
/// let device = CpuDevice::new();
/// let client = CpuRuntime::default_client(&device);
///
/// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
/// let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device)?;
///
/// let c = client.add(&a, &b)?;
/// ```
pub trait TensorOps<R: Runtime>:
    TypeConversionOps<R>
    + ConditionalOps<R>
    + ComplexOps<R>
    + DistanceOps<R>
    + NormalizationOps<R>
    + MatmulOps<R>
    + CumulativeOps<R>
    + ActivationOps<R>
    + UtilityOps<R>
    + ReduceOps<R>
    + IndexingOps<R>
    + LinalgOps<R>
    + ShapeOps<R>
    + SortingOps<R>
    + StatisticalOps<R>
    + UnaryOps<R>
    + BinaryOps<R>
    + SemiringMatmulOps<R>
{
    // All methods are provided by the individual trait implementations
}
