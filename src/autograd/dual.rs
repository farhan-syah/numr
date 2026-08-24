//! Dual number tensor for forward-mode automatic differentiation
//!
//! This module provides `DualTensor`, which carries both a primal (actual) value
//! and its tangent (derivative) through computation. This is the foundation
//! for computing Jacobian-vector products (JVP) in a single forward pass.
//!
//! # Mathematical Background
//!
//! Dual numbers extend real numbers with an infinitesimal element ε where ε² = 0.
//! Any dual number can be written as: a + b·ε
//!
//! When we compute f(a + b·ε), Taylor expansion gives:
//! f(a + b·ε) = f(a) + f'(a)·b·ε + O(ε²) = f(a) + f'(a)·b·ε
//!
//! The primal part gives f(a), and the tangent part gives f'(a)·b.
//! This allows computing both the function value and directional derivative
//! in a single forward pass.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Dual number tensor for forward-mode automatic differentiation
///
/// A `DualTensor` carries both:
/// - **primal**: The actual value of the tensor (what we'd compute without AD)
/// - **tangent**: The derivative of the value with respect to some input
///
/// During forward-mode AD, operations propagate both values simultaneously,
/// allowing computation of Jacobian-vector products (JVP) efficiently.
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # use numr::autograd::DualTensor;
/// # let device = CpuDevice::new();
/// // Create a dual tensor with primal x=3.0 and tangent v=1.0
/// let primal = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[], &device).unwrap();
/// let tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[], &device).unwrap();
/// let x = DualTensor::new(primal, Some(tangent));
///
/// // Access primal and tangent components
/// assert_eq!(x.primal().shape(), &[]);  // scalar
/// assert!(x.tangent().is_some());
/// ```
#[derive(Debug, Clone)]
pub struct DualTensor<R: Runtime> {
    /// The primal (actual) value
    primal: Tensor<R>,
    /// The tangent (derivative) value - same shape as primal
    /// None represents zero tangent (constant with respect to all inputs)
    tangent: Option<Tensor<R>>,
}

impl<R: Runtime> DualTensor<R> {
    /// Create a new dual tensor with explicit primal and tangent
    ///
    /// # Arguments
    ///
    /// * `primal` - The actual tensor value
    /// * `tangent` - The tangent vector (must have same shape as primal if Some)
    ///
    /// # Panics
    ///
    /// Panics if tangent shape doesn't match primal shape.
    pub fn new(primal: Tensor<R>, tangent: Option<Tensor<R>>) -> Self {
        if let Some(ref t) = tangent {
            assert_eq!(
                primal.shape(),
                t.shape(),
                "Tangent shape {:?} must match primal shape {:?}",
                t.shape(),
                primal.shape()
            );
        }
        Self { primal, tangent }
    }

    /// Create a dual tensor from primal with zero tangent (constant)
    ///
    /// Use this for values that don't depend on the inputs we're differentiating
    /// with respect to (e.g., constants, parameters when computing gradient w.r.t. data).
    pub fn constant(primal: Tensor<R>) -> Self {
        Self {
            primal,
            tangent: None,
        }
    }

    /// Create a dual tensor with unit tangent (for computing partial derivatives)
    ///
    /// The tangent is initialized to all ones with the same shape as the primal.
    /// This is useful when computing the derivative of a scalar function.
    pub fn with_unit_tangent(primal: Tensor<R>, device: &R::Device) -> Result<Self>
    where
        R: Runtime<DType = DType>,
    {
        let tangent = Tensor::ones(primal.shape(), primal.dtype(), device)?;
        Ok(Self {
            primal,
            tangent: Some(tangent),
        })
    }

    /// Create a dual tensor from a primal with a specific tangent
    ///
    /// This is the primary way to seed forward-mode AD: create dual tensors
    /// from your inputs where the tangent is the direction you want to differentiate along.
    pub fn with_tangent(primal: Tensor<R>, tangent: Tensor<R>) -> Self {
        Self::new(primal, Some(tangent))
    }

    /// Get a reference to the primal value
    #[inline]
    pub fn primal(&self) -> &Tensor<R> {
        &self.primal
    }

    /// Get a reference to the tangent value (if any)
    #[inline]
    pub fn tangent(&self) -> Option<&Tensor<R>> {
        self.tangent.as_ref()
    }

    /// Take ownership of the primal value
    pub fn into_primal(self) -> Tensor<R> {
        self.primal
    }

    /// Take ownership of both primal and tangent
    pub fn into_parts(self) -> (Tensor<R>, Option<Tensor<R>>) {
        (self.primal, self.tangent)
    }

    /// Check if this dual tensor has a non-zero tangent
    #[inline]
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Get the shape of the tensor
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.primal.shape()
    }

    /// Get the data type
    #[inline]
    pub fn dtype(&self) -> DType
    where
        R: Runtime<DType = DType>,
    {
        self.primal.dtype()
    }

    /// Get the device
    #[inline]
    pub fn device(&self) -> &R::Device {
        self.primal.device()
    }

    /// Get the number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.primal.numel()
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.primal.ndim()
    }

    /// Detach from AD - returns a constant dual tensor
    pub fn detach(&self) -> Self {
        Self {
            primal: self.primal.clone(),
            tangent: None,
        }
    }

    /// Create a zero tangent with the same shape as the primal
    ///
    /// This is useful when we need an explicit zero tangent for operations
    /// that can't handle `Option<Tensor>` directly.
    pub fn zero_tangent(&self, device: &R::Device) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
    {
        Tensor::zeros(self.primal.shape(), self.primal.dtype(), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_dual_tensor_new() {
        let device = CpuDevice::new();
        let primal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let tangent = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3], &[3], &device).unwrap();

        let dual = DualTensor::new(primal.clone(), Some(tangent.clone()));

        assert_eq!(dual.shape(), &[3]);
        assert!(dual.has_tangent());
        assert_eq!(dual.primal().to_vec::<f32>(), [1.0, 2.0, 3.0]);
        assert_eq!(dual.tangent().unwrap().to_vec::<f32>(), [0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_dual_tensor_constant() {
        let device = CpuDevice::new();
        let primal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        let dual = DualTensor::constant(primal);

        assert!(!dual.has_tangent());
        assert!(dual.tangent().is_none());
    }

    #[test]
    fn test_dual_tensor_with_unit_tangent() {
        let device = CpuDevice::new();
        let primal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        let dual = DualTensor::with_unit_tangent(primal, &device).unwrap();

        assert!(dual.has_tangent());
        assert_eq!(dual.tangent().unwrap().to_vec::<f32>(), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_dual_tensor_into_parts() {
        let device = CpuDevice::new();
        let primal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let tangent = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 0.5], &[2], &device).unwrap();

        let dual = DualTensor::new(primal, Some(tangent));
        let (p, t) = dual.into_parts();

        assert_eq!(p.to_vec::<f32>(), [1.0, 2.0]);
        assert_eq!(t.unwrap().to_vec::<f32>(), [0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "Tangent shape")]
    fn test_dual_tensor_shape_mismatch() {
        let device = CpuDevice::new();
        let primal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let tangent = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();

        // Should panic because shapes don't match
        DualTensor::new(primal, Some(tangent));
    }
}
