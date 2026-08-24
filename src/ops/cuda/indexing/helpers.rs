//! Indexing operation helpers for CUDA runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::ops::broadcast_shape;
use crate::runtime::cuda::kernels::compute_broadcast_strides;
use crate::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use crate::tensor::Tensor;

/// Validates that the mask tensor has dtype U8.
pub fn validate_mask_dtype(mask: &Tensor<CudaRuntime>) -> Result<()> {
    if mask.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: mask.dtype(),
        });
    }
    Ok(())
}

/// Normalize index tensor to I64 for CUDA indexing kernels.
///
/// CUDA indexing kernels consume i64 indices. This helper accepts both I32 and I64
/// public API inputs and casts to I64 on device when needed.
pub fn normalize_indices_to_i64(
    client: &CudaClient,
    indices: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    match indices.dtype() {
        DType::I64 => Ok(indices.clone()),
        DType::I32 => client.cast(indices, DType::I64),
        other => Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: other,
        }),
    }
}

/// Context for broadcast masked operations.
/// Holds the GPU tensors needed for broadcast stride computation.
pub struct BroadcastContext {
    /// Whether broadcasting is needed
    pub needs_broadcast: bool,
    /// Broadcast strides tensor (on GPU), None if no broadcast
    pub strides_tensor: Option<Tensor<CudaRuntime>>,
    /// Output shape tensor (on GPU), None if no broadcast
    pub shape_tensor: Option<Tensor<CudaRuntime>>,
    /// Number of dimensions
    pub ndim: usize,
}

impl BroadcastContext {
    /// Prepare broadcast context for a masked operation.
    /// Validates broadcast compatibility and allocates GPU tensors if needed.
    pub fn prepare(
        a: &Tensor<CudaRuntime>,
        mask: &Tensor<CudaRuntime>,
        device: &CudaDevice,
    ) -> Result<Self> {
        let needs_broadcast = a.shape() != mask.shape();

        if !needs_broadcast {
            return Ok(Self {
                needs_broadcast: false,
                strides_tensor: None,
                shape_tensor: None,
                ndim: a.shape().len(),
            });
        }

        // Validate broadcast compatibility - mask must broadcast to a's shape
        let broadcast_result = broadcast_shape(a.shape(), mask.shape());
        match broadcast_result {
            Some(ref bcast_shape) if bcast_shape == a.shape() => {
                // Mask broadcasts to a's shape - OK
            }
            _ => {
                return Err(Error::BroadcastError {
                    lhs: a.shape().to_vec(),
                    rhs: mask.shape().to_vec(),
                });
            }
        }

        // Compute broadcast strides for mask
        let mask_strides = compute_broadcast_strides(mask.shape(), a.shape());
        let out_shape_u32: Vec<u32> = a.shape().iter().map(|&x| x as u32).collect();
        let ndim = a.shape().len();

        // Allocate device memory for strides and shape
        let strides_tensor = Tensor::<CudaRuntime>::try_from_slice(&mask_strides, &[ndim], device)?;
        let shape_tensor = Tensor::<CudaRuntime>::try_from_slice(&out_shape_u32, &[ndim], device)?;

        Ok(Self {
            needs_broadcast: true,
            strides_tensor: Some(strides_tensor),
            shape_tensor: Some(shape_tensor),
            ndim,
        })
    }

    /// Get strides pointer.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `needs_broadcast` is true. In release builds,
    /// returns 0 if called incorrectly (safe but meaningless).
    #[inline]
    pub fn strides_ptr(&self) -> u64 {
        debug_assert!(
            self.needs_broadcast,
            "strides_ptr() called on non-broadcast context"
        );
        self.strides_tensor.as_ref().map(|t| t.ptr()).unwrap_or(0)
    }

    /// Get shape pointer.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `needs_broadcast` is true. In release builds,
    /// returns 0 if called incorrectly (safe but meaningless).
    #[inline]
    pub fn shape_ptr(&self) -> u64 {
        debug_assert!(
            self.needs_broadcast,
            "shape_ptr() called on non-broadcast context"
        );
        self.shape_tensor.as_ref().map(|t| t.ptr()).unwrap_or(0)
    }
}
