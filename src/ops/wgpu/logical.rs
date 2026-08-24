//! LogicalOps implementation for WebGPU runtime.
//!
//! WebGPU does not support U8, so logical operations use U32 tensors
//! where 0 = false and non-zero = true.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::LogicalOps;
use crate::runtime::wgpu::ops::native::logical::{
    native_logical_and, native_logical_not, native_logical_or, native_logical_xor,
};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl LogicalOps<WgpuRuntime> for WgpuClient {
    fn logical_and(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU uses U32 for boolean tensors (no U8 support)
        validate_logical_inputs(a, b)?;
        native_logical_and(self, a, b)
    }

    fn logical_or(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_logical_inputs(a, b)?;
        native_logical_or(self, a, b)
    }

    fn logical_xor(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_logical_inputs(a, b)?;
        native_logical_xor(self, a, b)
    }

    fn logical_not(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU uses U32 for boolean tensors (no U8 support)
        if a.dtype() != DType::U32 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U32,
                rhs: a.dtype(),
            });
        }
        native_logical_not(self, a)
    }
}

/// Validate inputs for binary logical operations.
/// WebGPU requires U32 dtype (no U8 support).
fn validate_logical_inputs(a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<()> {
    // WebGPU uses U32 for boolean tensors
    if a.dtype() != DType::U32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U32,
            rhs: a.dtype(),
        });
    }
    if b.dtype() != DType::U32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U32,
            rhs: b.dtype(),
        });
    }

    // Shapes must match
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::wgpu::device::WgpuDevice;
    use crate::runtime::wgpu::runtime::WgpuRuntime as WgpuRt;
    use crate::runtime::{Runtime, RuntimeClient};

    fn create_client() -> WgpuClient {
        let device = WgpuDevice::new(0);
        WgpuRt::default_client(&device)
    }

    #[test]
    fn test_logical_and() {
        let client = create_client();

        let a =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 0, 1, 0], &[4], client.device()).unwrap();
        let b =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 1, 0, 0], &[4], client.device()).unwrap();

        let result = client.logical_and(&a, &b).unwrap();
        let data: Vec<u32> = result.to_vec();

        assert_eq!(data, vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_logical_or() {
        let client = create_client();

        let a =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 0, 1, 0], &[4], client.device()).unwrap();
        let b =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 1, 0, 0], &[4], client.device()).unwrap();

        let result = client.logical_or(&a, &b).unwrap();
        let data: Vec<u32> = result.to_vec();

        assert_eq!(data, vec![1, 1, 1, 0]);
    }

    #[test]
    fn test_logical_xor() {
        let client = create_client();

        let a =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 0, 1, 0], &[4], client.device()).unwrap();
        let b =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 1, 0, 0], &[4], client.device()).unwrap();

        let result = client.logical_xor(&a, &b).unwrap();
        let data: Vec<u32> = result.to_vec();

        assert_eq!(data, vec![0, 1, 1, 0]);
    }

    #[test]
    fn test_logical_not() {
        let client = create_client();

        let a =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 0, 5, 0], &[4], client.device()).unwrap();

        let result = client.logical_not(&a).unwrap();
        let data: Vec<u32> = result.to_vec();

        assert_eq!(data, vec![0, 1, 0, 1]);
    }

    #[test]
    fn test_logical_rejects_non_u32() {
        let client = create_client();

        let a =
            Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 0.0], &[2], client.device()).unwrap();
        let b =
            Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], client.device()).unwrap();

        assert!(client.logical_and(&a, &b).is_err());
        assert!(client.logical_not(&a).is_err());
    }

    #[test]
    fn test_logical_shape_mismatch() {
        let client = create_client();

        let a =
            Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 0, 1], &[3], client.device()).unwrap();
        let b = Tensor::<WgpuRuntime>::try_from_slice(&[1u32, 1], &[2], client.device()).unwrap();

        assert!(client.logical_and(&a, &b).is_err());
    }
}
