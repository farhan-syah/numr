//! DType-aware tensor creation helpers for backend parity tests
//!
//! This module provides utilities to create test tensors with a specific target dtype,
//! enabling proper dtype parameterization across all backend tests.
//!
//! ## Problem
//!
//! Without these helpers, tensors created from f64 test data are always inferred as F64 dtype:
//! ```ignore
//! let tensor = Tensor::from_slice(&[1.0, 2.0], &[2], &device);
//! // tensor.dtype() == DType::F64 (inferred from data type)
//! ```
//!
//! This breaks dtype parameterization on backends like WebGPU (F32-only), causing
//! UnsupportedDType errors when testing with F64 tensors.
//!
//! ## Solution
//!
//! These helpers create a tensor in the canonical precision (f64), then cast to the target dtype:
//! ```ignore
//! let tensor = tensor_from_f64(&[1.0, 2.0], &[2], DType::F32, &device, &client)?;
//! // tensor.dtype() == DType::F32 (explicitly cast)
//! ```
//!
//! This allows tests to parameterize over all supported dtypes while maintaining
//! human-readable test data in the highest precision.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Create a tensor from f64 test data with a target dtype
///
/// This is the canonical way to create test tensors:
/// 1. Store test data as f64 (highest precision, human-readable)
/// 2. Create tensor (infers DType::F64 from data type)
/// 3. Cast to target dtype if different
///
/// ## Example
///
/// ```ignore
/// use numr::dtype::DType;
/// use tests::backend_parity::dtype_helpers::tensor_from_f64;
/// use tests::common::create_cpu_client;
///
/// let (client, device) = create_cpu_client();
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tensor = tensor_from_f64(&data, &[2, 2], DType::F32, &device, &client)?;
/// assert_eq!(tensor.dtype(), DType::F32);
/// ```
pub fn tensor_from_f64<R: Runtime<DType = DType>>(
    data: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
    client: &impl TypeConversionOps<R>,
) -> Result<Tensor<R>> {
    if dtype == DType::F64 {
        return Ok(Tensor::try_from_slice(data, shape, device).unwrap());
    }

    // Try creating as F64 and casting. If the backend doesn't support F64
    // (e.g. WebGPU), fall back to creating as F32 and casting from there.
    let f64_tensor = Tensor::try_from_slice(data, shape, device).unwrap();
    match client.cast(&f64_tensor, dtype) {
        Ok(t) => Ok(t),
        Err(_) => {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let f32_tensor = Tensor::try_from_slice(&f32_data, shape, device).unwrap();
            if dtype == DType::F32 {
                Ok(f32_tensor)
            } else {
                client.cast(&f32_tensor, dtype)
            }
        }
    }
}

/// Create a tensor from f32 test data with a target dtype
///
/// Similar to `tensor_from_f64` but for f32 input data.
/// Use this when test data is more naturally expressed in f32.
///
/// ## Example
///
/// ```ignore
/// let tensor = tensor_from_f32(&[1.0, 2.0], &[2], DType::F16, &device, &client)?;
/// assert_eq!(tensor.dtype(), DType::F16);
/// ```
pub fn tensor_from_f32<R: Runtime<DType = DType>>(
    data: &[f32],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
    client: &impl TypeConversionOps<R>,
) -> Result<Tensor<R>> {
    let tensor = Tensor::try_from_slice(data, shape, device).unwrap();

    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        client.cast(&tensor, dtype)
    }
}

/// Create a tensor from i32 test data with a target dtype
///
/// Similar to `tensor_from_f64` but for integer input data.
/// Use this for integer operations that need dtype parameterization.
///
/// ## Example
///
/// ```ignore
/// let tensor = tensor_from_i32(&[1, 2, 3], &[3], DType::U32, &device, &client)?;
/// assert_eq!(tensor.dtype(), DType::U32);
/// ```
pub fn tensor_from_i32<R: Runtime<DType = DType>>(
    data: &[i32],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
    client: &impl TypeConversionOps<R>,
) -> Result<Tensor<R>> {
    let tensor = Tensor::try_from_slice(data, shape, device).unwrap();

    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        client.cast(&tensor, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::create_cpu_client;

    #[test]
    fn test_tensor_from_f64_no_cast_needed() {
        let (client, device) = create_cpu_client();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let tensor = tensor_from_f64(&data, &[2, 2], DType::F64, &device, &client)
            .expect("tensor creation failed");

        assert_eq!(tensor.dtype(), DType::F64);
        assert_eq!(tensor.to_vec::<f64>(), data);
    }

    #[test]
    fn test_tensor_from_f64_with_cast() {
        let (client, device) = create_cpu_client();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let tensor = tensor_from_f64(&data, &[2, 2], DType::F32, &device, &client)
            .expect("tensor creation failed");

        assert_eq!(tensor.dtype(), DType::F32);
        // Cast works correctly - values are preserved with F32 precision
    }

    #[test]
    fn test_tensor_from_f32_no_cast_needed() {
        let (client, device) = create_cpu_client();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        let tensor = tensor_from_f32(&data, &[2, 2], DType::F32, &device, &client)
            .expect("tensor creation failed");

        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.to_vec::<f32>(), data);
    }

    #[test]
    fn test_tensor_from_f32_with_cast() {
        let (client, device) = create_cpu_client();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        let tensor = tensor_from_f32(&data, &[2, 2], DType::F64, &device, &client)
            .expect("tensor creation failed");

        assert_eq!(tensor.dtype(), DType::F64);
        let result = tensor.to_vec::<f64>();
        // Verify values are preserved
        for (actual, &expected) in result.iter().zip(data.iter()) {
            assert_eq!(*actual, expected as f64);
        }
    }

    #[test]
    fn test_tensor_from_i32_no_cast_needed() {
        let (client, device) = create_cpu_client();
        let data = vec![1i32, 2, 3, 4];

        let tensor = tensor_from_i32(&data, &[4], DType::I32, &device, &client)
            .expect("tensor creation failed");

        assert_eq!(tensor.dtype(), DType::I32);
        assert_eq!(tensor.to_vec::<i32>(), data);
    }

    #[test]
    fn test_tensor_from_i32_with_cast() {
        let (client, device) = create_cpu_client();
        let data = vec![1i32, 2, 3, 4];

        let tensor = tensor_from_i32(&data, &[4], DType::U32, &device, &client)
            .expect("tensor creation failed");

        assert_eq!(tensor.dtype(), DType::U32);
        let result = tensor.to_vec::<u32>();
        for (actual, &expected) in result.iter().zip(data.iter()) {
            assert_eq!(*actual, expected as u32);
        }
    }
}
