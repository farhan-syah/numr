//! CPU implementation of distance operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
#[cfg(feature = "fp8")]
use crate::ops::TypeConversionOps;
use crate::ops::distance_common::*;
use crate::ops::{DistanceMetric, DistanceOps};
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::ensure_contiguous, kernels};
use crate::tensor::Tensor;

/// Dispatch to distance kernel for float types only
macro_rules! dispatch_float_dtype {
    ($dtype:expr, $T:ident => $body:block, $op:expr) => {
        match $dtype {
            DType::F32 => {
                type $T = f32;
                $body
            }
            DType::F64 => {
                type $T = f64;
                $body
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                type $T = half::f16;
                $body
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $op,
                })
            }
        }
    };
}

impl DistanceOps<CpuRuntime> for CpuClient {
    fn cdist(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CpuRuntime>> {
        let x_shape = x.shape();
        let y_shape = y.shape();

        // Validate inputs using shared validators
        validate_2d_tensor(x_shape, "x", "cdist")?;
        validate_2d_tensor(y_shape, "y", "cdist")?;
        validate_same_dimension(x_shape, y_shape, "cdist")?;

        let dtype = x.dtype();
        validate_float_dtype(dtype, "cdist")?;
        validate_same_dtype(dtype, y.dtype(), "cdist")?;

        let n = x_shape[0];
        let m = y_shape[0];
        let d = x_shape[1];

        // Handle empty tensors
        if n == 0 || m == 0 {
            return Tensor::<CpuRuntime>::try_empty(&[n, m], dtype, &self.device);
        }

        // Ensure contiguous
        let x = ensure_contiguous(x)?;
        let y = ensure_contiguous(y)?;

        let out = Tensor::<CpuRuntime>::try_empty(&[n, m], dtype, &self.device)?;
        let x_ptr = x.ptr();
        let y_ptr = y.ptr();
        let out_ptr = out.ptr();

        // FP8 types: compute in F32, then cast result back
        #[cfg(feature = "fp8")]
        if dtype == DType::FP8E4M3 || dtype == DType::FP8E5M2 {
            let x_f32 = self.cast(&x, DType::F32)?;
            let y_f32 = self.cast(&y, DType::F32)?;
            let out_f32 = Tensor::<CpuRuntime>::try_empty(&[n, m], DType::F32, &self.device)?;
            unsafe {
                kernels::cdist_kernel::<f32>(
                    x_f32.ptr() as *const f32,
                    y_f32.ptr() as *const f32,
                    out_f32.ptr() as *mut f32,
                    n,
                    m,
                    d,
                    metric,
                );
            }
            return self.cast(&out_f32, dtype);
        }

        dispatch_float_dtype!(dtype, T => {
            unsafe {
                kernels::cdist_kernel::<T>(
                    x_ptr as *const T,
                    y_ptr as *const T,
                    out_ptr as *mut T,
                    n, m, d,
                    metric,
                );
            }
        }, "cdist");

        Ok(out)
    }

    fn pdist(&self, x: &Tensor<CpuRuntime>, metric: DistanceMetric) -> Result<Tensor<CpuRuntime>> {
        let x_shape = x.shape();

        // Validate inputs using shared validators
        validate_2d_tensor(x_shape, "x", "pdist")?;

        let n = x_shape[0];
        let d = x_shape[1];

        validate_min_points(n, 2, "x", "pdist")?;

        let dtype = x.dtype();
        validate_float_dtype(dtype, "pdist")?;

        // Output size: n*(n-1)/2
        let out_size = n * (n - 1) / 2;

        // Ensure contiguous
        let x = ensure_contiguous(x)?;

        let out = Tensor::<CpuRuntime>::try_empty(&[out_size], dtype, &self.device)?;
        let x_ptr = x.ptr();
        let out_ptr = out.ptr();

        // FP8 types: compute in F32, then cast result back
        #[cfg(feature = "fp8")]
        if dtype == DType::FP8E4M3 || dtype == DType::FP8E5M2 {
            let x_f32 = self.cast(&x, DType::F32)?;
            let out_f32 = Tensor::<CpuRuntime>::try_empty(&[out_size], DType::F32, &self.device)?;
            unsafe {
                kernels::pdist_kernel::<f32>(
                    x_f32.ptr() as *const f32,
                    out_f32.ptr() as *mut f32,
                    n,
                    d,
                    metric,
                );
            }
            return self.cast(&out_f32, dtype);
        }

        dispatch_float_dtype!(dtype, T => {
            unsafe {
                kernels::pdist_kernel::<T>(
                    x_ptr as *const T,
                    out_ptr as *mut T,
                    n, d,
                    metric,
                );
            }
        }, "pdist");

        Ok(out)
    }

    fn squareform(&self, condensed: &Tensor<CpuRuntime>, n: usize) -> Result<Tensor<CpuRuntime>> {
        let cond_shape = condensed.shape();

        // Validate inputs using shared validators
        validate_1d_tensor(cond_shape, "condensed", "squareform")?;
        validate_condensed_length(cond_shape[0], n, "condensed", "squareform")?;

        let dtype = condensed.dtype();
        validate_float_dtype(dtype, "squareform")?;

        // Handle edge case
        if n == 0 {
            return Tensor::<CpuRuntime>::try_empty(&[0, 0], dtype, &self.device);
        }
        if n == 1 {
            return Tensor::<CpuRuntime>::try_zeros(&[1, 1], dtype, &self.device);
        }

        // Ensure contiguous
        let condensed = ensure_contiguous(condensed)?;

        let out = Tensor::<CpuRuntime>::try_empty(&[n, n], dtype, &self.device)?;
        let cond_ptr = condensed.ptr();
        let out_ptr = out.ptr();

        dispatch_float_dtype!(dtype, T => {
            unsafe {
                kernels::squareform_kernel::<T>(
                    cond_ptr as *const T,
                    out_ptr as *mut T,
                    n,
                );
            }
        }, "squareform");

        Ok(out)
    }

    fn squareform_inverse(&self, square: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let sq_shape = square.shape();

        // Validate inputs using shared validators
        validate_2d_tensor(sq_shape, "square", "squareform_inverse")?;
        validate_square_matrix(sq_shape, "square", "squareform_inverse")?;

        let n = sq_shape[0];
        let dtype = square.dtype();
        validate_float_dtype(dtype, "squareform_inverse")?;

        // Handle edge cases
        if n == 0 {
            return Tensor::<CpuRuntime>::try_empty(&[0], dtype, &self.device);
        }
        if n == 1 {
            return Tensor::<CpuRuntime>::try_empty(&[0], dtype, &self.device);
        }

        // Ensure contiguous
        let square = ensure_contiguous(square)?;

        let out_size = n * (n - 1) / 2;
        let out = Tensor::<CpuRuntime>::try_empty(&[out_size], dtype, &self.device)?;
        let sq_ptr = square.ptr();
        let out_ptr = out.ptr();

        dispatch_float_dtype!(dtype, T => {
            unsafe {
                kernels::squareform_inverse_kernel::<T>(
                    sq_ptr as *const T,
                    out_ptr as *mut T,
                    n,
                );
            }
        }, "squareform_inverse");

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_cdist_euclidean() {
        let (device, client) = setup();

        // X = [[0, 0], [1, 1]]
        // Y = [[1, 0], [2, 2]]
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[2, 2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 2.0, 2.0], &[2, 2], &device);

        let dist = client.cdist(&x, &y, DistanceMetric::Euclidean).unwrap();
        assert_eq!(dist.shape(), &[2, 2]);

        let data: Vec<f32> = dist.to_vec();
        // d(x0, y0) = 1.0
        // d(x0, y1) = sqrt(8) = 2.828...
        // d(x1, y0) = 1.0
        // d(x1, y1) = sqrt(2) = 1.414...
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 8.0f32.sqrt()).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[3] - 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_pdist_euclidean() {
        let (device, client) = setup();

        // X = [[0, 0], [1, 0], [0, 1]] - 3 points
        let x =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device);

        let dist = client.pdist(&x, DistanceMetric::Euclidean).unwrap();
        assert_eq!(dist.shape(), &[3]); // n*(n-1)/2 = 3

        let data: Vec<f32> = dist.to_vec();
        // d(0,1) = 1.0, d(0,2) = 1.0, d(1,2) = sqrt(2)
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
        assert!((data[2] - 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_squareform_roundtrip() {
        let (device, client) = setup();

        // Create condensed form: [d(0,1), d(0,2), d(1,2)]
        let condensed = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        // Convert to square
        let square = client.squareform(&condensed, 3).unwrap();
        assert_eq!(square.shape(), &[3, 3]);

        // Convert back to condensed
        let condensed2 = client.squareform_inverse(&square).unwrap();
        assert_eq!(condensed2.shape(), &[3]);

        let data: Vec<f32> = condensed2.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cdist_cosine() {
        let (device, client) = setup();

        // Same direction vectors have cosine distance 0
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[1, 2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.0], &[1, 2], &device);

        let dist = client.cdist(&x, &y, DistanceMetric::Cosine).unwrap();
        let data: Vec<f32> = dist.to_vec();
        assert!(data[0].abs() < 1e-5);

        // Orthogonal vectors have cosine distance 1
        let y2 = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[1, 2], &device);
        let dist2 = client.cdist(&x, &y2, DistanceMetric::Cosine).unwrap();
        let data2: Vec<f32> = dist2.to_vec();
        assert!((data2[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cdist_manhattan() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[1, 3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);

        let dist = client.cdist(&x, &y, DistanceMetric::Manhattan).unwrap();
        let data: Vec<f32> = dist.to_vec();
        assert!((data[0] - 6.0).abs() < 1e-5); // |1| + |2| + |3| = 6
    }

    #[test]
    fn test_cdist_chebyshev() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[1, 3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0], &[1, 3], &device);

        let dist = client.cdist(&x, &y, DistanceMetric::Chebyshev).unwrap();
        let data: Vec<f32> = dist.to_vec();
        assert!((data[0] - 5.0).abs() < 1e-5); // max(|1|, |5|, |3|) = 5
    }

    #[test]
    fn test_error_on_non_2d() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        let result = client.cdist(&x, &y, DistanceMetric::Euclidean);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_on_dimension_mismatch() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device);

        let result = client.cdist(&x, &y, DistanceMetric::Euclidean);
        assert!(result.is_err());
    }

    #[test]
    fn test_pdist_requires_at_least_2_points() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device);

        let result = client.pdist(&x, DistanceMetric::Euclidean);
        assert!(result.is_err());
    }
}
