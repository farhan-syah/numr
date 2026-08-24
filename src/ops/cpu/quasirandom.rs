//! CPU implementation of quasi-random sequence operations.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::QuasiRandomOps;
use crate::ops::common::quasirandom::{
    validate_halton_params, validate_latin_hypercube_params, validate_sobol_params,
};
use crate::runtime::cpu::{CpuClient, CpuRuntime, kernels};
use crate::tensor::Tensor;

/// Supported data types for CPU quasi-random operations
const SUPPORTED_DTYPES: &[DType] = &[DType::F32, DType::F64];

impl QuasiRandomOps<CpuRuntime> for CpuClient {
    fn sobol(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_sobol_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "sobol")?;

        let out = Tensor::<CpuRuntime>::try_empty(&[n_points, dimension], dtype, &self.device)?;

        match dtype {
            DType::F32 => unsafe {
                kernels::sobol_f32(out.ptr() as *mut f32, n_points, dimension, skip);
            },
            DType::F64 => unsafe {
                kernels::sobol_f64(out.ptr() as *mut f64, n_points, dimension, skip);
            },
            _ => unreachable!("dtype validation should prevent this"),
        }

        Ok(out)
    }

    fn halton(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_halton_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "halton")?;

        let out = Tensor::<CpuRuntime>::try_empty(&[n_points, dimension], dtype, &self.device)?;

        match dtype {
            DType::F32 => unsafe {
                kernels::halton_f32(out.ptr() as *mut f32, n_points, dimension, skip);
            },
            DType::F64 => unsafe {
                kernels::halton_f64(out.ptr() as *mut f64, n_points, dimension, skip);
            },
            _ => unreachable!("dtype validation should prevent this"),
        }

        Ok(out)
    }

    fn latin_hypercube(
        &self,
        n_samples: usize,
        dimension: usize,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_latin_hypercube_params(
            n_samples,
            dimension,
            dtype,
            SUPPORTED_DTYPES,
            "latin_hypercube",
        )?;

        let out = Tensor::<CpuRuntime>::try_empty(&[n_samples, dimension], dtype, &self.device)?;

        match dtype {
            DType::F32 => unsafe {
                kernels::latin_hypercube_f32(out.ptr() as *mut f32, n_samples, dimension);
            },
            DType::F64 => unsafe {
                kernels::latin_hypercube_f64(out.ptr() as *mut f64, n_samples, dimension);
            },
            _ => unreachable!("dtype validation should prevent this"),
        }

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
    fn test_sobol_basic() {
        let (_device, client) = setup();

        let points = client.sobol(10, 2, 0, DType::F64).unwrap();
        assert_eq!(points.shape(), &[10, 2]);

        let data: Vec<f64> = points.to_vec();
        // All points should be in [0, 1)
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_halton_basic() {
        let (_device, client) = setup();

        let points = client.halton(10, 3, 0, DType::F64).unwrap();
        assert_eq!(points.shape(), &[10, 3]);

        let data: Vec<f64> = points.to_vec();
        // All points should be in [0, 1)
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_latin_hypercube_basic() {
        let (_device, client) = setup();

        let samples = client.latin_hypercube(20, 4, DType::F64).unwrap();
        assert_eq!(samples.shape(), &[20, 4]);

        let data: Vec<f64> = samples.to_vec();
        // All samples should be in [0, 1)
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Sample out of range: {}", val);
        }
    }

    #[test]
    fn test_sobol_deterministic() {
        let (_device, client) = setup();

        let points1 = client.sobol(5, 2, 0, DType::F64).unwrap();
        let points2 = client.sobol(5, 2, 0, DType::F64).unwrap();

        let data1: Vec<f64> = points1.to_vec();
        let data2: Vec<f64> = points2.to_vec();

        // Same parameters should produce identical sequences
        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert_eq!(v1, v2, "Sobol sequence not deterministic");
        }
    }

    #[test]
    fn test_halton_deterministic() {
        let (_device, client) = setup();

        let points1 = client.halton(5, 2, 0, DType::F64).unwrap();
        let points2 = client.halton(5, 2, 0, DType::F64).unwrap();

        let data1: Vec<f64> = points1.to_vec();
        let data2: Vec<f64> = points2.to_vec();

        // Same parameters should produce identical sequences
        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert_eq!(v1, v2, "Halton sequence not deterministic");
        }
    }

    #[test]
    fn test_error_zero_points() {
        let (_device, client) = setup();

        let result = client.sobol(0, 2, 0, DType::F64);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_zero_dimension() {
        let (_device, client) = setup();

        let result = client.sobol(10, 0, 0, DType::F64);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unsupported_dtype() {
        let (_device, client) = setup();

        let result = client.sobol(10, 2, 0, DType::I32);
        assert!(result.is_err());
    }

    #[test]
    fn test_sobol_dimension_limit() {
        let (_device, client) = setup();

        // Should work up to 21,201 dimensions (full Joe & Kuo dataset)
        let result = client.sobol(10, 100, 0, DType::F64);
        assert!(result.is_ok());

        let result = client.sobol(10, 1000, 0, DType::F64);
        assert!(result.is_ok());

        // Should fail beyond 21,201
        let result = client.sobol(10, 21202, 0, DType::F64);
        assert!(result.is_err());
    }

    #[test]
    fn test_halton_dimension_limit() {
        let (_device, client) = setup();

        // Should work up to 100 dimensions
        let result = client.halton(10, 100, 0, DType::F64);
        assert!(result.is_ok());

        // Should fail beyond 100
        let result = client.halton(10, 101, 0, DType::F64);
        assert!(result.is_err());
    }
}
