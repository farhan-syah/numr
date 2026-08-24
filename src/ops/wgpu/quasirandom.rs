//! Quasi-random sequence generation for WebGPU runtime

use wgpu::util::DeviceExt;

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::common::quasirandom::{
    compute_all_direction_vectors, validate_halton_params, validate_latin_hypercube_params,
    validate_sobol_params,
};
use crate::ops::traits::QuasiRandomOps;
use crate::runtime::wgpu::ops::helpers::{
    HaltonParams, LatinHypercubeParams, SobolParams, alloc_output, create_params_buffer,
    generate_wgpu_seed, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::quasirandom;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Supported data types for WebGPU quasi-random operations
/// Note: WebGPU currently only supports F32 for quasi-random sequences.
/// F64 support requires shader-emulated f64 or native WebGPU f64 extension.
const SUPPORTED_DTYPES: &[DType] = &[DType::F32];

impl QuasiRandomOps<WgpuRuntime> for WgpuClient {
    fn sobol(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_sobol_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "sobol")?;

        // Allocate output: shape [n_points, dimension]
        let shape = vec![n_points, dimension];
        let out = alloc_output(self, &shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Compute direction vectors and upload to GPU
        let direction_vectors = compute_all_direction_vectors(dimension);
        let dv_buf = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sobol_direction_vectors"),
                contents: bytemuck::cast_slice(&direction_vectors),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create params
        let params = SobolParams {
            n_points: n_points as u32,
            dimension: dimension as u32,
            skip: skip as u32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch shader
        quasirandom::launch_sobol(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &dv_buf,
            &params_buf,
            n_points,
            dtype,
        )?;

        Ok(out)
    }

    fn halton(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_halton_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "halton")?;

        // Allocate output: shape [n_points, dimension]
        let shape = vec![n_points, dimension];
        let out = alloc_output(self, &shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params
        let params = HaltonParams {
            n_points: n_points as u32,
            dimension: dimension as u32,
            skip: skip as u32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch shader
        let total_elements = n_points * dimension;
        quasirandom::launch_halton(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            total_elements,
            dtype,
        )?;

        Ok(out)
    }

    fn latin_hypercube(
        &self,
        n_samples: usize,
        dimension: usize,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_latin_hypercube_params(
            n_samples,
            dimension,
            dtype,
            SUPPORTED_DTYPES,
            "latin_hypercube",
        )?;

        // Allocate output: shape [n_samples, dimension]
        let shape = vec![n_samples, dimension];
        let out = alloc_output(self, &shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Generate random seed
        let seed = generate_wgpu_seed();

        // Create params
        let params = LatinHypercubeParams {
            n_samples: n_samples as u32,
            dimension: dimension as u32,
            seed,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch shader
        let total_workgroups = dimension; // One workgroup per dimension
        quasirandom::launch_latin_hypercube(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            total_workgroups,
            dtype,
        )?;

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::wgpu::WgpuDevice;

    fn setup() -> (WgpuDevice, WgpuClient) {
        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_sobol_basic() {
        let (_device, client) = setup();

        let points = client.sobol(10, 2, 0, DType::F32).unwrap();
        assert_eq!(points.shape(), &[10, 2]);

        // Verify points are in [0, 1)
        let data: Vec<f32> = points.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_halton_basic() {
        let (_device, client) = setup();

        let points = client.halton(10, 3, 0, DType::F32).unwrap();
        assert_eq!(points.shape(), &[10, 3]);

        // Verify points are in [0, 1)
        let data: Vec<f32> = points.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_latin_hypercube_basic() {
        let (_device, client) = setup();

        let samples = client.latin_hypercube(20, 4, DType::F32).unwrap();
        assert_eq!(samples.shape(), &[20, 4]);

        // Verify samples are in [0, 1)
        let data: Vec<f32> = samples.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Sample out of range: {}", val);
        }
    }

    #[test]
    fn test_error_unsupported_dtype() {
        let (_device, client) = setup();
        let result = client.sobol(10, 2, 0, DType::F64);
        assert!(result.is_err());
    }

    #[test]
    fn test_sobol_dimension_limit() {
        let (_device, client) = setup();

        // Should work up to 21,201 dimensions (full Joe & Kuo dataset)
        let result = client.sobol(10, 100, 0, DType::F32);
        assert!(result.is_ok());

        let result = client.sobol(10, 1000, 0, DType::F32);
        assert!(result.is_ok());

        // Should fail beyond 21,201
        let result = client.sobol(10, 21202, 0, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_halton_dimension_limit() {
        let (_device, client) = setup();

        // Should work up to 100 dimensions
        let result = client.halton(10, 100, 0, DType::F32);
        assert!(result.is_ok());

        // Should fail beyond 100
        let result = client.halton(10, 101, 0, DType::F32);
        assert!(result.is_err());
    }
}
