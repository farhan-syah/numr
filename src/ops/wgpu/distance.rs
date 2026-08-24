//! WebGPU implementation of distance operations.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::distance_common::*;
use crate::ops::{DistanceMetric, DistanceOps};
use crate::runtime::wgpu::ops::helpers::get_tensor_buffer;
use crate::runtime::wgpu::shaders::{
    distance_metric_p_value, distance_metric_to_index, launch_cdist, launch_pdist,
    launch_squareform, launch_squareform_inverse,
};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Parameters for cdist kernel
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CdistParams {
    n: u32,
    m: u32,
    d: u32,
    metric: u32,
    p: f32,
}

/// Parameters for pdist kernel
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PdistParams {
    n: u32,
    d: u32,
    metric: u32,
    p: f32,
}

/// Parameters for squareform kernels
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SquareformParams {
    n: u32,
}

impl DistanceOps<WgpuRuntime> for WgpuClient {
    fn cdist(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        let x_shape = x.shape();
        let y_shape = y.shape();

        // Validate inputs using shared validators
        validate_2d_tensor(x_shape, "x", "cdist")?;
        validate_2d_tensor(y_shape, "y", "cdist")?;
        validate_same_dimension(x_shape, y_shape, "cdist")?;

        let dtype = x.dtype();
        // WebGPU backend currently only supports F32
        validate_dtype_supported(dtype, &[DType::F32], "cdist")?;
        validate_same_dtype(dtype, y.dtype(), "cdist")?;

        let n = x_shape[0];
        let m = y_shape[0];
        let d = x_shape[1];

        // Handle empty tensors
        if n == 0 || m == 0 {
            return Ok(Tensor::<WgpuRuntime>::empty(
                &[n, m],
                dtype,
                &self.device_id,
            )?);
        }

        // Ensure contiguous
        let x = x.contiguous()?;
        let y = y.contiguous()?;

        let out = Tensor::<WgpuRuntime>::empty(&[n, m], dtype, &self.device_id)?;

        // Create params buffer
        let params = CdistParams {
            n: n as u32,
            m: m as u32,
            d: d as u32,
            metric: distance_metric_to_index(metric),
            p: distance_metric_p_value(metric),
        };
        let params_buffer =
            self.wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cdist_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let x_buf = get_tensor_buffer(&x)?;
        let y_buf = get_tensor_buffer(&y)?;
        let out_buf = get_tensor_buffer(&out)?;

        launch_cdist(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &x_buf,
            &y_buf,
            &out_buf,
            &params_buffer,
            n * m,
            dtype,
        )?;

        Ok(out)
    }

    fn pdist(
        &self,
        x: &Tensor<WgpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        let x_shape = x.shape();

        // Validate inputs using shared validators
        validate_2d_tensor(x_shape, "x", "pdist")?;

        let n = x_shape[0];
        let d = x_shape[1];

        validate_min_points(n, 2, "x", "pdist")?;

        let dtype = x.dtype();
        // WebGPU backend currently only supports F32
        validate_dtype_supported(dtype, &[DType::F32], "pdist")?;

        // Output size: n*(n-1)/2
        let out_size = n * (n - 1) / 2;

        // Ensure contiguous
        let x = x.contiguous()?;

        let out = Tensor::<WgpuRuntime>::empty(&[out_size], dtype, &self.device_id)?;

        // Create params buffer
        let params = PdistParams {
            n: n as u32,
            d: d as u32,
            metric: distance_metric_to_index(metric),
            p: distance_metric_p_value(metric),
        };
        let params_buffer =
            self.wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pdist_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let x_buf = get_tensor_buffer(&x)?;
        let out_buf = get_tensor_buffer(&out)?;

        launch_pdist(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &x_buf,
            &out_buf,
            &params_buffer,
            out_size,
            dtype,
        )?;

        Ok(out)
    }

    fn squareform(&self, condensed: &Tensor<WgpuRuntime>, n: usize) -> Result<Tensor<WgpuRuntime>> {
        let cond_shape = condensed.shape();

        // Validate inputs using shared validators
        validate_1d_tensor(cond_shape, "condensed", "squareform")?;
        validate_condensed_length(cond_shape[0], n, "condensed", "squareform")?;

        let dtype = condensed.dtype();
        // WebGPU backend currently only supports F32
        validate_dtype_supported(dtype, &[DType::F32], "squareform")?;

        // Handle edge cases
        if n == 0 {
            return Ok(Tensor::<WgpuRuntime>::empty(
                &[0, 0],
                dtype,
                &self.device_id,
            )?);
        }
        if n == 1 {
            return Ok(Tensor::<WgpuRuntime>::zeros(
                &[1, 1],
                dtype,
                &self.device_id,
            )?);
        }

        // Ensure contiguous
        let condensed = condensed.contiguous()?;

        let out = Tensor::<WgpuRuntime>::empty(&[n, n], dtype, &self.device_id)?;

        // Create params buffer
        let params = SquareformParams { n: n as u32 };
        let params_buffer =
            self.wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("squareform_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let cond_buf = get_tensor_buffer(&condensed)?;
        let out_buf = get_tensor_buffer(&out)?;

        launch_squareform(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &cond_buf,
            &out_buf,
            &params_buffer,
            n * n,
            dtype,
        )?;

        Ok(out)
    }

    fn squareform_inverse(&self, square: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let sq_shape = square.shape();

        // Validate inputs using shared validators
        validate_2d_tensor(sq_shape, "square", "squareform_inverse")?;
        validate_square_matrix(sq_shape, "square", "squareform_inverse")?;

        let n = sq_shape[0];
        let dtype = square.dtype();
        // WebGPU backend currently only supports F32
        validate_dtype_supported(dtype, &[DType::F32], "squareform_inverse")?;

        // Handle edge cases
        if n == 0 || n == 1 {
            return Ok(Tensor::<WgpuRuntime>::empty(&[0], dtype, &self.device_id)?);
        }

        // Ensure contiguous
        let square = square.contiguous()?;

        let out_size = n * (n - 1) / 2;
        let out = Tensor::<WgpuRuntime>::empty(&[out_size], dtype, &self.device_id)?;

        // Create params buffer
        let params = SquareformParams { n: n as u32 };
        let params_buffer =
            self.wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("squareform_inverse_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let sq_buf = get_tensor_buffer(&square)?;
        let out_buf = get_tensor_buffer(&out)?;

        launch_squareform_inverse(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &sq_buf,
            &out_buf,
            &params_buffer,
            out_size,
            dtype,
        )?;

        Ok(out)
    }
}
