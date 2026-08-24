//! Advanced PRNG operations for WebGPU runtime
//!
//! Implements Philox4x32-10, ThreeFry4x32-20, PCG64, and Xoshiro256++.
//! Only F32 dtype supported (WGSL has no native f64).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::AdvancedRandomOps;
use crate::runtime::RuntimeClient;
use crate::runtime::wgpu::ops::helpers::{alloc_output, create_params_buffer, get_tensor_buffer};
use crate::runtime::wgpu::shaders::advanced_random;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

// ============================================================================
// PRNG Parameter Structs (WGSL uses u32 pairs for 64-bit values)
// ============================================================================

/// Params for Philox4x32-10 and ThreeFry4x32-20 (counter-based PRNGs)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CounterPrngParams {
    numel: u32,
    key_lo: u32,
    key_hi: u32,
    counter_lo: u32,
    counter_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Params for PCG64 PRNG
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Pcg64Params {
    numel: u32,
    seed_lo: u32,
    seed_hi: u32,
    stream_lo: u32,
    stream_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Params for Xoshiro256++ PRNG
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Xoshiro256Params {
    numel: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn check_f32_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if !matches!(dtype, DType::F32) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

fn split_u64(value: u64) -> (u32, u32) {
    ((value & 0xFFFFFFFF) as u32, (value >> 32) as u32)
}

// ============================================================================
// Trait Implementation
// ============================================================================

impl AdvancedRandomOps<WgpuRuntime> for WgpuClient {
    fn philox_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "philox_randn (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (key_lo, key_hi) = split_u64(key);
        let (counter_lo, counter_hi) = split_u64(counter);
        let params = CounterPrngParams {
            numel: numel as u32,
            key_lo,
            key_hi,
            counter_lo,
            counter_hi,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_philox_randn(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn philox_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "philox_uniform (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (key_lo, key_hi) = split_u64(key);
        let (counter_lo, counter_hi) = split_u64(counter);
        let params = CounterPrngParams {
            numel: numel as u32,
            key_lo,
            key_hi,
            counter_lo,
            counter_hi,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_philox_uniform(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn threefry_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "threefry_randn (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (key_lo, key_hi) = split_u64(key);
        let (counter_lo, counter_hi) = split_u64(counter);
        let params = CounterPrngParams {
            numel: numel as u32,
            key_lo,
            key_hi,
            counter_lo,
            counter_hi,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_threefry_randn(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn threefry_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "threefry_uniform (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (key_lo, key_hi) = split_u64(key);
        let (counter_lo, counter_hi) = split_u64(counter);
        let params = CounterPrngParams {
            numel: numel as u32,
            key_lo,
            key_hi,
            counter_lo,
            counter_hi,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_threefry_uniform(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn pcg64_randn(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "pcg64_randn (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (seed_lo, seed_hi) = split_u64(seed);
        let (stream_lo, stream_hi) = split_u64(stream);
        let params = Pcg64Params {
            numel: numel as u32,
            seed_lo,
            seed_hi,
            stream_lo,
            stream_hi,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_pcg64_randn(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn pcg64_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "pcg64_uniform (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (seed_lo, seed_hi) = split_u64(seed);
        let (stream_lo, stream_hi) = split_u64(stream);
        let params = Pcg64Params {
            numel: numel as u32,
            seed_lo,
            seed_hi,
            stream_lo,
            stream_hi,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_pcg64_uniform(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn xoshiro256_randn(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "xoshiro256_randn (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (seed_lo, seed_hi) = split_u64(seed);
        let params = Xoshiro256Params {
            numel: numel as u32,
            seed_lo,
            seed_hi,
            _pad0: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_xoshiro256_randn(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn xoshiro256_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        check_f32_dtype(dtype, "xoshiro256_uniform (WebGPU: F32 only)")?;

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        let (seed_lo, seed_hi) = split_u64(seed);
        let params = Xoshiro256Params {
            numel: numel as u32,
            seed_lo,
            seed_hi,
            _pad0: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        advanced_random::launch_xoshiro256_uniform(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }
}
