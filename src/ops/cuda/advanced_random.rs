//! CUDA implementation of advanced PRNG operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::AdvancedRandomOps;
use crate::runtime::cuda::{CudaClient, CudaRuntime, kernels};
use crate::tensor::Tensor;

/// AdvancedRandomOps implementation for CUDA runtime.
impl AdvancedRandomOps<CudaRuntime> for CudaClient {
    fn philox_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "philox_randn",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_philox_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                key,
                counter,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn philox_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "philox_uniform",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_philox_uniform(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                key,
                counter,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn threefry_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "threefry_randn",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_threefry_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                key,
                counter,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn threefry_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "threefry_uniform",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_threefry_uniform(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                key,
                counter,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn pcg64_randn(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "pcg64_randn",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_pcg64_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                stream,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn pcg64_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "pcg64_uniform",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_pcg64_uniform(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                stream,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn xoshiro256_randn(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "xoshiro256_randn",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_xoshiro256_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn xoshiro256_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "xoshiro256_uniform",
            });
        }

        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            kernels::launch_xoshiro256_uniform(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }
}
