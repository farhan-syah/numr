//! CPU implementation of advanced PRNG operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::AdvancedRandomOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::dispatch_dtype, kernels};
use crate::tensor::Tensor;

/// AdvancedRandomOps implementation for CPU runtime.
impl AdvancedRandomOps<CpuRuntime> for CpuClient {
    fn philox_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "philox_randn",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::philox_randn_kernel::<T>(out_ptr as *mut T, numel, key, counter);
            }
        }, "philox_randn");

        Ok(out)
    }

    fn philox_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "philox_uniform",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::philox_uniform_kernel::<T>(out_ptr as *mut T, numel, key, counter);
            }
        }, "philox_uniform");

        Ok(out)
    }

    fn threefry_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "threefry_randn",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::threefry_randn_kernel::<T>(out_ptr as *mut T, numel, key, counter);
            }
        }, "threefry_randn");

        Ok(out)
    }

    fn threefry_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "threefry_uniform",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::threefry_uniform_kernel::<T>(out_ptr as *mut T, numel, key, counter);
            }
        }, "threefry_uniform");

        Ok(out)
    }

    fn pcg64_randn(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "pcg64_randn",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::pcg64_randn_kernel::<T>(out_ptr as *mut T, numel, seed, stream);
            }
        }, "pcg64_randn");

        Ok(out)
    }

    fn pcg64_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "pcg64_uniform",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::pcg64_uniform_kernel::<T>(out_ptr as *mut T, numel, seed, stream);
            }
        }, "pcg64_uniform");

        Ok(out)
    }

    fn xoshiro256_randn(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "xoshiro256_randn",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::xoshiro256_randn_kernel::<T>(out_ptr as *mut T, numel, seed);
            }
        }, "xoshiro256_randn");

        Ok(out)
    }

    fn xoshiro256_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "xoshiro256_uniform",
            });
        }

        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::xoshiro256_uniform_kernel::<T>(out_ptr as *mut T, numel, seed);
            }
        }, "xoshiro256_uniform");

        Ok(out)
    }
}
