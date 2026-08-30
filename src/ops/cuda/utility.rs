//! Utility operations for CUDA runtime
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, UtilityOps};
use crate::runtime::cuda::kernels::{
    launch_arange, launch_eye, launch_fill_with_f64, launch_linspace,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{validate_arange, validate_eye};
use crate::tensor::Tensor;

impl UtilityOps<CudaRuntime> for CudaClient {
    fn clamp(
        &self,
        a: &Tensor<CudaRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use native CUDA implementation via composition of maximum and minimum
        // clamp(x, min, max) = min(max(x, min), max)
        // This approach uses existing optimized kernels

        // Create scalar tensors for min and max
        let min_scalar = self.fill(&[], min_val, a.dtype())?;
        let max_scalar = self.fill(&[], max_val, a.dtype())?;

        // First: max(x, min_val)
        let clamped_low = self.maximum(a, &min_scalar)?;

        // Then: min(result, max_val)
        self.minimum(&clamped_low, &max_scalar)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<CudaRuntime>> {
        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        }

        // `FillValue::from_f64` stores Bool as a raw `value as u8`, which is
        // wrong for any fill value other than 0.0/1.0 (e.g. `fill(_, 2.0,
        // Bool)` would write byte 2, not the boolean collapse). Route through
        // the dtype-generic constructor instead, which applies the
        // crate-wide Bool convention (any nonzero, NaN included, is true)
        // and matches CPU/WebGPU `fill` above/below.
        if dtype == DType::Bool {
            return Tensor::<CudaRuntime>::full_scalar(shape, dtype, value, &self.device);
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // Launch native CUDA fill kernel
        unsafe {
            launch_fill_with_f64(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                value,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn arange(
        &self,
        start: f64,
        stop: f64,
        step: f64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use shared validation
        let numel = validate_arange(start, stop, step)?;

        // Handle empty tensor case
        if numel == 0 {
            return Tensor::<CudaRuntime>::empty(&[0], dtype, &self.device);
        }

        let out = Tensor::<CudaRuntime>::empty(&[numel], dtype, &self.device)?;

        unsafe {
            launch_arange(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                start,
                step,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn linspace(
        &self,
        start: f64,
        stop: f64,
        steps: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        // linspace supports all numeric dtypes - computation is done in higher precision,
        // then converted to the output dtype. This matches NumPy behavior.

        // Handle edge cases
        if steps == 0 {
            return Tensor::<CudaRuntime>::empty(&[0], dtype, &self.device);
        }

        if steps == 1 {
            return self.fill(&[1], start, dtype);
        }

        let out = Tensor::<CudaRuntime>::empty(&[steps], dtype, &self.device)?;

        unsafe {
            launch_linspace(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                start,
                stop,
                out.ptr(),
                steps,
            )?;
        }

        Ok(out)
    }

    fn one_hot(
        &self,
        indices: &Tensor<CudaRuntime>,
        num_classes: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        crate::ops::impl_generic::one_hot_impl(self, indices, num_classes)
    }

    fn meshgrid(
        &self,
        tensors: &[&Tensor<CudaRuntime>],
        indexing: crate::ops::MeshgridIndexing,
    ) -> Result<Vec<Tensor<CudaRuntime>>> {
        crate::ops::impl_generic::meshgrid_impl(tensors, indexing)
    }

    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // Use shared validation
        let (rows, cols) = validate_eye(n, m);

        // Handle edge cases
        if rows == 0 || cols == 0 {
            return Tensor::<CudaRuntime>::empty(&[rows, cols], dtype, &self.device);
        }

        let out = Tensor::<CudaRuntime>::empty(&[rows, cols], dtype, &self.device)?;

        unsafe {
            launch_eye(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                rows,
                cols,
                out.ptr(),
            )?;
        }

        Ok(out)
    }
}
