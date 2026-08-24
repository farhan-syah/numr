//! CPU implementation of utility operations.

use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::ops::UtilityOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels,
};
use crate::runtime::validate_arange;
use crate::tensor::Tensor;

use crate::error::Error;

/// UtilityOps implementation for CPU runtime.
impl UtilityOps<CpuRuntime> for CpuClient {
    fn clamp(
        &self,
        a: &Tensor<CpuRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &self.device)?;

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();
        let numel = a.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::clamp_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    numel,
                    min_val,
                    max_val,
                );
            }
        }, "clamp");

        Ok(out)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<CpuRuntime>> {
        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let out_ptr = out.ptr();
        let numel = out.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::fill_kernel::<T>(
                    out_ptr as *mut T,
                    T::from_f64(value),
                    numel,
                );
            }
        }, "fill");

        Ok(out)
    }

    fn arange(&self, start: f64, stop: f64, step: f64, dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Use shared validation
        let numel = validate_arange(start, stop, step)?;

        // Handle empty tensor case
        if numel == 0 {
            return Tensor::<CpuRuntime>::empty(&[0], dtype, &self.device);
        }

        let out = Tensor::<CpuRuntime>::empty(&[numel], dtype, &self.device)?;
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::arange_kernel::<T>(out_ptr as *mut T, start, step, numel);
            }
        }, "arange");

        Ok(out)
    }

    fn linspace(
        &self,
        start: f64,
        stop: f64,
        steps: usize,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // linspace supports all numeric dtypes - computation is done in f64,
        // then converted to the output dtype. This matches NumPy behavior.

        // Handle edge cases
        if steps == 0 {
            return Tensor::<CpuRuntime>::empty(&[0], dtype, &self.device);
        }

        if steps == 1 {
            let out = Tensor::<CpuRuntime>::empty(&[1], dtype, &self.device)?;
            let out_ptr = out.ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    *(out_ptr as *mut T) = T::from_f64(start);
                }
            }, "linspace");

            return Ok(out);
        }

        let out = Tensor::<CpuRuntime>::empty(&[steps], dtype, &self.device)?;
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::linspace_kernel::<T>(out_ptr as *mut T, start, stop, steps);
            }
        }, "linspace");

        Ok(out)
    }

    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Use shared validation
        use crate::runtime::validate_eye;
        let (rows, cols) = validate_eye(n, m);

        // Handle edge cases
        if rows == 0 || cols == 0 {
            return Tensor::<CpuRuntime>::empty(&[rows, cols], dtype, &self.device);
        }

        let out = Tensor::<CpuRuntime>::empty(&[rows, cols], dtype, &self.device)?;
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::eye_kernel::<T>(out_ptr as *mut T, rows, cols);
            }
        }, "eye");

        Ok(out)
    }

    fn one_hot(
        &self,
        indices: &Tensor<CpuRuntime>,
        num_classes: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = indices.dtype();

        // Validate indices are integer type
        if !dtype.is_int() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "one_hot",
            });
        }

        if num_classes == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_classes",
                reason: "one_hot requires num_classes > 0".to_string(),
            });
        }

        let indices = ensure_contiguous(indices)?;
        let numel = indices.numel();

        // Output shape = indices.shape() + [num_classes]
        let mut out_shape = indices.shape().to_vec();
        out_shape.push(num_classes);

        let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::F32, &self.device)?;
        let out_ptr = out.ptr() as *mut f32;

        // Zero-fill output
        unsafe {
            std::ptr::write_bytes(out_ptr, 0, numel * num_classes);
        }

        let indices_ptr = indices.ptr();

        // Dispatch on index dtype to read indices, write into f32 output
        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::one_hot_kernel::<T>(
                    indices_ptr as *const T,
                    out_ptr,
                    numel,
                    num_classes,
                );
            }
        }, "one_hot");

        Ok(out)
    }

    fn meshgrid(
        &self,
        tensors: &[&Tensor<CpuRuntime>],
        indexing: crate::ops::MeshgridIndexing,
    ) -> Result<Vec<Tensor<CpuRuntime>>> {
        crate::ops::impl_generic::meshgrid_impl(tensors, indexing)
    }
}
