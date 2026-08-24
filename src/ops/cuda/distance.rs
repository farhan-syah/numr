//! CUDA implementation of distance operations.

use crate::error::Result;
use crate::ops::distance_common::*;
use crate::ops::{DistanceMetric, DistanceOps};
use crate::runtime::cuda::{CudaClient, CudaRuntime, kernels};
use crate::tensor::Tensor;

impl DistanceOps<CudaRuntime> for CudaClient {
    fn cdist(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CudaRuntime>> {
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
            return Tensor::<CudaRuntime>::try_empty(&[n, m], dtype, &self.device);
        }

        // Ensure contiguous
        let x = x.contiguous()?;
        let y = y.contiguous()?;

        let out = Tensor::<CudaRuntime>::try_empty(&[n, m], dtype, &self.device)?;

        unsafe {
            kernels::launch_cdist(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                x.ptr(),
                y.ptr(),
                out.ptr(),
                n,
                m,
                d,
                metric,
            )?;
        }

        Ok(out)
    }

    fn pdist(
        &self,
        x: &Tensor<CudaRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let x = x.contiguous()?;

        let out = Tensor::<CudaRuntime>::try_empty(&[out_size], dtype, &self.device)?;

        unsafe {
            kernels::launch_pdist(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                x.ptr(),
                out.ptr(),
                n,
                d,
                metric,
            )?;
        }

        Ok(out)
    }

    fn squareform(&self, condensed: &Tensor<CudaRuntime>, n: usize) -> Result<Tensor<CudaRuntime>> {
        let cond_shape = condensed.shape();

        // Validate inputs using shared validators
        validate_1d_tensor(cond_shape, "condensed", "squareform")?;
        validate_condensed_length(cond_shape[0], n, "condensed", "squareform")?;

        let dtype = condensed.dtype();
        validate_float_dtype(dtype, "squareform")?;

        // Handle edge cases
        if n == 0 {
            return Tensor::<CudaRuntime>::try_empty(&[0, 0], dtype, &self.device);
        }
        if n == 1 {
            return Tensor::<CudaRuntime>::try_zeros(&[1, 1], dtype, &self.device);
        }

        // Ensure contiguous
        let condensed = condensed.contiguous()?;

        let out = Tensor::<CudaRuntime>::try_empty(&[n, n], dtype, &self.device)?;

        unsafe {
            kernels::launch_squareform(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                condensed.ptr(),
                out.ptr(),
                n,
            )?;
        }

        Ok(out)
    }

    fn squareform_inverse(&self, square: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let sq_shape = square.shape();

        // Validate inputs using shared validators
        validate_2d_tensor(sq_shape, "square", "squareform_inverse")?;
        validate_square_matrix(sq_shape, "square", "squareform_inverse")?;

        let n = sq_shape[0];
        let dtype = square.dtype();
        validate_float_dtype(dtype, "squareform_inverse")?;

        // Handle edge cases
        if n == 0 || n == 1 {
            return Tensor::<CudaRuntime>::try_empty(&[0], dtype, &self.device);
        }

        // Ensure contiguous
        let square = square.contiguous()?;

        let out_size = n * (n - 1) / 2;
        let out = Tensor::<CudaRuntime>::try_empty(&[out_size], dtype, &self.device)?;

        unsafe {
            kernels::launch_squareform_inverse(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                square.ptr(),
                out.ptr(),
                n,
            )?;
        }

        Ok(out)
    }
}
