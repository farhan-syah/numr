//! Conditional operations for CUDA runtime
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::ConditionalOps;
use crate::runtime::cuda::kernels::{
    launch_where_broadcast_generic_op, launch_where_broadcast_op, launch_where_generic_op,
    launch_where_op,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

impl ConditionalOps<CudaRuntime> for CudaClient {
    fn where_cond(
        &self,
        cond: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate that x and y have the same dtype
        let dtype = validate_binary_dtypes(x, y)?;
        let cond_dtype = cond.dtype();

        // For same shapes, use optimized element-wise kernel on GPU
        if cond.shape() == x.shape() && x.shape() == y.shape() {
            let cond_contig = ensure_contiguous(cond)?;
            let x_contig = ensure_contiguous(x)?;
            let y_contig = ensure_contiguous(y)?;
            let out = Tensor::<CudaRuntime>::empty(x.shape(), dtype, &self.device)?;

            unsafe {
                if cond_dtype == DType::U8 {
                    // Optimized U8 kernel
                    launch_where_op(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        dtype,
                        cond_contig.ptr(),
                        x_contig.ptr(),
                        y_contig.ptr(),
                        out.ptr(),
                        out.numel(),
                    )?;
                } else {
                    // Generic kernel for F32, F64, I32, I64, U32 conditions
                    launch_where_generic_op(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        cond_dtype,
                        dtype,
                        cond_contig.ptr(),
                        x_contig.ptr(),
                        y_contig.ptr(),
                        out.ptr(),
                        out.numel(),
                    )?;
                }
            }

            return Ok(out);
        }

        // For different shapes, use the broadcast kernel (stays on GPU)
        // Compute broadcast shape for all three tensors
        let xy_shape = compute_broadcast_shape(x, y)?;
        let out_shape = crate::ops::broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| {
            crate::error::Error::BroadcastError {
                lhs: cond.shape().to_vec(),
                rhs: xy_shape.clone(),
            }
        })?;

        let cond_contig = ensure_contiguous(cond)?;
        let x_contig = ensure_contiguous(x)?;
        let y_contig = ensure_contiguous(y)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device)?;

        unsafe {
            if cond_dtype == DType::U8 {
                // Optimized U8 broadcast kernel
                launch_where_broadcast_op(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    &self.device,
                    dtype,
                    cond_contig.ptr(),
                    x_contig.ptr(),
                    y_contig.ptr(),
                    out.ptr(),
                    cond.shape(),
                    x.shape(),
                    y.shape(),
                    &out_shape,
                )?;
            } else {
                // Generic broadcast kernel for non-U8 conditions
                launch_where_broadcast_generic_op(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    &self.device,
                    cond_dtype,
                    dtype,
                    cond_contig.ptr(),
                    x_contig.ptr(),
                    y_contig.ptr(),
                    out.ptr(),
                    cond.shape(),
                    x.shape(),
                    y.shape(),
                    &out_shape,
                )?;
            }
        }

        Ok(out)
    }
}
