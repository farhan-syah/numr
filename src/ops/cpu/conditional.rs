//! CPU implementation of conditional operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ConditionalOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels,
};
use crate::tensor::Tensor;

/// ConditionalOps implementation for CPU runtime.
impl ConditionalOps<CpuRuntime> for CpuClient {
    fn where_cond(
        &self,
        cond: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::broadcast_shape;

        // Validate that x and y have the same dtype
        if x.dtype() != y.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: x.dtype(),
                rhs: y.dtype(),
            });
        }
        let dtype = x.dtype();
        let cond_dtype = cond.dtype();

        // Compute broadcast shape (cond, x, y) -> out
        let xy_shape =
            broadcast_shape(x.shape(), y.shape()).ok_or_else(|| Error::BroadcastError {
                lhs: x.shape().to_vec(),
                rhs: y.shape().to_vec(),
            })?;
        let out_shape =
            broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| Error::BroadcastError {
                lhs: cond.shape().to_vec(),
                rhs: xy_shape.clone(),
            })?;

        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;
        let out_ptr = out.ptr();

        // Fast path: all same shape, use simple kernel
        if cond.shape() == x.shape() && x.shape() == y.shape() {
            let cond_contig = ensure_contiguous(cond)?;
            let x_contig = ensure_contiguous(x)?;
            let y_contig = ensure_contiguous(y)?;

            let cond_ptr = cond_contig.ptr();
            let x_ptr = x_contig.ptr();
            let y_ptr = y_contig.ptr();
            let numel = x.numel();

            // Double dispatch: cond dtype and value dtype
            // For U8 condition, use optimized SIMD kernel
            if cond_dtype == DType::U8 {
                dispatch_dtype!(dtype, T => {
                    unsafe {
                        kernels::where_kernel::<T>(
                            cond_ptr as *const u8,
                            x_ptr as *const T,
                            y_ptr as *const T,
                            out_ptr as *mut T,
                            numel,
                        );
                    }
                }, "where_cond");
            } else {
                // Generic kernel for any condition dtype (non-zero = true)
                dispatch_dtype!(cond_dtype, C => {
                    dispatch_dtype!(dtype, T => {
                        unsafe {
                            kernels::where_kernel_generic::<C, T>(
                                cond_ptr as *const C,
                                x_ptr as *const T,
                                y_ptr as *const T,
                                out_ptr as *mut T,
                                numel,
                            );
                        }
                    }, "where_cond");
                }, "where_cond");
            }
        } else {
            // Broadcasting path: use strided kernel
            // Broadcast all inputs to output shape (zero-copy views with stride 0 for broadcast dims)
            let cond_broadcast = cond.broadcast_to(&out_shape)?;
            let x_broadcast = x.broadcast_to(&out_shape)?;
            let y_broadcast = y.broadcast_to(&out_shape)?;

            let cond_ptr = cond_broadcast.ptr();
            let x_ptr = x_broadcast.ptr();
            let y_ptr = y_broadcast.ptr();

            // Get strides from broadcast layouts
            let cond_strides: Vec<isize> = cond_broadcast.layout().strides().to_vec();
            let x_strides: Vec<isize> = x_broadcast.layout().strides().to_vec();
            let y_strides: Vec<isize> = y_broadcast.layout().strides().to_vec();
            let cond_offset = cond_broadcast.layout().offset();
            let x_offset = x_broadcast.layout().offset();
            let y_offset = y_broadcast.layout().offset();

            // For U8 condition, use optimized kernel
            if cond_dtype == DType::U8 {
                dispatch_dtype!(dtype, T => {
                    unsafe {
                        kernels::where_strided_kernel::<T>(
                            cond_ptr as *const u8,
                            x_ptr as *const T,
                            y_ptr as *const T,
                            out_ptr as *mut T,
                            &out_shape,
                            &cond_strides,
                            &x_strides,
                            &y_strides,
                            cond_offset,
                            x_offset,
                            y_offset,
                        );
                    }
                }, "where_cond");
            } else {
                // Generic kernel for any condition dtype
                dispatch_dtype!(cond_dtype, C => {
                    dispatch_dtype!(dtype, T => {
                        unsafe {
                            kernels::where_strided_kernel_generic::<C, T>(
                                cond_ptr as *const C,
                                x_ptr as *const T,
                                y_ptr as *const T,
                                out_ptr as *mut T,
                                &out_shape,
                                &cond_strides,
                                &x_strides,
                                &y_strides,
                                cond_offset,
                                x_offset,
                                y_offset,
                            );
                        }
                    }, "where_cond");
                }, "where_cond");
            }
        }

        Ok(out)
    }
}
