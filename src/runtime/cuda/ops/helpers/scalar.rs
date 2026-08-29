//! Tensor-scalar op launcher for the CUDA client, and the dtype gate it uses.
//!
//! `scalar_op_has_kernel` is the single list of dtypes `scalar.cu`
//! instantiates, shared by the empty-tensor shortcut and the launch dispatch.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::launch_scalar_op_half;
use crate::runtime::cuda::kernels::{
    launch_pow_scalar_int, launch_scalar_op_c64, launch_scalar_op_c128, launch_scalar_op_f32,
    launch_scalar_op_f64, launch_scalar_op_int,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Whether `scalar.cu` instantiates this dtype's tensor-scalar kernels.
///
/// One list for both users in [`native_scalar_op`]: the empty-tensor shortcut,
/// which must not report success for a dtype the op cannot run at all, and the
/// launch dispatch, whose `_` arm reports the same dtypes as unsupported. A
/// gate that admitted a dtype with no `.cu` row would turn a clean
/// `UnsupportedDType` into a `named symbol not found` at launch.
fn scalar_op_has_kernel(dtype: DType) -> bool {
    match dtype {
        DType::F32
        | DType::F64
        | DType::FP8E4M3
        | DType::FP8E5M2
        | DType::Complex64
        | DType::Complex128 => true,
        // NUMR_SCALAR_ROW_INT covers all eight integer dtypes.
        d if d.is_int() => true,
        // The half-precision rows exist in the PTX either way, but the launcher
        // that reaches them is compiled out without the feature.
        DType::F16 | DType::BF16 => cfg!(feature = "f16"),
        _ => false,
    }
}

/// Launch a native CUDA tensor-scalar operation.
///
/// Dispatches to the CUDA kernels for add_scalar, sub_scalar, rsub_scalar,
/// mul_scalar, div_scalar and pow_scalar. Every dtype
/// [`scalar_op_has_kernel`] admits runs entirely on GPU; the rest report
/// [`Error::UnsupportedDType`].
///
/// # Arguments
/// * `op` - Operation name (e.g., "add_scalar", "mul_scalar")
/// * `scalar` - Scalar value to apply to each element
pub(crate) fn native_scalar_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
    scalar: f64,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;

    // A zero-element tensor has nothing to compute, and an empty launch grid
    // is itself invalid, so skip the launch entirely. Still fall through to
    // the dtype match below for a dtype this op doesn't support at all, so an
    // empty tensor errors exactly like a non-empty one would.
    if out.numel() == 0 && scalar_op_has_kernel(dtype) {
        return Ok(out);
    }

    // Integer `pow_scalar` takes the exponent as f64 rather than the element
    // type: `scalar as i32` would round 2.5 to 2 and silently answer a
    // different question. The op layer routes every non-integral exponent to an
    // F64 output before this point, so only a non-negative whole number arrives.
    if op == "pow_scalar" && dtype.is_int() {
        unsafe {
            launch_pow_scalar_int(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                scalar,
                out.ptr(),
                out.numel(),
            )?;
        }
        return Ok(out);
    }

    unsafe {
        match dtype {
            DType::F32 => launch_scalar_op_f32(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.ptr(),
                scalar as f32,
                out.ptr(),
                out.numel(),
            )?,
            DType::F64 => launch_scalar_op_f64(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.ptr(),
                scalar,
                out.ptr(),
                out.numel(),
            )?,
            // Every integer dtype: `launch_scalar_op_int` fans out to the
            // per-dtype launcher, since each kernel takes the scalar in its own
            // element type.
            d if d.is_int() => launch_scalar_op_int(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                d,
                a_contig.ptr(),
                scalar,
                out.ptr(),
                out.numel(),
            )?,
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => launch_scalar_op_half(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                scalar as f32,
                out.ptr(),
                out.numel(),
            )?,
            DType::FP8E4M3 | DType::FP8E5M2 => launch_scalar_op_half(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                scalar as f32,
                out.ptr(),
                out.numel(),
            )?,
            DType::Complex64 => launch_scalar_op_c64(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.ptr(),
                scalar as f32,
                out.ptr(),
                out.numel(),
            )?,
            DType::Complex128 => launch_scalar_op_c128(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.ptr(),
                scalar,
                out.ptr(),
                out.numel(),
            )?,
            _ => {
                // Bool, and F16/BF16 without the `f16` feature. Kept in step
                // with `scalar_op_has_kernel` above: a dtype admitted there
                // with no arm here would abort on an unreachable match instead
                // of reporting the dtype.
                return Err(Error::UnsupportedDType { dtype, op });
            }
        }
    }

    Ok(out)
}
