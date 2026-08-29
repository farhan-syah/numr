//! CUDA-specific helper functions for kernel launching and tensor operations

use super::super::kernels::launch_scalar_op_half;
use super::super::kernels::{
    AccumulationPrecision, int_matmul_output_dtype, launch_binary_op, launch_broadcast_binary_op,
    launch_broadcast_compare_op, launch_compare_op, launch_gemv_kernel_bt_mr,
    launch_matmul_batched_kernel, launch_matmul_bias_batched_kernel, launch_matmul_bias_kernel,
    launch_matmul_kernel, launch_reduce_dim_op, launch_scalar_op_f32, launch_scalar_op_f64,
    launch_semiring_matmul_batched_kernel, launch_semiring_matmul_kernel, launch_unary_op,
};
use super::super::kernels::{
    launch_pow_scalar_int, launch_scalar_op_c64, launch_scalar_op_c128, launch_scalar_op_int,
};
use super::super::{CudaClient, CudaRuntime};
use super::matmul_broadcast::resolve_batched_operands;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{matmul_bias_output_shape, matmul_output_shape, reduce_output_shape};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

// ============================================================================
// Native Tiled GEMM Implementation
// ============================================================================

/// Native matrix multiplication using tiled CUDA kernel.
///
/// Uses shared memory tiling for cache efficiency. This is the default
/// implementation that works without any vendor dependencies.
/// Detect if a 2D tensor is a simple transpose of a contiguous [N,K] matrix.
///
/// A tensor with shape [K, N] and strides [1, K] is a transpose view of
/// contiguous [N, K] data. We can pass the raw pointer directly to gemv_bt
/// instead of materializing the transpose (which copies the entire matrix).
fn is_simple_transpose_2d(tensor: &Tensor<CudaRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 2 {
        return false;
    }
    // shape=[K,N], strides=[1,K] means transpose of contiguous [N,K]
    strides[0] == 1 && strides[1] == shape[0] as isize
}

/// Whether this dtype may take the small-M GEMV fast path in a plain matmul.
///
/// FP8 has no GEMV kernel at all. I8 has none either, and could not use one: its
/// matmul widens to I32 (see `int_matmul_output_dtype`) while every GEMV kernel
/// writes the element type. CPU excludes I8 from its own GEMV-BT path for
/// the same reason, so both backends run the tiled kernel at every M.
#[inline]
fn has_gemv_kernel(dtype: DType) -> bool {
    !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2 | DType::I8)
}

pub(crate) fn matmul_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Fast path: if B is a transposed view of contiguous [N,K] and M is small,
    // use gemv_bt kernel directly — avoids copying the entire weight matrix.
    // FP8 and I8 are excluded: see `has_gemv_kernel`.
    if m <= 16 && has_gemv_kernel(dtype) && is_simple_transpose_2d(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_gemv_kernel_bt_mr(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(), // raw [N,K] pointer — no copy!
                out.ptr(),
                1, // batch
                m,
                n,
                k,
                1, // a_batch
                1, // b_batch
            )?;
        }

        return Ok(out);
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    // I8 is the one dtype whose matmul does not write its own dtype: it widens
    // to I32, matching CPU's quantized accumulation. Every other dtype maps to
    // itself here.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Detect if the last two dims of a 3D tensor are a simple transpose.
/// Shape [B, K, N] with strides [B_stride, 1, K] means each batch slice
/// is a transpose of contiguous [N, K].
fn is_batched_transpose_last2(tensor: &Tensor<CudaRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 3 {
        return false;
    }
    let k = shape[1];
    let n = shape[2];
    // strides: [n*k, 1, k] means transpose of contiguous [batch, N, K]
    strides[1] == 1 && strides[2] == k as isize && strides[0] == (n * k) as isize
}

/// Native batched matrix multiplication using tiled CUDA kernel.
pub(crate) fn matmul_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Pointers and batch counts must come from the same tensors, so both are taken
    // from one resolver rather than derived separately.
    let operands = resolve_batched_operands(a, b, &out_shape)?;
    let (a, b) = (&operands.a, &operands.b);
    let (a_batch, b_batch) = (operands.a_batch, operands.b_batch);

    // Fast path: transposed B with small M → gemv_bt. FP8 and I8 are excluded
    // for the same reason as in matmul_native.
    if m <= 16 && has_gemv_kernel(dtype) && is_batched_transpose_last2(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_gemv_kernel_bt_mr(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(),
                out.ptr(),
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            )?;
        }

        return Ok(out);
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    // I8 widens to I32 here too — see `matmul_native`.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            batch,
            m,
            n,
            k,
            a_batch,
            b_batch,
        )?;
    }

    Ok(out)
}

// ============================================================================
// Fused Matmul+Bias Native Implementation
// ============================================================================

/// Native fused matmul+bias using tiled CUDA kernel: C = A @ B + bias
///
/// Uses the same tiled GEMM algorithm as matmul_native, but fuses bias addition
/// into the epilogue to avoid an extra memory round-trip.
pub(crate) fn matmul_bias_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let bias_contig = ensure_contiguous(bias)?;

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
        Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        },
    )?;

    // I8 widens to I32 in the fused form exactly as in the plain one, and the
    // validator has already required an I32 bias to seed that accumulator.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_bias_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            bias_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Native batched fused matmul+bias using tiled CUDA kernel:
/// C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] + bias[N]
pub(crate) fn matmul_bias_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let bias_contig = ensure_contiguous(bias)?;

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
        Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        },
    )?;

    // Pointers and batch counts must come from the same tensors, so both are taken
    // from one resolver rather than derived separately.
    let operands = resolve_batched_operands(a, b, &out_shape)?;
    let (a_contig, b_contig) = operands.contiguous()?;
    let (a_batch, b_batch) = (operands.a_batch, operands.b_batch);

    // Widens at I8 like every other matmul entry point here.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_bias_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            bias_contig.ptr(),
            out.ptr(),
            batch,
            m,
            n,
            k,
            a_batch,
            b_batch,
        )?;
    }

    Ok(out)
}

// ============================================================================
// Native Kernel Helpers
// ============================================================================

/// Launch a native binary operation on GPU.
///
/// # Performance
///
/// - **Same shape**: Runs entirely on GPU (fast)
/// - **Different shapes**: Falls back to CPU with GPU↔CPU transfers (slow)
///
/// For broadcasting operations, consider pre-expanding tensors to matching shapes
/// using `broadcast_to()` or similar operations to avoid CPU fallback.
pub(crate) fn native_binary_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // A zero-element output has nothing to compute, and an empty launch grid
    // is itself invalid, so skip the launch entirely.
    if out_shape.iter().product::<usize>() == 0 {
        return Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
    }

    // For same-shape tensors, use the optimized element-wise kernel
    if a.shape() == b.shape() {
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_binary_op(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        return Ok(out);
    }

    // For different shapes, use the broadcast kernel (stays on GPU)
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    unsafe {
        launch_broadcast_binary_op(
            &client.context,
            &client.stream,
            client.device.index,
            &client.device,
            op,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            a.shape(),
            b.shape(),
            &out_shape,
        )?;
    }

    Ok(out)
}

/// Launch a native CUDA binary operation into a caller-provided destination.
///
/// Identical to [`native_binary_op`] but writes `op(a, b)` into the existing
/// `out` tensor instead of allocating. Required for destination-passing
/// workflows (e.g. CUDA graph capture) where the output buffer must be
/// allocated outside the captured region so its device address is stable.
///
/// `out` must be contiguous, share the inputs' dtype, and have a shape equal to
/// `broadcast(a, b)`.
pub(crate) fn native_binary_op_into(
    client: &CudaClient,
    out: &Tensor<CudaRuntime>,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<()> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    if out.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: out.dtype(),
        });
    }
    if out.shape() != out_shape.as_slice() {
        return Err(Error::ShapeMismatch {
            expected: out_shape,
            got: out.shape().to_vec(),
        });
    }
    if !out.is_contiguous() {
        return Err(Error::Backend(
            "native_binary_op_into: destination tensor must be contiguous".into(),
        ));
    }

    // A zero-element destination has nothing to compute, and an empty launch
    // grid is itself invalid, so skip the launch entirely.
    if out.numel() == 0 {
        return Ok(());
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    if a.shape() == b.shape() {
        unsafe {
            launch_binary_op(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }
    } else {
        unsafe {
            launch_broadcast_binary_op(
                &client.context,
                &client.stream,
                client.device.index,
                &client.device,
                op,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                a.shape(),
                b.shape(),
                &out_shape,
            )?;
        }
    }

    Ok(())
}

/// Launch a native CUDA unary operation (element-wise, single input).
///
/// Dispatches to CUDA kernels for operations like neg, abs, sqrt, exp, log,
/// sin, cos, sigmoid, relu, etc. The operation runs entirely on GPU.
///
/// # Arguments
/// * `op` - Operation name (must match kernel function suffix, e.g., "neg", "exp")
pub(crate) fn native_unary_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;

    // A zero-element tensor has nothing to compute, and an empty launch grid
    // is itself invalid, so skip the launch entirely.
    if out.numel() == 0 {
        return Ok(out);
    }

    unsafe {
        launch_unary_op(
            &client.context,
            &client.stream,
            client.device.index,
            op,
            dtype,
            a_contig.ptr(),
            out.ptr(),
            out.numel(),
        )?;
    }

    Ok(out)
}

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

/// Launch a native CUDA reduction operation (sum, max, min along dimensions).
///
/// # Performance
///
/// - **Single dimension**: Uses optimized CUDA kernel with warp-level reductions (fast)
/// - **Multiple dimensions**: Falls back to CPU with GPU↔CPU transfers (slow)
///
/// # Arguments
/// * `op` - Operation name ("sum", "max", "min")
/// * `dims` - Dimensions to reduce over
/// * `keepdim` - Whether to keep reduced dimensions as size 1
/// * `precision` - Optional accumulation precision (higher precision for sum)
pub(crate) fn native_reduce_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
    dims: &[usize],
    keepdim: bool,
    precision: Option<AccumulationPrecision>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let out_shape = reduce_output_shape(a.shape(), dims, keepdim);
    let acc_precision = precision.unwrap_or_default();

    // For single-dimension reduction, use optimized kernel
    if dims.len() == 1 {
        let dim = dims[0];
        let shape = a.shape();

        // Calculate outer, reduce, inner sizes
        let outer_size: usize = shape[..dim].iter().product();
        let reduce_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        // A zero-size output (some non-reduced dimension is 0) has nothing to
        // compute, and `outer_size`/`inner_size` were floored at 1 above, so
        // launching would write past the empty allocation.
        if out.numel() == 0 {
            return Ok(out);
        }

        unsafe {
            launch_reduce_dim_op(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                outer_size,
                reduce_size,
                inner_size,
                acc_precision,
            )?;
        }

        return Ok(out);
    }

    // For multiple dimensions: chain single-dimension reductions on GPU
    // This keeps all computation on the GPU instead of falling back to CPU

    // Sort dimensions from highest to lowest to avoid index shifting issues
    let mut sorted_dims: Vec<usize> = dims.to_vec();
    sorted_dims.sort_unstable();
    sorted_dims.reverse();

    // Reduce one dimension at a time, always keeping dims so each step's indexing
    // stays aligned with the original layout.
    let mut current = a.clone();
    for &dim in &sorted_dims {
        current = native_reduce_op(client, &current, op, &[dim], true, precision)?;
    }

    // Every reduced dim is still present as size 1, so drop them in one step.
    // Squeezing only at the end is what makes a full reduction with keepdim=false
    // collapse to a scalar rather than leaving a trailing size-1 dimension.
    if current.shape() != out_shape.as_slice() {
        current = current.reshape(&out_shape)?;
    }

    Ok(current)
}

/// Launch a native comparison operation on GPU.
///
/// # Performance
///
/// - **Same shape**: Uses optimized element-wise kernel (fast)
/// - **Different shapes**: Uses broadcast kernel with strided access (stays on GPU)
pub(crate) fn native_compare_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // A zero-element output has nothing to compute, and an empty launch grid
    // is itself invalid, so skip the launch entirely. Placed after the
    // dtype/shape checks so a mismatched pair still reports its error rather
    // than succeeding because it was empty.
    if out_shape.iter().product::<usize>() == 0 {
        return Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
    }

    // For same-shape tensors, use the optimized element-wise kernel
    if a.shape() == b.shape() {
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_compare_op(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        return Ok(out);
    }

    // For different shapes, use the broadcast kernel (stays on GPU)
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    unsafe {
        launch_broadcast_compare_op(
            &client.context,
            &client.stream,
            client.device.index,
            &client.device,
            op,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            a.shape(),
            b.shape(),
            &out_shape,
        )?;
    }

    Ok(out)
}

// ============================================================================
// Semiring Matrix Multiplication
// ============================================================================

/// Native semiring matrix multiplication using CUDA kernel.
pub(crate) fn semiring_matmul_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    semiring_op: u32,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    unsafe {
        launch_semiring_matmul_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
            semiring_op,
        )?;
    }

    Ok(out)
}

/// Native batched semiring matrix multiplication using CUDA kernel.
pub(crate) fn semiring_matmul_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    semiring_op: u32,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Pointers and batch counts must come from the same tensors, so both are taken
    // from one resolver rather than derived separately.
    let operands = resolve_batched_operands(a, b, &out_shape)?;
    let (a_contig, b_contig) = operands.contiguous()?;
    let (a_batch, b_batch) = (operands.a_batch, operands.b_batch);

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    unsafe {
        launch_semiring_matmul_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            batch,
            m,
            n,
            k,
            semiring_op,
            a_batch,
            b_batch,
        )?;
    }

    Ok(out)
}
