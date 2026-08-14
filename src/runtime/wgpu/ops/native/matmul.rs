//! Matrix multiplication operation implementations for WebGPU.

use super::helpers::*;
use super::matmul_broadcast::flatten_batched_operands;
use crate::error::Error;
use crate::error::Result;
use crate::ops::{matmul_bias_output_shape, matmul_output_shape, validate_matmul_bias_dtypes};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::{gemv_bt, matmul};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Detect if a 2D tensor is a simple transpose of a contiguous [N,K] matrix.
/// Shape [K, N] with strides [1, K] means it's a transpose view of contiguous [N, K].
fn is_simple_transpose_2d(tensor: &Tensor<WgpuRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 2 {
        return false;
    }
    strides[0] == 1 && strides[1] == shape[0] as isize
}

/// Detect if the last two dims of a 3D tensor are a simple transpose.
/// Shape [B, K, N] with strides [N*K, 1, K] means each batch slice
/// is a transpose of contiguous [N, K].
fn is_batched_transpose_last2(tensor: &Tensor<WgpuRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 3 {
        return false;
    }
    let k = shape[1];
    let n = shape[2];
    strides[1] == 1 && strides[2] == k as isize && strides[0] == (n * k) as isize
}

pub(crate) fn native_matmul(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    let out_shape = matmul_output_shape(a.shape(), b.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Operands the batched kernel cannot index directly are flattened to 3D first;
    // the recursive call then lands on the batched path below.
    if let Some((a3, b3)) = flatten_batched_operands(a, b, &out_shape)? {
        return native_matmul(client, &a3, &b3)?.reshape(&out_shape);
    }

    // Handle 2D case
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        // GEMV-BT fast path: transposed B with small M
        if m <= 16 && is_simple_transpose_2d(b) {
            let a_contig = ensure_contiguous(a)?;
            let out = alloc_output(client, &out_shape, dtype);

            let a_buf = get_tensor_buffer(&a_contig)?;
            let b_buf = get_tensor_buffer(b)?; // Use original [N,K] buffer directly
            let out_buf = get_tensor_buffer(&out)?;

            let params = MatmulParams {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                batch_size: 1,
            };
            let params_buf = create_params_buffer(client, &params);

            gemv_bt::launch_gemv_bt(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                dtype,
            )?;

            return Ok(out);
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: 1,
        };
        let params_buf = create_params_buffer(client, &params);

        // Use tiled for larger matrices, simple for small ones
        if m * n > 256 * 256 {
            matmul::launch_matmul(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                dtype,
            )?;
        } else {
            matmul::launch_matmul_simple(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                dtype,
            )?;
        }

        return Ok(out);
    }

    // Handle batched (3D) matmul natively
    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        // Validate batch dimensions match
        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        // GEMV-BT fast path: transposed B with small M
        if m <= 16 && is_batched_transpose_last2(b) {
            let a_contig = ensure_contiguous(a)?;
            let out = alloc_output(client, &out_shape, dtype);

            let a_buf = get_tensor_buffer(&a_contig)?;
            let b_buf = get_tensor_buffer(b)?;
            let out_buf = get_tensor_buffer(&out)?;

            let params = MatmulParams {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                batch_size: batch_size as u32,
            };
            let params_buf = create_params_buffer(client, &params);

            gemv_bt::launch_batched_gemv_bt(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                batch_size,
                dtype,
            )?;

            return Ok(out);
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: batch_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        matmul::launch_batched_matmul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            batch_size,
            dtype,
        )?;

        return Ok(out);
    }

    // >3D: flatten leading dims into batch, run 3D batched matmul, reshape back.
    // Same strategy as CUDA backend (which computes batch_size = product of leading dims).
    let ndim_a = a_shape.len();
    let ndim_b = b_shape.len();

    if ndim_a < 2 || ndim_b < 2 {
        return Err(Error::BackendLimitation {
            backend: "WebGPU",
            operation: "matmul",
            reason: format!(
                "requires at least 2D tensors, got shapes {:?} and {:?}",
                a_shape, b_shape
            ),
        });
    }

    let m = a_shape[ndim_a - 2];
    let k = a_shape[ndim_a - 1];
    let n = b_shape[ndim_b - 1];

    let batch_a: usize = a_shape[..ndim_a - 2].iter().product();
    let batch_b: usize = b_shape[..ndim_b - 2].iter().product();
    let batch_size = batch_a.max(batch_b);

    // Flatten to 3D
    let a_3d = ensure_contiguous(a)?
        .reshape(&[batch_a, m, k])
        .map_err(|_| Error::shape_mismatch(a_shape, b_shape))?;
    let b_3d = ensure_contiguous(b)?
        .reshape(&[batch_b, k, n])
        .map_err(|_| Error::shape_mismatch(a_shape, b_shape))?;

    // Broadcast if batch dims differ (one must be 1)
    let (a_batched, b_batched) = if batch_a == batch_b {
        (a_3d, b_3d)
    } else if batch_a == 1 {
        (
            a_3d.broadcast_to(&[batch_size, m, k])
                .map_err(|_| Error::shape_mismatch(a_shape, b_shape))?
                .contiguous()
                .map_err(|_| Error::shape_mismatch(a_shape, b_shape))?,
            b_3d,
        )
    } else if batch_b == 1 {
        (
            a_3d,
            b_3d.broadcast_to(&[batch_size, k, n])
                .map_err(|_| Error::shape_mismatch(a_shape, b_shape))?
                .contiguous()
                .map_err(|_| Error::shape_mismatch(a_shape, b_shape))?,
        )
    } else {
        return Err(Error::shape_mismatch(a_shape, b_shape));
    };

    let a_buf = get_tensor_buffer(&a_batched)?;
    let b_buf = get_tensor_buffer(&b_batched)?;
    let out_flat = alloc_output(client, &[batch_size, m, n], dtype);
    let out_buf = get_tensor_buffer(&out_flat)?;

    let params = MatmulParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        batch_size: batch_size as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    matmul::launch_batched_matmul(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &b_buf,
        &out_buf,
        &params_buf,
        m,
        n,
        batch_size,
        dtype,
    )?;

    // Reshape back to original leading dims + [m, n]
    let result = out_flat
        .reshape(&out_shape)
        .map_err(|_| Error::shape_mismatch(a_shape, b_shape))?;
    Ok(result)
}

/// Native WGPU implementation of fused matrix multiplication with bias.
///
/// Computes C = A @ B + bias where bias is a 1D tensor [N] broadcast across all rows.
/// The bias addition is fused into the GEMM epilogue for efficiency.
pub(crate) fn native_matmul_bias(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    // Validate dtypes using unified helper (ensures consistent error handling across backends)
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

    // Validate shapes and compute output shape
    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    // Operands the batched kernel cannot index directly are flattened to 3D first;
    // bias is 1-D over N and is unaffected by batch normalization.
    if let Some((a3, b3)) = flatten_batched_operands(a, b, &out_shape)? {
        return native_matmul_bias(client, &a3, &b3, bias)?.reshape(&out_shape);
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 2D case
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let bias_buf = get_tensor_buffer(&bias_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: 1,
        };
        let params_buf = create_params_buffer(client, &params);

        matmul::launch_matmul_bias(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            dtype,
        )?;

        return Ok(out);
    }

    // Handle batched matmul_bias (3D tensors)
    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        // Validate batch dimensions match
        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let bias_buf = get_tensor_buffer(&bias_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: batch_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        matmul::launch_batched_matmul_bias(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            batch_size,
            dtype,
        )?;

        return Ok(out);
    }

    // >3D tensors are not supported - return error instead of silent fallback
    // (WebGPU shader dispatch is limited to 3D workgroups)
    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "matmul_bias",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}
