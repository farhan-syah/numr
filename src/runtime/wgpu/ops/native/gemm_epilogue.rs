//! Native WGPU GEMM epilogue operations.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::ops::{GemmActivation, matmul_bias_output_shape, validate_matmul_bias_dtypes};
use crate::runtime::ensure_contiguous;
use crate::runtime::traits::client::RuntimeClient;
use crate::runtime::wgpu::shaders::gemm_epilogue;
use crate::runtime::wgpu::shaders::gemm_epilogue_bwd::{
    GemmEpilogueBwdBuffers, launch_gemm_bias_activation_bwd,
};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_gemm_bias_activation(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    activation: GemmActivation,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;
    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let out = alloc_output(client, &out_shape, dtype)?;

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_epilogue_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            1,
            activation,
        );

        gemm_epilogue::launch_gemm_bias_act(
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

    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let out = alloc_output(client, &out_shape, dtype)?;

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_epilogue_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            batch_size as u32,
            activation,
        );

        gemm_epilogue::launch_gemm_bias_act_batched(
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

    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "gemm_bias_activation",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}

pub(crate) fn native_gemm_bias_residual(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    residual: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;
    if residual.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: residual.dtype(),
        });
    }

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    if residual.shape() != out_shape.as_slice() {
        return Err(Error::ShapeMismatch {
            expected: out_shape.clone(),
            got: residual.shape().to_vec(),
        });
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let res_c = ensure_contiguous(residual)?;
        let out = alloc_output(client, &out_shape, dtype)?;

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let res_buf = get_tensor_buffer(&res_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_residual_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            1,
        );

        gemm_epilogue::launch_gemm_bias_residual(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &res_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            dtype,
        )?;

        return Ok(out);
    }

    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let res_c = ensure_contiguous(residual)?;
        let out = alloc_output(client, &out_shape, dtype)?;

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let res_buf = get_tensor_buffer(&res_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_residual_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            batch_size as u32,
        );

        gemm_epilogue::launch_gemm_bias_residual_batched(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &res_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            batch_size,
            dtype,
        )?;

        return Ok(out);
    }

    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "gemm_bias_residual",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}

pub(crate) fn native_gemm_bias_activation_bwd(
    client: &WgpuClient,
    grad: &Tensor<WgpuRuntime>,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    activation: GemmActivation,
) -> Result<(
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
)> {
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;
    if grad.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: grad.dtype(),
        });
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    let (batch_size, m, k, n) = match (a_shape.len(), b_shape.len()) {
        (2, 2) => (1usize, a_shape[0], a_shape[1], b_shape[1]),
        (3, 3) => {
            if b_shape[0] != a_shape[0] {
                return Err(Error::ShapeMismatch {
                    expected: vec![a_shape[0], a_shape[2], b_shape[2]],
                    got: b_shape.to_vec(),
                });
            }
            (a_shape[0], a_shape[1], a_shape[2], b_shape[2])
        }
        _ => {
            return Err(Error::BackendLimitation {
                backend: "WebGPU",
                operation: "gemm_bias_activation_bwd",
                reason: format!(
                    "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
                    a_shape, b_shape
                ),
            });
        }
    };

    if a_shape[a_shape.len() - 1] != k || b_shape[b_shape.len() - 2] != k {
        return Err(Error::shape_mismatch(a_shape, b_shape));
    }

    let a_c = ensure_contiguous(a)?;
    let b_c = ensure_contiguous(b)?;
    let bias_c = ensure_contiguous(bias)?;
    let grad_c = ensure_contiguous(grad)?;

    let d_a = alloc_output(client, a_shape, dtype)?;
    // The db shader sums the gradient over the batch dimension and writes only
    // the leading [K, N] slice of d_b; the CPU and CUDA references explicitly
    // allocate d_b with zeros, leaving the remaining [batch-1, K, N] elements
    // at zero. We match that contract explicitly here instead of relying on the
    // WebGPU allocator handing back freshly zero-initialized buffers, so the
    // trailing slices stay correct even if buffer pooling is added later.
    let d_b = if batch_size > 1 {
        Tensor::<WgpuRuntime>::zeros(b_shape, dtype, RuntimeClient::device(client))?
    } else {
        alloc_output(client, b_shape, dtype)?
    };
    let d_bias = alloc_output(client, &[n], dtype)?;
    // grad_pre scratch has the same shape as grad/output: [batch, M, N].
    let grad_pre = alloc_output(client, grad.shape(), dtype)?;

    let a_buf = get_tensor_buffer(&a_c)?;
    let b_buf = get_tensor_buffer(&b_c)?;
    let bias_buf = get_tensor_buffer(&bias_c)?;
    let grad_buf = get_tensor_buffer(&grad_c)?;
    let grad_pre_buf = get_tensor_buffer(&grad_pre)?;
    let d_a_buf = get_tensor_buffer(&d_a)?;
    let d_b_buf = get_tensor_buffer(&d_b)?;
    let d_bias_buf = get_tensor_buffer(&d_bias)?;

    let params_buf = gemm_epilogue::create_epilogue_params_buffer(
        client.pipeline_cache(),
        m as u32,
        k as u32,
        n as u32,
        batch_size as u32,
        activation,
    );

    let buffers = GemmEpilogueBwdBuffers {
        a: &a_buf,
        b: &b_buf,
        bias: &bias_buf,
        grad: &grad_buf,
        grad_pre: &grad_pre_buf,
        d_a: &d_a_buf,
        d_b: &d_b_buf,
        d_bias: &d_bias_buf,
    };

    launch_gemm_bias_activation_bwd(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &buffers,
        &params_buf,
        m,
        n,
        k,
        batch_size,
        dtype,
    )?;

    Ok((d_a, d_b, d_bias))
}
