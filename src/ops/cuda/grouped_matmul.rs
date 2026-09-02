//! CUDA implementation of `GroupedMatmulOps`.
//!
//! Launches the grouped entry points in `kernels/grouped_matmul.cu`, which wrap
//! the same compile-time-tiled GEMM core the dense F32 path uses.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{GemmActivation, GroupedMatmulOps};
use crate::runtime::Device;
use crate::runtime::cuda::kernels::launch_grouped_matmul_f32;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Validates the grouped shapes and returns `(total_rows, k, n, num_groups)`.
fn validate_grouped(
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    group_offsets: &Tensor<CudaRuntime>,
) -> Result<(usize, usize, usize, usize)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let o_shape = group_offsets.shape();

    if a_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!("expected 2-D [total_rows, k], got {}-D", a_shape.len()),
        });
    }
    if b_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "b",
            reason: format!("expected 3-D [num_groups, k, n], got {}-D", b_shape.len()),
        });
    }
    if o_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "group_offsets",
            reason: format!("expected 1-D [num_groups + 1], got {}-D", o_shape.len()),
        });
    }
    if group_offsets.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg: "group_offsets",
            reason: format!("expected I32, got {:?}", group_offsets.dtype()),
        });
    }
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!(
                "grouped matmul is F32 only, got a={:?} b={:?}",
                a.dtype(),
                b.dtype()
            ),
        });
    }

    let (total_rows, k) = (a_shape[0], a_shape[1]);
    let (num_groups, k_b, n) = (b_shape[0], b_shape[1], b_shape[2]);
    if k != k_b {
        return Err(Error::ShapeMismatch {
            expected: vec![total_rows, k_b],
            got: vec![total_rows, k],
        });
    }
    if o_shape[0] != num_groups + 1 {
        return Err(Error::InvalidArgument {
            arg: "group_offsets",
            reason: format!("expected {} entries, got {}", num_groups + 1, o_shape[0]),
        });
    }

    Ok((total_rows, k, n, num_groups))
}

fn grouped_matmul_launch(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    group_offsets: &Tensor<CudaRuntime>,
    activation: Option<GemmActivation>,
) -> Result<Tensor<CudaRuntime>> {
    let (total_rows, k, n, num_groups) = validate_grouped(a, b, group_offsets)?;
    let device = a.device();

    let output = Tensor::<CudaRuntime>::empty(&[total_rows, n], DType::F32, device)?;
    if total_rows == 0 || num_groups == 0 {
        return Ok(output);
    }

    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let o_c = group_offsets.contiguous()?;

    unsafe {
        launch_grouped_matmul_f32(
            client.context(),
            client.stream(),
            device.id(),
            a_c.ptr(),
            b_c.ptr(),
            o_c.ptr(),
            output.ptr(),
            total_rows,
            n,
            k,
            num_groups,
            activation.map(GemmActivation::code),
        )?;
    }

    Ok(output)
}

impl GroupedMatmulOps<CudaRuntime> for CudaClient {
    fn grouped_matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        group_offsets: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        grouped_matmul_launch(self, a, b, group_offsets, None)
    }

    fn grouped_matmul_activation(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        group_offsets: &Tensor<CudaRuntime>,
        activation: GemmActivation,
    ) -> Result<Tensor<CudaRuntime>> {
        // `None` has no epilogue work, so it takes the plain kernel rather than
        // paying the activation switch per output element.
        let act = match activation {
            GemmActivation::None => None,
            other => Some(other),
        };
        grouped_matmul_launch(self, a, b, group_offsets, act)
    }
}
