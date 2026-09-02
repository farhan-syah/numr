//! CPU implementation of `GroupedMatmulOps`.
//!
//! Each group is an independent dense matmul, so this slices the rows and
//! delegates to the existing CPU GEMM rather than carrying a second kernel.
//!
//! `#[path]`-included into `runtime::cpu::ops`, so `super` here is that module.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{GemmActivation, GroupedMatmulOps, MatmulOps, ShapeOps};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// Validates the grouped shapes and returns `(total_rows, k, n, num_groups)`.
fn validate_grouped(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    group_offsets: &Tensor<CpuRuntime>,
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
    if a.dtype() != b.dtype() {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("a is {:?} but b is {:?}", a.dtype(), b.dtype()),
        });
    }
    if !matches!(a.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("grouped matmul supports F32/F16/BF16, got {:?}", a.dtype()),
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

fn grouped_matmul_cpu(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    group_offsets: &Tensor<CpuRuntime>,
    activation: Option<GemmActivation>,
) -> Result<Tensor<CpuRuntime>> {
    let (total_rows, _k, n, num_groups) = validate_grouped(a, b, group_offsets)?;
    let device = a.device();

    if total_rows == 0 || num_groups == 0 {
        return Ok(Tensor::<CpuRuntime>::zeros(
            &[total_rows, n],
            a.dtype(),
            device,
        )?);
    }

    let offsets = group_offsets.to_vec::<i32>();

    // The groups partition the rows in order, so concatenating their results is
    // the output. Building it this way keeps the whole path dtype-generic
    // rather than routing every dtype through an f32 host buffer.
    let mut parts: Vec<Tensor<CpuRuntime>> = Vec::with_capacity(num_groups);
    for g in 0..num_groups {
        let start = offsets[g].max(0) as usize;
        let end = offsets[g + 1].max(0) as usize;
        if end <= start || end > total_rows {
            continue;
        }
        let rows = end - start;

        let a_g = a.narrow(0, start, rows)?;
        // `[1, k, n]` down to `[k, n]`: the group's own weight matrix.
        let b_g = b.narrow(0, g, 1)?.squeeze(Some(0));
        let c_g = client.matmul(&a_g, &b_g)?;
        parts.push(match activation {
            Some(act) => apply_activation(client, &c_g, act)?,
            None => c_g,
        });
    }

    if parts.is_empty() {
        return Ok(Tensor::<CpuRuntime>::zeros(
            &[total_rows, n],
            a.dtype(),
            device,
        )?);
    }

    let refs: Vec<&Tensor<CpuRuntime>> = parts.iter().collect();
    client.cat(&refs, 0)
}

/// Applies the activation with the same math every other backend's epilogue
/// uses, so the grouped path cannot disagree with the dense one.
fn apply_activation(
    client: &CpuClient,
    t: &Tensor<CpuRuntime>,
    activation: GemmActivation,
) -> Result<Tensor<CpuRuntime>> {
    use crate::ops::{ActivationOps, UnaryOps};
    match activation {
        GemmActivation::None => Ok(t.clone()),
        GemmActivation::ReLU => client.relu(t),
        GemmActivation::GELU => client.gelu(t),
        GemmActivation::SiLU => client.silu(t),
        GemmActivation::Sigmoid => client.sigmoid(t),
        GemmActivation::Tanh => client.tanh(t),
    }
}

impl GroupedMatmulOps<CpuRuntime> for CpuClient {
    fn grouped_matmul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        group_offsets: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        grouped_matmul_cpu(self, a, b, group_offsets, None)
    }

    fn grouped_matmul_activation(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        group_offsets: &Tensor<CpuRuntime>,
        activation: GemmActivation,
    ) -> Result<Tensor<CpuRuntime>> {
        let act = match activation {
            GemmActivation::None => None,
            other => Some(other),
        };
        grouped_matmul_cpu(self, a, b, group_offsets, act)
    }
}
