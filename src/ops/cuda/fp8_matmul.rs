//! CUDA implementation of FP8 matrix multiplication operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Fp8MatmulOps, matmul_output_shape};
use crate::runtime::cuda::kernels::{
    launch_fp8_matmul_e4m3, launch_fp8_matmul_e4m3_batched, launch_fp8_matmul_e5m2,
    launch_fp8_matmul_e5m2_batched,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Validate FP8 matmul inputs and extract dimensions.
fn validate_and_extract(
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    expected_a_dtype: DType,
    expected_b_dtype: DType,
    out_dtype: DType,
) -> Result<(Vec<usize>, usize, usize, usize, usize)> {
    if a.dtype() != expected_a_dtype {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: expected_a_dtype,
        });
    }
    if b.dtype() != expected_b_dtype {
        return Err(Error::DTypeMismatch {
            lhs: b.dtype(),
            rhs: expected_b_dtype,
        });
    }
    match out_dtype {
        DType::F32 | DType::F16 | DType::BF16 => {}
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: out_dtype,
                op: "fp8_matmul output",
            });
        }
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(Error::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let k_b = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];

    if k != k_b {
        return Err(Error::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }

    let out_shape = matmul_output_shape(a_shape, b_shape).ok_or(Error::ShapeMismatch {
        expected: a_shape.to_vec(),
        got: b_shape.to_vec(),
    })?;

    let batch_size: usize = out_shape
        .iter()
        .take(out_shape.len().saturating_sub(2))
        .product();
    // No `.max(1)`: an unbatched matmul takes 0 dims and already products to 1, so
    // a clamp would only fabricate a batch for a genuinely zero batch dim.

    Ok((out_shape, batch_size, m, k, n))
}

impl Fp8MatmulOps<CudaRuntime> for CudaClient {
    fn fp8_matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        let (out_shape, batch_size, m, k, n) =
            validate_and_extract(a, b, DType::FP8E4M3, DType::FP8E4M3, out_dtype)?;

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, out_dtype, &self.device)?;

        // A zero-element output has nothing to compute, and the launcher takes its
        // grid from `ceil(n / TILE_N)`, `ceil(m / TILE_M)` and the batch count, so a
        // zero `m`, `n` or batch is a launch error rather than a wrong answer.
        if out.numel() == 0 {
            return Ok(out);
        }

        // A zero-length contraction leaves a NON-empty output whose every element
        // sums over no term. CPU answers zeros; the kernel would read off the end of
        // the empty operands.
        if k == 0 {
            return Tensor::<CudaRuntime>::zeros(&out_shape, out_dtype, &self.device);
        }

        unsafe {
            if batch_size > 1 {
                launch_fp8_matmul_e4m3_batched(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    out_dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    out.ptr(),
                    scale_a,
                    scale_b,
                    batch_size,
                    m,
                    n,
                    k,
                )?;
            } else {
                launch_fp8_matmul_e4m3(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    out_dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    out.ptr(),
                    scale_a,
                    scale_b,
                    m,
                    n,
                    k,
                )?;
            }
        }

        Ok(out)
    }

    fn fp8_matmul_e5m2(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        let (out_shape, batch_size, m, k, n) =
            validate_and_extract(a, b, DType::FP8E5M2, DType::FP8E4M3, out_dtype)?;

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, out_dtype, &self.device)?;

        // A zero-element output has nothing to compute, and the launcher takes its
        // grid from `ceil(n / TILE_N)`, `ceil(m / TILE_M)` and the batch count, so a
        // zero `m`, `n` or batch is a launch error rather than a wrong answer.
        if out.numel() == 0 {
            return Ok(out);
        }

        // A zero-length contraction leaves a NON-empty output whose every element
        // sums over no term. CPU answers zeros; the kernel would read off the end of
        // the empty operands.
        if k == 0 {
            return Tensor::<CudaRuntime>::zeros(&out_shape, out_dtype, &self.device);
        }

        unsafe {
            if batch_size > 1 {
                launch_fp8_matmul_e5m2_batched(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    out_dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    out.ptr(),
                    scale_a,
                    scale_b,
                    batch_size,
                    m,
                    n,
                    k,
                )?;
            } else {
                launch_fp8_matmul_e5m2(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    out_dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    out.ptr(),
                    scale_a,
                    scale_b,
                    m,
                    n,
                    k,
                )?;
            }
        }

        Ok(out)
    }
}
