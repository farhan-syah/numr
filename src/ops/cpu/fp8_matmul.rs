//! CPU implementation of FP8 matrix multiplication operations.
//!
//! Fused kernel: reads FP8, converts to F32 inline during accumulation,
//! applies scaling, and writes output in the target dtype. No intermediate
//! tensor allocations.

use crate::dtype::{DType, FP8E4M3, FP8E5M2};
use crate::error::{Error, Result};
use crate::ops::Fp8MatmulOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// Validate FP8 matmul arguments.
fn validate_fp8_matmul(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
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

    let out_shape =
        crate::ops::matmul_output_shape(a_shape, b_shape).ok_or(Error::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        })?;

    let batch_size: usize = out_shape
        .iter()
        .take(out_shape.len().saturating_sub(2))
        .product();
    let batch_size = batch_size.max(1);

    Ok((out_shape, batch_size, m, k, n))
}

/// Fused FP8 matmul kernel: converts FP8→F32 inline during multiply-accumulate,
/// applies combined scale, writes output directly in target dtype.
///
/// `convert_a` and `convert_b` are FP8→f32 conversion functions.
fn fused_fp8_matmul_kernel(
    a_ptr: *const u8,
    b_ptr: *const u8,
    out_ptr: u64,
    convert_a: fn(u8) -> f32,
    convert_b: fn(u8) -> f32,
    combined_scale: f32,
    out_dtype: DType,
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let out_batch_stride = m * n;

    for batch in 0..batch_size {
        let a_base = unsafe { a_ptr.add(batch * a_batch_stride) };
        let b_base = unsafe { b_ptr.add(batch * b_batch_stride) };

        for i in 0..m {
            for j in 0..n {
                let mut acc: f32 = 0.0;
                for p in 0..k {
                    let a_val = convert_a(unsafe { *a_base.add(i * k + p) });
                    let b_val = convert_b(unsafe { *b_base.add(p * n + j) });
                    acc += a_val * b_val;
                }
                acc *= combined_scale;

                let out_idx = batch * out_batch_stride + i * n + j;
                match out_dtype {
                    DType::F32 => unsafe {
                        let ptr = out_ptr as *mut f32;
                        *ptr.add(out_idx) = acc;
                    },
                    #[cfg(feature = "f16")]
                    DType::F16 => unsafe {
                        let ptr = out_ptr as *mut half::f16;
                        *ptr.add(out_idx) = half::f16::from_f32(acc);
                    },
                    #[cfg(feature = "f16")]
                    DType::BF16 => unsafe {
                        let ptr = out_ptr as *mut half::bf16;
                        *ptr.add(out_idx) = half::bf16::from_f32(acc);
                    },
                    _ => {} // validated above
                }
            }
        }
    }
}

impl Fp8MatmulOps<CpuRuntime> for CpuClient {
    fn fp8_matmul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        let (out_shape, batch_size, m, k, n) =
            validate_fp8_matmul(a, b, DType::FP8E4M3, DType::FP8E4M3, out_dtype)?;

        let a_contig = crate::runtime::cpu::helpers::ensure_contiguous(a)?;
        let b_contig = crate::runtime::cpu::helpers::ensure_contiguous(b)?;
        let out = Tensor::<CpuRuntime>::try_empty(&out_shape, out_dtype, &self.device)?;

        fused_fp8_matmul_kernel(
            a_contig.ptr() as *const u8,
            b_contig.ptr() as *const u8,
            out.ptr(),
            |byte| FP8E4M3::from_bits(byte).to_f32(),
            |byte| FP8E4M3::from_bits(byte).to_f32(),
            scale_a * scale_b,
            out_dtype,
            batch_size,
            m,
            k,
            n,
        );

        Ok(out)
    }

    fn fp8_matmul_e5m2(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        let (out_shape, batch_size, m, k, n) =
            validate_fp8_matmul(a, b, DType::FP8E5M2, DType::FP8E4M3, out_dtype)?;

        let a_contig = crate::runtime::cpu::helpers::ensure_contiguous(a)?;
        let b_contig = crate::runtime::cpu::helpers::ensure_contiguous(b)?;
        let out = Tensor::<CpuRuntime>::try_empty(&out_shape, out_dtype, &self.device)?;

        fused_fp8_matmul_kernel(
            a_contig.ptr() as *const u8,
            b_contig.ptr() as *const u8,
            out.ptr(),
            |byte| FP8E5M2::from_bits(byte).to_f32(),
            |byte| FP8E4M3::from_bits(byte).to_f32(),
            scale_a * scale_b,
            out_dtype,
            batch_size,
            m,
            k,
            n,
        );

        Ok(out)
    }
}
