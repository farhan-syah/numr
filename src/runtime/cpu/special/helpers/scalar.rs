//! Generic scalar tensor operation adapters
//!
//! Provides `apply_unary`, `apply_binary`, and `apply_ternary` helpers that
//! apply scalar functions element-wise over tensors with proper error handling.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cpu::{CpuDevice, CpuRuntime};
use crate::tensor::Tensor;

// ============================================================================
// Core dispatch helpers (all dtype variants delegate to these)
// ============================================================================

/// Internal: apply a unary f64→f64 function over any float tensor.
fn apply_unary_via_f64<F>(
    x: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64) -> f64,
{
    match x.dtype() {
        DType::F32 => {
            let data: Vec<f32> = x.to_vec();
            let result: Vec<f32> = data.iter().map(|&v| f(v as f64) as f32).collect();
            Tensor::from_slice(&result, x.shape(), device)
        }
        DType::F64 => {
            let data: Vec<f64> = x.to_vec();
            let result: Vec<f64> = data.iter().map(|&v| f(v)).collect();
            Tensor::from_slice(&result, x.shape(), device)
        }
        #[cfg(feature = "f16")]
        DType::F16 => {
            let data: Vec<half::f16> = x.to_vec();
            let result: Vec<half::f16> = data
                .iter()
                .map(|&v| half::f16::from_f64(f(v.to_f64())))
                .collect();
            Tensor::from_slice(&result, x.shape(), device)
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            let data: Vec<half::bf16> = x.to_vec();
            let result: Vec<half::bf16> = data
                .iter()
                .map(|&v| half::bf16::from_f64(f(v.to_f64())))
                .collect();
            Tensor::from_slice(&result, x.shape(), device)
        }
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => {
            let data: Vec<crate::dtype::FP8E4M3> = x.to_vec();
            let result: Vec<crate::dtype::FP8E4M3> = data
                .iter()
                .map(|&v| crate::dtype::FP8E4M3::from_f32(f(v.to_f32() as f64) as f32))
                .collect();
            Tensor::from_slice(&result, x.shape(), device)
        }
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => {
            let data: Vec<crate::dtype::FP8E5M2> = x.to_vec();
            let result: Vec<crate::dtype::FP8E5M2> = data
                .iter()
                .map(|&v| crate::dtype::FP8E5M2::from_f32(f(v.to_f32() as f64) as f32))
                .collect();
            Tensor::from_slice(&result, x.shape(), device)
        }
        _ => unreachable!("dtype validated by caller"),
    }
}

/// Internal: apply a binary (f64,f64)→f64 function over any two float tensors.
fn apply_binary_via_f64<F>(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64, f64) -> f64,
{
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    match a.dtype() {
        DType::F32 => {
            let a_data: Vec<f32> = a.to_vec();
            let b_data: Vec<f32> = b.to_vec();
            let result: Vec<f32> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&av, &bv)| f(av as f64, bv as f64) as f32)
                .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        DType::F64 => {
            let a_data: Vec<f64> = a.to_vec();
            let b_data: Vec<f64> = b.to_vec();
            let result: Vec<f64> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&av, &bv)| f(av, bv))
                .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        #[cfg(feature = "f16")]
        DType::F16 => {
            let a_data: Vec<half::f16> = a.to_vec();
            let b_data: Vec<half::f16> = b.to_vec();
            let result: Vec<half::f16> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&av, &bv)| half::f16::from_f64(f(av.to_f64(), bv.to_f64())))
                .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            let a_data: Vec<half::bf16> = a.to_vec();
            let b_data: Vec<half::bf16> = b.to_vec();
            let result: Vec<half::bf16> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&av, &bv)| half::bf16::from_f64(f(av.to_f64(), bv.to_f64())))
                .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => {
            let a_data: Vec<crate::dtype::FP8E4M3> = a.to_vec();
            let b_data: Vec<crate::dtype::FP8E4M3> = b.to_vec();
            let result: Vec<crate::dtype::FP8E4M3> =
                a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&av, &bv)| {
                        crate::dtype::FP8E4M3::from_f32(
                            f(av.to_f32() as f64, bv.to_f32() as f64) as f32
                        )
                    })
                    .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => {
            let a_data: Vec<crate::dtype::FP8E5M2> = a.to_vec();
            let b_data: Vec<crate::dtype::FP8E5M2> = b.to_vec();
            let result: Vec<crate::dtype::FP8E5M2> =
                a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&av, &bv)| {
                        crate::dtype::FP8E5M2::from_f32(
                            f(av.to_f32() as f64, bv.to_f32() as f64) as f32
                        )
                    })
                    .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        _ => unreachable!("dtype validated by caller"),
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Apply a unary scalar function element-wise over a tensor.
pub fn apply_unary<F>(
    x: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64) -> f64,
{
    apply_unary_via_f64(x, device, f)
}

/// Apply a binary scalar function element-wise over two tensors.
///
/// Both tensors must have matching shapes (broadcasting not supported).
pub fn apply_binary<F>(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64, f64) -> f64,
{
    apply_binary_via_f64(a, b, device, f)
}

/// Apply a ternary scalar function element-wise over three tensors.
///
/// All tensors must have matching shapes (broadcasting not supported).
pub fn apply_ternary<F>(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    c: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64, f64, f64) -> f64,
{
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    if a.shape() != c.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: c.shape().to_vec(),
        });
    }

    match a.dtype() {
        DType::F32 => {
            let a_data: Vec<f32> = a.to_vec();
            let b_data: Vec<f32> = b.to_vec();
            let c_data: Vec<f32> = c.to_vec();
            let result: Vec<f32> = a_data
                .iter()
                .zip(b_data.iter())
                .zip(c_data.iter())
                .map(|((&av, &bv), &cv)| f(av as f64, bv as f64, cv as f64) as f32)
                .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        DType::F64 => {
            let a_data: Vec<f64> = a.to_vec();
            let b_data: Vec<f64> = b.to_vec();
            let c_data: Vec<f64> = c.to_vec();
            let result: Vec<f64> = a_data
                .iter()
                .zip(b_data.iter())
                .zip(c_data.iter())
                .map(|((&av, &bv), &cv)| f(av, bv, cv))
                .collect();
            Tensor::from_slice(&result, a.shape(), device)
        }
        _ => unreachable!("dtype validated by caller"),
    }
}

/// Apply a unary scalar function with one extra i32 parameter.
pub fn apply_unary_with_int<F>(
    x: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    n: i32,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(i32, f64) -> f64,
{
    apply_unary_via_f64(x, device, |v| f(n, v))
}

/// Apply a unary scalar function with two extra i32 parameters.
pub fn apply_unary_with_two_ints<F>(
    x: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    n: i32,
    m: i32,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(i32, i32, f64) -> f64,
{
    apply_unary_via_f64(x, device, |v| f(n, m, v))
}

/// Apply a binary scalar function with two extra i32 parameters (for sph_harm).
pub fn apply_binary_with_two_ints<F>(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    n: i32,
    m: i32,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(i32, i32, f64, f64) -> f64,
{
    apply_binary_via_f64(a, b, device, |av, bv| f(n, m, av, bv))
}

/// Apply a unary scalar function with three extra f64 parameters (for hyp2f1).
pub fn apply_unary_with_three_f64s<F>(
    z: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    a: f64,
    b: f64,
    c: f64,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64, f64, f64, f64) -> f64,
{
    apply_unary_via_f64(z, device, |v| f(a, b, c, v))
}

/// Apply a unary scalar function with two extra f64 parameters (for hyp1f1).
pub fn apply_unary_with_two_f64s<F>(
    z: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    a: f64,
    b: f64,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64, f64, f64) -> f64,
{
    apply_unary_via_f64(z, device, |v| f(a, b, v))
}
