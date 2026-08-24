//! Optimized 1D convolution for polynomial multiplication
//!
//! This module provides convolution algorithms for polynomial operations:
//! - Direct convolution: O(n*m) for small polynomials
//! - FFT-based convolution: O(n log n) for large polynomials
//!
//! # Algorithm Selection
//!
//! The `convolve_impl` function automatically selects the optimal algorithm:
//! - n*m < 64: Direct convolution (cache-friendly, no padding overhead)
//! - n*m >= 64: FFT convolution (amortizes FFT cost for larger sizes)
//!
//! # Implementation
//!
//! All operations stay on-device using tensor operations only.
//! No GPU↔CPU transfers occur during computation.

use super::DTypeSupport;
use crate::algorithm::fft::{FftAlgorithms, FftNormalization};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, ComplexOps, IndexingOps, ReduceOps, ShapeOps, UtilityOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Threshold for switching from direct to FFT convolution.
/// Based on typical FFT overhead (padding, complex conversion).
const FFT_THRESHOLD: usize = 64;

/// Convolve two 1D tensors (polynomial multiplication).
///
/// Auto-selects direct vs FFT based on input sizes:
/// - Direct for n*m < 64 (O(n*m) complexity)
/// - FFT for n*m >= 64 (O(n log n) complexity)
///
/// # Arguments
///
/// * `client` - The runtime client
/// * `a` - First polynomial coefficients [a₀, a₁, ..., aₙ₋₁]
/// * `b` - Second polynomial coefficients [b₀, b₁, ..., bₘ₋₁]
/// * `dtype_support` - Backend dtype support flags
///
/// # Returns
///
/// Convolution result with length n+m-1, where:
/// `c[k]` = Σᵢ `a[i]` * `b[k-i]` for valid i
///
/// # Example
///
/// ```ignore
/// // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
/// let a = Tensor::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
/// let b = Tensor::try_from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();
/// let c = convolve_impl(&client, &a, &b, DTypeSupport::FULL)?;
/// // c = [3.0, 10.0, 8.0]
/// ```
pub fn convolve_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + ReduceOps<R>
        + FftAlgorithms<R>
        + ComplexOps<R>,
{
    let n_a = a.shape()[0];
    let n_b = b.shape()[0];
    let dtype = a.dtype();
    let device = client.device();

    // Handle edge cases
    if n_a == 0 || n_b == 0 {
        return Tensor::try_zeros(&[0], dtype, device);
    }

    // Select algorithm based on size
    if n_a * n_b < FFT_THRESHOLD {
        convolve_direct(client, a, b, dtype_support)
    } else {
        convolve_fft(client, a, b, dtype_support)
    }
}

/// Direct convolution via outer product + scatter_reduce.
///
/// Complexity: O(n*m) where n = len(a), m = len(b)
///
/// This method is efficient for small polynomials because:
/// - No padding overhead
/// - Cache-friendly access patterns
/// - Uses existing optimized tensor operations
fn convolve_direct<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + ReduceOps<R>,
{
    let n_a = a.shape()[0];
    let n_b = b.shape()[0];
    let dtype = a.dtype();
    let device = client.device();
    let out_len = n_a + n_b - 1;

    // Convolution via outer product + scatter_reduce
    //
    // 1. Compute outer product: outer[i,j] = a[i] * b[j]
    // 2. Create index tensor: indices[i,j] = i + j
    // 3. Flatten both and use scatter_reduce(Sum) to accumulate

    // Reshape a to [n_a, 1] and b to [1, n_b] for broadcasting
    let a_col = a.reshape(&[n_a, 1])?;
    let b_row = b.reshape(&[1, n_b])?;

    // Outer product via broadcasting: [n_a, 1] * [1, n_b] = [n_a, n_b]
    let outer = client.mul(&a_col, &b_row)?;

    // Create index tensor for output positions
    // indices[i,j] = i + j
    // Use backend-appropriate index dtype (I64 for CPU/CUDA, I32 for WebGPU)
    let index_dtype = dtype_support.index_dtype;
    let i_indices = client.arange(0.0, n_a as f64, 1.0, index_dtype)?;
    let j_indices = client.arange(0.0, n_b as f64, 1.0, index_dtype)?;

    let i_col = i_indices.reshape(&[n_a, 1])?;
    let j_row = j_indices.reshape(&[1, n_b])?;

    // Broadcast add to get output indices: [n_a, n_b]
    let out_indices = client.add(&i_col, &j_row)?;

    // Flatten both outer product and indices
    let outer_flat = outer.reshape(&[n_a * n_b])?;
    let indices_flat = out_indices.reshape(&[n_a * n_b])?;

    // Create output tensor of zeros
    let output = Tensor::try_zeros(&[out_len], dtype, device)?;

    // Use scatter_reduce with Sum to accumulate products at correct positions
    client.scatter_reduce(
        &output,
        0,
        &indices_flat,
        &outer_flat,
        crate::ops::ScatterReduceOp::Sum,
        true, // include_self (start with zeros)
    )
}

/// FFT-based convolution.
///
/// Complexity: O(n log n) where n = next_power_of_2(len(a) + len(b) - 1)
///
/// Algorithm:
/// 1. Pad a and b to length N = next_power_of_2(n_a + n_b - 1)
/// 2. A = rfft(a_padded)  // Real → Complex, shape [N/2+1]
/// 3. B = rfft(b_padded)  // Real → Complex, shape [N/2+1]
/// 4. C = A * B           // Element-wise complex multiply
/// 5. c = irfft(C, n=N)   // Complex → Real, shape [N]
/// 6. Return c[0:n_a+n_b-1]  // Trim padding
fn convolve_fft<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + FftAlgorithms<R>
        + ComplexOps<R>,
{
    let n_a = a.shape()[0];
    let n_b = b.shape()[0];
    let dtype = a.dtype();

    let out_len = n_a + n_b - 1;
    let fft_size = out_len.next_power_of_two();

    // Check FFT support for dtype
    dtype_support.check(dtype, "convolve_fft")?;

    // Pad a to fft_size
    let a_padded = if n_a < fft_size {
        let pad_amount = fft_size - n_a;
        // pad() takes pairs from last dimension: [pad_before, pad_after]
        client.pad(a, &[0, pad_amount], 0.0)?
    } else {
        a.clone()
    };

    // Pad b to fft_size
    let b_padded = if n_b < fft_size {
        let pad_amount = fft_size - n_b;
        client.pad(b, &[0, pad_amount], 0.0)?
    } else {
        b.clone()
    };

    // Forward FFT with no normalization (we'll normalize in irfft)
    let a_fft = client.rfft(&a_padded, FftNormalization::None)?;
    let b_fft = client.rfft(&b_padded, FftNormalization::None)?;

    // Element-wise complex multiplication
    // (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
    let c_fft = complex_mul(client, &a_fft, &b_fft)?;

    // Inverse FFT with backward normalization (divide by N)
    let c_full = client.irfft(&c_fft, Some(fft_size), FftNormalization::Backward)?;

    // Trim to actual output length
    if out_len < fft_size {
        // Use index_select to extract first out_len elements
        // Use backend-appropriate index dtype (I64 for CPU/CUDA, I32 for WebGPU)
        let indices = client.arange(0.0, out_len as f64, 1.0, dtype_support.index_dtype)?;
        client.index_select(&c_full, 0, &indices)
    } else {
        Ok(c_full)
    }
}

/// Element-wise complex multiplication.
///
/// For complex tensors a and b:
/// (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
///
/// This uses BinaryOps::mul which handles complex types via the Element trait.
fn complex_mul<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R>,
{
    // BinaryOps::mul handles complex multiplication natively
    client.mul(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn create_client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    #[test]
    fn test_convolve_direct_simple() {
        let (client, device) = create_client();

        // (1 + x) * (1 + x) = 1 + 2x + x²
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

        let c = convolve_direct(&client, &a, &b, DTypeSupport::FULL).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_convolve_direct_asymmetric() {
        let (client, device) = create_client();

        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();

        let c = convolve_direct(&client, &a, &b, DTypeSupport::FULL).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);
        assert!((data[2] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_convolve_fft_simple() {
        let (client, device) = create_client();

        // (1 + x) * (1 + x) = 1 + 2x + x²
        // Need size >= FFT_THRESHOLD to use FFT path
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

        let c = convolve_fft(&client, &a, &b, DTypeSupport::FULL).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve_impl_selects_direct() {
        let (client, device) = create_client();

        // Small inputs should use direct convolution
        // 2 * 2 = 4 < FFT_THRESHOLD
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();

        let c = convolve_impl(&client, &a, &b, DTypeSupport::FULL).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);
        assert!((data[2] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_convolve_impl_selects_fft() {
        let (client, device) = create_client();

        // Large inputs should use FFT convolution
        // 10 * 10 = 100 >= FFT_THRESHOLD
        let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..10).map(|i| i as f32 + 1.0).collect();

        let a = Tensor::<CpuRuntime>::try_from_slice(&a_data, &[10], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&b_data, &[10], &device).unwrap();

        let c = convolve_impl(&client, &a, &b, DTypeSupport::FULL).unwrap();

        // Result should have length 10 + 10 - 1 = 19
        assert_eq!(c.shape()[0], 19);
    }

    #[test]
    fn test_convolve_direct_vs_fft_equivalence() {
        let (client, device) = create_client();

        // Test that direct and FFT give the same results
        let a =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::try_from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[4], &device).unwrap();

        let c_direct = convolve_direct(&client, &a, &b, DTypeSupport::FULL).unwrap();
        let c_fft = convolve_fft(&client, &a, &b, DTypeSupport::FULL).unwrap();

        let direct_data: Vec<f32> = c_direct.to_vec();
        let fft_data: Vec<f32> = c_fft.to_vec();

        assert_eq!(direct_data.len(), fft_data.len());

        for (i, (d, f)) in direct_data.iter().zip(fft_data.iter()).enumerate() {
            assert!(
                (d - f).abs() < 1e-4,
                "Mismatch at index {}: direct={}, fft={}",
                i,
                d,
                f
            );
        }
    }

    #[test]
    fn test_convolve_f64() {
        let (client, device) = create_client();

        // Test F64 convolution
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[4.0f64, 5.0], &[2], &device).unwrap();

        // (1 + 2x + 3x²) * (4 + 5x) = 4 + 13x + 22x² + 15x³
        let c = convolve_impl(&client, &a, &b, DTypeSupport::FULL).unwrap();
        let data: Vec<f64> = c.to_vec();

        assert_eq!(data.len(), 4);
        assert!((data[0] - 4.0).abs() < 1e-12);
        assert!((data[1] - 13.0).abs() < 1e-12);
        assert!((data[2] - 22.0).abs() < 1e-12);
        assert!((data[3] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_empty() {
        let (client, device) = create_client();

        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[] as &[f32], &[0], &device).unwrap();

        let c = convolve_impl(&client, &a, &b, DTypeSupport::FULL).unwrap();
        assert_eq!(c.shape()[0], 0);
    }
}
