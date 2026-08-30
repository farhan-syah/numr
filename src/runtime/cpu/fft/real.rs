//! Real FFT helpers (rfft, irfft)

use super::super::{CpuClient, CpuRuntime, kernels};
use crate::algorithm::fft::{
    FftDirection, FftNormalization, complex_dtype_for_real, real_dtype_for_complex,
    validate_fft_complex_dtype, validate_rfft_real_dtype,
};
use crate::dtype::{Complex64, Complex128, DType};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub(super) fn rfft_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    norm: FftNormalization,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = input.dtype();
    validate_rfft_real_dtype(dtype, "rfft")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "rfft requires at least 1D input".to_string(),
        });
    }

    let n = input.shape()[ndim - 1];
    // The CPU backend handles any size: power-of-two via Stockham, everything
    // else via Bluestein. Only an empty transform axis is invalid.
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "rfft requires size >= 1 along the last dim, got 0".to_string(),
        });
    }

    let input_contig = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };

    let output_dtype = complex_dtype_for_real(dtype)?;
    let normalize_factor = norm.factor(FftDirection::Forward, n);

    let mut out_shape = input_contig.shape().to_vec();
    out_shape[ndim - 1] = n / 2 + 1;

    let output = Tensor::<CpuRuntime>::empty(&out_shape, output_dtype, &client.device)?;

    // Unclamped: rank-1 already products to 1. Clamping a zero batch dim to 1 would
    // build a `from_raw_parts` slice longer than the zero-element allocation.
    let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
    #[cfg(feature = "rayon")]
    let min_len = client.rayon_min_len();

    let input_ptr = input_contig.ptr();
    let output_ptr = output.ptr();

    match dtype {
        DType::F32 => {
            let input_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(input_ptr as *const f32, batch_size * n) };
            let output_slice: &mut [Complex64] = unsafe {
                std::slice::from_raw_parts_mut(
                    output_ptr as *mut Complex64,
                    batch_size * (n / 2 + 1),
                )
            };
            let norm_f32 = normalize_factor as f32;

            client.install_parallelism(|| {
                #[cfg(feature = "rayon")]
                if batch_size > 1 {
                    output_slice
                        .par_chunks_mut(n / 2 + 1)
                        .enumerate()
                        .with_min_len(min_len)
                        .for_each(|(batch_idx, out_chunk)| {
                            let in_start = batch_idx * n;
                            unsafe {
                                kernels::rfft_c64(
                                    &input_slice[in_start..in_start + n],
                                    out_chunk,
                                    norm_f32,
                                );
                            }
                        });
                    return;
                }

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * n;
                    let out_start = batch_idx * (n / 2 + 1);
                    unsafe {
                        kernels::rfft_c64(
                            &input_slice[in_start..in_start + n],
                            &mut output_slice[out_start..out_start + n / 2 + 1],
                            norm_f32,
                        );
                    }
                }
            });
        }
        DType::F64 => {
            let input_slice: &[f64] =
                unsafe { std::slice::from_raw_parts(input_ptr as *const f64, batch_size * n) };
            let output_slice: &mut [Complex128] = unsafe {
                std::slice::from_raw_parts_mut(
                    output_ptr as *mut Complex128,
                    batch_size * (n / 2 + 1),
                )
            };

            client.install_parallelism(|| {
                #[cfg(feature = "rayon")]
                if batch_size > 1 {
                    output_slice
                        .par_chunks_mut(n / 2 + 1)
                        .enumerate()
                        .with_min_len(min_len)
                        .for_each(|(batch_idx, out_chunk)| {
                            let in_start = batch_idx * n;
                            unsafe {
                                kernels::rfft_c128(
                                    &input_slice[in_start..in_start + n],
                                    out_chunk,
                                    normalize_factor,
                                );
                            }
                        });
                    return;
                }

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * n;
                    let out_start = batch_idx * (n / 2 + 1);
                    unsafe {
                        kernels::rfft_c128(
                            &input_slice[in_start..in_start + n],
                            &mut output_slice[out_start..out_start + n / 2 + 1],
                            normalize_factor,
                        );
                    }
                }
            });
        }
        _ => unreachable!(),
    }

    Ok(output)
}

pub(super) fn irfft_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    n: Option<usize>,
    norm: FftNormalization,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = input.dtype();
    validate_fft_complex_dtype(dtype, "irfft")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "irfft requires at least 1D input".to_string(),
        });
    }

    let input_n = input.shape()[ndim - 1];
    if input_n == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "irfft requires size >= 1 along the last dim, got 0".to_string(),
        });
    }
    let output_n = n.unwrap_or(2 * (input_n - 1));

    if output_n / 2 + 1 != input_n {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: format!(
                "For irfft with n={}, input must have size {}, got {}",
                output_n,
                output_n / 2 + 1,
                input_n
            ),
        });
    }

    // The CPU backend handles any size: power-of-two via Stockham, everything
    // else via Bluestein. Only an empty transform axis is invalid.
    if output_n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "irfft requires output size >= 1, got 0".to_string(),
        });
    }

    let input_contig = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };

    let output_dtype = real_dtype_for_complex(dtype)?;
    let normalize_factor = norm.factor(FftDirection::Inverse, output_n);

    let mut out_shape = input_contig.shape().to_vec();
    out_shape[ndim - 1] = output_n;

    let output = Tensor::<CpuRuntime>::empty(&out_shape, output_dtype, &client.device)?;

    // Unclamped: rank-1 already products to 1. Clamping a zero batch dim to 1 would
    // build a `from_raw_parts` slice longer than the zero-element allocation.
    let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
    #[cfg(feature = "rayon")]
    let min_len = client.rayon_min_len();

    let input_ptr = input_contig.ptr();
    let output_ptr = output.ptr();

    match dtype {
        DType::Complex64 => {
            let input_slice: &[Complex64] = unsafe {
                std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * input_n)
            };
            let output_slice: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, batch_size * output_n)
            };
            let norm_f32 = normalize_factor as f32;

            client.install_parallelism(|| {
                #[cfg(feature = "rayon")]
                if batch_size > 1 {
                    output_slice
                        .par_chunks_mut(output_n)
                        .enumerate()
                        .with_min_len(min_len)
                        .for_each(|(batch_idx, out_chunk)| {
                            let in_start = batch_idx * input_n;
                            unsafe {
                                kernels::irfft_c64(
                                    &input_slice[in_start..in_start + input_n],
                                    out_chunk,
                                    norm_f32,
                                );
                            }
                        });
                    return;
                }

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * input_n;
                    let out_start = batch_idx * output_n;
                    unsafe {
                        kernels::irfft_c64(
                            &input_slice[in_start..in_start + input_n],
                            &mut output_slice[out_start..out_start + output_n],
                            norm_f32,
                        );
                    }
                }
            });
        }
        DType::Complex128 => {
            let input_slice: &[Complex128] = unsafe {
                std::slice::from_raw_parts(input_ptr as *const Complex128, batch_size * input_n)
            };
            let output_slice: &mut [f64] = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, batch_size * output_n)
            };

            client.install_parallelism(|| {
                #[cfg(feature = "rayon")]
                if batch_size > 1 {
                    output_slice
                        .par_chunks_mut(output_n)
                        .enumerate()
                        .with_min_len(min_len)
                        .for_each(|(batch_idx, out_chunk)| {
                            let in_start = batch_idx * input_n;
                            unsafe {
                                kernels::irfft_c128(
                                    &input_slice[in_start..in_start + input_n],
                                    out_chunk,
                                    normalize_factor,
                                );
                            }
                        });
                    return;
                }

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * input_n;
                    let out_start = batch_idx * output_n;
                    unsafe {
                        kernels::irfft_c128(
                            &input_slice[in_start..in_start + input_n],
                            &mut output_slice[out_start..out_start + output_n],
                            normalize_factor,
                        );
                    }
                }
            });
        }
        _ => unreachable!(),
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::fft::FftAlgorithms;
    use crate::runtime::cpu::CpuDevice;

    /// Whisper's mel front end needs exactly `n_fft = 400`, which is not a
    /// power of two. The kernel-level tests cover Bluestein directly; this one
    /// pins the PUBLIC `rfft`/`irfft` path that consumers actually call, which
    /// used to reject the size outright.
    const N: usize = 400;

    fn signal() -> Vec<f64> {
        (0..N)
            .map(|i| {
                let t = i as f64 / N as f64;
                (2.0 * std::f64::consts::PI * 3.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 47.0 * t).cos()
            })
            .collect()
    }

    /// Naive O(n^2) DFT, first `N/2 + 1` bins.
    fn naive_rdft(x: &[f64]) -> Vec<Complex128> {
        let n = x.len();
        (0..n / 2 + 1)
            .map(|k| {
                let mut acc = Complex128::new(0.0, 0.0);
                for (j, v) in x.iter().enumerate() {
                    let theta = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    acc += Complex128::new(v * theta.cos(), v * theta.sin());
                }
                acc
            })
            .collect()
    }

    #[test]
    fn rfft_at_400_matches_naive_dft() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x = signal();
        let input = Tensor::<CpuRuntime>::from_slice(&x, &[N], &device).expect("input");

        let spectrum = client
            .rfft(&input, FftNormalization::None)
            .expect("rfft must accept a non-power-of-two size");
        assert_eq!(spectrum.shape(), &[N / 2 + 1]);

        let got: Vec<Complex128> = spectrum.contiguous().expect("contiguous").to_vec();
        let expected = naive_rdft(&x);
        for (k, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a.re - b.re).abs() < 1e-9 && (a.im - b.im).abs() < 1e-9,
                "bin {k}: got ({}, {}), want ({}, {})",
                a.re,
                a.im,
                b.re,
                b.im
            );
        }
    }

    #[test]
    fn rfft_irfft_at_400_round_trips() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x = signal();
        let input = Tensor::<CpuRuntime>::from_slice(&x, &[N], &device).expect("input");

        // `Backward` is the numpy-matching default: the inverse divides by N,
        // so the round trip returns the original samples. `None` would return
        // `N * x` by design.
        let spectrum = client
            .rfft(&input, FftNormalization::Backward)
            .expect("rfft");
        let restored = client
            .irfft(&spectrum, Some(N), FftNormalization::Backward)
            .expect("irfft must accept a non-power-of-two size");
        assert_eq!(restored.shape(), &[N]);

        let got: Vec<f64> = restored.contiguous().expect("contiguous").to_vec();
        for (i, (a, b)) in got.iter().zip(x.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "sample {i}: got {a}, want {b}");
        }
    }

    /// A batched call must build one plan per batch and still produce the same
    /// spectrum each row would get on its own.
    #[test]
    fn batched_rfft_at_400_matches_per_row() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x = signal();
        let mut batched = Vec::with_capacity(3 * N);
        for row in 0..3 {
            batched.extend(x.iter().map(|v| v * (row as f64 + 1.0)));
        }
        let input = Tensor::<CpuRuntime>::from_slice(&batched, &[3, N], &device).expect("input");
        let spectrum = client.rfft(&input, FftNormalization::None).expect("rfft");
        assert_eq!(spectrum.shape(), &[3, N / 2 + 1]);

        let got: Vec<Complex128> = spectrum.contiguous().expect("contiguous").to_vec();
        let base = naive_rdft(&x);
        let bins = N / 2 + 1;
        for row in 0..3 {
            let scale = row as f64 + 1.0;
            for k in 0..bins {
                let a = got[row * bins + k];
                let want = base[k];
                assert!(
                    (a.re - want.re * scale).abs() < 1e-9 && (a.im - want.im * scale).abs() < 1e-9,
                    "row {row} bin {k}: got ({}, {}), want ({}, {})",
                    a.re,
                    a.im,
                    want.re * scale,
                    want.im * scale
                );
            }
        }
    }
}
