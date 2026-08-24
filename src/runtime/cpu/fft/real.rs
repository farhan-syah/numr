//! Real FFT helpers (rfft, irfft)

use super::super::{CpuClient, CpuRuntime, kernels};
use crate::algorithm::fft::{
    FftDirection, FftNormalization, complex_dtype_for_real, real_dtype_for_complex,
    validate_fft_complex_dtype, validate_fft_size, validate_rfft_real_dtype,
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
    validate_fft_size(n, "rfft")?;

    let input_contig = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };

    let output_dtype = complex_dtype_for_real(dtype)?;
    let normalize_factor = norm.factor(FftDirection::Forward, n);

    let mut out_shape = input_contig.shape().to_vec();
    out_shape[ndim - 1] = n / 2 + 1;

    let output = Tensor::<CpuRuntime>::try_empty(&out_shape, output_dtype, &client.device)?;

    let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);
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

    validate_fft_size(output_n, "irfft")?;

    let input_contig = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };

    let output_dtype = real_dtype_for_complex(dtype)?;
    let normalize_factor = norm.factor(FftDirection::Inverse, output_n);

    let mut out_shape = input_contig.shape().to_vec();
    out_shape[ndim - 1] = output_n;

    let output = Tensor::<CpuRuntime>::try_empty(&out_shape, output_dtype, &client.device)?;

    let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);
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
