//! FFT shift and frequency helpers

use super::super::{CpuClient, CpuRuntime, kernels};
use crate::dtype::{Complex64, Complex128, DType};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub(super) fn fftshift_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    shift_impl(client, input, false)
}

pub(super) fn ifftshift_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    shift_impl(client, input, true)
}

fn shift_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    inverse: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = input.dtype();
    let ndim = input.ndim();

    if ndim == 0 {
        return Ok(input.clone());
    }

    let input_contig = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };

    let n = input_contig.shape()[ndim - 1];
    let output = Tensor::<CpuRuntime>::empty(input_contig.shape(), dtype, &client.device)?;

    // Unclamped: rank-1 already products to 1. Clamping a zero batch dim to 1 would
    // build a `from_raw_parts` slice longer than the zero-element allocation.
    let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
    #[cfg(feature = "rayon")]
    let min_len = client.rayon_min_len();

    let input_ptr = input_contig.ptr();
    let output_ptr = output.ptr();

    let op_name = if inverse { "ifftshift" } else { "fftshift" };

    match dtype {
        DType::Complex64 => {
            let input_slice: &[Complex64] = unsafe {
                std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * n)
            };
            let output_slice: &mut [Complex64] = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut Complex64, batch_size * n)
            };

            let kernel_fn = if inverse {
                kernels::ifftshift_c64
            } else {
                kernels::fftshift_c64
            };

            client.install_parallelism(|| {
                #[cfg(feature = "rayon")]
                if batch_size > 1 {
                    output_slice
                        .par_chunks_mut(n)
                        .enumerate()
                        .with_min_len(min_len)
                        .for_each(|(batch_idx, out_chunk)| {
                            let start = batch_idx * n;
                            unsafe {
                                kernel_fn(&input_slice[start..start + n], out_chunk);
                            }
                        });
                    return;
                }

                for batch_idx in 0..batch_size {
                    let start = batch_idx * n;
                    unsafe {
                        kernel_fn(
                            &input_slice[start..start + n],
                            &mut output_slice[start..start + n],
                        );
                    }
                }
            });
        }
        DType::Complex128 => {
            let input_slice: &[Complex128] = unsafe {
                std::slice::from_raw_parts(input_ptr as *const Complex128, batch_size * n)
            };
            let output_slice: &mut [Complex128] = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut Complex128, batch_size * n)
            };

            let kernel_fn = if inverse {
                kernels::ifftshift_c128
            } else {
                kernels::fftshift_c128
            };

            client.install_parallelism(|| {
                #[cfg(feature = "rayon")]
                if batch_size > 1 {
                    output_slice
                        .par_chunks_mut(n)
                        .enumerate()
                        .with_min_len(min_len)
                        .for_each(|(batch_idx, out_chunk)| {
                            let start = batch_idx * n;
                            unsafe {
                                kernel_fn(&input_slice[start..start + n], out_chunk);
                            }
                        });
                    return;
                }

                for batch_idx in 0..batch_size {
                    let start = batch_idx * n;
                    unsafe {
                        kernel_fn(
                            &input_slice[start..start + n],
                            &mut output_slice[start..start + n],
                        );
                    }
                }
            });
        }
        _ => {
            return Err(Error::UnsupportedDType { dtype, op: op_name });
        }
    }

    Ok(output)
}

pub(super) fn fftfreq_impl(
    n: usize,
    d: f64,
    dtype: DType,
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "n must be positive".to_string(),
        });
    }

    let output = Tensor::<CpuRuntime>::empty(&[n], dtype, device)?;
    let scale = 1.0 / (d * n as f64);
    let output_ptr = output.ptr();

    match dtype {
        DType::F32 => {
            let output_slice: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, n) };

            #[allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
            for i in 0..n {
                let freq = if i < (n + 1) / 2 {
                    i as f64
                } else {
                    (i as isize - n as isize) as f64
                };
                output_slice[i] = (freq * scale) as f32;
            }
        }
        DType::F64 => {
            let output_slice: &mut [f64] =
                unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, n) };

            #[allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
            for i in 0..n {
                let freq = if i < (n + 1) / 2 {
                    i as f64
                } else {
                    (i as isize - n as isize) as f64
                };
                output_slice[i] = freq * scale;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "fftfreq",
            });
        }
    }

    Ok(output)
}

pub(super) fn rfftfreq_impl(
    n: usize,
    d: f64,
    dtype: DType,
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "n must be positive".to_string(),
        });
    }

    let output_len = n / 2 + 1;
    let output = Tensor::<CpuRuntime>::empty(&[output_len], dtype, device)?;
    let scale = 1.0 / (d * n as f64);
    let output_ptr = output.ptr();

    match dtype {
        DType::F32 => {
            let output_slice: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_len) };

            #[allow(clippy::needless_range_loop)]
            for i in 0..output_len {
                output_slice[i] = (i as f64 * scale) as f32;
            }
        }
        DType::F64 => {
            let output_slice: &mut [f64] =
                unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_len) };

            #[allow(clippy::needless_range_loop)]
            for i in 0..output_len {
                output_slice[i] = i as f64 * scale;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "rfftfreq",
            });
        }
    }

    Ok(output)
}
