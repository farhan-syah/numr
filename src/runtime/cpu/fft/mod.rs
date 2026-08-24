//! FFT algorithm implementation for CPU runtime
//!
//! This module implements the FftAlgorithms trait for CpuClient using
//! the Stockham autosort algorithm.

mod real;
mod shift;

use super::{CpuClient, CpuRuntime, kernels};
use crate::algorithm::fft::{
    FftAlgorithms, FftDirection, FftNormalization, validate_fft_complex_dtype, validate_fft_size,
};
use crate::dtype::{Complex64, Complex128, DType};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl FftAlgorithms<CpuRuntime> for CpuClient {
    fn fft(
        &self,
        input: &Tensor<CpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "FFT requires at least 1D input".to_string(),
            });
        }
        self.fft_dim(input, -1, direction, norm)
    }

    fn fft_dim(
        &self,
        input: &Tensor<CpuRuntime>,
        dim: isize,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();
        validate_fft_complex_dtype(dtype, "fft")?;

        let ndim = input.ndim();
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        let n = input.shape()[dim];
        validate_fft_size(n, "fft")?;

        if dim == ndim - 1 && input.is_contiguous() {
            return self.fft_last_dim_contiguous(input, direction, norm);
        }

        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(dim, ndim - 1);

        let transposed = input.permute(&perm)?;
        let transposed_contig = transposed.contiguous()?;

        let result = self.fft_last_dim_contiguous(&transposed_contig, direction, norm)?;
        result.permute(&perm)
    }

    fn rfft(
        &self,
        input: &Tensor<CpuRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        real::rfft_impl(self, input, norm)
    }

    fn irfft(
        &self,
        input: &Tensor<CpuRuntime>,
        n: Option<usize>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        real::irfft_impl(self, input, n, norm)
    }

    fn fft2(
        &self,
        input: &Tensor<CpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "fft2 requires at least 2D input".to_string(),
            });
        }

        let result = self.fft_dim(input, -1, direction, norm)?;
        self.fft_dim(&result, -2, direction, norm)
    }

    fn rfft2(
        &self,
        input: &Tensor<CpuRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "rfft2 requires at least 2D input".to_string(),
            });
        }

        let result = self.rfft(input, norm)?;
        self.fft_dim(&result, -2, FftDirection::Forward, FftNormalization::None)
    }

    fn irfft2(
        &self,
        input: &Tensor<CpuRuntime>,
        s: Option<(usize, usize)>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "irfft2 requires at least 2D input".to_string(),
            });
        }

        let input_shape = input.shape();
        let (m, n) = if let Some((m, n)) = s {
            (m, n)
        } else {
            (input_shape[ndim - 2], 2 * (input_shape[ndim - 1] - 1))
        };

        let result = self.fft_dim(input, -2, FftDirection::Inverse, FftNormalization::None)?;

        if result.shape()[ndim - 2] != m {
            return Err(Error::InvalidArgument {
                arg: "s",
                reason: format!(
                    "M dimension mismatch: expected {}, got {}",
                    m,
                    result.shape()[ndim - 2]
                ),
            });
        }

        self.irfft(&result, Some(n), norm)
    }

    fn fftshift(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        shift::fftshift_impl(self, input)
    }

    fn ifftshift(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        shift::ifftshift_impl(self, input)
    }

    fn fftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        shift::fftfreq_impl(n, d, dtype, device)
    }

    fn rfftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        shift::rfftfreq_impl(n, d, dtype, device)
    }
}

impl CpuClient {
    /// FFT along last dimension for contiguous tensor
    fn fft_last_dim_contiguous(
        &self,
        input: &Tensor<CpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();
        let ndim = input.ndim();
        let n = input.shape()[ndim - 1];

        let inverse = matches!(direction, FftDirection::Inverse);
        let normalize_factor = norm.factor(direction, n);

        let output = Tensor::<CpuRuntime>::try_empty(input.shape(), dtype, &self.device)?;

        let batch_size: usize = input.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);
        let min_len = self.chunk_size_hint();

        let input_ptr = input.ptr();
        let output_ptr = output.ptr();

        match dtype {
            DType::Complex64 => {
                let input_slice: &[Complex64] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * n)
                };
                let output_slice: &mut [Complex64] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex64, batch_size * n)
                };

                if batch_size > 1 {
                    self.install_parallelism(|| unsafe {
                        kernels::stockham_fft_batched_c64(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor as f32,
                            min_len,
                        );
                    });
                } else {
                    unsafe {
                        kernels::stockham_fft_batched_c64(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor as f32,
                            min_len,
                        );
                    }
                }
            }
            DType::Complex128 => {
                let input_slice: &[Complex128] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex128, batch_size * n)
                };
                let output_slice: &mut [Complex128] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex128, batch_size * n)
                };

                if batch_size > 1 {
                    self.install_parallelism(|| unsafe {
                        kernels::stockham_fft_batched_c128(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor,
                            min_len,
                        );
                    });
                } else {
                    unsafe {
                        kernels::stockham_fft_batched_c128(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor,
                            min_len,
                        );
                    }
                }
            }
            _ => unreachable!(),
        }

        Ok(output)
    }
}
