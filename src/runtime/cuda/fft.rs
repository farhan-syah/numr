//! CUDA implementation of FFT algorithms
//!
//! This module implements the [`FftAlgorithms`] trait for CUDA using native
//! Stockham FFT kernels. All algorithms follow the exact specification in
//! the trait documentation to ensure backend parity with CPU implementations.
//!
//! Native CUDA kernels are used - NO cuFFT dependency.

use super::CudaRuntime;
use super::client::CudaClient;
use super::kernels;
use crate::algorithm::fft::{
    FftAlgorithms, FftDirection, FftNormalization, complex_dtype_for_real, real_dtype_for_complex,
    validate_fft_complex_dtype, validate_fft_size, validate_rfft_real_dtype,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::{Layout, Storage, Tensor};

impl FftAlgorithms<CudaRuntime> for CudaClient {
    fn fft(
        &self,
        input: &Tensor<CudaRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate input
        let dtype = input.dtype();
        validate_fft_complex_dtype(dtype, "fft")?;

        let ndim = input.ndim();
        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "FFT requires at least 1D input".to_string(),
            });
        }

        let n = input.shape()[ndim - 1];
        validate_fft_size(n, "fft")?;

        // Ensure contiguous input
        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()?
        };

        // Calculate batch size and scale factor
        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);
        let scale = norm.factor(direction, n);
        let inverse = direction == FftDirection::Inverse;

        let device = self.device();
        let total_elements = batch_size * n;

        // Allocate output
        let output_size = total_elements * dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();

        let input_ptr = input_contig.ptr();

        // Choose small FFT (shared memory) or large FFT (multi-stage) based on size
        if n <= kernels::MAX_SHARED_MEM_FFT_SIZE {
            // Small FFT: single kernel with shared memory
            unsafe {
                kernels::launch_stockham_fft_batched(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    input_ptr,
                    output_ptr,
                    n,
                    batch_size,
                    inverse,
                    scale,
                )?;
            }
        } else {
            // Large FFT: multi-stage with ping-pong buffers
            let log_n = (n as f64).log2() as usize;

            // Allocate temp buffer for ping-pong
            let temp_guard = AllocGuard::new(self.allocator(), output_size)?;
            let temp_ptr = temp_guard.ptr();

            // Copy input to one buffer
            CudaRuntime::copy_within_device(input_ptr, output_ptr, output_size, device)?;

            let mut src_ptr = output_ptr;
            let mut dst_ptr = temp_ptr;

            // Run all stages
            for stage in 0..log_n {
                unsafe {
                    kernels::launch_stockham_fft_stage(
                        self.context(),
                        self.stream(),
                        device.index,
                        dtype,
                        src_ptr,
                        dst_ptr,
                        n,
                        stage,
                        batch_size,
                        inverse,
                    )?;
                }
                std::mem::swap(&mut src_ptr, &mut dst_ptr);
            }

            // Apply scale if needed
            if (scale - 1.0).abs() > 1e-10 {
                unsafe {
                    kernels::launch_scale_complex(
                        self.context(),
                        self.stream(),
                        device.index,
                        dtype,
                        src_ptr,
                        scale,
                        total_elements,
                    )?;
                }
            }

            // If result is in temp buffer, copy to output
            if src_ptr == temp_ptr {
                CudaRuntime::copy_within_device(temp_ptr, output_ptr, output_size, device)?;
            }
        }

        self.synchronize();

        // Create output tensor
        let output = unsafe {
            Self::tensor_from_raw(output_guard.release(), input_contig.shape(), dtype, device)
        };

        Ok(output)
    }

    fn fft_dim(
        &self,
        input: &Tensor<CudaRuntime>,
        dim: isize,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();
        validate_fft_complex_dtype(dtype, "fft_dim")?;

        let ndim = input.ndim();
        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "FFT requires at least 1D input".to_string(),
            });
        }

        // Normalize dimension
        let dim_usize = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim_usize >= ndim {
            return Err(Error::InvalidDimension { dim, ndim });
        }

        // If dim is already the last dimension, use regular fft
        if dim_usize == ndim - 1 {
            return self.fft(input, direction, norm);
        }

        // Otherwise, permute to move target dim to last, compute FFT, permute back
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(dim_usize);
        perm.push(dim_usize);

        let permuted = input.permute(&perm)?;
        let fft_result = self.fft(&permuted, direction, norm)?;

        // Inverse permutation
        let mut inv_perm = vec![0; ndim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        fft_result.permute(&inv_perm)
    }

    fn rfft(
        &self,
        input: &Tensor<CudaRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<CudaRuntime>> {
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

        let complex_dtype = complex_dtype_for_real(dtype)?;
        let device = self.device();

        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        // Allocate complex buffer for full FFT
        let complex_size = batch_size * n * complex_dtype.size_in_bytes();
        let complex_guard = AllocGuard::new(self.allocator(), complex_size)?;
        let complex_ptr = complex_guard.ptr();

        // Pack real to complex (zero imaginary parts)
        unsafe {
            kernels::launch_rfft_pack(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                input_contig.ptr(),
                complex_ptr,
                n,
                batch_size,
            )?;
        }

        // Compute full complex FFT
        let scale = norm.factor(FftDirection::Forward, n);

        if n <= kernels::MAX_SHARED_MEM_FFT_SIZE {
            // In-place for small FFTs
            let temp_guard = AllocGuard::new(self.allocator(), complex_size)?;
            let temp_ptr = temp_guard.ptr();
            unsafe {
                kernels::launch_stockham_fft_batched(
                    self.context(),
                    self.stream(),
                    device.index,
                    complex_dtype,
                    complex_ptr,
                    temp_ptr,
                    n,
                    batch_size,
                    false,
                    scale,
                )?;
            }
            CudaRuntime::copy_within_device(temp_ptr, complex_ptr, complex_size, device)?;
        } else {
            // Multi-stage for large FFTs
            let log_n = (n as f64).log2() as usize;
            let temp_guard = AllocGuard::new(self.allocator(), complex_size)?;
            let temp_ptr = temp_guard.ptr();

            let mut src_ptr = complex_ptr;
            let mut dst_ptr = temp_ptr;

            for stage in 0..log_n {
                unsafe {
                    kernels::launch_stockham_fft_stage(
                        self.context(),
                        self.stream(),
                        device.index,
                        complex_dtype,
                        src_ptr,
                        dst_ptr,
                        n,
                        stage,
                        batch_size,
                        false,
                    )?;
                }
                std::mem::swap(&mut src_ptr, &mut dst_ptr);
            }

            if (scale - 1.0).abs() > 1e-10 {
                unsafe {
                    kernels::launch_scale_complex(
                        self.context(),
                        self.stream(),
                        device.index,
                        complex_dtype,
                        src_ptr,
                        scale,
                        batch_size * n,
                    )?;
                }
            }

            if src_ptr != complex_ptr {
                CudaRuntime::copy_within_device(src_ptr, complex_ptr, complex_size, device)?;
            }
        }

        // Truncate to N/2 + 1 elements (Hermitian symmetry)
        let output_n = n / 2 + 1;
        let output_size = batch_size * output_n * complex_dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();

        unsafe {
            kernels::launch_rfft_truncate(
                self.context(),
                self.stream(),
                device.index,
                complex_dtype,
                complex_ptr,
                output_ptr,
                n,
                output_n,
                batch_size,
            )?;
        }
        self.synchronize();

        // Build output shape
        let mut out_shape = input_contig.shape().to_vec();
        out_shape[ndim - 1] = output_n;

        let output = unsafe {
            Self::tensor_from_raw(output_guard.release(), &out_shape, complex_dtype, device)
        };

        Ok(output)
    }

    fn irfft(
        &self,
        input: &Tensor<CudaRuntime>,
        n: Option<usize>,
        norm: FftNormalization,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let output_n = n.unwrap_or_else(|| 2 * (input_n - 1));
        validate_fft_size(output_n, "irfft")?;

        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()?
        };

        let device = self.device();
        let real_dtype = real_dtype_for_complex(dtype)?;

        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        // Extend Hermitian spectrum to full spectrum
        let full_complex_size = batch_size * output_n * dtype.size_in_bytes();
        let full_complex_guard = AllocGuard::new(self.allocator(), full_complex_size)?;
        let full_complex_ptr = full_complex_guard.ptr();

        unsafe {
            kernels::launch_hermitian_extend(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                input_contig.ptr(),
                full_complex_ptr,
                input_n,
                output_n,
                batch_size,
            )?;
        }

        // Compute inverse FFT (scale handled separately)
        if output_n <= kernels::MAX_SHARED_MEM_FFT_SIZE {
            let temp_guard = AllocGuard::new(self.allocator(), full_complex_size)?;
            let temp_ptr = temp_guard.ptr();
            unsafe {
                kernels::launch_stockham_fft_batched(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    full_complex_ptr,
                    temp_ptr,
                    output_n,
                    batch_size,
                    true,
                    1.0, // Scale applied in unpack
                )?;
            }
            CudaRuntime::copy_within_device(temp_ptr, full_complex_ptr, full_complex_size, device)?;
        } else {
            let log_n = (output_n as f64).log2() as usize;
            let temp_guard = AllocGuard::new(self.allocator(), full_complex_size)?;
            let temp_ptr = temp_guard.ptr();

            let mut src_ptr = full_complex_ptr;
            let mut dst_ptr = temp_ptr;

            for stage in 0..log_n {
                unsafe {
                    kernels::launch_stockham_fft_stage(
                        self.context(),
                        self.stream(),
                        device.index,
                        dtype,
                        src_ptr,
                        dst_ptr,
                        output_n,
                        stage,
                        batch_size,
                        true,
                    )?;
                }
                std::mem::swap(&mut src_ptr, &mut dst_ptr);
            }

            if src_ptr != full_complex_ptr {
                CudaRuntime::copy_within_device(
                    src_ptr,
                    full_complex_ptr,
                    full_complex_size,
                    device,
                )?;
            }
        }

        // Unpack to real with scaling
        let scale = norm.factor(FftDirection::Inverse, output_n);
        let output_size = batch_size * output_n * real_dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();

        unsafe {
            kernels::launch_irfft_unpack(
                self.context(),
                self.stream(),
                device.index,
                real_dtype,
                full_complex_ptr,
                output_ptr,
                output_n,
                scale,
                batch_size,
            )?;
        }
        self.synchronize();

        // Build output shape
        let mut out_shape = input_contig.shape().to_vec();
        out_shape[ndim - 1] = output_n;

        let output = unsafe {
            Self::tensor_from_raw(output_guard.release(), &out_shape, real_dtype, device)
        };

        Ok(output)
    }

    fn fft2(
        &self,
        input: &Tensor<CudaRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CudaRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "fft2 requires at least 2D input".to_string(),
            });
        }

        // FFT along last two dimensions (dim=-1, then dim=-2)
        // Apply normalization to both dimensions
        let result = self.fft_dim(input, -1, direction, norm)?;
        self.fft_dim(&result, -2, direction, norm)
    }

    fn rfft2(
        &self,
        input: &Tensor<CudaRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<CudaRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "rfft2 requires at least 2D input".to_string(),
            });
        }

        // rfft along last dimension, then fft along second-to-last
        let result = self.rfft(input, norm)?;
        self.fft_dim(&result, -2, FftDirection::Forward, norm)
    }

    fn irfft2(
        &self,
        input: &Tensor<CudaRuntime>,
        s: Option<(usize, usize)>,
        norm: FftNormalization,
    ) -> Result<Tensor<CudaRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "irfft2 requires at least 2D input".to_string(),
            });
        }

        // Inverse FFT along second-to-last dimension first
        let result = self.fft_dim(input, -2, FftDirection::Inverse, norm)?;

        // Then irfft along last dimension
        let n = s.map(|(_, n)| n);
        self.irfft(&result, n, norm)
    }

    fn fftshift(&self, input: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();
        validate_fft_complex_dtype(dtype, "fftshift")?;

        let ndim = input.ndim();
        if ndim == 0 {
            return Ok(input.clone());
        }

        let n = input.shape()[ndim - 1];
        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()?
        };

        let device = self.device();
        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        let total_elements = batch_size * n;
        let output_size = total_elements * dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();

        unsafe {
            kernels::launch_fftshift(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                input_contig.ptr(),
                output_ptr,
                n,
                batch_size,
            )?;
        }

        self.synchronize();

        let output = unsafe {
            Self::tensor_from_raw(output_guard.release(), input_contig.shape(), dtype, device)
        };

        Ok(output)
    }

    fn ifftshift(&self, input: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();
        validate_fft_complex_dtype(dtype, "ifftshift")?;

        let ndim = input.ndim();
        if ndim == 0 {
            return Ok(input.clone());
        }

        let n = input.shape()[ndim - 1];
        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()?
        };

        let device = self.device();
        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        let total_elements = batch_size * n;
        let output_size = total_elements * dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();

        unsafe {
            kernels::launch_ifftshift(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                input_contig.ptr(),
                output_ptr,
                n,
                batch_size,
            )?;
        }

        self.synchronize();

        let output = unsafe {
            Self::tensor_from_raw(output_guard.release(), input_contig.shape(), dtype, device)
        };

        Ok(output)
    }

    fn fftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &super::CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_rfft_real_dtype(dtype, "fftfreq")?;

        // Generate frequencies on CPU and transfer
        // [0, 1, 2, ..., N/2-1, -N/2, ..., -1] / (d*N)
        let factor = 1.0 / (d * n as f64);
        let half = (n + 1) / 2;

        let freqs: Vec<f64> = (0..n)
            .map(|i| {
                if i < half {
                    i as f64 * factor
                } else {
                    (i as isize - n as isize) as f64 * factor
                }
            })
            .collect();

        match dtype {
            DType::F32 => {
                let freqs_f32: Vec<f32> = freqs.iter().map(|&x| x as f32).collect();
                Tensor::<CudaRuntime>::from_slice(&freqs_f32, &[n], device)
            }
            DType::F64 => Tensor::<CudaRuntime>::from_slice(&freqs, &[n], device),
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "fftfreq",
            }),
        }
    }

    fn rfftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &super::CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_rfft_real_dtype(dtype, "rfftfreq")?;

        // Generate frequencies on CPU and transfer
        // [0, 1, 2, ..., N/2] / (d*N)
        let factor = 1.0 / (d * n as f64);
        let out_n = n / 2 + 1;

        let freqs: Vec<f64> = (0..out_n).map(|i| i as f64 * factor).collect();

        match dtype {
            DType::F32 => {
                let freqs_f32: Vec<f32> = freqs.iter().map(|&x| x as f32).collect();
                Tensor::<CudaRuntime>::from_slice(&freqs_f32, &[out_n], device)
            }
            DType::F64 => Tensor::<CudaRuntime>::from_slice(&freqs, &[out_n], device),
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "rfftfreq",
            }),
        }
    }
}

// Helper method - reuse from linalg.rs pattern
impl CudaClient {
    /// Create a tensor from a raw CUDA GPU pointer for FFT operations.
    ///
    /// This wraps the tensor_from_raw helper defined in linalg.rs.
    #[allow(dead_code)]
    unsafe fn fft_tensor_from_raw(
        ptr: u64,
        shape: &[usize],
        dtype: DType,
        device: &super::CudaDevice,
    ) -> Tensor<CudaRuntime> {
        let len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let storage = unsafe { Storage::<CudaRuntime>::from_ptr(ptr, len, dtype, device) };
        let layout = Layout::contiguous(shape);
        Tensor::from_parts(storage, layout)
    }
}
