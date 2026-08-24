//! WebGPU implementation of FFT algorithms
//!
//! This module implements the [`FftAlgorithms`] trait for WebGPU using native
//! WGSL compute shaders. All algorithms follow the exact specification in
//! the trait documentation to ensure backend parity with CPU implementations.
//!
//! Native WGSL shaders are used - NO vendor dependencies.
//!
//! # Limitations
//!
//! - Only F32 (Complex64) is supported (WGSL doesn't support F64)
//! - FFT size limited to power of 2
//! - Non-last dimension FFT uses permute-based approach (transpose, FFT, transpose back)

use super::client::get_buffer;
use super::shaders::fft as kernels;
const MAX_WORKGROUP_FFT_SIZE: usize = 256;
use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::fft::{
    FftAlgorithms, FftDirection, FftNormalization, complex_dtype_for_real, real_dtype_for_complex,
    validate_fft_complex_dtype, validate_fft_size, validate_rfft_real_dtype,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::{Layout, Storage, Tensor};

/// Helper macro to get a GPU buffer from a pointer with proper error context.
macro_rules! get_buffer_or_err {
    ($ptr:expr, $name:expr) => {
        get_buffer($ptr).ok_or_else(|| {
            Error::Internal(format!(
                "Failed to get {} buffer from GPU allocation",
                $name
            ))
        })?
    };
}

impl FftAlgorithms<WgpuRuntime> for WgpuClient {
    fn fft(
        &self,
        input: &Tensor<WgpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<WgpuRuntime>> {
        // For multi-dim input, apply FFT to last dimension
        self.fft_dim(input, -1, direction, norm)
    }

    fn fft_dim(
        &self,
        input: &Tensor<WgpuRuntime>,
        dim: isize,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_fft_complex_dtype(input.dtype(), "wgpu_fft")?;

        let dtype = input.dtype();
        let device = self.device();

        // WGSL only supports F32 (Complex64)
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU FFT (only Complex64 supported)",
            });
        }

        // Normalize dimension
        let ndim = input.ndim();
        let dim_usize = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim_usize >= ndim {
            return Err(Error::InvalidDimension { dim, ndim });
        }

        // Ensure contiguous input
        let input_contig = input.contiguous()?;

        let n = input_contig.shape()[dim_usize];
        validate_fft_size(n, "wgpu_fft")?;

        // Calculate batch size (product of all dims except the FFT dim)
        let mut batch_size = 1usize;
        for (i, &s) in input_contig.shape().iter().enumerate() {
            if i != dim_usize {
                batch_size *= s;
            }
        }

        // Calculate scale factor
        let scale = match (&direction, &norm) {
            (FftDirection::Forward, FftNormalization::None) => 1.0,
            (FftDirection::Forward, FftNormalization::Backward) => 1.0,
            (FftDirection::Forward, FftNormalization::Ortho) => 1.0 / (n as f64).sqrt(),
            (FftDirection::Forward, FftNormalization::Forward) => 1.0 / n as f64,
            (FftDirection::Inverse, FftNormalization::None) => 1.0,
            (FftDirection::Inverse, FftNormalization::Forward) => 1.0,
            (FftDirection::Inverse, FftNormalization::Ortho) => 1.0 / (n as f64).sqrt(),
            (FftDirection::Inverse, FftNormalization::Backward) => 1.0 / n as f64,
        };

        let inverse = matches!(direction, FftDirection::Inverse);
        let log_n = (n as f64).log2() as u32;
        let total_elements = input_contig.numel();
        let element_size = dtype.size_in_bytes();

        // Allocate output buffer
        let output_size = total_elements * element_size;
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();
        let output_buffer = get_buffer_or_err!(output_ptr, "FFT output");

        let input_buffer = get_buffer_or_err!(input_contig.ptr(), "FFT input");

        // If FFT is on last dimension and data is contiguous, we can do batched FFT directly
        if dim_usize == ndim - 1 {
            // Create params buffer
            // FftParams: n, log_n, inverse, scale, batch_size, pad1, pad2, pad3
            let params: [u32; 8] = [
                n as u32,
                log_n,
                if inverse { 1 } else { 0 },
                (scale as f32).to_bits(),
                batch_size as u32,
                0,
                0,
                0,
            ];
            let params_buffer = self.create_uniform_buffer("fft_params", 32);
            self.write_buffer(&params_buffer, &params);

            if n <= MAX_WORKGROUP_FFT_SIZE {
                // Small FFT - use shared memory kernel
                kernels::launch_stockham_fft_batched(
                    self.pipeline_cache(),
                    &self.queue,
                    &input_buffer,
                    &output_buffer,
                    &params_buffer,
                    n,
                    batch_size,
                )?;
            } else {
                // Large FFT - use multi-stage approach
                // Need temp buffer for ping-pong
                let temp_guard = AllocGuard::new(self.allocator(), output_size)?;
                let temp_ptr = temp_guard.ptr();
                let temp_buffer = get_buffer_or_err!(temp_ptr, "FFT temp");

                // Copy input to temp buffer initially
                WgpuRuntime::copy_within_device(input_contig.ptr(), temp_ptr, output_size, device)?;

                // Run stages
                let mut use_temp_as_input = true;
                for stage in 0..log_n {
                    // Update params with current stage
                    let stage_params: [u32; 8] = [
                        n as u32,
                        stage, // Use log_n field for stage
                        if inverse { 1 } else { 0 },
                        1.0f32.to_bits(), // No scaling during stages
                        batch_size as u32,
                        0,
                        0,
                        0,
                    ];
                    self.write_buffer(&params_buffer, &stage_params);

                    let (src, dst) = if use_temp_as_input {
                        (&temp_buffer, &output_buffer)
                    } else {
                        (&output_buffer, &temp_buffer)
                    };

                    kernels::launch_stockham_fft_stage(
                        self.pipeline_cache(),
                        &self.queue,
                        src,
                        dst,
                        &params_buffer,
                        n,
                        batch_size,
                    )?;

                    use_temp_as_input = !use_temp_as_input;
                }

                // Apply scaling if needed
                if scale != 1.0 {
                    let scale_params: [u32; 8] = [
                        total_elements as u32,
                        0,
                        0,
                        (scale as f32).to_bits(),
                        0,
                        0,
                        0,
                        0,
                    ];
                    self.write_buffer(&params_buffer, &scale_params);

                    let final_src = if use_temp_as_input {
                        &temp_buffer
                    } else {
                        &output_buffer
                    };

                    // If final result is in temp, copy with scale to output
                    if use_temp_as_input {
                        kernels::launch_scale_complex(
                            self.pipeline_cache(),
                            &self.queue,
                            final_src,
                            &output_buffer,
                            &params_buffer,
                            total_elements,
                        )?;
                    }
                } else if use_temp_as_input {
                    // Result is in temp, copy to output
                    WgpuRuntime::copy_within_device(temp_ptr, output_ptr, output_size, device)?;
                }
            }
        } else {
            // FFT on non-last dimension - permute, FFT on last dim, permute back
            // Create permutation to move target dim to last position
            let mut perm: Vec<usize> = (0..ndim).collect();
            perm.swap(dim_usize, ndim - 1);

            // Permute input and make contiguous
            let transposed = input_contig.permute(&perm)?;
            let transposed_contig = transposed.contiguous()?;

            // Compute FFT on last dimension (recursive call)
            let result = self.fft_dim(&transposed_contig, -1, direction, norm)?;

            // Permute back to original dimension order
            return result.permute(&perm);
        }

        // Create output tensor
        let storage = unsafe {
            Storage::<WgpuRuntime>::from_ptr(output_guard.release(), total_elements, dtype, device)
        };
        let layout = Layout::contiguous(input_contig.shape());
        Ok(Tensor::from_parts(storage, layout))
    }

    fn rfft(
        &self,
        input: &Tensor<WgpuRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_rfft_real_dtype(input.dtype(), "wgpu_rfft")?;

        let dtype = input.dtype();
        let device = self.device();

        // WGSL only supports F32
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU rfft (only F32 supported)",
            });
        }

        let complex_dtype = complex_dtype_for_real(dtype)?;

        // Ensure contiguous input
        let input_contig = input.contiguous()?;

        let shape = input_contig.shape().to_vec();
        let n = *shape.last().ok_or_else(|| Error::InvalidArgument {
            arg: "input",
            reason: format!("expected at least 1D tensor, got shape {:?}", shape),
        })?;
        validate_fft_size(n, "wgpu_rfft")?;

        // Calculate batch size
        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1);

        // Output has shape [..., N/2 + 1]
        let out_n = n / 2 + 1;
        let mut out_shape = shape.clone();
        *out_shape.last_mut().unwrap() = out_n;

        let total_input = input_contig.numel();
        let total_output = out_shape.iter().product::<usize>();

        // Step 1: Pack real input to complex
        let complex_size = total_input * complex_dtype.size_in_bytes();
        let complex_guard = AllocGuard::new(self.allocator(), complex_size)?;
        let complex_ptr = complex_guard.ptr();
        let complex_buffer = get_buffer_or_err!(complex_ptr, "rfft complex");

        let input_buffer = get_buffer_or_err!(input_contig.ptr(), "rfft input");

        let pack_params: [u32; 4] = [n as u32, batch_size as u32, 0, 0];
        let params_buffer = self.create_uniform_buffer("rfft_params", 16);
        self.write_buffer(&params_buffer, &pack_params);

        kernels::launch_rfft_pack(
            self.pipeline_cache(),
            &self.queue,
            &input_buffer,
            &complex_buffer,
            &params_buffer,
            n,
            batch_size,
        )?;

        // Step 2: Full complex FFT
        let complex_storage = unsafe {
            Storage::<WgpuRuntime>::from_ptr(
                complex_guard.release(),
                total_input,
                complex_dtype,
                device,
            )
        };
        let complex_layout = Layout::contiguous(&shape);
        let complex_tensor = Tensor::from_parts(complex_storage, complex_layout);

        let fft_result = self.fft(&complex_tensor, FftDirection::Forward, norm)?;

        // Step 3: Truncate to N/2 + 1
        let output_size = total_output * complex_dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();
        let output_buffer = get_buffer_or_err!(output_ptr, "rfft output");

        let fft_buffer = get_buffer_or_err!(fft_result.ptr(), "rfft fft result");

        let truncate_params: [u32; 4] = [n as u32, out_n as u32, batch_size as u32, 0];
        self.write_buffer(&params_buffer, &truncate_params);

        kernels::launch_rfft_truncate(
            self.pipeline_cache(),
            &self.queue,
            &fft_buffer,
            &output_buffer,
            &params_buffer,
            out_n,
            batch_size,
        )?;

        // Create output tensor
        let storage = unsafe {
            Storage::<WgpuRuntime>::from_ptr(
                output_guard.release(),
                total_output,
                complex_dtype,
                device,
            )
        };
        let layout = Layout::contiguous(&out_shape);
        Ok(Tensor::from_parts(storage, layout))
    }

    fn irfft(
        &self,
        input: &Tensor<WgpuRuntime>,
        n: Option<usize>,
        norm: FftNormalization,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_fft_complex_dtype(input.dtype(), "wgpu_irfft")?;

        let dtype = input.dtype();
        let device = self.device();

        // WGSL only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU irfft (only Complex64 supported)",
            });
        }

        let real_dtype = real_dtype_for_complex(dtype)?;

        // Ensure contiguous input
        let input_contig = input.contiguous()?;

        let shape = input_contig.shape().to_vec();
        let half_n = *shape.last().ok_or_else(|| Error::InvalidArgument {
            arg: "input",
            reason: format!("expected at least 1D tensor, got shape {:?}", shape),
        })?;

        // Determine full FFT size
        let full_n = n.unwrap_or_else(|| 2 * (half_n - 1));
        validate_fft_size(full_n, "wgpu_irfft")?;

        // Calculate batch size
        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1);

        // Output shape
        let mut out_shape = shape.clone();
        *out_shape.last_mut().unwrap() = full_n;

        let total_output = out_shape.iter().product::<usize>();

        // Step 1: Extend Hermitian symmetric to full complex
        let extended_size = batch_size * full_n * dtype.size_in_bytes();
        let extended_guard = AllocGuard::new(self.allocator(), extended_size)?;
        let extended_ptr = extended_guard.ptr();
        let extended_buffer = get_buffer_or_err!(extended_ptr, "irfft extended");

        let input_buffer = get_buffer_or_err!(input_contig.ptr(), "irfft input");

        let extend_params: [u32; 4] = [full_n as u32, half_n as u32, batch_size as u32, 0];
        let params_buffer = self.create_uniform_buffer("irfft_params", 16);
        self.write_buffer(&params_buffer, &extend_params);

        kernels::launch_hermitian_extend(
            self.pipeline_cache(),
            &self.queue,
            &input_buffer,
            &extended_buffer,
            &params_buffer,
            full_n,
            batch_size,
        )?;

        // Step 2: Full inverse FFT
        let extended_storage = unsafe {
            Storage::<WgpuRuntime>::from_ptr(
                extended_guard.release(),
                batch_size * full_n,
                dtype,
                device,
            )
        };
        let mut extended_shape = shape.clone();
        *extended_shape.last_mut().unwrap() = full_n;
        let extended_layout = Layout::contiguous(&extended_shape);
        let extended_tensor = Tensor::from_parts(extended_storage, extended_layout);

        let ifft_result = self.fft(&extended_tensor, FftDirection::Inverse, norm)?;

        // Step 3: Extract real part
        let output_size = total_output * real_dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();
        let output_buffer = get_buffer_or_err!(output_ptr, "irfft output");

        let ifft_buffer = get_buffer_or_err!(ifft_result.ptr(), "irfft ifft result");

        let unpack_params: [u32; 4] = [full_n as u32, batch_size as u32, 0, 0];
        self.write_buffer(&params_buffer, &unpack_params);

        kernels::launch_irfft_unpack(
            self.pipeline_cache(),
            &self.queue,
            &ifft_buffer,
            &output_buffer,
            &params_buffer,
            full_n,
            batch_size,
        )?;

        // Create output tensor
        let storage = unsafe {
            Storage::<WgpuRuntime>::from_ptr(
                output_guard.release(),
                total_output,
                real_dtype,
                device,
            )
        };
        let layout = Layout::contiguous(&out_shape);
        Ok(Tensor::from_parts(storage, layout))
    }

    fn fft2(
        &self,
        input: &Tensor<WgpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<WgpuRuntime>> {
        // FFT on last dimension, then second-to-last
        let result = self.fft_dim(input, -1, direction, norm)?;
        self.fft_dim(&result, -2, direction, norm)
    }

    fn rfft2(
        &self,
        input: &Tensor<WgpuRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Real FFT on last dimension
        let result = self.rfft(input, norm)?;
        // Complex FFT on second-to-last dimension
        self.fft_dim(&result, -2, FftDirection::Forward, norm)
    }

    fn irfft2(
        &self,
        input: &Tensor<WgpuRuntime>,
        s: Option<(usize, usize)>,
        norm: FftNormalization,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Inverse FFT on second-to-last dimension
        let result = self.fft_dim(input, -2, FftDirection::Inverse, norm)?;
        // Inverse real FFT on last dimension
        let n = s.map(|(_, cols)| cols);
        self.irfft(&result, n, norm)
    }

    fn fftshift(&self, input: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_fft_complex_dtype(input.dtype(), "wgpu_fftshift")?;

        let dtype = input.dtype();
        let device = self.device();

        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU fftshift (only Complex64 supported)",
            });
        }

        let input_contig = input.contiguous()?;

        let shape = input_contig.shape().to_vec();
        let n = *shape.last().ok_or_else(|| Error::InvalidArgument {
            arg: "input",
            reason: format!("expected at least 1D tensor, got shape {:?}", shape),
        })?;

        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1);
        let total_elements = input_contig.numel();

        let output_size = total_elements * dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();
        let output_buffer = get_buffer_or_err!(output_ptr, "fftshift output");

        let input_buffer = get_buffer_or_err!(input_contig.ptr(), "fftshift input");

        let params: [u32; 4] = [n as u32, batch_size as u32, 0, 0];
        let params_buffer = self.create_uniform_buffer("fftshift_params", 16);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_fftshift(
            self.pipeline_cache(),
            &self.queue,
            &input_buffer,
            &output_buffer,
            &params_buffer,
            n,
            batch_size,
        )?;

        let storage = unsafe {
            Storage::<WgpuRuntime>::from_ptr(output_guard.release(), total_elements, dtype, device)
        };
        let layout = Layout::contiguous(&shape);
        Ok(Tensor::from_parts(storage, layout))
    }

    fn ifftshift(&self, input: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_fft_complex_dtype(input.dtype(), "wgpu_ifftshift")?;

        let dtype = input.dtype();
        let device = self.device();

        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU ifftshift (only Complex64 supported)",
            });
        }

        let input_contig = input.contiguous()?;

        let shape = input_contig.shape().to_vec();
        let n = *shape.last().ok_or_else(|| Error::InvalidArgument {
            arg: "input",
            reason: format!("expected at least 1D tensor, got shape {:?}", shape),
        })?;

        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1);
        let total_elements = input_contig.numel();

        let output_size = total_elements * dtype.size_in_bytes();
        let output_guard = AllocGuard::new(self.allocator(), output_size)?;
        let output_ptr = output_guard.ptr();
        let output_buffer = get_buffer_or_err!(output_ptr, "ifftshift output");

        let input_buffer = get_buffer_or_err!(input_contig.ptr(), "ifftshift input");

        let params: [u32; 4] = [n as u32, batch_size as u32, 0, 0];
        let params_buffer = self.create_uniform_buffer("ifftshift_params", 16);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_ifftshift(
            self.pipeline_cache(),
            &self.queue,
            &input_buffer,
            &output_buffer,
            &params_buffer,
            n,
            batch_size,
        )?;

        let storage = unsafe {
            Storage::<WgpuRuntime>::from_ptr(output_guard.release(), total_elements, dtype, device)
        };
        let layout = Layout::contiguous(&shape);
        Ok(Tensor::from_parts(storage, layout))
    }

    fn fftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &super::WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: "n must be positive".to_string(),
            });
        }

        // Generate frequency array directly (host-side arithmetic, then upload to GPU)
        // Frequencies: [0, 1, ..., N/2-1, -N/2, ..., -1] / (d*N)
        let scale = 1.0 / (d * n as f64);

        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..n)
                    .map(|i| {
                        let freq = if i < (n + 1) / 2 {
                            i as f64
                        } else {
                            (i as isize - n as isize) as f64
                        };
                        (freq * scale) as f32
                    })
                    .collect();
                Tensor::<WgpuRuntime>::from_slice(&data, &[n], device)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..n)
                    .map(|i| {
                        let freq = if i < (n + 1) / 2 {
                            i as f64
                        } else {
                            (i as isize - n as isize) as f64
                        };
                        freq * scale
                    })
                    .collect();
                Tensor::<WgpuRuntime>::from_slice(&data, &[n], device)
            }
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
        device: &super::WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: "n must be positive".to_string(),
            });
        }

        // Generate frequency array directly (host-side arithmetic, then upload to GPU)
        // Frequencies: [0, 1, ..., N/2] / (d*N)
        let output_len = n / 2 + 1;
        let scale = 1.0 / (d * n as f64);

        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..output_len).map(|i| (i as f64 * scale) as f32).collect();
                Tensor::<WgpuRuntime>::from_slice(&data, &[output_len], device)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..output_len).map(|i| i as f64 * scale).collect();
                Tensor::<WgpuRuntime>::from_slice(&data, &[output_len], device)
            }
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "rfftfreq",
            }),
        }
    }
}
