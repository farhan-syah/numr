//! Complex number operations for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ComplexOps;
use crate::ops::common::{validate_complex_real_inputs, validate_make_complex_inputs};
use crate::runtime::cuda::kernels::{
    launch_angle, launch_angle_real, launch_complex_div_real, launch_complex_mul_real, launch_conj,
    launch_fill_with_f64, launch_from_real_imag, launch_imag, launch_real,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl ComplexOps<CudaRuntime> for CudaClient {
    fn conj(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // For real types, conjugate is identity
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_conj(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn real(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // For real types, return copy
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        // Determine output dtype: Complex64 → F32, Complex128 → F64
        let out_dtype = match dtype {
            DType::Complex64 => DType::F32,
            DType::Complex128 => DType::F64,
            _ => return Err(Error::UnsupportedDType { dtype, op: "real" }),
        };

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device)?;

        unsafe {
            launch_real(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn imag(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // Determine output dtype
        let out_dtype = if dtype.is_complex() {
            match dtype {
                DType::Complex64 => DType::F32,
                DType::Complex128 => DType::F64,
                _ => return Err(Error::UnsupportedDType { dtype, op: "imag" }),
            }
        } else {
            // For real types, return zeros with same dtype
            dtype
        };

        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device)?;

        // For real types, fill with zeros
        if !dtype.is_complex() {
            unsafe {
                launch_fill_with_f64(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    out_dtype,
                    0.0,
                    out.ptr(),
                    out.numel(),
                )?;
            }
            return Ok(out);
        }

        // For complex types, extract imaginary part
        let a_contig = ensure_contiguous(a)?;

        unsafe {
            launch_imag(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn angle(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // Determine output dtype
        let out_dtype = if dtype.is_complex() {
            match dtype {
                DType::Complex64 => DType::F32,
                DType::Complex128 => DType::F64,
                _ => return Err(Error::UnsupportedDType { dtype, op: "angle" }),
            }
        } else {
            // For real types, return zeros with same dtype
            dtype
        };

        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device)?;
        let a_contig = ensure_contiguous(a)?;

        // For real types: angle(x) = 0 if x >= 0, π if x < 0
        if !dtype.is_complex() {
            match dtype {
                DType::F32 | DType::F64 => unsafe {
                    launch_angle_real(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        dtype,
                        a_contig.ptr(),
                        out.ptr(),
                        a.numel(),
                    )?;
                },
                _ => {
                    // For integer types, return zeros (π as integer doesn't make mathematical sense)
                    unsafe {
                        launch_fill_with_f64(
                            &self.context,
                            &self.stream,
                            self.device.index,
                            out_dtype,
                            0.0,
                            out.ptr(),
                            out.numel(),
                        )?;
                    }
                }
            }
            return Ok(out);
        }

        // For complex types, compute phase angle
        unsafe {
            launch_angle(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn make_complex(
        &self,
        real: &Tensor<CudaRuntime>,
        imag: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_make_complex_inputs(real, imag)?;

        let input_dtype = real.dtype();
        let shape = real.shape();
        let numel = real.numel();

        // Determine output dtype
        let out_dtype = match input_dtype {
            DType::F32 => DType::Complex64,
            DType::F64 => DType::Complex128,
            _ => unreachable!("validated above"),
        };

        let real_contig = ensure_contiguous(real)?;
        let imag_contig = ensure_contiguous(imag)?;
        let out = Tensor::<CudaRuntime>::empty(shape, out_dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            launch_from_real_imag(
                &self.context,
                &self.stream,
                self.device.index,
                input_dtype,
                real_contig.ptr(),
                imag_contig.ptr(),
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn complex_mul_real(
        &self,
        complex: &Tensor<CudaRuntime>,
        real: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_complex_real_inputs(complex, real, "complex_mul_real")?;

        let dtype = complex.dtype();
        let shape = complex.shape();
        let numel = complex.numel();

        let complex_contig = ensure_contiguous(complex)?;
        let real_contig = ensure_contiguous(real)?;
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            launch_complex_mul_real(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                complex_contig.ptr(),
                real_contig.ptr(),
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn complex_div_real(
        &self,
        complex: &Tensor<CudaRuntime>,
        real: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_complex_real_inputs(complex, real, "complex_div_real")?;

        let dtype = complex.dtype();
        let shape = complex.shape();
        let numel = complex.numel();

        let complex_contig = ensure_contiguous(complex)?;
        let real_contig = ensure_contiguous(real)?;
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        unsafe {
            launch_complex_div_real(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                complex_contig.ptr(),
                real_contig.ptr(),
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }
}
