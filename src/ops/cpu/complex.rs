//! CPU implementation of complex number operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ComplexOps;
use crate::ops::common::{validate_complex_real_inputs, validate_make_complex_inputs};
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::ensure_contiguous, kernels};
use crate::tensor::Tensor;

/// ComplexOps implementation for CPU runtime.
impl ComplexOps<CpuRuntime> for CpuClient {
    fn conj(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();

        // For real types, conjugate is identity
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();
        let chunk_size = self.chunk_size_hint();

        match dtype {
            DType::Complex64 => {
                self.install_parallelism(|| unsafe {
                    kernels::conj_complex64(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            DType::Complex128 => {
                self.install_parallelism(|| unsafe {
                    kernels::conj_complex128(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            _ => unreachable!("conj called on non-complex dtype"),
        }

        Ok(out)
    }

    fn real(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();

        // For real types, return copy
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a)?;

        // Determine output dtype (F32 for Complex64, F64 for Complex128)
        let out_dtype = dtype
            .complex_component_dtype()
            .ok_or_else(|| Error::Internal("Expected complex dtype".to_string()))?;

        let out = Tensor::<CpuRuntime>::try_empty(shape, out_dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();
        let chunk_size = self.chunk_size_hint();

        match dtype {
            DType::Complex64 => {
                self.install_parallelism(|| unsafe {
                    kernels::real_complex64(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            DType::Complex128 => {
                self.install_parallelism(|| unsafe {
                    kernels::real_complex128(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            _ => unreachable!("real called on non-complex dtype"),
        }

        Ok(out)
    }

    fn imag(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let numel = a.numel();

        // For real types, return zeros
        if !dtype.is_complex() {
            return Ok(Tensor::<CpuRuntime>::try_zeros(shape, dtype, &self.device)?);
        }

        let a_contig = ensure_contiguous(a)?;

        // Determine output dtype (F32 for Complex64, F64 for Complex128)
        let out_dtype = dtype
            .complex_component_dtype()
            .ok_or_else(|| Error::Internal("Expected complex dtype".to_string()))?;

        let out = Tensor::<CpuRuntime>::try_empty(shape, out_dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();
        let chunk_size = self.chunk_size_hint();

        match dtype {
            DType::Complex64 => {
                self.install_parallelism(|| unsafe {
                    kernels::imag_complex64(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            DType::Complex128 => {
                self.install_parallelism(|| unsafe {
                    kernels::imag_complex128(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            _ => unreachable!("imag called on non-complex dtype"),
        }

        Ok(out)
    }

    fn angle(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let numel = a.numel();
        let chunk_size = self.chunk_size_hint();

        let a_contig = ensure_contiguous(a)?;

        // For real types: angle(x) = 0 if x >= 0, π if x < 0
        if !dtype.is_complex() {
            let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;

            // Handle empty tensors
            if numel == 0 {
                return Ok(out);
            }

            let a_ptr = a_contig.ptr();
            let out_ptr = out.ptr();

            match dtype {
                DType::F32 => {
                    self.install_parallelism(|| unsafe {
                        kernels::angle_real_f32(
                            a_ptr as *const _,
                            out_ptr as *mut _,
                            numel,
                            chunk_size,
                        );
                    });
                }
                DType::F64 => {
                    self.install_parallelism(|| unsafe {
                        kernels::angle_real_f64(
                            a_ptr as *const _,
                            out_ptr as *mut _,
                            numel,
                            chunk_size,
                        );
                    });
                }
                _ => {
                    // For integer types, angle doesn't make mathematical sense
                    // Return zeros
                    return Ok(Tensor::<CpuRuntime>::try_zeros(shape, dtype, &self.device)?);
                }
            }
            return Ok(out);
        }

        // Determine output dtype (F32 for Complex64, F64 for Complex128)
        let out_dtype = dtype
            .complex_component_dtype()
            .ok_or_else(|| Error::Internal("Expected complex dtype".to_string()))?;

        let out = Tensor::<CpuRuntime>::try_empty(shape, out_dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();

        match dtype {
            DType::Complex64 => {
                self.install_parallelism(|| unsafe {
                    kernels::angle_complex64(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            DType::Complex128 => {
                self.install_parallelism(|| unsafe {
                    kernels::angle_complex128(
                        a_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            _ => unreachable!("angle called on non-complex dtype"),
        }

        Ok(out)
    }

    fn make_complex(
        &self,
        real: &Tensor<CpuRuntime>,
        imag: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
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
        let out = Tensor::<CpuRuntime>::try_empty(shape, out_dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let real_ptr = real_contig.ptr();
        let imag_ptr = imag_contig.ptr();
        let out_ptr = out.ptr();
        let chunk_size = self.chunk_size_hint();

        match input_dtype {
            DType::F32 => {
                self.install_parallelism(|| unsafe {
                    kernels::from_real_imag_f32(
                        real_ptr as *const _,
                        imag_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            DType::F64 => {
                self.install_parallelism(|| unsafe {
                    kernels::from_real_imag_f64(
                        real_ptr as *const _,
                        imag_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            _ => unreachable!("validated above"),
        }

        Ok(out)
    }

    fn complex_mul_real(
        &self,
        complex: &Tensor<CpuRuntime>,
        real: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_complex_real_inputs(complex, real, "complex_mul_real")?;

        let dtype = complex.dtype();
        let shape = complex.shape();
        let numel = complex.numel();

        let complex_contig = ensure_contiguous(complex)?;
        let real_contig = ensure_contiguous(real)?;
        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let complex_ptr = complex_contig.ptr();
        let real_ptr = real_contig.ptr();
        let out_ptr = out.ptr();
        let chunk_size = self.chunk_size_hint();

        match dtype {
            DType::Complex64 => {
                self.install_parallelism(|| unsafe {
                    kernels::complex64_mul_real(
                        complex_ptr as *const _,
                        real_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            DType::Complex128 => {
                self.install_parallelism(|| unsafe {
                    kernels::complex128_mul_real(
                        complex_ptr as *const _,
                        real_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            _ => unreachable!("validated above"),
        }

        Ok(out)
    }

    fn complex_div_real(
        &self,
        complex: &Tensor<CpuRuntime>,
        real: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_complex_real_inputs(complex, real, "complex_div_real")?;

        let dtype = complex.dtype();
        let shape = complex.shape();
        let numel = complex.numel();

        let complex_contig = ensure_contiguous(complex)?;
        let real_contig = ensure_contiguous(real)?;
        let out = Tensor::<CpuRuntime>::try_empty(shape, dtype, &self.device)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let complex_ptr = complex_contig.ptr();
        let real_ptr = real_contig.ptr();
        let out_ptr = out.ptr();
        let chunk_size = self.chunk_size_hint();

        match dtype {
            DType::Complex64 => {
                self.install_parallelism(|| unsafe {
                    kernels::complex64_div_real(
                        complex_ptr as *const _,
                        real_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            DType::Complex128 => {
                self.install_parallelism(|| unsafe {
                    kernels::complex128_div_real(
                        complex_ptr as *const _,
                        real_ptr as *const _,
                        out_ptr as *mut _,
                        numel,
                        chunk_size,
                    );
                });
            }
            _ => unreachable!("validated above"),
        }

        Ok(out)
    }
}
