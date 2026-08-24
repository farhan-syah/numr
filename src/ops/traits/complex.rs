//! Complex number operations trait.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Complex number operations
pub trait ComplexOps<R: Runtime> {
    /// Complex conjugate: conj(a + bi) = a - bi
    ///
    /// Returns the complex conjugate of the input tensor.
    /// For real tensors, returns the input unchanged.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (Complex64, Complex128, or real types)
    ///
    /// # Returns
    ///
    /// * Complex types: Tensor with same shape and dtype, imaginary part negated
    /// * Real types: Returns input tensor unchanged (real numbers equal their conjugate)
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends (identity operation)
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    /// use numr::dtype::Complex64;
    ///
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    ///     &[2],
    ///     &device
    /// )?;
    /// let conj_z = client.conj(&z)?;
    /// // Result: [1.0 - 2.0i, 3.0 + 4.0i]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn conj(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ComplexOps::conj",
        })
    }

    /// Extract real part of complex tensor: real(a + bi) = a
    ///
    /// Extracts the real component from a complex tensor.
    /// For real tensors, returns a copy of the input.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with same shape
    /// * Complex128 input → F64 tensor with same shape
    /// * Real input → Copy of input tensor
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    /// use numr::dtype::Complex64;
    ///
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// )?;
    /// let re = client.real(&z)?;  // F32 tensor: [1.0, 3.0]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn real(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ComplexOps::real",
        })
    }

    /// Extract imaginary part of complex tensor: imag(a + bi) = b
    ///
    /// Extracts the imaginary component from a complex tensor.
    /// For real tensors, returns a zero tensor with the same shape.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with same shape
    /// * Complex128 input → F64 tensor with same shape
    /// * Real input → Zero tensor with same shape and dtype
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    /// use numr::dtype::Complex64;
    ///
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// )?;
    /// let im = client.imag(&z)?;  // F32 tensor: [2.0, 4.0]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn imag(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ComplexOps::imag",
        })
    }

    /// Compute phase angle of complex tensor: angle(a + bi) = atan2(b, a)
    ///
    /// Returns the phase angle (argument) of complex numbers in radians.
    /// The result is in the range [-π, π].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with angles in radians
    /// * Complex128 input → F64 tensor with angles in radians
    /// * Real input → Zero tensor (real numbers have phase angle 0 for positive, π for negative)
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    /// use numr::dtype::Complex64;
    ///
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 1.0), Complex64::new(-1.0, 0.0)],
    ///     &[2],
    ///     &device
    /// )?;
    /// let angles = client.angle(&z)?;  // F32 tensor: [π/4, π]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Mathematical Notes
    ///
    /// For complex z = a + bi, returns atan2(b, a) in radians [-π, π].
    /// For real x, returns 0 if x ≥ 0, π if x < 0.
    /// To compute magnitude, use abs(z) = sqrt(re² + im²) separately.
    fn angle(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ComplexOps::angle",
        })
    }

    /// Construct complex tensor from separate real and imaginary part tensors.
    ///
    /// Creates a complex tensor where each element is `real[i] + imag[i]*i`.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor containing real parts (F32 or F64)
    /// * `imag` - Tensor containing imaginary parts (must match `real` dtype and shape)
    ///
    /// # Returns
    ///
    /// * F32 inputs → Complex64 tensor with same shape
    /// * F64 inputs → Complex128 tensor with same shape
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if `real` and `imag` have different shapes
    /// * `DTypeMismatch` - if `real` and `imag` have different dtypes
    /// * `UnsupportedDType` - if input dtype is not F32 or F64
    ///
    /// # Supported Types
    ///
    /// * F32 → Complex64: All backends (CPU, CUDA, WebGPU)
    /// * F64 → Complex128: CPU and CUDA only (WebGPU does not support F64)
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    ///
    /// let real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    /// let imag = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device)?;
    /// let complex = client.make_complex(&real, &imag)?;
    /// // Result: [1.0+4.0i, 2.0+5.0i, 3.0+6.0i]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn make_complex(&self, real: &Tensor<R>, imag: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (real, imag);
        Err(Error::NotImplemented {
            feature: "ComplexOps::make_complex",
        })
    }

    /// Multiply complex tensor by real tensor element-wise.
    ///
    /// Computes (a + bi) * r = ar + br*i for each element.
    ///
    /// # Arguments
    ///
    /// * `complex` - Complex tensor (Complex64 or Complex128)
    /// * `real` - Real tensor (F32 for Complex64, F64 for Complex128)
    ///
    /// # Returns
    ///
    /// Complex tensor with same dtype and shape as input complex tensor.
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if shapes don't match (no broadcasting)
    /// * `DTypeMismatch` - if real dtype doesn't match complex component dtype
    /// * `UnsupportedDType` - if complex is not Complex64/Complex128
    ///
    /// # Supported Types
    ///
    /// * Complex64 × F32: All backends (CPU, CUDA, WebGPU)
    /// * Complex128 × F64: CPU and CUDA only (WebGPU does not support F64)
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    /// use numr::dtype::Complex64;
    ///
    /// let complex = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// )?;
    /// let scale = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.5], &[2], &device)?;
    /// let result = client.complex_mul_real(&complex, &scale)?;
    /// // Result: [2.0+4.0i, 1.5+2.0i]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn complex_mul_real(&self, complex: &Tensor<R>, real: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (complex, real);
        Err(Error::NotImplemented {
            feature: "ComplexOps::complex_mul_real",
        })
    }

    /// Divide complex tensor by real tensor element-wise.
    ///
    /// Computes (a + bi) / r = (a/r) + (b/r)*i for each element.
    ///
    /// # Arguments
    ///
    /// * `complex` - Complex tensor (Complex64 or Complex128)
    /// * `real` - Real tensor (F32 for Complex64, F64 for Complex128)
    ///
    /// # Returns
    ///
    /// Complex tensor with same dtype and shape as input complex tensor.
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if shapes don't match (no broadcasting)
    /// * `DTypeMismatch` - if real dtype doesn't match complex component dtype
    /// * `UnsupportedDType` - if complex is not Complex64/Complex128
    ///
    /// # Supported Types
    ///
    /// * Complex64 / F32: All backends (CPU, CUDA, WebGPU)
    /// * Complex128 / F64: CPU and CUDA only (WebGPU does not support F64)
    ///
    /// # Note
    ///
    /// Division by zero will result in NaN/Inf values, following IEEE 754 semantics.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    /// use numr::dtype::Complex64;
    ///
    /// let complex = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(4.0, 6.0), Complex64::new(2.0, 4.0)],
    ///     &[2],
    ///     &device
    /// )?;
    /// let divisor = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0], &[2], &device)?;
    /// let result = client.complex_div_real(&complex, &divisor)?;
    /// // Result: [2.0+3.0i, 1.0+2.0i]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn complex_div_real(&self, complex: &Tensor<R>, real: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (complex, real);
        Err(Error::NotImplemented {
            feature: "ComplexOps::complex_div_real",
        })
    }

    /// Multiply real tensor by complex tensor element-wise.
    ///
    /// Computes r * (a + bi) = ra + rb*i for each element.
    /// This is equivalent to `complex_mul_real` (multiplication is commutative).
    ///
    /// # Arguments
    ///
    /// * `real` - Real tensor (F32 for Complex64, F64 for Complex128)
    /// * `complex` - Complex tensor (Complex64 or Complex128)
    ///
    /// # Returns
    ///
    /// Complex tensor with same dtype and shape as input complex tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ComplexOps;
    /// use numr::dtype::Complex64;
    ///
    /// let scale = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.5], &[2], &device)?;
    /// let complex = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// )?;
    /// let result = client.real_mul_complex(&scale, &complex)?;
    /// // Result: [2.0+4.0i, 1.5+2.0i]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn real_mul_complex(&self, real: &Tensor<R>, complex: &Tensor<R>) -> Result<Tensor<R>> {
        // Multiplication is commutative
        self.complex_mul_real(complex, real)
    }
}
