//! CUDA implementation of special mathematical functions
//!
//! This module implements the [`SpecialFunctions`] trait for CUDA using native
//! CUDA math library functions. CUDA provides built-in functions for erf, erfc,
//! lgamma, tgamma (gamma), etc.
//!
//! Native CUDA kernels are used - NO cuBLAS/cuSPARSE dependency.

use super::CudaRuntime;
use super::client::CudaClient;
use super::kernels;
use crate::algorithm::special::{SpecialFunctions, validate_special_dtype};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

impl SpecialFunctions<CudaRuntime> for CudaClient {
    fn erf(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_erf(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn erfc(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_erfc(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn erfinv(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_erfinv(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn gamma(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_gamma(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn lgamma(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_lgamma(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn digamma(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_digamma(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn beta(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), a.dtype(), device)?;

        unsafe {
            kernels::launch_beta(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.ptr(),
                b.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn betainc(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() || a.dtype() != x.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() || a.shape() != x.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), a.dtype(), device)?;

        unsafe {
            kernels::launch_betainc(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.ptr(),
                b.ptr(),
                x.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn gammainc(
        &self,
        a: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), a.dtype(), device)?;

        unsafe {
            kernels::launch_gammainc(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.ptr(),
                x.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn gammaincc(
        &self,
        a: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), a.dtype(), device)?;

        unsafe {
            kernels::launch_gammaincc(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.ptr(),
                x.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn gammaincinv(
        &self,
        a: &Tensor<CudaRuntime>,
        p: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != p.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: p.dtype(),
            });
        }
        if a.shape() != p.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: p.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), a.dtype(), device)?;

        unsafe {
            kernels::launch_gammaincinv(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.ptr(),
                p.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn betaincinv(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        p: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() || a.dtype() != p.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() || a.shape() != p.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: p.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), a.dtype(), device)?;

        unsafe {
            kernels::launch_betaincinv(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.ptr(),
                b.ptr(),
                p.ptr(),
                out.ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_j0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_j0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_j1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_j1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_y0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_y0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_y1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_y1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_i0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_i0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_i1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_i1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_k0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_k0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_k1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_bessel_k1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    // ========================================================================
    // Extended Special Functions
    // ========================================================================

    fn ellipk(&self, m: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(m.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(m.shape(), m.dtype(), device)?;

        unsafe {
            kernels::launch_ellipk(
                self.context(),
                self.stream(),
                device.index,
                m.dtype(),
                m.ptr(),
                out.ptr(),
                m.numel(),
            )?;
        }

        Ok(out)
    }

    fn ellipe(&self, m: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(m.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(m.shape(), m.dtype(), device)?;

        unsafe {
            kernels::launch_ellipe(
                self.context(),
                self.stream(),
                device.index,
                m.dtype(),
                m.ptr(),
                out.ptr(),
                m.numel(),
            )?;
        }

        Ok(out)
    }

    fn hyp2f1(
        &self,
        a: f64,
        b: f64,
        c: f64,
        z: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(z.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(z.shape(), z.dtype(), device)?;

        unsafe {
            kernels::launch_hyp2f1(
                self.context(),
                self.stream(),
                device.index,
                z.dtype(),
                a,
                b,
                c,
                z.ptr(),
                out.ptr(),
                z.numel(),
            )?;
        }

        Ok(out)
    }

    fn hyp1f1(&self, a: f64, b: f64, z: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(z.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(z.shape(), z.dtype(), device)?;

        unsafe {
            kernels::launch_hyp1f1(
                self.context(),
                self.stream(),
                device.index,
                z.dtype(),
                a,
                b,
                z.ptr(),
                out.ptr(),
                z.numel(),
            )?;
        }

        Ok(out)
    }

    fn airy_ai(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_airy_ai(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn airy_bi(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_airy_bi(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn legendre_p(&self, n: i32, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_legendre_p(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                n,
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn legendre_p_assoc(
        &self,
        n: i32,
        m: i32,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_legendre_p_assoc(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                n,
                m,
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn sph_harm(
        &self,
        n: i32,
        m: i32,
        theta: &Tensor<CudaRuntime>,
        phi: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(theta.dtype())?;
        if theta.dtype() != phi.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: theta.dtype(),
                rhs: phi.dtype(),
            });
        }
        if theta.shape() != phi.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: theta.shape().to_vec(),
                got: phi.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(theta.shape(), theta.dtype(), device)?;

        unsafe {
            kernels::launch_sph_harm(
                self.context(),
                self.stream(),
                device.index,
                theta.dtype(),
                n,
                m,
                theta.ptr(),
                phi.ptr(),
                out.ptr(),
                theta.numel(),
            )?;
        }

        Ok(out)
    }

    fn fresnel_s(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_fresnel_s(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn fresnel_c(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::try_empty(x.shape(), x.dtype(), device)?;

        unsafe {
            kernels::launch_fresnel_c(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.ptr(),
                out.ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }
}
