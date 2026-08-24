//! CUDA implementation of polynomial algorithms
//!
//! This module implements the [`PolynomialAlgorithms`] trait for CUDA.
//! All algorithms delegate to the shared core implementations to ensure
//! backend parity with CPU/WebGPU.
//!
//! Uses dtype promotion for reduced-precision types (F16, BF16, FP8).

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::algorithm::polynomial::PolynomialAlgorithms;
use crate::algorithm::polynomial::core::{self, DTypeSupport};
use crate::algorithm::polynomial::types::PolynomialRoots;
use crate::error::Result;
use crate::tensor::Tensor;

impl PolynomialAlgorithms<CudaRuntime> for CudaClient {
    fn polyroots(&self, coeffs: &Tensor<CudaRuntime>) -> Result<PolynomialRoots<CudaRuntime>> {
        let (coeffs_p, orig_dtype) = linalg_promote(self, coeffs)?;
        let roots = core::polyroots_impl(self, &coeffs_p, DTypeSupport::FULL)?;
        Ok(PolynomialRoots {
            roots_real: linalg_demote(self, roots.roots_real, orig_dtype)?,
            roots_imag: linalg_demote(self, roots.roots_imag, orig_dtype)?,
        })
    }

    fn polyval(
        &self,
        coeffs: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let (coeffs_p, orig_dtype) = linalg_promote(self, coeffs)?;
        let (x_p, _) = linalg_promote(self, x)?;
        let result = core::polyval_impl(self, &coeffs_p, &x_p, DTypeSupport::FULL)?;
        linalg_demote(self, result, orig_dtype)
    }

    fn polyfromroots(
        &self,
        roots_real: &Tensor<CudaRuntime>,
        roots_imag: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let (rr_p, orig_dtype) = linalg_promote(self, roots_real)?;
        let (ri_p, _) = linalg_promote(self, roots_imag)?;
        let result = core::polyfromroots_impl(self, &rr_p, &ri_p, DTypeSupport::FULL)?;
        linalg_demote(self, result, orig_dtype)
    }

    fn polymul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let (a_p, orig_dtype) = linalg_promote(self, a)?;
        let (b_p, _) = linalg_promote(self, b)?;
        let result = core::polymul_impl(self, &a_p, &b_p, DTypeSupport::FULL)?;
        linalg_demote(self, result, orig_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;
    use std::panic;

    fn create_cuda_client() -> Option<(CudaClient, CudaDevice)> {
        let result = panic::catch_unwind(|| {
            let device = CudaDevice::new(0);
            let client = CudaRuntime::default_client(&device);
            (client, device)
        });
        result.ok()
    }

    #[test]
    fn test_cuda_polyroots_quadratic_real() {
        let Some((client, device)) = create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        // x² - 3x + 2 = (x-1)(x-2), roots: 1, 2
        let coeffs =
            Tensor::<CudaRuntime>::from_slice(&[2.0f32, -3.0, 1.0], &[3], &device).unwrap();

        let roots = client.polyroots(&coeffs).unwrap();

        let real: Vec<f32> = roots.roots_real.to_vec();
        let imag: Vec<f32> = roots.roots_imag.to_vec();

        assert_eq!(real.len(), 2);

        let mut sorted_real: Vec<f32> = real.clone();
        sorted_real.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(
            (sorted_real[0] - 1.0).abs() < 1e-4,
            "Expected root 1, got {}",
            sorted_real[0]
        );
        assert!(
            (sorted_real[1] - 2.0).abs() < 1e-4,
            "Expected root 2, got {}",
            sorted_real[1]
        );

        for &im in &imag {
            assert!(im.abs() < 1e-4, "Expected real root");
        }
    }

    #[test]
    fn test_cuda_polyval_quadratic() {
        let Some((client, device)) = create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        // p(x) = 1 + 2x + 3x² → p(2) = 1 + 4 + 12 = 17
        let coeffs = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let x = Tensor::<CudaRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();

        let result = client.polyval(&coeffs, &x).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert!((data[0] - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_cuda_polymul() {
        let Some((client, device)) = create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        // (1 + x) * (1 + x) = 1 + 2x + x² = [1, 2, 1]
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();
        let b = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

        let c = client.polymul(&a, &b).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cuda_polyfromroots() {
        let Some((client, device)) = create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        // Roots: 1, 2 → (x-1)(x-2) = x² - 3x + 2 = [2, -3, 1]
        let roots_real = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let roots_imag = Tensor::<CudaRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();

        let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
        let data: Vec<f32> = coeffs.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - (-3.0)).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cuda_polyroots_f64() {
        let Some((client, device)) = create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        // x² - 3x + 2, roots: 1, 2
        let coeffs =
            Tensor::<CudaRuntime>::from_slice(&[2.0f64, -3.0, 1.0], &[3], &device).unwrap();

        let roots = client.polyroots(&coeffs).unwrap();

        let real: Vec<f64> = roots.roots_real.to_vec();
        let imag: Vec<f64> = roots.roots_imag.to_vec();

        assert_eq!(real.len(), 2);

        let mut sorted_real: Vec<f64> = real.clone();
        sorted_real.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted_real[0] - 1.0).abs() < 1e-10);
        assert!((sorted_real[1] - 2.0).abs() < 1e-10);

        for im in &imag {
            assert!(im.abs() < 1e-10);
        }
    }
}
