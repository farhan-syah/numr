//! WebGPU implementation of polynomial algorithms
//!
//! This module implements the [`PolynomialAlgorithms`] trait for WebGPU.
//! All algorithms delegate to the shared core implementations to ensure
//! backend parity with CPU/CUDA.
//!
//! # Supported DTypes
//!
//! WebGPU only supports F32 for polynomial operations. F64 is not available
//! because WGSL (WebGPU Shading Language) does not support 64-bit floats.

use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::polynomial::PolynomialAlgorithms;
use crate::algorithm::polynomial::core::{self, DTypeSupport};
use crate::algorithm::polynomial::types::PolynomialRoots;
use crate::error::Result;
use crate::tensor::Tensor;

impl PolynomialAlgorithms<WgpuRuntime> for WgpuClient {
    fn polyroots(&self, coeffs: &Tensor<WgpuRuntime>) -> Result<PolynomialRoots<WgpuRuntime>> {
        core::polyroots_impl(self, coeffs, DTypeSupport::F32_ONLY)
    }

    fn polyval(
        &self,
        coeffs: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        core::polyval_impl(self, coeffs, x, DTypeSupport::F32_ONLY)
    }

    fn polyfromroots(
        &self,
        roots_real: &Tensor<WgpuRuntime>,
        roots_imag: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        core::polyfromroots_impl(self, roots_real, roots_imag, DTypeSupport::F32_ONLY)
    }

    fn polymul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        core::polymul_impl(self, a, b, DTypeSupport::F32_ONLY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::wgpu::{WgpuDevice, is_wgpu_available};

    fn create_wgpu_client() -> Option<(WgpuClient, WgpuDevice)> {
        if !is_wgpu_available() {
            return None;
        }
        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        Some((client, device))
    }

    #[test]
    fn test_wgpu_polyroots_quadratic_real() {
        let Some((client, device)) = create_wgpu_client() else {
            println!("WebGPU not available, skipping test");
            return;
        };

        // x² - 3x + 2 = (x-1)(x-2), roots: 1, 2
        let coeffs =
            Tensor::<WgpuRuntime>::from_slice(&[2.0f32, -3.0, 1.0], &[3], &device).unwrap();

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
    fn test_wgpu_polyval_quadratic() {
        let Some((client, device)) = create_wgpu_client() else {
            println!("WebGPU not available, skipping test");
            return;
        };

        // p(x) = 1 + 2x + 3x² → p(2) = 1 + 4 + 12 = 17
        let coeffs = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let x = Tensor::<WgpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();

        let result = client.polyval(&coeffs, &x).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert!((data[0] - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_wgpu_polymul() {
        let Some((client, device)) = create_wgpu_client() else {
            println!("WebGPU not available, skipping test");
            return;
        };

        // (1 + x) * (1 + x) = 1 + 2x + x² = [1, 2, 1]
        let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();
        let b = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

        let c = client.polymul(&a, &b).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wgpu_polyfromroots() {
        let Some((client, device)) = create_wgpu_client() else {
            println!("WebGPU not available, skipping test");
            return;
        };

        // Roots: 1, 2 → (x-1)(x-2) = x² - 3x + 2 = [2, -3, 1]
        let roots_real = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let roots_imag = Tensor::<WgpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();

        let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
        let data: Vec<f32> = coeffs.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - (-3.0)).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_wgpu_f64_unsupported() {
        let Some((client, device)) = create_wgpu_client() else {
            println!("WebGPU not available, skipping test");
            return;
        };

        // F64 should return UnsupportedDType error
        let coeffs =
            Tensor::<WgpuRuntime>::from_slice(&[2.0f64, -3.0, 1.0], &[3], &device).unwrap();

        let result = client.polyroots(&coeffs);
        assert!(result.is_err(), "F64 should not be supported on WebGPU");
    }
}
