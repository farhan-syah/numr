//! CPU implementation of polynomial algorithms
//!
//! This module implements the [`PolynomialAlgorithms`] trait for CPU.
//! All algorithms delegate to the shared core implementations to ensure
//! backend parity with CUDA/WebGPU.

use super::super::{CpuClient, CpuRuntime};
use crate::algorithm::polynomial::PolynomialAlgorithms;
use crate::algorithm::polynomial::core::{self, DTypeSupport};
use crate::algorithm::polynomial::types::PolynomialRoots;
use crate::error::Result;
use crate::tensor::Tensor;

impl PolynomialAlgorithms<CpuRuntime> for CpuClient {
    fn polyroots(&self, coeffs: &Tensor<CpuRuntime>) -> Result<PolynomialRoots<CpuRuntime>> {
        core::polyroots_impl(self, coeffs, DTypeSupport::FULL)
    }

    fn polyval(
        &self,
        coeffs: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        core::polyval_impl(self, coeffs, x, DTypeSupport::FULL)
    }

    fn polyfromroots(
        &self,
        roots_real: &Tensor<CpuRuntime>,
        roots_imag: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        core::polyfromroots_impl(self, roots_real, roots_imag, DTypeSupport::FULL)
    }

    fn polymul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        core::polymul_impl(self, a, b, DTypeSupport::FULL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    fn create_client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    #[test]
    fn test_polyroots_quadratic_real() {
        let (client, device) = create_client();

        // x² - 3x + 2 = (x-1)(x-2), roots: 1, 2
        // coeffs: [2, -3, 1] (constant, x, x²)
        let coeffs =
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, -3.0, 1.0], &[3], &device).unwrap();

        let roots = client.polyroots(&coeffs).unwrap();

        let real: Vec<f32> = roots.roots_real.to_vec();
        let imag: Vec<f32> = roots.roots_imag.to_vec();

        assert_eq!(real.len(), 2);
        assert_eq!(imag.len(), 2);

        // Roots should be 1 and 2 (in some order), imaginary parts ~0
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

        for (i, &im) in imag.iter().enumerate() {
            assert!(
                im.abs() < 1e-4,
                "Expected real root, got imag={} at {}",
                im,
                i
            );
        }
    }

    #[test]
    fn test_polyroots_quadratic_complex() {
        let (client, device) = create_client();

        // x² + 1 = 0, roots: ±i
        // coeffs: [1, 0, 1]
        let coeffs =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 1.0], &[3], &device).unwrap();

        let roots = client.polyroots(&coeffs).unwrap();

        let real: Vec<f32> = roots.roots_real.to_vec();
        let imag: Vec<f32> = roots.roots_imag.to_vec();

        assert_eq!(real.len(), 2);
        assert_eq!(imag.len(), 2);

        // Roots should be ±i (real parts ~0, imaginary parts ±1)
        for r in &real {
            assert!(r.abs() < 1e-4, "Expected real part ~0, got {}", r);
        }

        let mut sorted_imag: Vec<f32> = imag.clone();
        sorted_imag.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted_imag[0] - (-1.0)).abs() < 1e-4);
        assert!((sorted_imag[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_polyval_constant() {
        let (client, device) = create_client();

        // p(x) = 5 (constant)
        let coeffs = Tensor::<CpuRuntime>::try_from_slice(&[5.0f32], &[1], &device).unwrap();
        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        let result = client.polyval(&coeffs, &x).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data.len(), 3);
        for &v in &data {
            assert!((v - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_polyval_linear() {
        let (client, device) = create_client();

        // p(x) = 2 + 3x
        let coeffs = Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 3.0], &[2], &device).unwrap();
        let x = Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 1.0, 2.0], &[3], &device).unwrap();

        let result = client.polyval(&coeffs, &x).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert!((data[0] - 2.0).abs() < 1e-6); // 2 + 3*0 = 2
        assert!((data[1] - 5.0).abs() < 1e-6); // 2 + 3*1 = 5
        assert!((data[2] - 8.0).abs() < 1e-6); // 2 + 3*2 = 8
    }

    #[test]
    fn test_polyval_quadratic() {
        let (client, device) = create_client();

        // p(x) = 1 + 2x + 3x² → p(2) = 1 + 4 + 12 = 17
        let coeffs =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let x = Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap();

        let result = client.polyval(&coeffs, &x).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert!((data[0] - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_polyfromroots_real() {
        let (client, device) = create_client();

        // Roots: 1, 2 → (x-1)(x-2) = x² - 3x + 2 = [2, -3, 1]
        let roots_real =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let roots_imag =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();

        let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
        let data: Vec<f32> = coeffs.to_vec();

        assert_eq!(data.len(), 3);
        assert!(
            (data[0] - 2.0).abs() < 1e-5,
            "Expected c0=2, got {}",
            data[0]
        );
        assert!(
            (data[1] - (-3.0)).abs() < 1e-5,
            "Expected c1=-3, got {}",
            data[1]
        );
        assert!(
            (data[2] - 1.0).abs() < 1e-5,
            "Expected c2=1, got {}",
            data[2]
        );
    }

    #[test]
    fn test_polyfromroots_complex() {
        let (client, device) = create_client();

        // Roots: ±i → (x-i)(x+i) = x² + 1 = [1, 0, 1]
        let roots_real =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();
        let roots_imag =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, -1.0], &[2], &device).unwrap();

        let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
        let data: Vec<f32> = coeffs.to_vec();

        assert_eq!(data.len(), 3);
        assert!(
            (data[0] - 1.0).abs() < 1e-5,
            "Expected c0=1, got {}",
            data[0]
        );
        assert!(data[1].abs() < 1e-5, "Expected c1=0, got {}", data[1]);
        assert!(
            (data[2] - 1.0).abs() < 1e-5,
            "Expected c2=1, got {}",
            data[2]
        );
    }

    #[test]
    fn test_polymul_linear() {
        let (client, device) = create_client();

        // (1 + x) * (1 + x) = 1 + 2x + x² = [1, 2, 1]
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

        let c = client.polymul(&a, &b).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_polymul_difference_of_squares() {
        let (client, device) = create_client();

        // (1 - x) * (1 + x) = 1 - x² = [1, 0, -1]
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, -1.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

        let c = client.polymul(&a, &b).unwrap();
        let data: Vec<f32> = c.to_vec();

        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!(data[1].abs() < 1e-6);
        assert!((data[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip_roots_coeffs() {
        let (client, device) = create_client();

        // Original polynomial: x² - 5x + 6 = (x-2)(x-3)
        let original =
            Tensor::<CpuRuntime>::try_from_slice(&[6.0f32, -5.0, 1.0], &[3], &device).unwrap();

        // Find roots
        let roots = client.polyroots(&original).unwrap();

        // Reconstruct polynomial from roots
        let reconstructed = client
            .polyfromroots(&roots.roots_real, &roots.roots_imag)
            .unwrap();
        let data: Vec<f32> = reconstructed.to_vec();

        // Should get back [6, -5, 1] (monic, so we need to check ratios)
        // polyfromroots returns monic polynomial, original is also monic
        assert_eq!(data.len(), 3);
        assert!(
            (data[0] - 6.0).abs() < 1e-4,
            "c0: expected 6, got {}",
            data[0]
        );
        assert!(
            (data[1] - (-5.0)).abs() < 1e-4,
            "c1: expected -5, got {}",
            data[1]
        );
        assert!(
            (data[2] - 1.0).abs() < 1e-4,
            "c2: expected 1, got {}",
            data[2]
        );
    }

    #[test]
    fn test_polyroots_f64() {
        let (client, device) = create_client();

        // x² - 3x + 2, roots: 1, 2
        let coeffs =
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f64, -3.0, 1.0], &[3], &device).unwrap();

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
