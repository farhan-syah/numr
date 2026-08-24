//! Quasi-random sequence generation for CUDA runtime

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::common::quasirandom::{
    validate_halton_params, validate_latin_hypercube_params, validate_sobol_params,
};
use crate::ops::traits::QuasiRandomOps;
use crate::runtime::Allocator;
use crate::runtime::cuda::kernels::{
    launch_halton_f32, launch_halton_f64, launch_latin_hypercube_f32, launch_latin_hypercube_f64,
    launch_sobol_f32, launch_sobol_f64,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Supported data types for CUDA quasi-random operations
const SUPPORTED_DTYPES: &[DType] = &[DType::F32, DType::F64];

impl QuasiRandomOps<CudaRuntime> for CudaClient {
    fn sobol(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_sobol_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "sobol")?;

        // ── Sobol direction-vector cache ──────────────────────────────────────
        //
        // H2D copies of host Vecs inside a captured stream are unsafe: the copy
        // is recorded as a `cuMemcpyHtoD` graph-memcpy node whose source pointer
        // is the stack-local Vec's address at capture time. On `cuGraphLaunch`
        // replay the driver re-reads from that address, which by then points to
        // freed memory → CUDA_ERROR_ILLEGAL_ADDRESS.
        //
        // Fix: use a persistent device buffer (allocated outside any capture
        // region) whose address is stable across replays. The buffer is stored
        // in `self.sobol_dv_cache` and keyed by `dimension`.
        //
        // If the allocator is currently frozen (we are inside a capture region)
        // and the dimension is not cached yet, we cannot safely do the H2D copy
        // here. The caller must call `client.warmup_sobol(dimension)` first.
        let dim_u32 = dimension as u32;
        let dv_ptr: u64 = match self.sobol_dv_cache.get(dim_u32) {
            Some((ptr, _)) => ptr,
            None => {
                if self.allocator.is_frozen() {
                    return Err(crate::error::Error::BackendLimitation {
                        backend: "cuda",
                        operation: "sobol",
                        reason: format!(
                            "Sobol direction vectors for dimension {} must be warmed up before \
                             graph capture; call client.warmup_sobol({}) once outside the \
                             capture region first",
                            dimension, dimension
                        ),
                    });
                }
                // Not in capture mode: populate the cache now. This performs the
                // H2D copy, synchronises the stream, and stores the persistent
                // device pointer. Subsequent calls (including inside capture) will
                // hit the fast path above.
                self.warmup_sobol(dimension)?;
                // The entry is now guaranteed to be present.
                self.sobol_dv_cache
                    .get(dim_u32)
                    .expect("warmup_sobol succeeded but cache entry is missing")
                    .0
            }
        };

        // Allocate output: shape [n_points, dimension]
        let shape = vec![n_points, dimension];
        let out = Tensor::<CudaRuntime>::empty(&shape, dtype, &self.device)?;

        // Launch CUDA kernel — no H2D copy occurs here.
        unsafe {
            match dtype {
                DType::F32 => {
                    launch_sobol_f32(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        dv_ptr,
                        out.ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                DType::F64 => {
                    launch_sobol_f64(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        dv_ptr,
                        out.ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                _ => unreachable!("dtype validation should prevent this"),
            }
        }

        Ok(out)
    }

    fn halton(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_halton_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "halton")?;

        // Allocate output: shape [n_points, dimension]
        let shape = vec![n_points, dimension];
        let out = Tensor::<CudaRuntime>::empty(&shape, dtype, &self.device)?;

        // Launch CUDA kernel
        unsafe {
            match dtype {
                DType::F32 => {
                    launch_halton_f32(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                DType::F64 => {
                    launch_halton_f64(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                _ => unreachable!("dtype validation should prevent this"),
            }
        }

        Ok(out)
    }

    fn latin_hypercube(
        &self,
        n_samples: usize,
        dimension: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_latin_hypercube_params(
            n_samples,
            dimension,
            dtype,
            SUPPORTED_DTYPES,
            "latin_hypercube",
        )?;

        // Allocate output: shape [n_samples, dimension]
        let shape = vec![n_samples, dimension];
        let out = Tensor::<CudaRuntime>::empty(&shape, dtype, &self.device)?;

        // Generate random seed for shuffling
        let seed = generate_random_seed();

        // Launch CUDA kernel
        unsafe {
            match dtype {
                DType::F32 => {
                    launch_latin_hypercube_f32(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.ptr(),
                        n_samples,
                        dimension,
                        seed,
                    )?;
                }
                DType::F64 => {
                    launch_latin_hypercube_f64(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.ptr(),
                        n_samples,
                        dimension,
                        seed,
                    )?;
                }
                _ => unreachable!("dtype validation should prevent this"),
            }
        }

        Ok(out)
    }
}

// ============================================================================
// Random Seed Generation Helper
// ============================================================================

/// Global atomic counter for generating unique seeds
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a random seed combining atomic counter and system time.
///
/// This provides good entropy for parallel random number generation:
/// - Atomic counter ensures uniqueness across calls
/// - System time adds unpredictability
#[inline]
fn generate_random_seed() -> u64 {
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time_component = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Combine counter and time using splitmix64-style mixing
    let mut z = counter.wrapping_add(time_component);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaDevice, CudaClient)> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        Some((device, client))
    }

    #[test]
    fn test_sobol_basic() {
        let Some((_device, client)) = setup() else {
            return;
        };

        let points = client.sobol(10, 2, 0, DType::F32).unwrap();
        assert_eq!(points.shape(), &[10, 2]);

        // Verify points are in [0, 1)
        let data: Vec<f32> = points.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_halton_basic() {
        let Some((_device, client)) = setup() else {
            return;
        };

        let points = client.halton(10, 3, 0, DType::F32).unwrap();
        assert_eq!(points.shape(), &[10, 3]);

        // Verify points are in [0, 1)
        let data: Vec<f32> = points.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_latin_hypercube_basic() {
        let Some((_device, client)) = setup() else {
            return;
        };

        let samples = client.latin_hypercube(20, 4, DType::F32).unwrap();
        assert_eq!(samples.shape(), &[20, 4]);

        // Verify samples are in [0, 1)
        let data: Vec<f32> = samples.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Sample out of range: {}", val);
        }
    }

    #[test]
    fn test_sobol_deterministic() {
        let Some((_device, client)) = setup() else {
            return;
        };

        let points1 = client.sobol(5, 2, 0, DType::F32).unwrap();
        let points2 = client.sobol(5, 2, 0, DType::F32).unwrap();

        let data1: Vec<f32> = points1.to_vec();
        let data2: Vec<f32> = points2.to_vec();

        // Same parameters should produce identical sequences
        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert_eq!(v1, v2, "Sobol sequence not deterministic");
        }
    }

    #[test]
    fn test_error_zero_points() {
        let Some((_device, client)) = setup() else {
            return;
        };
        let result = client.sobol(0, 2, 0, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unsupported_dtype() {
        let Some((_device, client)) = setup() else {
            return;
        };
        let result = client.sobol(10, 2, 0, DType::I32);
        assert!(result.is_err());
    }

    #[test]
    fn test_sobol_dimension_limit() {
        let Some((_device, client)) = setup() else {
            return;
        };

        // Should work up to 21,201 dimensions (full Joe & Kuo dataset)
        let result = client.sobol(10, 100, 0, DType::F32);
        assert!(result.is_ok());

        let result = client.sobol(10, 1000, 0, DType::F32);
        assert!(result.is_ok());

        // Should fail beyond 21,201
        let result = client.sobol(10, 21202, 0, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_halton_dimension_limit() {
        let Some((_device, client)) = setup() else {
            return;
        };

        // Should work up to 100 dimensions
        let result = client.halton(10, 100, 0, DType::F32);
        assert!(result.is_ok());

        // Should fail beyond 100
        let result = client.halton(10, 101, 0, DType::F32);
        assert!(result.is_err());
    }

    /// Test that `warmup_sobol` pre-populates the cache and subsequent sobol
    /// calls reuse the cached pointer (no second allocation occurs).
    ///
    /// This test requires a live CUDA GPU.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn test_warmup_sobol_cache_population() {
        let Some((_device, client)) = setup() else {
            return;
        };

        // Cache should be empty before warmup.
        assert!(
            client.sobol_dv_cache.get(4).is_none(),
            "cache should be empty before warmup"
        );

        // Warmup for dimension 4.
        client.warmup_sobol(4).expect("warmup_sobol should succeed");

        // Cache should now contain the entry.
        let entry = client
            .sobol_dv_cache
            .get(4)
            .expect("cache entry should exist after warmup");
        assert_ne!(entry.0, 0, "cached device pointer must be non-null");
        assert_eq!(entry.1, 4 * 32, "num_u32s = dimension * SOBOL_BITS");

        // Second warmup is idempotent — same pointer returned.
        let ptr_before = entry.0;
        client
            .warmup_sobol(4)
            .expect("second warmup_sobol should succeed");
        let entry2 = client.sobol_dv_cache.get(4).expect("still cached");
        assert_eq!(
            entry2.0, ptr_before,
            "pointer must be stable across repeated warmup calls"
        );

        // Sobol call after warmup must succeed and return correct-shape output.
        let points = client.sobol(8, 4, 0, DType::F32).unwrap();
        assert_eq!(points.shape(), &[8, 4]);
        let data: Vec<f32> = points.to_vec();
        for &v in &data {
            assert!(v >= 0.0 && v < 1.0, "point out of range: {}", v);
        }
    }

    /// Test that calling sobol without warmup outside capture mode succeeds
    /// (auto-warmup path).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn test_sobol_auto_warmup_outside_capture() {
        let Some((_device, client)) = setup() else {
            return;
        };

        // No explicit warmup — sobol should auto-warmup.
        let result = client.sobol(5, 3, 0, DType::F32);
        assert!(
            result.is_ok(),
            "sobol should auto-warmup outside capture: {:?}",
            result
        );

        // Cache should now be populated.
        assert!(
            client.sobol_dv_cache.get(3).is_some(),
            "auto-warmup should have populated cache for dimension 3"
        );
    }
}
