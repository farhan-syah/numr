//! Common test utilities
#![allow(dead_code)]

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

/// Create a CPU client and device for testing
pub fn create_cpu_client() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

/// Assert two f64 slices are close within tolerance
///
/// Uses the formula: |a - b| <= atol + rtol * |b|
pub fn assert_allclose_f64(a: &[f64], b: &[f64], rtol: f64, atol: f64, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {} (diff={}, tol={})",
            msg,
            i,
            x,
            y,
            diff,
            tol
        );
    }
}

/// Create a CUDA client and device, returning None if CUDA is unavailable
#[cfg(feature = "cuda")]
pub fn create_cuda_client() -> Option<(CudaClient, CudaDevice)> {
    if !numr::runtime::cuda::is_cuda_available() {
        return None;
    }
    let init = std::panic::catch_unwind(|| {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        (client, device)
    });
    init.ok()
}

/// Create a WebGPU client and device, returning None if WebGPU is unavailable
#[cfg(feature = "wgpu")]
pub fn create_wgpu_client() -> Option<(WgpuClient, WgpuDevice)> {
    if !numr::runtime::wgpu::is_wgpu_available() {
        return None;
    }
    let init = std::panic::catch_unwind(|| {
        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        (client, device)
    });
    init.ok()
}

/// Assert two f32 slices are close within tolerance
#[allow(dead_code)]
pub fn assert_allclose_f32(a: &[f32], b: &[f32], rtol: f32, atol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {} (diff={}, tol={})",
            msg,
            i,
            x,
            y,
            diff,
            tol
        );
    }
}

// ============================================================================
// DType Support Framework
// ============================================================================

/// Returns list of dtypes supported by a specific backend
///
/// Used internally by `supported_dtypes` to determine which dtypes to test.
/// This is the source of truth for backend capabilities.
pub fn backend_supported_dtypes(backend: &str) -> Vec<DType> {
    match backend {
        // CUDA's unsigned coverage is partial: `cumsum` and `cumprod` have U32
        // kernels, but `compare`, `cast`, and the binary ops do not. U32 is left
        // out because a per-backend list cannot express per-op support, and
        // claiming it produces failures rather than skips.
        #[cfg(feature = "cuda")]
        "cuda" => build_dtype_list(&[DType::F32, DType::F64, DType::I32]),
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            // WebGPU: 32-bit types only (F32, I32, U32)
            vec![DType::F32, DType::I32, DType::U32]
        }
        _ => build_dtype_list(&[DType::F32, DType::F64, DType::I32, DType::U32]),
    }
}

/// Build a dtype list from base types, appending feature-gated types
fn build_dtype_list(base: &[DType]) -> Vec<DType> {
    let mut dtypes = base.to_vec();

    if cfg!(feature = "f16") {
        dtypes.push(DType::F16);
        dtypes.push(DType::BF16);
    }
    if cfg!(feature = "fp8") {
        dtypes.push(DType::FP8E4M3);
        dtypes.push(DType::FP8E5M2);
    }

    dtypes
}

/// Check if a dtype is supported on a given backend
///
/// ## Example
///
/// ```ignore
/// if is_dtype_supported("wgpu", DType::F32) {
///     // Run WebGPU test for F32
/// }
/// ```
pub fn is_dtype_supported(backend: &str, dtype: DType) -> bool {
    backend_supported_dtypes(backend).contains(&dtype)
}

/// Returns list of dtypes to test for a given backend
///
/// This is used by test macros to determine which dtypes to parameterize over.
/// For testing purposes, we test:
/// - CPU: All supported dtypes (F32, F64 always; F16/BF16 if f16 feature; FP8 if fp8 feature)
/// - CUDA: All supported dtypes
/// - WebGPU: F32 only (32-bit types only)
pub fn supported_dtypes(backend: &str) -> Vec<DType> {
    match backend {
        #[cfg(feature = "cuda")]
        "cuda" => build_dtype_list(&[DType::F32, DType::F64]),
        #[cfg(feature = "wgpu")]
        "wgpu" => vec![DType::F32],
        _ => build_dtype_list(&[DType::F32, DType::F64]),
    }
}

/// Returns (rtol, atol) tolerance pair for a given dtype
///
/// See `assert_allclose_for_dtype` for precision details per dtype.
pub fn tolerance_for_dtype(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F32 => (1e-5, 1e-6),   // 0.001% relative, 1e-6 absolute
        DType::F64 => (1e-12, 1e-14), // Machine epsilon-level tolerance
        DType::F16 => (0.01, 0.1),    // 1% relative tolerance for half-precision
        DType::BF16 => (0.01, 0.1),   // 1% relative tolerance for BF16
        DType::FP8E4M3 => (0.3, 2.5), // 30% relative — 4-bit mantissa; atol=2.5 for compound ops (norm bwd, gemm)
        DType::FP8E5M2 => (1.0, 2.5), // Very coarse — 2-bit mantissa; atol=2.5 because scatter_reduce/cov accumulate rounding error
        _ => (1e-5, 1e-6),            // Default tolerance
    }
}

/// Assert two f64 slices are close, with tolerance based on dtype
///
/// This handles different precision levels appropriately:
/// - F64: Machine epsilon-level tolerance
/// - F32: Standard single-precision tolerance
/// - F16/BF16: Relaxed tolerance due to reduced precision (1%)
/// - FP8E4M3: Coarse tolerance (10%) — 4-bit mantissa
/// - FP8E5M2: Very coarse tolerance (100%) — 2-bit mantissa
pub fn assert_allclose_for_dtype(actual: &[f64], expected: &[f64], dtype: DType, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: dtype={:?}: length mismatch",
        msg,
        dtype
    );
    let (rtol, atol) = tolerance_for_dtype(dtype);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        assert!(
            diff <= tol,
            "{}: dtype={:?}: element {} differs: {} vs {} (diff={:.2e}, tol={:.2e})",
            msg,
            dtype,
            i,
            a,
            e,
            diff,
            tol
        );
    }
}

/// Assert two tensors are close by reading back in native dtype and comparing.
///
/// Dispatches on `dtype` to call `to_vec::<T>()` with the correct native type,
/// then compares element-wise using dtype-appropriate tolerance.
/// No unnecessary casting - F32 compares as f32, F64 as f64, F16 as f16, etc.
pub fn assert_tensor_allclose<R1: Runtime, R2: Runtime>(
    actual: &numr::tensor::Tensor<R1>,
    expected: &numr::tensor::Tensor<R2>,
    dtype: DType,
    msg: &str,
) {
    let (rtol, atol) = tolerance_for_dtype(dtype);

    macro_rules! compare_native {
        ($T:ty) => {{
            let a_vec = actual.to_vec::<$T>();
            let e_vec = expected.to_vec::<$T>();
            assert_eq!(
                a_vec.len(),
                e_vec.len(),
                "{}: dtype={:?}: length mismatch ({} vs {})",
                msg,
                dtype,
                a_vec.len(),
                e_vec.len()
            );
            for (i, (a, e)) in a_vec.iter().zip(e_vec.iter()).enumerate() {
                let a_f64 = <$T as ToF64>::to_f64(*a);
                let e_f64 = <$T as ToF64>::to_f64(*e);
                let diff = (a_f64 - e_f64).abs();
                let tol = atol + rtol * e_f64.abs();
                assert!(
                    diff <= tol,
                    "{}: dtype={:?}: element {} differs: {} vs {} (diff={:.2e}, tol={:.2e})",
                    msg,
                    dtype,
                    i,
                    a_f64,
                    e_f64,
                    diff,
                    tol
                );
            }
        }};
    }

    match dtype {
        DType::F64 => compare_native!(f64),
        DType::F32 => compare_native!(f32),
        #[cfg(feature = "f16")]
        DType::F16 => compare_native!(half::f16),
        #[cfg(feature = "f16")]
        DType::BF16 => compare_native!(half::bf16),
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => compare_native!(numr::dtype::FP8E4M3),
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => compare_native!(numr::dtype::FP8E5M2),
        DType::I64 => compare_native!(i64),
        DType::I32 => compare_native!(i32),
        DType::U32 => compare_native!(u32),
        DType::Bool => compare_native!(u8),
        _ => panic!("assert_tensor_allclose: unsupported dtype {dtype:?}"),
    }
}

/// Helper trait to convert numeric types to f64 for tolerance comparison
pub trait ToF64: Copy {
    fn to_f64(self) -> f64;
}

impl ToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
impl ToF64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for u8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
#[cfg(feature = "f16")]
impl ToF64 for half::f16 {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
}
#[cfg(feature = "f16")]
impl ToF64 for half::bf16 {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
}
#[cfg(feature = "fp8")]
impl ToF64 for numr::dtype::FP8E4M3 {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
}
#[cfg(feature = "fp8")]
impl ToF64 for numr::dtype::FP8E5M2 {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
}

/// Read back a tensor as a boolean mask (Vec<bool>), regardless of its dtype.
///
/// Compare ops may return different dtypes depending on the backend and input dtype
/// (Bool/u8 on CPU, U32 on WebGPU, or the input dtype with 0/1 values).
/// This function normalizes all of them to Vec<bool> for uniform comparison.
///
/// Nonzero = true, zero = false.
pub fn readback_as_bool<R: Runtime<DType = DType>>(tensor: &numr::tensor::Tensor<R>) -> Vec<bool> {
    macro_rules! nonzero {
        ($T:ty) => {
            tensor
                .to_vec::<$T>()
                .iter()
                .map(|x| <$T as ToF64>::to_f64(*x) != 0.0)
                .collect()
        };
    }

    match tensor.dtype() {
        DType::Bool | DType::U8 => tensor.to_vec::<u8>().iter().map(|&x| x != 0).collect(),
        DType::U32 => tensor.to_vec::<u32>().iter().map(|&x| x != 0).collect(),
        DType::I32 => tensor.to_vec::<i32>().iter().map(|&x| x != 0).collect(),
        DType::F32 => nonzero!(f32),
        DType::F64 => nonzero!(f64),
        #[cfg(feature = "f16")]
        DType::F16 => nonzero!(half::f16),
        #[cfg(feature = "f16")]
        DType::BF16 => nonzero!(half::bf16),
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => nonzero!(numr::dtype::FP8E4M3),
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => nonzero!(numr::dtype::FP8E5M2),
        other => panic!("readback_as_bool: unsupported dtype {other:?}"),
    }
}

/// Macro for parameterized testing across dtypes
///
/// Usage:
/// ```ignore
/// #[test]
/// fn test_add_parity() {
///     test_all_dtypes!("cuda", dtype => {
///         // test body using `dtype`
///         let result = client.add(&a, &b)?;
///         assert_eq!(result.dtype(), dtype);
///     });
/// }
/// ```
#[macro_export]
macro_rules! test_all_dtypes {
    ($backend:expr, $dtype:ident => $body:block) => {
        for $dtype in $crate::common::supported_dtypes($backend) {
            $body
        }
    };
}

/// Macro for conditional dtype testing (only on CUDA)
///
/// Useful for tests that only work on specific backends
#[macro_export]
macro_rules! test_cuda_dtypes {
    ($dtype:ident => $body:block) => {
        #[cfg(feature = "cuda")]
        for $dtype in $crate::common::supported_dtypes("cuda") {
            $body
        }
    };
}
