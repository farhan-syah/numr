//! Common test utilities
#![allow(dead_code)]

pub mod backend_lock;

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
/// Uses the formula: |a - b| <= atol + rtol * |b|, with non-finite values
/// compared by identity (see [`values_close`]).
pub fn assert_allclose_f64(a: &[f64], b: &[f64], rtol: f64, atol: f64, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            values_close(*x, *y, rtol, atol),
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
///
/// Non-finite values are compared by identity (see [`values_close`]).
#[allow(dead_code)]
pub fn assert_allclose_f32(a: &[f32], b: &[f32], rtol: f32, atol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            values_close(*x as f64, *y as f64, rtol as f64, atol as f64),
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

/// The backend's design scope: which dtypes it can represent at all.
///
/// This is one of the two axes a parity test intersects, the other being
/// [`DTypeDomain`]. Keep it narrow and principled. WebGPU is 32-bit by design,
/// so F64 and the narrow floats are out of scope there permanently. A dtype the
/// backend could represent but has no kernel for does NOT belong here — that is
/// a gap the tests exist to surface.
pub fn backend_supported_dtypes(backend: &str) -> Vec<DType> {
    match backend {
        // CUDA carries native kernels for every float and every integer width,
        // so all of them are in scope. `matmul` on I8 widens to I32, which is
        // CPU's contract too, not a CUDA quirk.
        #[cfg(feature = "cuda")]
        "cuda" => build_dtype_list(&[
            DType::F32,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::U32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U16,
            DType::U8,
        ]),
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            // WebGPU: 32-bit types only (F32, I32, U32)
            vec![DType::F32, DType::I32, DType::U32]
        }
        _ => build_dtype_list(&[
            DType::F32,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::U32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U16,
            DType::U8,
        ]),
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

// ============================================================================
// Operation DType Domain
// ============================================================================

/// The dtypes an operation is mathematically defined for.
///
/// This is the operation's OWN domain and is independent of any backend. `log`
/// is undefined on an integer tensor however capable the hardware is, and `add`
/// is defined on every numeric dtype however few kernels a backend ships.
///
/// Keeping this separate from [`backend_supported_dtypes`] is what lets a parity
/// test run integer dtypes without claiming every op accepts them. Variants
/// derive their membership from `DType`'s own predicates, so a new dtype joins
/// the right sets without editing a list here.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DTypeDomain {
    /// Floats and integers alike: binary arithmetic, compare, reductions,
    /// cumulative, matmul, indexing, shape ops, sort, cast, logical.
    AllNumeric,
    /// Floats only: transcendentals, activations, normalization, decompositions,
    /// matrix functions, conv, distance, statistics, FFT, random distributions.
    FloatsOnly,
    /// Floats and signed integers: ops that need a representable negation.
    SignedOnly,
    /// Signed and unsigned integers only.
    IntsOnly,
}

impl DTypeDomain {
    /// True when the operation is mathematically defined for `dtype`.
    pub fn admits(self, dtype: DType) -> bool {
        match self {
            Self::AllNumeric => dtype.is_float() || dtype.is_int(),
            Self::FloatsOnly => dtype.is_float(),
            Self::SignedOnly => dtype.is_float() || dtype.is_signed_int(),
            Self::IntsOnly => dtype.is_int(),
        }
    }
}

/// Dtypes a parity test runs: the operation's domain intersected with the
/// backend's design scope.
///
/// A dtype that survives the intersection is expected to work. If the backend
/// then fails on it, the test FAILS — that gap is the reason the intersection
/// exists. There is no third axis here for a missing kernel to opt out through.
pub fn parity_dtypes(domain: DTypeDomain, backend: &str) -> Vec<DType> {
    backend_supported_dtypes(backend)
        .into_iter()
        .filter(|&dtype| domain.admits(dtype))
        .collect()
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

/// Compare one pair of values with dtype tolerance, treating non-finite values
/// by identity.
///
/// `inf - inf` is NaN and `NaN <= tol` is false, so a plain tolerance check
/// reports two backends that both produced infinity as differing. Backends
/// agreeing on infinity, or on NaN, is the correct outcome. An infinity against
/// a finite value, or a NaN against a number, still fails.
pub fn values_close(actual: f64, expected: f64, rtol: f64, atol: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    (actual - expected).abs() <= atol + rtol * expected.abs()
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
            values_close(*a, *e, rtol, atol),
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

/// Assert two tensors are close by reading each back in ITS OWN dtype.
///
/// `dtype` is the dtype the test was parameterised on. It selects the tolerance,
/// because the input precision is what bounds the achievable error, and it
/// labels the failure message. It does NOT select the read type: an op may
/// promote, and `pow_scalar` on an I32 tensor with a fractional exponent returns
/// F64. Reading those F64 bytes as I32 reinterprets the mantissa as an integer,
/// so two results agreeing to 1e-16 report as `0` vs `-1`.
///
/// The two tensors must carry the same dtype. A dtype divergence between
/// backends is itself a parity failure, and it is reported as one.
///
/// Tolerance applies to float results only. An integer result is compared
/// exactly, whatever `dtype` the test was parameterised on.
pub fn assert_tensor_allclose<R1: Runtime<DType = DType>, R2: Runtime<DType = DType>>(
    actual: &numr::tensor::Tensor<R1>,
    expected: &numr::tensor::Tensor<R2>,
    dtype: DType,
    msg: &str,
) {
    let (rtol, atol) = tolerance_for_dtype(dtype);
    let result_dtype = actual.dtype();
    assert_eq!(
        result_dtype,
        expected.dtype(),
        "{}: input dtype={:?}: result dtype divergence: {:?} vs {:?}",
        msg,
        dtype,
        result_dtype,
        expected.dtype()
    );

    macro_rules! compare_native {
        ($T:ty) => {{
            let a_vec = actual.to_vec::<$T>();
            let e_vec = expected.to_vec::<$T>();
            assert_eq!(
                a_vec.len(),
                e_vec.len(),
                "{}: dtype={:?}: length mismatch ({} vs {})",
                msg,
                result_dtype,
                a_vec.len(),
                e_vec.len()
            );
            for (i, (a, e)) in a_vec.iter().zip(e_vec.iter()).enumerate() {
                let a_f64 = <$T as ToF64>::to_f64(*a);
                let e_f64 = <$T as ToF64>::to_f64(*e);
                let diff = (a_f64 - e_f64).abs();
                let tol = atol + rtol * e_f64.abs();
                assert!(
                    values_close(a_f64, e_f64, rtol, atol),
                    "{}: dtype={:?}: element {} differs: {} vs {} (diff={:.2e}, tol={:.2e})",
                    msg,
                    result_dtype,
                    i,
                    a_f64,
                    e_f64,
                    diff,
                    tol
                );
            }
        }};
    }

    /// Integers are compared EXACTLY. They carry no rounding error, so any
    /// difference is a real divergence, and a relative tolerance would hide a
    /// large one: `1e-5` of a value near `i64::MAX` is ~9.2e13. The `as f64`
    /// conversion is itself lossy past 2^53, where distinct integers land on
    /// the same float and cannot be told apart at all.
    macro_rules! compare_int_exact {
        ($T:ty) => {{
            let a_vec = actual.to_vec::<$T>();
            let e_vec = expected.to_vec::<$T>();
            assert_eq!(
                a_vec.len(),
                e_vec.len(),
                "{}: dtype={:?}: length mismatch ({} vs {})",
                msg,
                result_dtype,
                a_vec.len(),
                e_vec.len()
            );
            for (i, (a, e)) in a_vec.iter().zip(e_vec.iter()).enumerate() {
                assert!(
                    a == e,
                    "{}: dtype={:?}: element {} differs: {} vs {}",
                    msg,
                    result_dtype,
                    i,
                    a,
                    e
                );
            }
        }};
    }

    match result_dtype {
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
        DType::I64 => compare_int_exact!(i64),
        DType::I32 => compare_int_exact!(i32),
        DType::I16 => compare_int_exact!(i16),
        DType::I8 => compare_int_exact!(i8),
        DType::U64 => compare_int_exact!(u64),
        DType::U32 => compare_int_exact!(u32),
        DType::U16 => compare_int_exact!(u16),
        DType::U8 | DType::Bool => compare_int_exact!(u8),
        other => panic!("assert_tensor_allclose: unsupported result dtype {other:?}"),
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
impl ToF64 for i16 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for i8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for u16 {
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
