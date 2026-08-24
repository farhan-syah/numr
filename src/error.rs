//! Error types for numr

use crate::dtype::DType;
use thiserror::Error;

/// Result type alias using numr's Error
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in numr operations
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Shape mismatch in an operation
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        got: Vec<usize>,
    },

    /// Shapes cannot be broadcast together
    #[error("Cannot broadcast shapes {lhs:?} and {rhs:?}")]
    BroadcastError {
        /// Left-hand side shape
        lhs: Vec<usize>,
        /// Right-hand side shape
        rhs: Vec<usize>,
    },

    /// Invalid dimension index
    #[error("Invalid dimension {dim} for tensor with {ndim} dimensions")]
    InvalidDimension {
        /// The invalid dimension
        dim: isize,
        /// Number of dimensions
        ndim: usize,
    },

    /// Unsupported dtype for an operation
    #[error("Unsupported dtype {dtype:?} for operation '{op}'")]
    UnsupportedDType {
        /// The unsupported dtype
        dtype: DType,
        /// The operation name
        op: &'static str,
    },

    /// DType mismatch between operands
    #[error("DType mismatch: {lhs:?} vs {rhs:?}")]
    DTypeMismatch {
        /// Left-hand side dtype
        lhs: DType,
        /// Right-hand side dtype
        rhs: DType,
    },

    /// Device mismatch between operands
    #[error("Device mismatch: tensors must be on the same device")]
    DeviceMismatch,

    /// Out of memory
    #[error("Out of memory: failed to allocate {size} bytes")]
    OutOfMemory {
        /// Requested size in bytes
        size: usize,
    },

    /// Tensor allocation failed, with the shape/dtype/device context
    ///
    /// Wraps the underlying allocation error (usually [`Error::OutOfMemory`])
    /// so the message names what was being allocated, not only a byte count.
    #[error("failed to allocate tensor {shape:?} {dtype} on {device}: {source}")]
    AllocFailed {
        /// Shape of the tensor that could not be allocated
        shape: Vec<usize>,
        /// Element type short name (e.g. "bf16", "q4_0")
        dtype: String,
        /// Device name (from [`Device::name`](crate::runtime::Device::name))
        device: String,
        /// The underlying allocation error
        #[source]
        source: Box<Error>,
    },

    /// Index out of bounds
    #[error("Index {index} out of bounds for dimension of size {size}")]
    IndexOutOfBounds {
        /// The invalid index
        index: usize,
        /// Size of the dimension
        size: usize,
    },

    /// Invalid argument provided to an operation
    #[error("Invalid argument '{arg}': {reason}")]
    InvalidArgument {
        /// The argument name
        arg: &'static str,
        /// Reason for invalidity
        reason: String,
    },

    /// Tensor is not contiguous when contiguous memory is required
    #[error("Operation requires contiguous tensor")]
    NotContiguous,

    /// Missing gradient in backward pass
    #[error("Missing gradient for tensor")]
    MissingGradient,

    /// Backend-specific error
    #[error("Backend error: {0}")]
    Backend(String),

    /// Backend limitation - operation valid but exceeds backend capabilities
    #[error("{backend} limitation: {operation} - {reason}")]
    BackendLimitation {
        /// The backend that has the limitation
        backend: &'static str,
        /// The operation being attempted
        operation: &'static str,
        /// Description of the limitation
        reason: String,
    },

    /// CUDA-specific error
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    /// Generic message error
    #[error("{0}")]
    Msg(String),

    /// Generic internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Feature not yet implemented
    #[error("Not implemented: {feature}")]
    NotImplemented {
        /// Description of the unimplemented feature
        feature: &'static str,
    },

    /// Cargo feature required but not enabled
    #[error(
        "{dtype:?} requires the \"{feature}\" feature. Enable it with: cargo build --features {feature}"
    )]
    FeatureRequired {
        /// The dtype that needs the feature
        dtype: DType,
        /// The cargo feature name to enable
        feature: &'static str,
    },

    /// Allocator cannot reset while allocations are still live
    #[error("Allocator busy: {active_allocations} allocations still active")]
    AllocatorBusy {
        /// Number of allocations that are still live
        active_allocations: usize,
    },

    /// Allocator is frozen — no new allocations permitted
    #[error("Allocator frozen: allocation rejected while frozen")]
    AllocatorFrozen,
}

impl Error {
    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: &[usize], got: &[usize]) -> Self {
        Self::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        }
    }

    /// Create a broadcast error
    pub fn broadcast(lhs: &[usize], rhs: &[usize]) -> Self {
        Self::BroadcastError {
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        }
    }

    /// Create an unsupported dtype error
    pub fn unsupported_dtype(dtype: DType, op: &'static str) -> Self {
        Self::UnsupportedDType { dtype, op }
    }

    /// Create a backend limitation error
    pub fn backend_limitation(
        backend: &'static str,
        operation: &'static str,
        reason: impl Into<String>,
    ) -> Self {
        Self::BackendLimitation {
            backend,
            operation,
            reason: reason.into(),
        }
    }
}
