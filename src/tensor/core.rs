//! Core Tensor type

use super::{Layout, Storage, TensorId};
use crate::dtype::{DType, DataType, Element};
use crate::error::{Error, Result};
use crate::runtime::{Device, Runtime};
use std::fmt;

/// N-dimensional array stored on a compute device
///
/// `Tensor` is the fundamental data structure in numr. It consists of:
/// - **Storage**: Reference-counted device memory
/// - **Layout**: Shape, strides, and offset defining the view into storage
/// - **DType**: Element type (determined at runtime)
///
/// # Zero-Copy Views
///
/// Operations like `transpose`, `slice`, and `reshape` create new tensors
/// that share the same underlying storage. This is achieved through:
/// - Arc-wrapped storage (reference counting)
/// - Modified layout (different strides/offset)
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # let device = CpuDevice::new();
/// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
/// let b = a.transpose(-1, -2); // Zero-copy, shares storage with a
/// # Ok::<(), numr::error::Error>(())
/// ```
pub struct Tensor<R: Runtime> {
    /// Unique ID for autograd tracking
    id: TensorId,
    /// Device memory
    storage: Storage<R>,
    /// Shape, strides, offset
    layout: Layout,
}

// ============================================================================
// Generic methods — work with ANY R::DType via DataType trait
// ============================================================================

impl<R: Runtime> Tensor<R> {
    /// Create a tensor from storage and layout
    pub fn from_parts(storage: Storage<R>, layout: Layout) -> Self {
        Self {
            id: TensorId::new(),
            storage,
            layout,
        }
    }

    /// Wrap an allocation error with the shape, dtype and device being allocated.
    ///
    /// The raw [`Error::OutOfMemory`] carries only a byte count, which forces the
    /// reader to hand-derive what failed. This attaches the context.
    fn alloc_failed(shape: &[usize], dtype: R::DType, device: &R::Device, source: Error) -> Error {
        Error::AllocFailed {
            shape: shape.to_vec(),
            dtype: dtype.short_name().to_string(),
            device: device.name(),
            source: Box::new(source),
        }
    }

    /// Create an uninitialized tensor (fallible version)
    pub fn empty(shape: &[usize], dtype: R::DType, device: &R::Device) -> Result<Self> {
        let len: usize = shape.iter().product();
        let storage = Storage::new(len, dtype, device)
            .map_err(|e| Self::alloc_failed(shape, dtype, device, e))?;
        let layout = Layout::contiguous(shape);

        Ok(Self {
            id: TensorId::new(),
            storage,
            layout,
        })
    }

    // ===== Accessors =====

    /// Get the internal tensor ID for autograd graph tracking.
    #[inline]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Get the storage
    #[inline]
    pub fn storage(&self) -> &Storage<R> {
        &self.storage
    }

    /// Get the layout
    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Get the shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Get the strides
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Get the number of dimensions (rank)
    #[inline]
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Get the total number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.layout.elem_count()
    }

    /// Get the element type
    #[inline]
    pub fn dtype(&self) -> R::DType {
        self.storage.dtype()
    }

    /// Get the device
    #[inline]
    pub fn device(&self) -> &R::Device {
        self.storage.device()
    }

    /// Check if the tensor is contiguous in memory
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Check if this is a scalar (0-dimensional tensor)
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.layout.is_scalar()
    }

    /// Get size along a dimension (supports negative indexing)
    pub fn size(&self, dim: isize) -> Option<usize> {
        self.layout.dim(dim)
    }

    /// Get size along a dimension, returning error on invalid index
    pub fn dim(&self, index: isize) -> Result<usize> {
        self.layout.dim(index).ok_or(Error::InvalidDimension {
            dim: index,
            ndim: self.ndim(),
        })
    }

    // ===== Aliases (common across tensor libraries) =====

    /// Number of dimensions (alias for `ndim`)
    #[inline]
    pub fn rank(&self) -> usize {
        self.layout.ndim()
    }

    /// Total number of elements (alias for `numel`)
    #[inline]
    pub fn elem_count(&self) -> usize {
        self.layout.elem_count()
    }

    /// Shape as slice (alias for `shape`)
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Total number of elements (alias for `numel`)
    #[inline]
    pub fn len(&self) -> usize {
        self.layout.elem_count()
    }

    /// Whether the tensor has zero elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layout.elem_count() == 0
    }

    /// Layout offset into storage (in elements)
    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset()
    }

    /// Data pointer adjusted for layout offset.
    /// This is the pointer to the first element of this tensor's view.
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.storage.ptr() + (self.layout.offset() * self.dtype().size_in_bytes()) as u64
    }

    /// Whether the underlying storage is owned (will deallocate on drop)
    #[inline]
    pub fn owns_memory(&self) -> bool {
        self.storage.is_owned()
    }

    /// Check if two tensors share the same storage
    pub fn shares_storage_with(&self, other: &Tensor<R>) -> bool {
        self.storage.ptr() == other.storage.ptr()
    }

    /// Storage reference count
    pub fn ref_count(&self) -> usize {
        self.storage.ref_count()
    }

    // ===== Dimension Unpacking =====

    /// Unpack shape of a 1D tensor
    pub fn dims1(&self) -> Result<usize> {
        let s = self.shape();
        if s.len() == 1 {
            Ok(s[0])
        } else {
            Err(Error::ShapeMismatch {
                expected: vec![0],
                got: s.to_vec(),
            })
        }
    }

    /// Unpack shape of a 2D tensor
    pub fn dims2(&self) -> Result<(usize, usize)> {
        let s = self.shape();
        if s.len() == 2 {
            Ok((s[0], s[1]))
        } else {
            Err(Error::ShapeMismatch {
                expected: vec![0, 0],
                got: s.to_vec(),
            })
        }
    }

    /// Unpack shape of a 3D tensor
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        let s = self.shape();
        if s.len() == 3 {
            Ok((s[0], s[1], s[2]))
        } else {
            Err(Error::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: s.to_vec(),
            })
        }
    }

    /// Unpack shape of a 4D tensor
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let s = self.shape();
        if s.len() == 4 {
            Ok((s[0], s[1], s[2], s[3]))
        } else {
            Err(Error::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: s.to_vec(),
            })
        }
    }

    /// Unpack shape of a 5D tensor
    pub fn dims5(&self) -> Result<(usize, usize, usize, usize, usize)> {
        let s = self.shape();
        if s.len() == 5 {
            Ok((s[0], s[1], s[2], s[3], s[4]))
        } else {
            Err(Error::ShapeMismatch {
                expected: vec![0, 0, 0, 0, 0],
                got: s.to_vec(),
            })
        }
    }

    // ===== Construction Helpers =====

    /// Create tensor from storage and contiguous layout
    pub fn from_storage_contiguous(storage: Storage<R>, shape: &[usize]) -> Self {
        Self {
            id: TensorId::new(),
            storage,
            layout: Layout::contiguous(shape),
        }
    }

    // ===== View Operations (Zero-Copy) =====

    /// Transpose two dimensions (zero-copy)
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Result<Self> {
        let new_layout =
            self.layout
                .transpose(dim0, dim1)
                .ok_or_else(|| Error::InvalidDimension {
                    dim: dim0,
                    ndim: self.ndim(),
                })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Transpose last two dimensions (matrix transpose)
    pub fn t(&self) -> Result<Self> {
        self.transpose(-2, -1)
    }

    /// Reshape to a new shape (zero-copy if contiguous)
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let new_layout = self.layout.reshape(shape).ok_or(Error::NotContiguous)?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Flatten to 1D (zero-copy if contiguous)
    pub fn flatten(&self) -> Result<Self> {
        self.reshape(&[self.numel()])
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self, dim: Option<isize>) -> Self {
        Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.squeeze(dim),
        }
    }

    /// Add a dimension of size 1
    pub fn unsqueeze(&self, dim: isize) -> Result<Self> {
        let new_layout = self
            .layout
            .unsqueeze(dim)
            .ok_or_else(|| Error::InvalidDimension {
                dim,
                ndim: self.ndim(),
            })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// View tensor with different shape (alias for reshape)
    pub fn view(&self, shape: &[usize]) -> Result<Self> {
        self.reshape(shape)
    }

    /// Permute dimensions (zero-copy)
    ///
    /// Reorders the dimensions of the tensor according to the given permutation.
    /// This is a view operation - no data is copied.
    ///
    /// # Arguments
    /// * `dims` - New order of dimensions. Must be a permutation of 0..ndim.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let data = vec![0.0f32; 24];
    /// let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device)?;
    /// let permuted = tensor.permute(&[2, 0, 1])?; // Shape becomes [4, 2, 3]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let new_layout = self
            .layout
            .permute(dims)
            .ok_or_else(|| Error::InvalidDimension {
                dim: dims.first().copied().unwrap_or(0) as isize,
                ndim: self.ndim(),
            })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Narrow a dimension (zero-copy slice)
    ///
    /// Returns a view of the tensor narrowed to a contiguous subset of elements
    /// along a single dimension. This is a view operation - no data is copied.
    ///
    /// # Arguments
    /// * `dim` - Dimension to narrow (supports negative indexing)
    /// * `start` - Starting index in that dimension
    /// * `length` - Number of elements to keep
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let data = vec![0.0f32; 120];
    /// let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[4, 5, 6], &device)?;
    /// let narrowed = tensor.narrow(1, 1, 3)?; // Shape becomes [4, 3, 6]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn narrow(&self, dim: isize, start: usize, length: usize) -> Result<Self> {
        let dim_idx = self
            .layout
            .normalize_dim(dim)
            .ok_or(Error::InvalidDimension {
                dim,
                ndim: self.ndim(),
            })?;

        let new_layout =
            self.layout
                .narrow(dim_idx, start, length)
                .ok_or_else(|| Error::ShapeMismatch {
                    expected: vec![self.shape()[dim_idx]],
                    got: vec![start, length],
                })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Broadcast to a target shape (zero-copy)
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let new_layout = self
            .layout
            .broadcast_to(shape)
            .ok_or_else(|| Error::BroadcastError {
                lhs: self.shape().to_vec(),
                rhs: shape.to_vec(),
            })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Flip (reverse) tensor along a dimension (zero-copy)
    ///
    /// Reverses the order of elements along the specified dimension.
    /// This is a view operation - no data is copied.
    ///
    /// # Arguments
    /// * `dim` - Dimension to flip (supports negative indexing)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let flipped = tensor.flip(0)?; // Reverse rows: [[3, 4], [1, 2]]
    /// let flipped = tensor.flip(-1)?; // Reverse columns: [[2, 1], [4, 3]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn flip(&self, dim: isize) -> Result<Self> {
        let new_layout = self.layout.flip(dim).ok_or(Error::InvalidDimension {
            dim,
            ndim: self.ndim(),
        })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Flip (reverse) tensor along multiple dimensions (zero-copy)
    ///
    /// # Arguments
    /// * `dims` - Dimensions to flip (supports negative indexing)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let data = vec![0.0f32; 24];
    /// let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device)?;
    /// let flipped = tensor.flip_dims(&[0, 2])?; // Flip first and last dimensions
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn flip_dims(&self, dims: &[isize]) -> Result<Self> {
        let new_layout = self
            .layout
            .flip_dims(dims)
            .ok_or_else(|| Error::InvalidDimension {
                dim: dims.first().copied().unwrap_or(0),
                ndim: self.ndim(),
            })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Make tensor contiguous (copy if needed)
    ///
    /// If the tensor is already contiguous, returns a clone (zero-copy view).
    /// Otherwise, allocates new storage and copies the data to a contiguous layout.
    ///
    /// # Errors
    ///
    /// Returns `Err` if device memory allocation fails (OOM) or if the strided
    /// copy operation fails.
    ///
    /// # Backend Support
    ///
    /// This method uses `Runtime::copy_strided` which handles strided copies
    /// correctly for each backend:
    /// - CPU/CUDA: Uses pointer arithmetic (handles can be offset directly)
    /// - WGPU: Uses compute shader (buffer IDs don't support arithmetic)
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() && self.layout.offset() == 0 {
            Ok(self.clone())
        } else {
            // Need to copy data to a new contiguous storage
            let dtype = self.dtype();
            let device = self.storage.device();
            let numel = self.numel();

            // Allocate new contiguous storage
            let new_storage = Storage::new(numel, dtype, device)?;
            let new_layout = Layout::contiguous(self.shape());

            // Use Runtime::copy_strided which handles buffer ID vs pointer correctly
            let elem_size = dtype.size_in_bytes();
            let src_handle = self.storage.ptr();
            let dst_handle = new_storage.ptr();
            let src_byte_offset = self.layout.offset() * elem_size;

            R::copy_strided(
                src_handle,
                src_byte_offset,
                dst_handle,
                self.shape(),
                self.strides(),
                elem_size,
                device,
            )?;

            Ok(Self {
                id: TensorId::new(),
                storage: new_storage,
                layout: new_layout,
            })
        }
    }

    /// Detach from computation graph (for autograd)
    pub fn detach(&self) -> Self {
        Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }

    // ===== Data Access =====

    /// Copy tensor data to a Vec on the host
    ///
    /// For contiguous tensors, this copies only the viewed portion of the storage,
    /// respecting the tensor's shape and offset.
    /// # Panics
    /// Panics if the tensor is not contiguous, or if the device-to-host copy
    /// fails. Use [`Self::try_to_vec`] in fallible contexts (training loops,
    /// servers) where a device error must not abort the process.
    #[track_caller]
    pub fn to_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        assert!(
            self.is_contiguous(),
            "Tensor must be contiguous to copy to vec"
        );

        self.try_to_vec()
            .expect("copy_from_device failed in to_vec()")
    }

    /// Copy tensor data to a Vec on the host (fallible version)
    ///
    /// Returns [`Error::NotContiguous`] if the tensor is not contiguous, and
    /// propagates any device-to-host copy failure instead of panicking.
    pub fn try_to_vec<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
        if !self.is_contiguous() {
            return Err(Error::NotContiguous);
        }

        let numel = self.numel();
        let offset = self.layout.offset();
        let elem_size = std::mem::size_of::<T>();
        let byte_offset = offset * elem_size;

        // Allocate with correct alignment for T, then cast to bytes for copy.
        // This avoids alignment violations that would occur if we allocated
        // a Vec<u8> and cast to stricter-aligned types like f64/i64.
        let mut result = vec![T::zeroed(); numel];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        let src_ptr = self.storage.ptr() as usize + byte_offset;
        R::copy_from_device(src_ptr as u64, bytes, self.storage.device())?;
        Ok(result)
    }

    /// Record an event on the compute stream for this tensor's device.
    ///
    /// Call this BEFORE launching additional compute work, then pass the event
    /// to `to_vec_pipelined` AFTER launching the compute work. This allows the
    /// copy to proceed as soon as the event fires, while compute continues.
    pub fn record_event(&self) -> crate::error::Result<u64> {
        R::record_compute_event(self.storage.device())
    }

    /// Copy tensor data to a Vec using the pipelined copy stream, synchronized
    /// via a previously recorded event.
    ///
    /// On CUDA, syncs only the copy stream — compute stream keeps running.
    pub fn to_vec_pipelined<T: bytemuck::Pod>(&self, event: u64) -> crate::error::Result<Vec<T>> {
        if !self.is_contiguous() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: vec![self.numel()],
                got: self.shape().to_vec(),
            });
        }

        let numel = self.numel();
        let offset = self.layout.offset();
        let elem_size = std::mem::size_of::<T>();
        let byte_offset = offset * elem_size;

        let mut result = vec![T::zeroed(); numel];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        let src_ptr = self.storage.ptr() as usize + byte_offset;
        R::copy_from_device_pipelined(src_ptr as u64, bytes, self.storage.device(), event)?;
        Ok(result)
    }

    /// Extract the scalar value from a single-element tensor
    ///
    /// This is the idiomatic way to get a scalar value from a tensor for use
    /// in Rust control flow (convergence checks, comparisons, etc.).
    ///
    /// # Returns
    ///
    /// The single element as type `T`, or an error if the tensor doesn't
    /// contain exactly one element.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let loss = Tensor::<CpuRuntime>::from_slice(&[0.5f32], &[], &device)?;
    /// let loss_val: f32 = loss.item()?;
    /// if loss_val < 1.0 {
    ///     // training converged
    /// }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn item<T: bytemuck::Pod + Copy>(&self) -> Result<T> {
        if self.numel() != 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![1],
                got: self.shape().to_vec(),
            });
        }

        // For non-contiguous single-element tensors, compute the actual offset
        // from strides (the element may not be at offset 0 in storage)
        let tensor = if self.is_contiguous() {
            std::borrow::Cow::Borrowed(self)
        } else {
            std::borrow::Cow::Owned(self.contiguous()?)
        };

        // Direct single-element copy without Vec allocation
        let offset = tensor.layout.offset();
        let elem_size = std::mem::size_of::<T>();
        let byte_offset = offset * elem_size;
        let src_ptr = (tensor.storage.ptr() as usize + byte_offset) as u64;

        let mut result = T::zeroed();
        let bytes: &mut [u8] = bytemuck::bytes_of_mut(&mut result);
        R::copy_from_device(src_ptr, bytes, tensor.storage.device())?;
        Ok(result)
    }
}

// ============================================================================
// Generic constructors (work with ANY R::DType via DataType trait)
// ============================================================================

impl<R: Runtime> Tensor<R> {
    /// Create a tensor filled with zeros (generic, works with any DType)
    pub fn zeros_generic(shape: &[usize], dtype: R::DType, device: &R::Device) -> Result<Self> {
        Self::full_scalar_generic(shape, dtype, 0.0, device)
    }

    /// Create a tensor filled with ones (generic, works with any DType)
    pub fn ones_generic(shape: &[usize], dtype: R::DType, device: &R::Device) -> Result<Self> {
        Self::full_scalar_generic(shape, dtype, 1.0, device)
    }

    /// Create a tensor filled with a scalar value (generic, works with any DType)
    ///
    /// Uses `DataType::fill_bytes` to generate the fill pattern, so it works
    /// with any DType that implements the trait (including boostr's quantized types).
    pub fn full_scalar_generic(
        shape: &[usize],
        dtype: R::DType,
        value: f64,
        device: &R::Device,
    ) -> Result<Self> {
        let len: usize = shape.iter().product();
        if len == 0 {
            return Self::empty(shape, dtype, device);
        }

        let bytes = dtype.fill_bytes(value, len).ok_or_else(|| {
            Error::Msg(format!(
                "fill not supported for dtype {}",
                dtype.short_name()
            ))
        })?;

        let storage = Storage::from_bytes(&bytes, dtype, device)
            .map_err(|e| Self::alloc_failed(shape, dtype, device, e))?;
        let layout = Layout::contiguous(shape);

        Ok(Self {
            id: TensorId::new(),
            storage,
            layout,
        })
    }

    /// Create a tensor from raw bytes with specified dtype (generic)
    pub fn from_bytes(
        bytes: &[u8],
        shape: &[usize],
        dtype: R::DType,
        device: &R::Device,
    ) -> Result<Self> {
        let storage = Storage::from_bytes(bytes, dtype, device)?;
        let layout = Layout::contiguous(shape);
        Ok(Self {
            id: TensorId::new(),
            storage,
            layout,
        })
    }
}

// ============================================================================
// Constructors that require numr's standard DType (for variant matching)
// ============================================================================

impl<R: Runtime<DType = DType>> Tensor<R> {
    /// Create a tensor from a slice of data
    ///
    /// Returns an error if `data.len()` does not equal the product of the `shape` dimensions,
    /// or if memory allocation fails.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn from_slice<T: Element>(data: &[T], shape: &[usize], device: &R::Device) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }

        let storage = Storage::from_slice(data, device)
            .map_err(|e| Self::alloc_failed(shape, T::DTYPE, device, e))?;
        let layout = Layout::contiguous(shape);

        Ok(Self {
            id: TensorId::new(),
            storage,
            layout,
        })
    }

    /// Create a tensor filled with zeros
    ///
    /// This properly initializes memory to zero on all backends (CPU and GPU).
    pub fn zeros(shape: &[usize], dtype: DType, device: &R::Device) -> Result<Self> {
        Self::full_scalar(shape, dtype, 0.0, device)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize], dtype: DType, device: &R::Device) -> Result<Self> {
        Self::full_scalar(shape, dtype, 1.0, device)
    }

    /// Create a tensor filled with a scalar value
    ///
    /// The scalar is converted to the target dtype.
    pub fn full_scalar(
        shape: &[usize],
        dtype: DType,
        value: f64,
        device: &R::Device,
    ) -> Result<Self> {
        // Helper to convert a typed Vec to bytes safely.
        // Allocates with correct alignment for T, then copies to u8 vec.
        #[inline]
        fn typed_to_bytes<T: bytemuck::NoUninit>(v: Vec<T>) -> Vec<u8> {
            bytemuck::cast_slice::<T, u8>(&v).to_vec()
        }

        let len: usize = shape.iter().product();
        if len == 0 {
            return Self::empty(shape, dtype, device);
        }

        // Allocate with correct type alignment, then convert to bytes.
        // This avoids alignment violations that would occur if we allocated
        // a Vec<u8> and cast to stricter-aligned types like f64/i64.
        let bytes: Vec<u8> = match dtype {
            DType::F64 => typed_to_bytes(vec![value; len]),
            DType::F32 => typed_to_bytes(vec![value as f32; len]),
            DType::F16 => {
                #[cfg(feature = "f16")]
                {
                    use half::f16;
                    typed_to_bytes(vec![f16::from_f64(value); len])
                }
                #[cfg(not(feature = "f16"))]
                {
                    let half_bits = half_from_f32(value as f32, dtype);
                    typed_to_bytes(vec![half_bits; len])
                }
            }
            DType::BF16 => {
                #[cfg(feature = "f16")]
                {
                    use half::bf16;
                    typed_to_bytes(vec![bf16::from_f64(value); len])
                }
                #[cfg(not(feature = "f16"))]
                {
                    let half_bits = half_from_f32(value as f32, dtype);
                    typed_to_bytes(vec![half_bits; len])
                }
            }
            DType::FP8E4M3 => {
                vec![crate::dtype::FP8E4M3::from_f32(value as f32).to_bits(); len]
            }
            DType::FP8E5M2 => {
                vec![crate::dtype::FP8E5M2::from_f32(value as f32).to_bits(); len]
            }
            DType::I64 => typed_to_bytes(vec![value as i64; len]),
            DType::I32 => typed_to_bytes(vec![value as i32; len]),
            DType::I16 => typed_to_bytes(vec![value as i16; len]),
            DType::I8 => typed_to_bytes(vec![value as i8; len]),
            DType::U64 => typed_to_bytes(vec![value as u64; len]),
            DType::U32 => typed_to_bytes(vec![value as u32; len]),
            DType::U16 => typed_to_bytes(vec![value as u16; len]),
            DType::U8 => vec![value as u8; len],
            DType::Bool => vec![if value != 0.0 { 1u8 } else { 0u8 }; len],
            DType::Complex64 => {
                typed_to_bytes(vec![crate::dtype::Complex64::new(value as f32, 0.0); len])
            }
            DType::Complex128 => {
                typed_to_bytes(vec![crate::dtype::Complex128::new(value, 0.0); len])
            }
        };

        // Allocate and copy to device
        let storage = Storage::from_bytes(&bytes, dtype, device)
            .map_err(|e| Self::alloc_failed(shape, dtype, device, e))?;
        let layout = Layout::contiguous(shape);

        Ok(Self {
            id: TensorId::new(),
            storage,
            layout,
        })
    }
}

// ============================================================================
// Foundational ops (generic — work with any R::DType)
// ============================================================================

impl<R: Runtime> Tensor<R> {
    /// Serialize tensor data to raw bytes
    ///
    /// Makes tensor contiguous first if needed, then copies raw bytes from device.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let tensor = if self.is_contiguous() && self.offset() == 0 {
            std::borrow::Cow::Borrowed(self)
        } else {
            std::borrow::Cow::Owned(self.contiguous()?)
        };
        let size = tensor.numel() * tensor.dtype().size_in_bytes();
        let mut data = vec![0u8; size];
        R::copy_from_device(tensor.storage().ptr(), &mut data, tensor.storage().device())
            .map_err(|e| Error::Msg(format!("to_bytes copy failed: {}", e)))?;
        Ok(data)
    }

    /// Clone tensor with new storage (deep copy)
    pub fn clone_deep(&self) -> Result<Self> {
        let bytes = self.to_bytes()?;
        Self::from_bytes(&bytes, self.shape(), self.dtype(), self.device())
    }
}

impl<R: Runtime> Clone for Tensor<R> {
    /// Clone creates a new tensor sharing the same storage (zero-copy)
    fn clone(&self) -> Self {
        Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<R: Runtime> fmt::Debug for Tensor<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id)
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("contiguous", &self.is_contiguous())
            .finish()
    }
}

impl<R: Runtime> fmt::Display for Tensor<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({:?}, dtype={})",
            self.shape(),
            self.dtype().short_name()
        )
    }
}

/// Convert f32 to half-precision bit representation
///
/// This is a simple conversion that handles common cases.
/// For full IEEE 754 compliance, use the `half` crate (enabled with `f16` feature).
#[cfg(not(feature = "f16"))]
fn half_from_f32(value: f32, dtype: DType) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if dtype == DType::BF16 {
        // BF16: truncate mantissa, keep exponent
        ((bits >> 16) & 0xFFFF) as u16
    } else {
        // F16: IEEE 754 half precision
        if exp == 0 {
            // Zero or subnormal
            (sign << 15) as u16
        } else if exp == 0xFF {
            // Inf or NaN
            ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16
        } else {
            // Normal number
            let new_exp = exp - 127 + 15;
            if new_exp <= 0 {
                // Underflow to zero
                (sign << 15) as u16
            } else if new_exp >= 31 {
                // Overflow to infinity
                ((sign << 15) | 0x7C00) as u16
            } else {
                ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_from_slice() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.numel(), 6);

        let result: Vec<f32> = tensor.to_vec();
        assert_eq!(result, data);
    }

    #[test]
    fn test_transpose() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();

        let transposed = tensor.transpose(0, 1).unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert!(!transposed.is_contiguous()); // Transpose makes it non-contiguous
        assert_eq!(transposed.numel(), 6);
    }

    #[test]
    fn test_contiguous_already_contiguous() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &device).unwrap();

        assert!(tensor.is_contiguous());
        let contiguous = tensor.contiguous().unwrap();
        assert!(contiguous.is_contiguous());

        let result: Vec<f32> = contiguous.to_vec();
        assert_eq!(result, data);
    }

    #[test]
    fn test_contiguous_from_transpose() {
        let device = CpuDevice::new();
        // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();

        // Transpose to 3x2: [[1, 4], [2, 5], [3, 6]]
        let transposed = tensor.transpose(0, 1).unwrap();
        assert!(!transposed.is_contiguous());

        // Make contiguous - should copy data
        let contiguous = transposed.contiguous().unwrap();
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape(), &[3, 2]);

        // Verify the data is in the correct order for row-major layout
        // Row 0: [1, 4], Row 1: [2, 5], Row 2: [3, 6]
        let result: Vec<f32> = contiguous.to_vec();
        assert_eq!(result, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_reshape() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();

        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert!(reshaped.is_contiguous());

        let result: Vec<f32> = reshaped.to_vec();
        assert_eq!(result, data); // Data unchanged, just reinterpreted
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[1, 3, 1], &device).unwrap();

        // Squeeze all dimensions of size 1
        let squeezed = tensor.squeeze(None);
        assert_eq!(squeezed.shape(), &[3]);

        // Unsqueeze at dimension 0
        let unsqueezed = squeezed.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3]);
    }

    #[test]
    fn test_zeros() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::zeros(&[2, 3], DType::F32, &device).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.is_contiguous());

        let result: Vec<f32> = tensor.to_vec();
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.is_contiguous());

        let result: Vec<f32> = tensor.to_vec();
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_full_scalar() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::full_scalar(&[2, 2], DType::I32, 42.0, &device).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), DType::I32);

        let result: Vec<i32> = tensor.to_vec();
        assert_eq!(result, [42, 42, 42, 42]);
    }

    #[test]
    fn test_item_scalar() {
        let device = CpuDevice::new();

        // 0-dimensional scalar
        let tensor =
            Tensor::<CpuRuntime>::from_slice(&[std::f32::consts::PI], &[], &device).unwrap();
        let val: f32 = tensor.item().unwrap();
        assert!((val - std::f32::consts::PI).abs() < 1e-6);

        // Shape [1] tensor
        let tensor = Tensor::<CpuRuntime>::from_slice(&[42.0f64], &[1], &device).unwrap();
        let val: f64 = tensor.item().unwrap();
        assert!((val - 42.0).abs() < 1e-10);

        // Shape [1, 1, 1] tensor
        let tensor = Tensor::<CpuRuntime>::from_slice(&[7i32], &[1, 1, 1], &device).unwrap();
        let val: i32 = tensor.item().unwrap();
        assert_eq!(val, 7);
    }

    #[test]
    fn test_item_error_on_multi_element() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();

        let result: Result<f32> = tensor.item();
        assert!(result.is_err());
    }

    /// An allocation failure must name what was being allocated.
    ///
    /// 2^60 F32 elements is 4 EiB — representable as a `Layout` (under
    /// `isize::MAX`) so the request reaches the system allocator, which cannot
    /// satisfy it. The resulting error must carry shape, dtype and device;
    /// the raw byte count alone forces the reader to hand-derive the tensor.
    ///
    /// Sabotage check: drop the `map_err` in `empty` so the bare
    /// `Error::OutOfMemory` propagates. The message becomes
    /// `Out of memory: failed to allocate 4611686018427387904 bytes` and every
    /// assertion below fails — it contains no shape, no dtype, no device.
    #[test]
    fn alloc_failure_names_shape_dtype_and_device() {
        let device = CpuDevice::new();
        let shape = [1usize << 40, 1usize << 20];

        let err = Tensor::<CpuRuntime>::empty(&shape, DType::F32, &device)
            .expect_err("4 EiB must not allocate");
        let msg = err.to_string();

        assert!(msg.contains("[1099511627776, 1048576]"), "no shape: {msg}");
        assert!(msg.contains("f32"), "no dtype: {msg}");
        assert!(msg.contains("cpu"), "no device: {msg}");
        assert!(
            matches!(err, Error::AllocFailed { ref source, .. }
                if matches!(**source, Error::OutOfMemory { .. })),
            "the underlying OutOfMemory must be preserved as the source: {msg}"
        );
    }

    /// `try_to_vec` is the fallible twin of `to_vec`, not a different readback.
    ///
    /// Sabotage check: make `try_to_vec` return `Ok(Vec::new())` and both the
    /// value assertion and the `to_vec` equality assertion fail.
    #[test]
    fn try_to_vec_matches_to_vec() {
        let device = CpuDevice::new();
        let data = [1.5f32, -2.25, 3.75, 0.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &device).unwrap();

        let got: Vec<f32> = tensor.try_to_vec().expect("readback runs");
        assert_eq!(got, vec![1.5f32, -2.25, 3.75, 0.0]);
        assert_eq!(got, tensor.to_vec::<f32>());
    }

    /// A non-contiguous tensor is an error for `try_to_vec`, where `to_vec`
    /// asserts. Both refuse to read a strided view as if it were packed.
    #[test]
    fn try_to_vec_rejects_a_non_contiguous_tensor() {
        let device = CpuDevice::new();
        let tensor =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let transposed = tensor.transpose(0, 1).expect("transpose runs");

        assert!(matches!(
            transposed.try_to_vec::<f32>(),
            Err(Error::NotContiguous)
        ));
    }
}
