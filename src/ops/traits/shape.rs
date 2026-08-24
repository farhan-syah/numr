//! Shape manipulation operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Shape manipulation operations
pub trait ShapeOps<R: Runtime> {
    /// Concatenate tensors along a dimension
    ///
    /// Joins a sequence of tensors along an existing dimension. All tensors must
    /// have the same shape except in the concatenation dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensor references to concatenate
    /// * `dim` - Dimension along which to concatenate (supports negative indexing)
    ///
    /// # Returns
    ///
    /// New tensor containing the concatenated data
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device)?;
    /// let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0, 5.0], &[3], &device)?;
    /// let c = client.cat(&[&a, &b], 0)?; // Shape: [5]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn cat(&self, tensors: &[&Tensor<R>], dim: isize) -> Result<Tensor<R>>;

    /// Stack tensors along a new dimension
    ///
    /// Joins a sequence of tensors along a new dimension. All tensors must have
    /// exactly the same shape.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensor references to stack
    /// * `dim` - Dimension at which to insert the new stacking dimension
    ///
    /// # Returns
    ///
    /// New tensor with an additional dimension
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device)?;
    /// let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], &device)?;
    /// let c = client.stack(&[&a, &b], 0)?; // Shape: [2, 2]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn stack(&self, tensors: &[&Tensor<R>], dim: isize) -> Result<Tensor<R>>;

    /// Split a tensor into chunks of a given size along a dimension
    ///
    /// Splits the tensor into chunks. The last chunk will be smaller if the
    /// dimension size is not evenly divisible by split_size.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to split
    /// * `split_size` - Size of each chunk (except possibly the last)
    /// * `dim` - Dimension along which to split (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Vector of tensor views (zero-copy) into the original tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device)?;
    /// let chunks = client.split(&a, 2, 0)?; // [2], [2], [1]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn split(&self, tensor: &Tensor<R>, split_size: usize, dim: isize) -> Result<Vec<Tensor<R>>>;

    /// Split a tensor into a specific number of chunks along a dimension
    ///
    /// Splits the tensor into approximately equal chunks. If the dimension
    /// is not evenly divisible, earlier chunks will be one element larger.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to chunk
    /// * `chunks` - Number of chunks to create
    /// * `dim` - Dimension along which to chunk (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Vector of tensor views (zero-copy) into the original tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device)?;
    /// let chunks = client.chunk(&a, 2, 0)?; // [3], [2]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn chunk(&self, tensor: &Tensor<R>, chunks: usize, dim: isize) -> Result<Vec<Tensor<R>>>;

    /// Repeat tensor along each dimension
    ///
    /// Creates a new tensor by repeating the input tensor along each dimension.
    /// The `repeats` slice specifies how many times to repeat along each dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `repeats` - Number of repetitions for each dimension. Length must match tensor ndim.
    ///
    /// # Returns
    ///
    /// New tensor with shape `` `[dim_0 * repeats[0], dim_1 * repeats[1], ...]` ``
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let repeated = client.repeat(&a, &[2, 3])?; // Shape: [4, 6]
    /// // Result: [[1,2,1,2,1,2], [3,4,3,4,3,4], [1,2,1,2,1,2], [3,4,3,4,3,4]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn repeat(&self, tensor: &Tensor<R>, repeats: &[usize]) -> Result<Tensor<R>>;

    /// Pad tensor with a constant value
    ///
    /// Adds padding to the tensor along specified dimensions. The `padding` slice
    /// contains pairs of (before, after) padding sizes, starting from the last dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `padding` - Padding sizes as pairs: `` `[last_before, last_after, second_last_before, ...]` ``
    /// * `value` - Value to use for padding
    ///
    /// # Returns
    ///
    /// New tensor with padded dimensions
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// // Pad last dim by 1 on each side
    /// let padded = client.pad(&a, &[1, 1], 0.0)?; // Shape: [2, 4]
    /// // Result: [[0,1,2,0], [0,3,4,0]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn pad(&self, tensor: &Tensor<R>, padding: &[usize], value: f64) -> Result<Tensor<R>>;

    /// Roll tensor elements along a dimension
    ///
    /// Shifts elements circularly along a dimension. Elements that roll beyond
    /// the last position wrap around to the first position.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `shift` - Number of positions to shift (negative = shift left, positive = shift right)
    /// * `dim` - Dimension along which to roll (supports negative indexing)
    ///
    /// # Returns
    ///
    /// New tensor with rolled elements
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)?;
    /// let rolled = client.roll(&a, 1, 0)?; // [4, 1, 2, 3]
    /// let rolled = client.roll(&a, -1, 0)?; // [2, 3, 4, 1]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn roll(&self, tensor: &Tensor<R>, shift: isize, dim: isize) -> Result<Tensor<R>>;

    /// Extract sliding local windows along a dimension.
    ///
    /// Returns a tensor containing all windows of length `size` sampled every `step`
    /// elements along `dim`. The output has one extra dimension, with the window-size
    /// dimension appended at the end.
    ///
    /// If input shape is `` `[d0, ..., d_dim, ..., dn]` ``, output shape is
    /// `` `[d0, ..., num_windows, ..., dn, size]` `` where:
    /// `` `num_windows = (d_dim - size) / step + 1` ``.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `dim` - Dimension along which to extract windows (supports negative indexing)
    /// * `size` - Window size (must be > 0 and <= dimension size)
    /// * `step` - Stride between window starts (must be > 0)
    ///
    /// # Returns
    ///
    /// New tensor containing extracted windows
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device)?;
    /// let windows = client.unfold(&a, 0, 3, 1)?; // Shape: [3, 3]
    /// // Result: [[1,2,3], [2,3,4], [3,4,5]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn unfold(&self, tensor: &Tensor<R>, dim: isize, size: usize, step: usize)
    -> Result<Tensor<R>>;

    /// Repeat each element along a dimension.
    ///
    /// Unlike `repeat`, which tiles whole tensor blocks along each dimension,
    /// `repeat_interleave` repeats individual elements in-place along one dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `repeats` - Number of times to repeat each element (must be > 0)
    /// * `dim` - Dimension to repeat along (supports negative indexing). If `None`, input is flattened first.
    ///
    /// # Returns
    ///
    /// New tensor with repeated elements
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ShapeOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    /// let out = client.repeat_interleave(&a, 2, Some(0))?;
    /// // Result: [1, 1, 2, 2, 3, 3]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn repeat_interleave(
        &self,
        tensor: &Tensor<R>,
        repeats: usize,
        dim: Option<isize>,
    ) -> Result<Tensor<R>>;
}
