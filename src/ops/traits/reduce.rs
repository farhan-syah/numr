//! Reduction operations trait.

use crate::error::Result;
use crate::ops::reduce::AccumulationPrecision;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Reduction operations
pub trait ReduceOps<R: Runtime> {
    /// Sum along specified dimensions
    fn sum(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Sum along specified dimensions with explicit accumulation precision.
    ///
    /// See `AccumulationPrecision` for details on precision options.
    fn sum_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    /// Mean along specified dimensions
    fn mean(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Maximum along specified dimensions
    fn max(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Maximum along specified dimensions with explicit accumulation precision.
    ///
    /// See `AccumulationPrecision` for details on precision options.
    fn max_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    /// Minimum along specified dimensions
    fn min(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Minimum along specified dimensions with explicit accumulation precision.
    ///
    /// See `AccumulationPrecision` for details on precision options.
    fn min_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    /// Product along specified dimensions
    ///
    /// Computes the product of elements along the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor containing product values
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device)?;
    /// let result = client.prod(&a, &[0], false)?; // 24.0
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn prod(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Product along specified dimensions with explicit accumulation precision.
    ///
    /// Accumulation precision is especially important for products as values
    /// can grow or shrink exponentially, causing overflow or underflow.
    ///
    /// See `AccumulationPrecision` for details on precision options.
    fn prod_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    /// Test if any element is true (non-zero) along specified dimensions.
    ///
    /// Returns true (1) if any element is non-zero along the specified dimensions,
    /// false (0) otherwise. This operation performs logical OR reduction.
    ///
    /// # Truth Value Semantics by DType
    ///
    /// The "truthiness" of a value depends on its dtype:
    ///
    /// | DType | False | True |
    /// |-------|-------|------|
    /// | Bool | `false` / 0 | `true` / 1 |
    /// | F32, F64, F16, BF16 | `0.0`, `-0.0` | Any non-zero value, including **NaN** and **±Inf** |
    /// | I8, I16, I32, I64 | `0` | Any non-zero value (positive or negative) |
    /// | U8, U16, U32, U64 | `0` | Any non-zero value |
    /// | FP8 variants | `0.0` | Any non-zero value |
    ///
    /// # Important: NaN Handling
    ///
    /// **NaN is considered truthy (non-zero).** This follows the convention that
    /// `any` checks if values are non-zero, not whether they are valid numbers.
    /// If you need to exclude NaN values, filter them first with `nan_to_num` or
    /// check with `isnan`.
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // NaN is truthy
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[0.0, f32::NAN, 0.0], &[3], &device)?;
    /// let result = client.any(&a, &[0], false)?; // 1.0 (true, because NaN ≠ 0)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of any supported dtype
    /// * `dims` - Dimensions to reduce over (empty = reduce over all dimensions)
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor with the same dtype as input, containing:
    /// - `1` (or `1.0` for floats) where any element is non-zero
    /// - `0` (or `0.0` for floats) where all elements are zero
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ReduceOps;
    ///
    /// // Float tensor - standard case
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 1.0, 0.0], &[4], &device)?;
    /// let result = client.any(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // Integer tensor
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[0i32, 0, -5, 0], &[4], &device)?;
    /// let result = client.any(&a, &[0], false)?; // 1 (true, -5 ≠ 0)
    ///
    /// // All zeros
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0], &[3], &device)?;
    /// let result = client.any(&a, &[0], false)?; // 0.0 (false)
    ///
    /// // With infinity
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, f32::INFINITY], &[2], &device)?;
    /// let result = client.any(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // 2D tensor - reduce along rows
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 1.0, 0.0, 0.0], &[2, 2], &device)?;
    /// let result = client.any(&a, &[1], false)?; // [1.0, 0.0]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # See Also
    ///
    /// * `all` - Test if all elements are true (logical AND)
    /// * `sum` - For counting non-zero elements, consider `sum(a != 0)`
    fn any(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Test if all elements are true (non-zero) along specified dimensions.
    ///
    /// Returns true (1) if all elements are non-zero along the specified dimensions,
    /// false (0) otherwise. This operation performs logical AND reduction.
    ///
    /// # Truth Value Semantics by DType
    ///
    /// The "truthiness" of a value depends on its dtype:
    ///
    /// | DType | False | True |
    /// |-------|-------|------|
    /// | Bool | `false` / 0 | `true` / 1 |
    /// | F32, F64, F16, BF16 | `0.0`, `-0.0` | Any non-zero value, including **NaN** and **±Inf** |
    /// | I8, I16, I32, I64 | `0` | Any non-zero value (positive or negative) |
    /// | U8, U16, U32, U64 | `0` | Any non-zero value |
    /// | FP8 variants | `0.0` | Any non-zero value |
    ///
    /// # Important: NaN Handling
    ///
    /// **NaN is considered truthy (non-zero).** This follows the convention that
    /// `all` checks if values are non-zero, not whether they are valid numbers.
    /// A tensor of all NaN values will return true. If you need to check for
    /// valid (non-NaN) values, use `isnan` first.
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // All NaN values → true (all are non-zero)
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[f32::NAN, f32::NAN], &[2], &device)?;
    /// let result = client.all(&a, &[0], false)?; // 1.0 (true, because NaN ≠ 0)
    ///
    /// // NaN mixed with zero → false
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[f32::NAN, 0.0], &[2], &device)?;
    /// let result = client.all(&a, &[0], false)?; // 0.0 (false, because 0.0 == 0)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of any supported dtype
    /// * `dims` - Dimensions to reduce over (empty = reduce over all dimensions)
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor with the same dtype as input, containing:
    /// - `1` (or `1.0` for floats) where all elements are non-zero
    /// - `0` (or `0.0` for floats) where any element is zero
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::ReduceOps;
    ///
    /// // Float tensor - all non-zero
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)?;
    /// let result = client.all(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // Float tensor - contains zero
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 3.0, 4.0], &[4], &device)?;
    /// let result = client.all(&a, &[0], false)?; // 0.0 (false)
    ///
    /// // Integer tensor - negative values are truthy
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[-1i32, -2, -3], &[3], &device)?;
    /// let result = client.all(&a, &[0], false)?; // 1 (true)
    ///
    /// // With infinity - Inf is truthy
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[f32::INFINITY, f32::NEG_INFINITY], &[2], &device)?;
    /// let result = client.all(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // 2D tensor - reduce along rows
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 0.0], &[2, 2], &device)?;
    /// let result = client.all(&a, &[1], false)?; // [1.0, 0.0]
    ///
    /// // Empty dimension reduction - edge case
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2, 1], &device)?;
    /// let result = client.all(&a, &[1], false)?; // [1.0, 1.0] (single element is truthy)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # See Also
    ///
    /// * `any` - Test if any element is true (logical OR)
    /// * `prod` - Product reduction (different from logical AND)
    fn all(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;
}
