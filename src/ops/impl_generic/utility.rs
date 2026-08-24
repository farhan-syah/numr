//! Generic implementations of utility composite operations.
//!
//! These implementations are shared across GPU backends (CUDA, WebGPU) to ensure
//! numerical parity and eliminate code duplication.

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::MeshgridIndexing;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::ops::{CompareOps, TypeConversionOps, UtilityOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// One-hot encoding via broadcasting and comparison (GPU-native)
///
/// Algorithm:
/// 1. Cast integer indices to F32 for comparison
/// 2. Create class indices `[0, 1, ..., num_classes-1]`
/// 3. Reshape indices `[...] -> [..., 1]` and class indices `[1, ..., 1, num_classes]`
/// 4. Broadcast compare: `indices == class_indices`
/// 5. Cast boolean mask to F32
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn one_hot_impl<R, C>(client: &C, indices: &Tensor<R>, num_classes: usize) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: UtilityOps<R> + TypeConversionOps<R> + CompareOps<R>,
{
    if num_classes == 0 {
        return Err(Error::InvalidArgument {
            arg: "num_classes",
            reason: "one_hot requires num_classes > 0".to_string(),
        });
    }

    let idx_dtype = indices.dtype();
    if !idx_dtype.is_int() {
        return Err(Error::UnsupportedDType {
            dtype: idx_dtype,
            op: "one_hot (requires integer indices)",
        });
    }

    let numel: usize = indices.shape().iter().product();
    if numel == 0 {
        let mut out_shape = indices.shape().to_vec();
        out_shape.push(num_classes);
        return Ok(Tensor::<R>::try_empty(
            &out_shape,
            DType::F32,
            indices.device(),
        )?);
    }

    // Cast indices to F32 for comparison (GPU compare ops work on same dtype)
    let indices_f32 = if idx_dtype != DType::F32 {
        client.cast(indices, DType::F32)?
    } else {
        indices.clone()
    };

    // Create class indices [0, 1, ..., num_classes-1]
    let class_indices = client.arange(0.0, num_classes as f64, 1.0, DType::F32)?;

    // Reshape for broadcasting: indices [...] -> [..., 1], class_indices -> [1, ..., 1, num_classes]
    let mut idx_shape = indices.shape().to_vec();
    idx_shape.push(1);
    let indices_expanded = indices_f32.reshape(&idx_shape)?;

    let class_shape: Vec<usize> = std::iter::repeat(1)
        .take(indices.shape().len())
        .chain(std::iter::once(num_classes))
        .collect();
    let class_expanded = class_indices.reshape(&class_shape)?;

    // Compare: indices == class_indices (broadcasts to [..., num_classes])
    let mask = client.eq(&indices_expanded, &class_expanded)?;

    // Cast bool mask to F32
    client.cast(&mask, DType::F32)
}

/// Meshgrid implementation via reshape + broadcast_to + contiguous
///
/// Algorithm:
/// 1. For Xy indexing, swap first two input tensors
/// 2. Compute output shape from all input lengths
/// 3. For each input: reshape to have size-1 dims everywhere except its own axis,
///    then broadcast_to the full shape, then contiguous() to materialize
pub fn meshgrid_impl<R: Runtime>(
    tensors: &[&Tensor<R>],
    indexing: MeshgridIndexing,
) -> Result<Vec<Tensor<R>>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    // Validate all inputs are 1-D
    for (i, t) in tensors.iter().enumerate() {
        if t.ndim() != 1 {
            return Err(Error::InvalidArgument {
                arg: "tensors",
                reason: format!(
                    "meshgrid requires 1-D inputs, but tensor {} has shape {:?}",
                    i,
                    t.shape()
                ),
            });
        }
    }

    // For Xy indexing, swap first two inputs
    let inputs: Vec<&Tensor<R>> = if indexing == MeshgridIndexing::Xy && tensors.len() >= 2 {
        let mut v: Vec<&Tensor<R>> = tensors.to_vec();
        v.swap(0, 1);
        v
    } else {
        tensors.to_vec()
    };

    let ndim = inputs.len();

    // Compute output shape
    let output_shape: Vec<usize> = inputs.iter().map(|t| t.shape()[0]).collect();

    let mut grids = Vec::with_capacity(ndim);
    for (i, t) in inputs.iter().enumerate() {
        // Build reshape: size-1 everywhere except axis i
        let mut reshape_dims = vec![1usize; ndim];
        reshape_dims[i] = t.shape()[0];

        let reshaped = t.reshape(&reshape_dims)?;
        let broadcasted = reshaped.broadcast_to(&output_shape)?;
        let materialized = broadcasted.contiguous()?;

        grids.push(materialized);
    }

    // For Xy indexing, swap first two outputs back so output[0] corresponds to input x, output[1] to input y
    if indexing == MeshgridIndexing::Xy && grids.len() >= 2 {
        grids.swap(0, 1);
    }

    Ok(grids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuRuntime;
    use crate::tensor::Tensor;

    fn cpu_device() -> crate::runtime::cpu::CpuDevice {
        crate::runtime::cpu::CpuDevice::default()
    }

    #[test]
    fn test_meshgrid_2d_ij() {
        let device = cpu_device();
        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let y = Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 5.0], &[2], &device).unwrap();

        let grids = meshgrid_impl(&[&x, &y], MeshgridIndexing::Ij).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[3, 2]);
        assert_eq!(grids[1].shape(), &[3, 2]);

        let g0: Vec<f32> = grids[0].to_vec();
        let g1: Vec<f32> = grids[1].to_vec();
        // g0: [[1,1],[2,2],[3,3]]
        assert_eq!(g0, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // g1: [[4,5],[4,5],[4,5]]
        assert_eq!(g1, vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    #[test]
    fn test_meshgrid_2d_xy() {
        let device = cpu_device();
        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let y = Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 5.0], &[2], &device).unwrap();

        let grids = meshgrid_impl(&[&x, &y], MeshgridIndexing::Xy).unwrap();
        assert_eq!(grids.len(), 2);
        // Xy: output shape is [len(y), len(x)] = [2, 3]
        assert_eq!(grids[0].shape(), &[2, 3]);
        assert_eq!(grids[1].shape(), &[2, 3]);

        let g0: Vec<f32> = grids[0].to_vec();
        let g1: Vec<f32> = grids[1].to_vec();
        // g0 (x values): [[1,2,3],[1,2,3]]
        assert_eq!(g0, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        // g1 (y values): [[4,4,4],[5,5,5]]
        assert_eq!(g1, vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_meshgrid_3d() {
        let device = cpu_device();
        let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0, 5.0], &[3], &device).unwrap();
        let c = Tensor::<CpuRuntime>::try_from_slice(&[6.0f32, 7.0], &[2], &device).unwrap();

        let grids = meshgrid_impl(&[&a, &b, &c], MeshgridIndexing::Ij).unwrap();
        assert_eq!(grids.len(), 3);
        assert_eq!(grids[0].shape(), &[2, 3, 2]);
        assert_eq!(grids[1].shape(), &[2, 3, 2]);
        assert_eq!(grids[2].shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_meshgrid_single_input() {
        let device = cpu_device();
        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        let grids = meshgrid_impl(&[&x], MeshgridIndexing::Ij).unwrap();
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].shape(), &[3]);
        let g: Vec<f32> = grids[0].to_vec();
        assert_eq!(g, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_meshgrid_empty() {
        let grids = meshgrid_impl::<CpuRuntime>(&[], MeshgridIndexing::Ij).unwrap();
        assert!(grids.is_empty());
    }

    #[test]
    fn test_meshgrid_non_1d_error() {
        let device = cpu_device();
        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)
            .unwrap();
        let result = meshgrid_impl(&[&x], MeshgridIndexing::Ij);
        assert!(result.is_err());
    }
}
