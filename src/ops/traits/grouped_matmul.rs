//! Grouped matrix multiplication with device-side group boundaries.

use super::gemm_epilogue::GemmActivation;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// One independent matmul per group, with the row boundaries held on device.
///
/// Rows of `a` are partitioned into consecutive groups and each group is
/// multiplied by its own matrix:
///
/// ```text
/// c[offsets[g] .. offsets[g+1]] = a[offsets[g] .. offsets[g+1]] @ b[g]
/// ```
///
/// # Layout
///
/// - `a`: `[total_rows, k]`
/// - `b`: `[num_groups, k, n]`
/// - `group_offsets`: `[num_groups + 1]`, I32, monotonically non-decreasing,
///   first entry `0` and last entry `total_rows`
/// - output: `[total_rows, n]`
///
/// # Why the offsets are a tensor
///
/// They live wherever the tensors live. A caller that already knew the group
/// sizes on the host could launch one matmul per group instead; this operation
/// exists for the case where the partition was produced on the device and
/// reading it back would stall the queue.
pub trait GroupedMatmulOps<R: Runtime> {
    /// Grouped matmul: `c[g] = a[g] @ b[g]` for every group.
    fn grouped_matmul(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        group_offsets: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Grouped matmul with an activation fused into the epilogue:
    /// `c[g] = activation(a[g] @ b[g])`.
    ///
    /// Fusing saves a full read and write of the output compared with a
    /// separate activation pass.
    fn grouped_matmul_activation(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        group_offsets: &Tensor<R>,
        activation: GemmActivation,
    ) -> Result<Tensor<R>>;
}
