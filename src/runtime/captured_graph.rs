//! Destination-passing CUDA graph capture result.
//!
//! [`CapturedGraph`] bundles a compiled computation graph together with the
//! externally-owned tensors it reads from and writes into. Holding `Arc` clones
//! of those tensors guarantees their device addresses stay valid for the
//! lifetime of the graph.

use crate::runtime::{Graph, Runtime};
use crate::tensor::Tensor;

/// A captured computation graph plus the externally-owned I/O tensors.
///
/// Produced by [`Runtime::capture_graph_into`] or
/// `CudaRuntime::capture_graph_into_with_arena`. The caller passes explicit
/// `inputs` and `outputs` slices before capture begins, and the closure writes
/// into those tensors in-place. `CapturedGraph` then holds `Arc` clones of all
/// I/O tensors so their device addresses cannot be freed while replay is
/// possible.
///
/// # Drop ordering
///
/// Fields are declared in the order `graph`, `inputs`, `outputs`, `arena`.
/// Rust drops struct fields in **declaration order**, so `graph` is always
/// destroyed before `inputs`, `outputs`, and `arena`. This is **load-bearing**:
///
/// - On some NVIDIA driver versions `cuGraphExecDestroy` dereferences the
///   device pointers encoded into the graph at capture time. The I/O tensors
///   must therefore still be alive when the graph handle is destroyed.
/// - The `arena` tensor holds the device buffer whose addresses are baked into
///   the graph's kernel-parameter blocks for graph-internal intermediate
///   allocations. It must outlive the graph handle AND the I/O tensors to
///   guarantee no use-after-free during `cuGraphExecDestroy`.
///
/// **DO NOT REORDER THESE FIELDS.**
pub struct CapturedGraph<R: Runtime> {
    // Field order is load-bearing. Rust drops fields in declaration order.
    // graph → inputs → outputs → arena
    //
    // `graph` drops first because cuGraphExecDestroy dereferences device
    // pointers from the I/O tensors (and arena) on some driver versions.
    //
    // `arena` drops LAST so that the device buffer it owns (whose addresses
    // are baked into the graph's kernel nodes) outlives both the graph handle
    // and any references the I/O tensors may have into it during destruction.
    //
    // DO NOT REORDER.
    graph: R::Graph,
    inputs: Vec<Tensor<R>>,
    outputs: Vec<Tensor<R>>,
    /// Graph-internal scratch arena buffer. When present, every intermediate
    /// allocation made during graph capture was served from this buffer. The
    /// device addresses baked into the graph point inside this tensor's storage.
    ///
    /// `None` for graphs captured without an arena (the original path).
    /// `Some` for graphs captured via `capture_graph_into_with_arena`.
    arena: Option<Tensor<R>>,
}

impl<R: Runtime> CapturedGraph<R> {
    /// Construct a new `CapturedGraph` without an arena.
    ///
    /// `inputs` and `outputs` should be clones of the tensors that were passed
    /// to [`Runtime::capture_graph_into`]; holding them here prevents the
    /// underlying device memory from being freed while this graph is alive.
    pub fn new(graph: R::Graph, inputs: Vec<Tensor<R>>, outputs: Vec<Tensor<R>>) -> Self {
        Self {
            graph,
            inputs,
            outputs,
            arena: None,
        }
    }

    /// Construct a new `CapturedGraph` with a dedicated scratch arena.
    ///
    /// `arena` is a pre-allocated device buffer whose interior holds all
    /// graph-internal intermediate tensors captured during the freeze window.
    /// The device addresses baked into the graph's kernel-parameter blocks
    /// point inside this buffer; the buffer must outlive the graph handle.
    ///
    /// # Drop ordering
    ///
    /// The field order guarantees `graph` drops before `arena` (declaration
    /// order), so `cuGraphExecDestroy` always executes while the arena buffer
    /// is still mapped on the device.
    pub fn new_with_arena(
        graph: R::Graph,
        inputs: Vec<Tensor<R>>,
        outputs: Vec<Tensor<R>>,
        arena: Tensor<R>,
    ) -> Self {
        Self {
            graph,
            inputs,
            outputs,
            arena: Some(arena),
        }
    }

    /// Replay the captured computation.
    ///
    /// On CUDA this calls `cuGraphLaunch`, re-executing every recorded kernel
    /// against the same fixed-address I/O buffers. The caller is responsible
    /// for writing fresh input data (H2D copy) before calling `launch`, and for
    /// reading results from the output tensors after `launch` completes.
    pub fn launch(&self) -> crate::error::Result<()> {
        self.graph.launch()
    }

    /// Access the underlying graph handle.
    pub fn graph(&self) -> &R::Graph {
        &self.graph
    }

    /// The input tensors whose device addresses are encoded in the graph.
    pub fn inputs(&self) -> &[Tensor<R>] {
        &self.inputs
    }

    /// The output tensors that the graph writes into on each replay.
    pub fn outputs(&self) -> &[Tensor<R>] {
        &self.outputs
    }
}

// Both bounds are required because Tensor<R> needs them and R::Graph does too.
// SAFETY: graph drops before inputs/outputs (declaration order), so device
// pointers held by the tensors outlive the graph handle in all cases.
unsafe impl<R: Runtime> Send for CapturedGraph<R>
where
    R::Graph: Send,
    Tensor<R>: Send,
{
}

unsafe impl<R: Runtime> Sync for CapturedGraph<R>
where
    R::Graph: Sync,
    Tensor<R>: Sync,
{
}

impl<R: Runtime> std::fmt::Debug for CapturedGraph<R>
where
    R::Graph: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedGraph")
            .field("graph", &self.graph)
            .field("inputs_len", &self.inputs.len())
            .field("outputs_len", &self.outputs.len())
            .field("has_arena", &self.arena.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::NoOpGraph;

    /// Verify that `CapturedGraph` compiles with `NoOpGraph` and that `launch`
    /// delegates correctly.
    #[test]
    fn test_captured_graph_noop_launch() {
        let graph = NoOpGraph;
        let captured: CapturedGraph<crate::runtime::cpu::CpuRuntime> =
            CapturedGraph::new(graph, vec![], vec![]);
        assert!(captured.launch().is_ok());
    }

    /// Verify that `Send + Sync` bounds hold for `CapturedGraph<CpuRuntime>`.
    #[test]
    fn test_captured_graph_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CapturedGraph<crate::runtime::cpu::CpuRuntime>>();
    }

    /// Verify that `Send + Sync` bounds hold for `CapturedGraph<CudaRuntime>`.
    ///
    /// The unsafe `Send`/`Sync` impls are the load-bearing part of this type, so
    /// the real backend (whose `Graph` owns raw driver handles) must satisfy them
    /// — not just the trivial `CpuRuntime`/`NoOpGraph`. This is a compile-time
    /// assertion; it needs no GPU.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_captured_graph_send_sync_cuda() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CapturedGraph<crate::runtime::cuda::CudaRuntime>>();
    }

    /// Drop ordering test: dropping a `CapturedGraph` with non-empty I/O vecs
    /// completes without panic. This exercises the Rust field-drop order
    /// (graph first, then inputs, then outputs).
    #[test]
    fn test_captured_graph_drop_ordering() {
        use crate::runtime::cpu::CpuRuntime;
        use crate::tensor::Tensor;

        let device = CpuRuntime::default_device();
        let a = Tensor::<CpuRuntime>::zeros(&[4], crate::dtype::DType::F32, &device).unwrap();
        let b = a.clone();
        let c = a.clone();

        let captured: CapturedGraph<CpuRuntime> =
            CapturedGraph::new(NoOpGraph, vec![a, b], vec![c]);

        // Dropping `captured` here: graph drops first (NoOpGraph), then
        // inputs Vec, then outputs Vec. No panic expected.
        drop(captured);
    }
}
