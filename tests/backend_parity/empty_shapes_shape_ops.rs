//! Backend parity for zero-element inputs in the shape, in-place-index and
//! sorting families.
//!
//! Companion to `empty_shapes_elementwise.rs`, `empty_shapes_structural.rs` and
//! `empty_shapes_normalization.rs`. Each op here sizes its launch from an
//! `outer_size`/`inner_size` pair taken from the shape, so a zero in either used
//! to be floored to 1 — which turns a no-op into a write past the end of an
//! empty allocation on CPU and CUDA, and a bind of a buffer that never existed
//! on WebGPU.
//!
//! CPU is the reference throughout.

use numr::dtype::DType;
use numr::ops::{IndexingOps, ShapeOps, SortingOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

type CpuDev = <CpuRuntime as Runtime>::Device;

// ============================================================================
// cat
// ============================================================================

fn check_cat<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: ShapeOps<R>,
{
    // Concatenating along dim 1 with a zero leading dim leaves `outer_size` at 0
    // and an empty result. Flooring it made CPU take its single-memcpy fast path
    // and copy a whole row out of an empty tensor.
    let a_cpu = Tensor::<CpuRuntime>::zeros(&[0, 3], dtype, cpu_device).expect("cpu a");
    let b_cpu = Tensor::<CpuRuntime>::zeros(&[0, 2], dtype, cpu_device).expect("cpu b");
    let a = Tensor::<R>::zeros(&[0, 3], dtype, device).expect("a");
    let b = Tensor::<R>::zeros(&[0, 2], dtype, device).expect("b");

    let expected = cpu_client
        .cat(&[&a_cpu, &b_cpu], 1)
        .expect("cpu cat of empty tensors");
    let actual = client
        .cat(&[&a, &b], 1)
        .unwrap_or_else(|e| panic!("{backend} cat empty {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), &[0, 5], "{backend} cat output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("cat [0,3]+[0,2] dim 1 {backend} vs cpu"),
    );

    // One empty operand beside a non-empty one: the result is NOT empty, so the
    // empty operand must be skipped rather than the whole call short-circuited.
    let e_cpu = Tensor::<CpuRuntime>::zeros(&[0, 3], dtype, cpu_device).expect("cpu empty");
    let f_cpu = Tensor::<CpuRuntime>::zeros(&[2, 3], dtype, cpu_device).expect("cpu full");
    let e = Tensor::<R>::zeros(&[0, 3], dtype, device).expect("empty");
    let f = Tensor::<R>::zeros(&[2, 3], dtype, device).expect("full");

    let expected = cpu_client
        .cat(&[&e_cpu, &f_cpu], 0)
        .expect("cpu cat with one empty operand");
    let actual = client
        .cat(&[&e, &f], 0)
        .unwrap_or_else(|e| panic!("{backend} cat mixed {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), &[2, 3], "{backend} cat mixed output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("cat [0,3]+[2,3] dim 0 {backend} vs cpu"),
    );
}

// ============================================================================
// roll
// ============================================================================

fn check_roll<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: ShapeOps<R>,
{
    // Rolling along a non-empty dim of an empty tensor. `validate_roll` rejects a
    // zero-length roll dimension, so the zero has to sit in `outer_size`.
    let a_cpu = Tensor::<CpuRuntime>::zeros(&[0, 3], dtype, cpu_device).expect("cpu a");
    let a = Tensor::<R>::zeros(&[0, 3], dtype, device).expect("a");

    let expected = cpu_client
        .roll(&a_cpu, 1, 1)
        .expect("cpu roll on an empty tensor");
    let actual = client
        .roll(&a, 1, 1)
        .unwrap_or_else(|e| panic!("{backend} roll {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), &[0, 3], "{backend} roll output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("roll [0,3] dim 1 {backend} vs cpu"),
    );
}

// ============================================================================
// index_put / slice_assign
// ============================================================================

fn check_in_place_index<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    // Empty destination: nothing is copied and nothing is written back.
    let idx: [i32; 2] = [0, 2];
    let a_cpu = Tensor::<CpuRuntime>::zeros(&[0, 5], dtype, cpu_device).expect("cpu a");
    let i_cpu = Tensor::<CpuRuntime>::from_slice(&idx, &[2], cpu_device).expect("cpu idx");
    let s_cpu = Tensor::<CpuRuntime>::zeros(&[0, 2], dtype, cpu_device).expect("cpu src");
    let a = Tensor::<R>::zeros(&[0, 5], dtype, device).expect("a");
    let i = Tensor::<R>::from_slice(&idx, &[2], device).expect("idx");
    let s = Tensor::<R>::zeros(&[0, 2], dtype, device).expect("src");

    let expected = cpu_client
        .index_put(&a_cpu, 1, &i_cpu, &s_cpu)
        .expect("cpu index_put on an empty destination");
    let actual = client
        .index_put(&a, 1, &i, &s)
        .unwrap_or_else(|e| panic!("{backend} index_put empty dst {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), &[0, 5], "{backend} index_put output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("index_put empty dst {backend} vs cpu"),
    );

    // Empty index set against a non-empty destination: the destination survives
    // unchanged, so the copy must still run and only the put be skipped.
    let empty_idx: [i32; 0] = [];
    let dst_cpu = Tensor::<CpuRuntime>::zeros(&[3, 4], dtype, cpu_device).expect("cpu dst");
    let ei_cpu = Tensor::<CpuRuntime>::from_slice(&empty_idx, &[0], cpu_device).expect("cpu eidx");
    let es_cpu = Tensor::<CpuRuntime>::zeros(&[0, 4], dtype, cpu_device).expect("cpu esrc");
    let dst = Tensor::<R>::zeros(&[3, 4], dtype, device).expect("dst");
    let ei = Tensor::<R>::from_slice(&empty_idx, &[0], device).expect("eidx");
    let es = Tensor::<R>::zeros(&[0, 4], dtype, device).expect("esrc");

    let expected = cpu_client
        .index_put(&dst_cpu, 0, &ei_cpu, &es_cpu)
        .expect("cpu index_put with an empty index");
    let actual = client
        .index_put(&dst, 0, &ei, &es)
        .unwrap_or_else(|e| panic!("{backend} index_put empty idx {dtype:?}: {e:?}"));
    assert_eq!(
        actual.shape(),
        &[3, 4],
        "{backend} index_put unchanged output shape"
    );
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("index_put empty idx {backend} vs cpu"),
    );

    // slice_assign: an empty destination, then an empty source slice written into
    // a non-empty destination.
    let sa_src_cpu = Tensor::<CpuRuntime>::zeros(&[0, 2], dtype, cpu_device).expect("cpu sa src");
    let sa_src = Tensor::<R>::zeros(&[0, 2], dtype, device).expect("sa src");
    let expected = cpu_client
        .slice_assign(&a_cpu, &sa_src_cpu, 1, 1)
        .expect("cpu slice_assign on an empty destination");
    let actual = client
        .slice_assign(&a, &sa_src, 1, 1)
        .unwrap_or_else(|e| panic!("{backend} slice_assign empty dst {dtype:?}: {e:?}"));
    assert_eq!(
        actual.shape(),
        &[0, 5],
        "{backend} slice_assign output shape"
    );
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("slice_assign empty dst {backend} vs cpu"),
    );

    let expected = cpu_client
        .slice_assign(&dst_cpu, &es_cpu, 0, 1)
        .expect("cpu slice_assign with an empty source");
    let actual = client
        .slice_assign(&dst, &es, 0, 1)
        .unwrap_or_else(|e| panic!("{backend} slice_assign empty src {dtype:?}: {e:?}"));
    assert_eq!(
        actual.shape(),
        &[3, 4],
        "{backend} slice_assign unchanged output shape"
    );
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("slice_assign empty src {backend} vs cpu"),
    );
}

// ============================================================================
// gather_nd
// ============================================================================

fn check_gather_nd<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    // No index vector at all: `num_slices` is 0 and the output is empty.
    let empty_idx: [i32; 0] = [];
    let src_cpu = Tensor::<CpuRuntime>::zeros(&[4, 5], dtype, cpu_device).expect("cpu src");
    let i_cpu = Tensor::<CpuRuntime>::from_slice(&empty_idx, &[0, 1], cpu_device).expect("cpu idx");
    let src = Tensor::<R>::zeros(&[4, 5], dtype, device).expect("src");
    let i = Tensor::<R>::from_slice(&empty_idx, &[0, 1], device).expect("idx");

    let expected = cpu_client
        .gather_nd(&src_cpu, &i_cpu)
        .expect("cpu gather_nd with no index vector");
    let actual = client
        .gather_nd(&src, &i)
        .unwrap_or_else(|e| panic!("{backend} gather_nd empty index {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), &[0, 5], "{backend} gather_nd output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("gather_nd empty index {backend} vs cpu"),
    );

    // Index vectors that address a zero-length trailing dimension: `slice_size`
    // is 0 while `num_slices` is 2, so the output is empty for the other reason.
    let idx: [i32; 2] = [0, 1];
    let flat_cpu = Tensor::<CpuRuntime>::zeros(&[3, 0], dtype, cpu_device).expect("cpu flat");
    let fi_cpu = Tensor::<CpuRuntime>::from_slice(&idx, &[2, 1], cpu_device).expect("cpu fidx");
    let flat = Tensor::<R>::zeros(&[3, 0], dtype, device).expect("flat");
    let fi = Tensor::<R>::from_slice(&idx, &[2, 1], device).expect("fidx");

    let expected = cpu_client
        .gather_nd(&flat_cpu, &fi_cpu)
        .expect("cpu gather_nd over a zero-length slice");
    let actual = client
        .gather_nd(&flat, &fi)
        .unwrap_or_else(|e| panic!("{backend} gather_nd empty slice {dtype:?}: {e:?}"));
    assert_eq!(
        actual.shape(),
        &[2, 0],
        "{backend} gather_nd empty-slice output shape"
    );
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("gather_nd empty slice {backend} vs cpu"),
    );
}

// ============================================================================
// sort / argsort / topk
// ============================================================================

fn check_sorting<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: SortingOps<R>,
{
    // `[0, 5]` zeroes `outer_size`, which is the CUDA grid's x extent.
    // `[3, 0]` zeroes the sort length, which is the block extent.
    for shape in [&[0usize, 5][..], &[3, 0][..]] {
        let a_cpu = Tensor::<CpuRuntime>::zeros(shape, dtype, cpu_device).expect("cpu a");
        let a = Tensor::<R>::zeros(shape, dtype, device).expect("a");

        let expected = cpu_client
            .sort(&a_cpu, 1, false)
            .expect("cpu sort on an empty tensor");
        let actual = client
            .sort(&a, 1, false)
            .unwrap_or_else(|e| panic!("{backend} sort {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "{backend} sort output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("sort {shape:?} {backend} vs cpu"),
        );

        let expected = cpu_client
            .argsort(&a_cpu, 1, false)
            .expect("cpu argsort on an empty tensor");
        let actual = client
            .argsort(&a, 1, false)
            .unwrap_or_else(|e| panic!("{backend} argsort {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "{backend} argsort output shape");
        assert_eq!(
            actual.numel(),
            expected.numel(),
            "{backend} argsort element count"
        );
    }

    // topk needs `1 <= k <= sort_size`, so the zero has to sit in `outer_size`.
    let a_cpu = Tensor::<CpuRuntime>::zeros(&[0, 5], dtype, cpu_device).expect("cpu a");
    let a = Tensor::<R>::zeros(&[0, 5], dtype, device).expect("a");

    let (expected, _) = cpu_client
        .topk(&a_cpu, 2, 1, true, true)
        .expect("cpu topk on an empty tensor");
    let (actual, actual_idx) = client
        .topk(&a, 2, 1, true, true)
        .unwrap_or_else(|e| panic!("{backend} topk {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), &[0, 2], "{backend} topk values shape");
    assert_eq!(actual_idx.shape(), &[0, 2], "{backend} topk indices shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("topk [0,5] {backend} vs cpu"),
    );
}

// ============================================================================
// Per-backend entry points
// ============================================================================

fn run_all<R, C>(client: &C, device: &R::Device, backend: &str)
where
    R: Runtime<DType = DType>,
    C: ShapeOps<R> + IndexingOps<R> + SortingOps<R>,
{
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, backend) {
        check_cat::<R, _>(client, device, &cpu_client, &cpu_device, dtype, backend);
        check_roll::<R, _>(client, device, &cpu_client, &cpu_device, dtype, backend);
        check_in_place_index::<R, _>(client, device, &cpu_client, &cpu_device, dtype, backend);
        check_gather_nd::<R, _>(client, device, &cpu_client, &cpu_device, dtype, backend);
        check_sorting::<R, _>(client, device, &cpu_client, &cpu_device, dtype, backend);
    }
}

#[test]
fn test_empty_shape_ops_cpu_is_self_consistent() {
    let (client, device) = create_cpu_client();
    run_all::<CpuRuntime, _>(&client, &device, "cpu");
}

#[cfg(feature = "cuda")]
#[test]
fn test_empty_shape_ops_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        run_all::<CudaRuntime, _>(&client, &device, "cuda");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_empty_shape_ops_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        run_all::<WgpuRuntime, _>(&client, &device, "wgpu");
    });
}
