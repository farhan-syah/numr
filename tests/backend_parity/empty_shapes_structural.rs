//! Backend parity for zero-element inputs: matmul, semiring matmul, indexing,
//! scatter_reduce and bincount.
//!
//! Companion to `empty_shapes_elementwise.rs`. These are the families whose
//! launch geometry comes from shape extents rather than a flat element count,
//! so an empty operand can produce a grid extent of zero — an invalid CUDA
//! launch — or bind a buffer that a zero-byte allocation never registered.
//!
//! `semiring_matmul` in particular had no `numel() == 0` guard on either GPU
//! backend before these tests existed.
//!
//! CPU is the reference throughout.

use numr::dtype::DType;
use numr::ops::SemiringOp;
use numr::ops::{IndexingOps, MatmulOps, ScatterReduceOp, SemiringMatmulOps, TypeConversionOps};
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
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

type CpuDev = <CpuRuntime as Runtime>::Device;

// ============================================================================
// matmul
// ============================================================================

fn check_matmul<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: MatmulOps<R>,
{
    // Two shapes of empty. `[0, 5] x [5, 3]` leaves an empty OUTPUT, which the
    // launch geometry must not turn into a zero grid extent. `[3, 0] x [0, 4]`
    // leaves a full `[3, 4]` output whose every element sums over no terms, so
    // the answer is zero — an early return that skips writing the output gets
    // this one wrong.
    for (a_shape, b_shape, out_shape) in [
        (
            &[0usize, 5usize][..],
            &[5usize, 3usize][..],
            &[0usize, 3usize][..],
        ),
        (&[3, 0][..], &[0, 4][..], &[3, 4][..]),
    ] {
        let a_cpu = Tensor::<CpuRuntime>::zeros(a_shape, dtype, cpu_device).expect("cpu a");
        let b_cpu = Tensor::<CpuRuntime>::zeros(b_shape, dtype, cpu_device).expect("cpu b");
        let a = Tensor::<R>::zeros(a_shape, dtype, device).expect("a");
        let b = Tensor::<R>::zeros(b_shape, dtype, device).expect("b");

        let expected = cpu_client
            .matmul(&a_cpu, &b_cpu)
            .expect("cpu matmul with an empty operand");
        assert_eq!(expected.shape(), out_shape, "cpu matmul output shape");

        let actual = client.matmul(&a, &b).unwrap_or_else(|e| {
            panic!("{backend} matmul {a_shape:?}x{b_shape:?} {dtype:?}: {e:?}")
        });
        assert_eq!(actual.shape(), out_shape, "{backend} matmul output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("matmul {a_shape:?}x{b_shape:?} {backend} vs cpu"),
        );
    }
}

// ============================================================================
// semiring_matmul
// ============================================================================

/// The semiring dtypes every backend implements: the float and integer widths
/// `SemiringOp::validate_dtype` admits, intersected with the backend's scope.
/// `OrAnd` is excluded because it is Bool-only and Bool is outside
/// `backend_supported_dtypes`.
fn semiring_dtypes(backend: &str) -> Vec<DType> {
    [DType::F32, DType::F64, DType::I32, DType::I64]
        .into_iter()
        .filter(|&d| is_dtype_supported(backend, d))
        .collect()
}

fn check_semiring_matmul<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: SemiringMatmulOps<R>,
{
    // Only the empty-OUTPUT shape is covered. A zero-length CONTRACTION
    // (`[3, 0] x [0, 4]`) leaves every output element at the semiring's reduce
    // identity — `+inf` for MinPlus — which no backend can produce without a
    // fill kernel it does not have; that gap is reported, not tested here.
    for op in [SemiringOp::MinPlus, SemiringOp::MaxPlus, SemiringOp::MaxMin] {
        let a_cpu = Tensor::<CpuRuntime>::zeros(&[0, 3], dtype, cpu_device).expect("cpu a");
        let b_cpu = Tensor::<CpuRuntime>::zeros(&[3, 4], dtype, cpu_device).expect("cpu b");
        let a = Tensor::<R>::zeros(&[0, 3], dtype, device).expect("a");
        let b = Tensor::<R>::zeros(&[3, 4], dtype, device).expect("b");

        let expected = cpu_client
            .semiring_matmul(&a_cpu, &b_cpu, op)
            .expect("cpu semiring_matmul with an empty operand");
        assert_eq!(expected.shape(), &[0, 4], "cpu semiring output shape");

        let actual = client
            .semiring_matmul(&a, &b, op)
            .unwrap_or_else(|e| panic!("{backend} semiring_matmul {op:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), &[0, 4], "{backend} semiring output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("semiring_matmul {op:?} [0,3]x[3,4] {backend} vs cpu"),
        );
    }
}

// ============================================================================
// indexing: index_select, gather
// ============================================================================

fn check_indexing<R, C>(
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
    // Empty source along dim 0, selecting along the non-empty dim 1. The output
    // keeps the zero dimension, so nothing is read and nothing is written.
    let idx: [i32; 2] = [0, 2];
    let a_cpu = Tensor::<CpuRuntime>::zeros(&[0, 5], dtype, cpu_device).expect("cpu a");
    let i_cpu = Tensor::<CpuRuntime>::from_slice(&idx, &[2], cpu_device).expect("cpu idx");
    let a = Tensor::<R>::zeros(&[0, 5], dtype, device).expect("a");
    let i = Tensor::<R>::from_slice(&idx, &[2], device).expect("idx");

    let expected = cpu_client
        .index_select(&a_cpu, 1, &i_cpu)
        .expect("cpu index_select on an empty source");
    let actual = client
        .index_select(&a, 1, &i)
        .unwrap_or_else(|e| panic!("{backend} index_select {dtype:?}: {e:?}"));
    assert_eq!(
        actual.shape(),
        &[0, 2],
        "{backend} index_select output shape"
    );
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("index_select [0,5] dim 1 {backend} vs cpu"),
    );

    // Empty index set against a non-empty source: the output takes the index
    // shape, so it is empty while the source is not.
    let empty_idx: [i32; 0] = [];
    let src_cpu = Tensor::<CpuRuntime>::zeros(&[3, 4], dtype, cpu_device).expect("cpu src");
    let ei_cpu =
        Tensor::<CpuRuntime>::from_slice(&empty_idx, &[0, 4], cpu_device).expect("cpu empty idx");
    let src = Tensor::<R>::zeros(&[3, 4], dtype, device).expect("src");
    let ei = Tensor::<R>::from_slice(&empty_idx, &[0, 4], device).expect("empty idx");

    let expected = cpu_client
        .gather(&src_cpu, 0, &ei_cpu)
        .expect("cpu gather with an empty index");
    let actual = client
        .gather(&src, 0, &ei)
        .unwrap_or_else(|e| panic!("{backend} gather {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), &[0, 4], "{backend} gather output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("gather empty index {backend} vs cpu"),
    );
}

// ============================================================================
// scatter_reduce
// ============================================================================

fn check_scatter_reduce<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: IndexingOps<R> + TypeConversionOps<R>,
{
    // An empty source scatters nothing, so the destination survives unchanged.
    // `include_self` is true so the expected value is the destination itself
    // rather than an op-dependent identity.
    let empty_idx: [i32; 0] = [];
    let dst_vals: [i32; 4] = [1, 2, 3, 4];

    let dst_cpu = Tensor::<CpuRuntime>::from_slice(&dst_vals, &[4], cpu_device).expect("cpu dst");
    let dst_cpu = cast_to(cpu_client, &dst_cpu, dtype);
    let idx_cpu = Tensor::<CpuRuntime>::from_slice(&empty_idx, &[0], cpu_device).expect("cpu idx");
    let src_cpu = Tensor::<CpuRuntime>::zeros(&[0], dtype, cpu_device).expect("cpu src");

    let dst = Tensor::<R>::from_slice(&dst_vals, &[4], device).expect("dst");
    let dst = cast_to(client, &dst, dtype);
    let idx = Tensor::<R>::from_slice(&empty_idx, &[0], device).expect("idx");
    let src = Tensor::<R>::zeros(&[0], dtype, device).expect("src");

    let expected = cpu_client
        .scatter_reduce(&dst_cpu, 0, &idx_cpu, &src_cpu, ScatterReduceOp::Sum, true)
        .expect("cpu scatter_reduce with an empty source");
    let actual = client
        .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Sum, true)
        .unwrap_or_else(|e| panic!("{backend} scatter_reduce {dtype:?}: {e:?}"));
    assert_eq!(
        actual.shape(),
        &[4],
        "{backend} scatter_reduce output shape"
    );
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("scatter_reduce empty src {backend} vs cpu"),
    );
}

/// Cast an I32 tensor to `dtype`, used to build the same values in whichever
/// dtype the case runs.
fn cast_to<R: Runtime<DType = DType>>(
    client: &impl TypeConversionOps<R>,
    t: &Tensor<R>,
    dtype: DType,
) -> Tensor<R> {
    if t.dtype() == dtype {
        return t.clone();
    }
    client.cast(t, dtype).expect("cast to case dtype")
}

// ============================================================================
// bincount
// ============================================================================

fn check_bincount<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    // `bincount_with_len` is the entry point covered here because it takes the
    // output length from the caller. Plain `bincount` derives it from the
    // input's maximum, and CPU's `max_i64_kernel` answers -1 for an empty input,
    // which the caller then rejects as negative — an error, not an empty
    // histogram. That behaviour is reported, not exercised.
    let empty: [i32; 0] = [];
    let input_cpu = Tensor::<CpuRuntime>::from_slice(&empty, &[0], cpu_device).expect("cpu input");
    let input = Tensor::<R>::from_slice(&empty, &[0], device).expect("input");

    let expected = cpu_client
        .bincount_with_len(&input_cpu, None, 3)
        .expect("cpu bincount_with_len on an empty input");
    let actual = client
        .bincount_with_len(&input, None, 3)
        .unwrap_or_else(|e| panic!("{backend} bincount_with_len: {e:?}"));
    assert_eq!(actual.shape(), &[3], "{backend} bincount output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        expected.dtype(),
        &format!("bincount_with_len empty input {backend} vs cpu"),
    );
}

// ============================================================================
// Per-backend entry points
// ============================================================================

#[test]
fn test_empty_structural_cpu_is_self_consistent() {
    let (client, device) = create_cpu_client();
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        check_matmul::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_indexing::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_scatter_reduce::<CpuRuntime, _>(
            &client,
            &device,
            &cpu_client,
            &cpu_device,
            dtype,
            "cpu",
        );
    }
    for dtype in semiring_dtypes("cpu") {
        check_semiring_matmul::<CpuRuntime, _>(
            &client,
            &device,
            &cpu_client,
            &cpu_device,
            dtype,
            "cpu",
        );
    }
    check_bincount::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, "cpu");
}

#[cfg(feature = "cuda")]
#[test]
fn test_empty_structural_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cuda") {
            check_matmul::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_indexing::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_scatter_reduce::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
        }
        for dtype in semiring_dtypes("cuda") {
            check_semiring_matmul::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
        }
        check_bincount::<CudaRuntime, _>(&client, &device, &cpu_client, &cpu_device, "cuda");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_empty_structural_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::AllNumeric, "wgpu") {
            check_matmul::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_indexing::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_scatter_reduce::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
        }
        for dtype in semiring_dtypes("wgpu") {
            check_semiring_matmul::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
        }
        check_bincount::<WgpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, "wgpu");
    });
}
