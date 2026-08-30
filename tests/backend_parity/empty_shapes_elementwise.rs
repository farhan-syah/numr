//! Backend parity for zero-element and 0-dim inputs: element-wise, ternary,
//! cumulative and reduction families.
//!
//! Empty-shape coverage in `tests/backend_parity/` used to be six CUDA F32
//! tests, and no integer dtype had one. Every case here runs for the full
//! `parity_dtypes` intersection instead, and includes `&[0, 5]` (a zero
//! dimension beside a non-zero one) and `&[]` (a 0-dim scalar) alongside `&[0]`.
//!
//! CPU is the reference. Where a backend cannot bind a zero-byte allocation —
//! WebGPU's allocator registers no buffer for one — the op must return the empty
//! result without dispatching, not an error.

use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, CumulativeOps, ReduceOps};
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

/// The shapes every element-wise case runs: a 1-D empty tensor, and a zero
/// dimension sitting beside a non-zero one so a kernel that only checks
/// `numel == 0` on the flattened length is still exercised on its stride math.
const EMPTY_SHAPES: [&[usize]; 2] = [&[0], &[0, 5]];

/// Everything a backend client must provide for this file's cases.
trait EmptyOps<R: Runtime<DType = DType>>:
    BinaryOps<R> + ConditionalOps<R> + CumulativeOps<R> + ReduceOps<R>
{
}

impl<R: Runtime<DType = DType>, C> EmptyOps<R> for C where
    C: BinaryOps<R> + ConditionalOps<R> + CumulativeOps<R> + ReduceOps<R>
{
}

// ============================================================================
// Fused element-wise: fused_mul_add, fused_add_mul
// ============================================================================

fn check_fused<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: EmptyOps<R>,
{
    for shape in EMPTY_SHAPES {
        let a_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu a");
        let b_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu b");
        let c_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu c");
        let a = Tensor::<R>::empty(shape, dtype, device).expect("a");
        let b = Tensor::<R>::empty(shape, dtype, device).expect("b");
        let c = Tensor::<R>::empty(shape, dtype, device).expect("c");

        let expected = cpu_client
            .fused_mul_add(&a_cpu, &b_cpu, &c_cpu)
            .expect("cpu fused_mul_add on an empty tensor");
        let actual = client
            .fused_mul_add(&a, &b, &c)
            .unwrap_or_else(|e| panic!("{backend} fused_mul_add {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(
            actual.shape(),
            shape,
            "fused_mul_add {backend} output shape"
        );
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("fused_mul_add {shape:?} {backend} vs cpu"),
        );

        let expected = cpu_client
            .fused_add_mul(&a_cpu, &b_cpu, &c_cpu)
            .expect("cpu fused_add_mul on an empty tensor");
        let actual = client
            .fused_add_mul(&a, &b, &c)
            .unwrap_or_else(|e| panic!("{backend} fused_add_mul {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(
            actual.shape(),
            shape,
            "fused_add_mul {backend} output shape"
        );
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("fused_add_mul {shape:?} {backend} vs cpu"),
        );
    }
}

// ============================================================================
// Ternary: where_cond
// ============================================================================

fn check_where<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: EmptyOps<R>,
{
    for shape in EMPTY_SHAPES {
        // The condition is U8, the dtype every backend's `where_cond` accepts
        // for a mask.
        let cond_cpu = Tensor::<CpuRuntime>::empty(shape, DType::U8, cpu_device).expect("cpu cond");
        let x_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu x");
        let y_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu y");
        let cond = Tensor::<R>::empty(shape, DType::U8, device).expect("cond");
        let x = Tensor::<R>::empty(shape, dtype, device).expect("x");
        let y = Tensor::<R>::empty(shape, dtype, device).expect("y");

        let expected = cpu_client
            .where_cond(&cond_cpu, &x_cpu, &y_cpu)
            .expect("cpu where_cond on an empty tensor");
        let actual = client
            .where_cond(&cond, &x, &y)
            .unwrap_or_else(|e| panic!("{backend} where_cond {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "where_cond {backend} output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("where_cond {shape:?} {backend} vs cpu"),
        );
    }
}

// ============================================================================
// Cumulative: cumsum, cumprod
// ============================================================================

fn check_cumulative<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: EmptyOps<R>,
{
    for shape in EMPTY_SHAPES {
        let a_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu a");
        let a = Tensor::<R>::empty(shape, dtype, device).expect("a");

        let expected = cpu_client
            .cumsum(&a_cpu, 0)
            .expect("cpu cumsum on an empty tensor");
        let actual = client
            .cumsum(&a, 0)
            .unwrap_or_else(|e| panic!("{backend} cumsum {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "cumsum {backend} output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("cumsum {shape:?} {backend} vs cpu"),
        );

        let expected = cpu_client
            .cumprod(&a_cpu, 0)
            .expect("cpu cumprod on an empty tensor");
        let actual = client
            .cumprod(&a, 0)
            .unwrap_or_else(|e| panic!("{backend} cumprod {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "cumprod {backend} output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("cumprod {shape:?} {backend} vs cpu"),
        );
    }
}

// ============================================================================
// Reductions over a zero-length dimension
// ============================================================================

fn check_reduce<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: EmptyOps<R>,
{
    // Two distinct shapes of the same problem:
    //
    // - `[0, 5]` reduced over dim 1 leaves `[0]`: the OUTPUT is empty too.
    // - `[3, 0]` reduced over dim 1 leaves `[3]`: the output is NOT empty, and
    //   each of its elements folds over no input, so the value is the
    //   reduction's identity. This is the case a `numel == 0` early return gets
    //   wrong if it hands back an unwritten allocation.
    for (shape, dims) in [(&[0usize, 5usize][..], 1usize), (&[3usize, 0usize][..], 1)] {
        let a_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu a");
        let a = Tensor::<R>::empty(shape, dtype, device).expect("a");

        let expected = cpu_client
            .sum(&a_cpu, &[dims], false)
            .expect("cpu sum over a zero-length dim");
        let actual = client
            .sum(&a, &[dims], false)
            .unwrap_or_else(|e| panic!("{backend} sum {shape:?} {dtype:?}: {e:?}"));
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("sum {shape:?} dim {dims} {backend} vs cpu"),
        );

        let expected = cpu_client
            .prod(&a_cpu, &[dims], false)
            .expect("cpu prod over a zero-length dim");
        let actual = client
            .prod(&a, &[dims], false)
            .unwrap_or_else(|e| panic!("{backend} prod {shape:?} {dtype:?}: {e:?}"));
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("prod {shape:?} dim {dims} {backend} vs cpu"),
        );
    }
}

// ============================================================================
// 0-dim scalar tensors
// ============================================================================

/// A `&[]` shape holds exactly one element, so nothing here is empty. It is
/// covered because a rank of zero is the other degenerate shape, and no op
/// family in `tests/backend_parity/` had a case for it.
fn check_scalar_rank0<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: EmptyOps<R>,
{
    let cond_cpu = Tensor::<CpuRuntime>::empty(&[], DType::U8, cpu_device).expect("cpu cond");
    let x_cpu = Tensor::<CpuRuntime>::empty(&[], dtype, cpu_device).expect("cpu x");
    let y_cpu = Tensor::<CpuRuntime>::empty(&[], dtype, cpu_device).expect("cpu y");
    // WebGPU has no U8 dtype, so its `where_cond` takes a 32-bit mask. CPU and
    // CUDA read a U8 condition. The result dtype is `dtype` either way, which is
    // what this case asserts.
    let cond_dtype = if backend == "wgpu" {
        DType::U32
    } else {
        DType::U8
    };
    let cond = Tensor::<R>::empty(&[], cond_dtype, device).expect("cond");
    let x = Tensor::<R>::empty(&[], dtype, device).expect("x");
    let y = Tensor::<R>::empty(&[], dtype, device).expect("y");

    // A 0-dim `where_cond` reads one condition byte, so both sides must agree on
    // the same bytes. `empty` is uninitialized, so the comparison is on shape
    // and dtype only — the value is whatever each allocator handed back.
    let expected = cpu_client
        .where_cond(&cond_cpu, &x_cpu, &y_cpu)
        .expect("cpu where_cond on a 0-dim tensor");
    let actual = client
        .where_cond(&cond, &x, &y)
        .unwrap_or_else(|e| panic!("{backend} where_cond rank-0 {dtype:?}: {e:?}"));
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "where_cond rank-0 {backend} output shape"
    );
    assert_eq!(
        actual.dtype(),
        expected.dtype(),
        "where_cond rank-0 {backend} output dtype"
    );
}

// ============================================================================
// Per-backend entry points
// ============================================================================

#[test]
fn test_empty_elementwise_cpu_is_self_consistent() {
    let (client, device) = create_cpu_client();
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        check_fused::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_where::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_cumulative::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_reduce::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_scalar_rank0::<CpuRuntime, _>(
            &client,
            &device,
            &cpu_client,
            &cpu_device,
            dtype,
            "cpu",
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_empty_elementwise_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cuda") {
            check_fused::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_where::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_cumulative::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_reduce::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_scalar_rank0::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_empty_elementwise_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::AllNumeric, "wgpu") {
            check_fused::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_where::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_cumulative::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_reduce::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_scalar_rank0::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
        }
    });
}
