//! Backend parity for zero-element inputs in the normalization, softmax and
//! GEMM-epilogue families.
//!
//! Companion to `empty_shapes_elementwise.rs` and `empty_shapes_structural.rs`.
//! Everything here takes its launch geometry from a `batch_size` (the product of
//! every dimension but the last) and a per-row extent, so a zero in either gives
//! a CUDA grid or block extent of 0 — a launch error the driver rejects — or a
//! WebGPU dispatch binding a buffer a zero-byte allocation never registered.
//!
//! CPU is the reference throughout.
//!
//! One neighbouring case is deliberately NOT covered, because CPU does not
//! answer it and the divergence is reported instead of pinned: softmax over a
//! zero-length dimension with a non-empty batch (`[3, 0]`), where CPU's scalar
//! softmax kernel seeds its running max from `a[base]` — a read past the end of
//! the empty allocation.
//!
//! `k == 0` IS covered on all three backends, forward and backward. WebGPU
//! cannot bind the zero-byte `a`/`b` operands to the epilogue shader, so it
//! never dispatches one: `bias_only_activation` in `runtime/wgpu/ops/native/
//! gemm_epilogue.rs` builds `act(bias)` from a broadcast plus the activation op,
//! and `matmul_bias_residual` takes the matching `bias + residual` branch.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{ActivationOps, GemmActivation, GemmEpilogueOps, NormalizationOps};
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

const EPS: f32 = 1e-5;

// ============================================================================
// rms_norm / layer_norm
// ============================================================================

/// Two shapes of empty per row-wise norm: a zero batch beside a non-zero hidden
/// size, which makes the CUDA grid extent 0, and a zero hidden size beside a
/// non-zero batch, which makes the CUDA block extent 0. Both are launch errors,
/// not wrong answers.
const ROW_NORM_SHAPES: [(&[usize], usize); 2] = [(&[0, 4], 4), (&[2, 0], 0)];

fn check_row_norms<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: NormalizationOps<R>,
{
    for (shape, hidden) in ROW_NORM_SHAPES {
        let param = [hidden];
        let x_cpu = Tensor::<CpuRuntime>::zeros(shape, dtype, cpu_device).expect("cpu x");
        let w_cpu = Tensor::<CpuRuntime>::zeros(&param, dtype, cpu_device).expect("cpu weight");
        let b_cpu = Tensor::<CpuRuntime>::zeros(&param, dtype, cpu_device).expect("cpu bias");
        let x = Tensor::<R>::zeros(shape, dtype, device).expect("x");
        let w = Tensor::<R>::zeros(&param, dtype, device).expect("weight");
        let b = Tensor::<R>::zeros(&param, dtype, device).expect("bias");

        let expected = cpu_client
            .rms_norm(&x_cpu, &w_cpu, EPS)
            .expect("cpu rms_norm on an empty input");
        let actual = client
            .rms_norm(&x, &w, EPS)
            .unwrap_or_else(|e| panic!("{backend} rms_norm {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "{backend} rms_norm output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("rms_norm {shape:?} {backend} vs cpu"),
        );

        let expected = cpu_client
            .layer_norm(&x_cpu, &w_cpu, &b_cpu, EPS)
            .expect("cpu layer_norm on an empty input");
        let actual = client
            .layer_norm(&x, &w, &b, EPS)
            .unwrap_or_else(|e| panic!("{backend} layer_norm {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "{backend} layer_norm output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("layer_norm {shape:?} {backend} vs cpu"),
        );
    }
}

// ============================================================================
// group_norm
// ============================================================================

fn check_group_norm<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: NormalizationOps<R>,
{
    // `[0, 4, 2]` zeroes the batch, which zeroes the `batch * num_groups` grid.
    // `[2, 4, 0]` zeroes `spatial`, which zeroes the `channels_per_group *
    // spatial` block — the extent whose `.max(1)` used to hide it.
    for shape in [&[0usize, 4, 2][..], &[2, 4, 0][..]] {
        let param = [4usize];
        let x_cpu = Tensor::<CpuRuntime>::zeros(shape, dtype, cpu_device).expect("cpu x");
        let w_cpu = Tensor::<CpuRuntime>::zeros(&param, dtype, cpu_device).expect("cpu weight");
        let b_cpu = Tensor::<CpuRuntime>::zeros(&param, dtype, cpu_device).expect("cpu bias");
        let x = Tensor::<R>::zeros(shape, dtype, device).expect("x");
        let w = Tensor::<R>::zeros(&param, dtype, device).expect("weight");
        let b = Tensor::<R>::zeros(&param, dtype, device).expect("bias");

        let expected = cpu_client
            .group_norm(&x_cpu, &w_cpu, &b_cpu, 2, EPS)
            .expect("cpu group_norm on an empty input");
        let actual = client
            .group_norm(&x, &w, &b, 2, EPS)
            .unwrap_or_else(|e| panic!("{backend} group_norm {shape:?} {dtype:?}: {e:?}"));
        assert_eq!(actual.shape(), shape, "{backend} group_norm output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("group_norm {shape:?} {backend} vs cpu"),
        );
    }
}

/// `group_norm` with `num_groups == 0` must be an `InvalidArgument`, not a panic.
///
/// Not an empty shape, but the same degenerate-divisor family:
/// `channels.is_multiple_of(0)` answers `true` when `channels == 0`, so a zero
/// group count used to pass validation and then divide by zero. Both channel
/// counts are covered because only the zero one slips through `is_multiple_of`.
fn check_group_norm_zero_groups<R, C>(client: &C, device: &R::Device, dtype: DType, backend: &str)
where
    R: Runtime<DType = DType>,
    C: NormalizationOps<R>,
{
    for &channels in &[4usize, 0] {
        let param = [channels];
        let x = Tensor::<R>::zeros(&[2usize, channels, 3][..], dtype, device).expect("x");
        let w = Tensor::<R>::zeros(&param, dtype, device).expect("weight");
        let b = Tensor::<R>::zeros(&param, dtype, device).expect("bias");
        let err = client.group_norm(&x, &w, &b, 0, EPS).expect_err(&format!(
            "{backend} group_norm num_groups 0, channels {channels}, {dtype:?}: must be rejected"
        ));
        assert!(
            matches!(err, Error::InvalidArgument { .. }),
            "{backend} group_norm num_groups 0, channels {channels}: \
             want InvalidArgument, got {err:?}"
        );
    }
}

// ============================================================================
// softmax / softmax_bwd
// ============================================================================

fn check_softmax<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: ActivationOps<R>,
{
    // Only the zero-batch shape: see the module note on `[3, 0]`.
    let shape: &[usize] = &[0, 5];
    let x_cpu = Tensor::<CpuRuntime>::zeros(shape, dtype, cpu_device).expect("cpu x");
    let x = Tensor::<R>::zeros(shape, dtype, device).expect("x");

    let expected = cpu_client
        .softmax(&x_cpu, -1)
        .expect("cpu softmax on an empty input");
    let actual = client
        .softmax(&x, -1)
        .unwrap_or_else(|e| panic!("{backend} softmax {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), shape, "{backend} softmax output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("softmax [0,5] {backend} vs cpu"),
    );

    let expected = cpu_client
        .softmax_bwd(&x_cpu, &x_cpu, -1)
        .expect("cpu softmax_bwd on an empty input");
    let actual = client
        .softmax_bwd(&x, &x, -1)
        .unwrap_or_else(|e| panic!("{backend} softmax_bwd {dtype:?}: {e:?}"));
    assert_eq!(actual.shape(), shape, "{backend} softmax_bwd output shape");
    assert_tensor_allclose(
        &actual,
        &expected,
        dtype,
        &format!("softmax_bwd [0,5] {backend} vs cpu"),
    );
}

// ============================================================================
// GEMM epilogue
// ============================================================================
/// One gemm-epilogue shape case: `a` shape, `b` shape, expected output shape,
/// and the bias length.
/// One gemm-epilogue backward case: `a`, `b`, grad and bias shapes.
type GemmBwdCase = (
    &'static [usize],
    &'static [usize],
    &'static [usize],
    &'static [usize],
);

type GemmCase = (&'static [usize], &'static [usize], &'static [usize], usize);

fn check_gemm_epilogue<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: GemmEpilogueOps<R>,
{
    // `[0, 5] x [5, 3]` leaves an empty output. `[3, 0] x [0, 4]` leaves a full
    // `[3, 4]` output whose every element sums over no term, so the product part
    // is zero and only the bias — then the activation — remains.
    let cases: [GemmCase; 2] = [
        (&[0usize, 5][..], &[5usize, 3][..], &[0usize, 3][..], 3),
        (&[3, 0][..], &[0, 4][..], &[3, 4][..], 4),
    ];

    for (a_shape, b_shape, out_shape, n) in cases {
        let bias_shape = [n];
        let a_cpu = Tensor::<CpuRuntime>::zeros(a_shape, dtype, cpu_device).expect("cpu a");
        let b_cpu = Tensor::<CpuRuntime>::zeros(b_shape, dtype, cpu_device).expect("cpu b");
        let bias_cpu = Tensor::<CpuRuntime>::zeros(&bias_shape, dtype, cpu_device).expect("cpu c");
        let res_cpu = Tensor::<CpuRuntime>::zeros(out_shape, dtype, cpu_device).expect("cpu res");
        let a = Tensor::<R>::zeros(a_shape, dtype, device).expect("a");
        let b = Tensor::<R>::zeros(b_shape, dtype, device).expect("b");
        let bias = Tensor::<R>::zeros(&bias_shape, dtype, device).expect("bias");
        let res = Tensor::<R>::zeros(out_shape, dtype, device).expect("res");

        let expected = cpu_client
            .matmul_bias_activation(&a_cpu, &b_cpu, &bias_cpu, GemmActivation::ReLU)
            .expect("cpu matmul_bias_activation with an empty operand");
        assert_eq!(expected.shape(), out_shape, "cpu gemm output shape");
        let actual = client
            .matmul_bias_activation(&a, &b, &bias, GemmActivation::ReLU)
            .unwrap_or_else(|e| {
                panic!("{backend} matmul_bias_activation {a_shape:?}x{b_shape:?} {dtype:?}: {e:?}")
            });
        assert_eq!(actual.shape(), out_shape, "{backend} gemm output shape");
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("matmul_bias_activation {a_shape:?}x{b_shape:?} {backend} vs cpu"),
        );

        let expected = cpu_client
            .matmul_bias_residual(&a_cpu, &b_cpu, &bias_cpu, &res_cpu)
            .expect("cpu matmul_bias_residual with an empty operand");
        let actual = client
            .matmul_bias_residual(&a, &b, &bias, &res)
            .unwrap_or_else(|e| {
                panic!("{backend} matmul_bias_residual {a_shape:?}x{b_shape:?} {dtype:?}: {e:?}")
            });
        assert_eq!(
            actual.shape(),
            out_shape,
            "{backend} gemm residual output shape"
        );
        assert_tensor_allclose(
            &actual,
            &expected,
            dtype,
            &format!("matmul_bias_residual {a_shape:?}x{b_shape:?} {backend} vs cpu"),
        );
    }
}

/// The backward over a degenerate operand, in the two cases that differ:
///
/// - Zero batch (`[0, 2, 5] x [0, 5, 3]`): the gradient is empty, no batch
///   contributes, and all three gradients are the additive identity.
/// - `k == 0` with a NON-empty gradient (`[2, 0] x [0, 3]`, and its batched
///   counterpart): `d_a` and `d_b` are empty tensors of their own shapes, but
///   `d_bias` is a REAL reduction of `grad * act'(bias)` — the contraction adds
///   nothing, so the pre-activation is the bias alone. `a` and `b` are zero-byte
///   allocations that no GPU dispatch can bind, so the value has to be produced
///   another way. CPU is the reference.
///
/// `bias` is 1.0 and `grad` is 2.0 so ReLU's derivative at the pre-activation is
/// 1 and `d_bias` is a non-zero `2 * batch * M` — zeros would pass even if the
/// reduction were skipped entirely.
fn check_gemm_epilogue_bwd<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: GemmEpilogueOps<R>,
{
    // (a shape, b shape, grad shape, bias shape)
    let cases: [GemmBwdCase; 3] = [
        (&[0, 2, 5], &[0, 5, 3], &[0, 2, 3], &[3]),
        (&[2, 0], &[0, 3], &[2, 3], &[3]),
        (&[2, 3, 0], &[2, 0, 4], &[2, 3, 4], &[4]),
    ];

    for (a_shape, b_shape, g_shape, bias_shape) in cases {
        let a_cpu = Tensor::<CpuRuntime>::zeros(a_shape, dtype, cpu_device).expect("cpu a");
        let b_cpu = Tensor::<CpuRuntime>::zeros(b_shape, dtype, cpu_device).expect("cpu b");
        let g_cpu =
            Tensor::<CpuRuntime>::full_scalar(g_shape, dtype, 2.0, cpu_device).expect("cpu grad");
        let bias_cpu = Tensor::<CpuRuntime>::full_scalar(bias_shape, dtype, 1.0, cpu_device)
            .expect("cpu bias");
        let a = Tensor::<R>::zeros(a_shape, dtype, device).expect("a");
        let b = Tensor::<R>::zeros(b_shape, dtype, device).expect("b");
        let g = Tensor::<R>::full_scalar(g_shape, dtype, 2.0, device).expect("grad");
        let bias = Tensor::<R>::full_scalar(bias_shape, dtype, 1.0, device).expect("bias");

        let (e_da, e_db, e_dbias) = cpu_client
            .matmul_bias_activation_bwd(&g_cpu, &a_cpu, &b_cpu, &bias_cpu, GemmActivation::ReLU)
            .expect("cpu matmul_bias_activation_bwd over a degenerate operand");
        let (da, db, dbias) = client
            .matmul_bias_activation_bwd(&g, &a, &b, &bias, GemmActivation::ReLU)
            .unwrap_or_else(|e| {
                panic!("{backend} matmul_bias_activation_bwd {a_shape:?} {dtype:?}: {e:?}")
            });

        assert_eq!(da.shape(), a_shape, "{backend} d_a shape {a_shape:?}");
        assert_eq!(db.shape(), b_shape, "{backend} d_b shape {a_shape:?}");
        assert_eq!(
            dbias.shape(),
            bias_shape,
            "{backend} d_bias shape {a_shape:?}"
        );
        assert_tensor_allclose(
            &da,
            &e_da,
            dtype,
            &format!("gemm bwd d_a {a_shape:?} {backend} vs cpu"),
        );
        assert_tensor_allclose(
            &db,
            &e_db,
            dtype,
            &format!("gemm bwd d_b {a_shape:?} {backend} vs cpu"),
        );
        assert_tensor_allclose(
            &dbias,
            &e_dbias,
            dtype,
            &format!("gemm bwd d_bias {a_shape:?} {backend} vs cpu"),
        );
    }
}

// ============================================================================
// Per-backend entry points
// ============================================================================

#[test]
fn test_empty_normalization_cpu_is_self_consistent() {
    let (client, device) = create_cpu_client();
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        check_row_norms::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_group_norm::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_group_norm_zero_groups::<CpuRuntime, _>(&client, &device, dtype, "cpu");
        check_softmax::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
        check_gemm_epilogue::<CpuRuntime, _>(
            &client,
            &device,
            &cpu_client,
            &cpu_device,
            dtype,
            "cpu",
        );
        check_gemm_epilogue_bwd::<CpuRuntime, _>(
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
fn test_empty_normalization_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cuda") {
            check_row_norms::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_group_norm::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_group_norm_zero_groups::<CudaRuntime, _>(&client, &device, dtype, "cuda");
            check_softmax::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_gemm_epilogue::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_gemm_epilogue_bwd::<CudaRuntime, _>(
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
fn test_empty_normalization_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "wgpu") {
            check_row_norms::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_group_norm::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_group_norm_zero_groups::<WgpuRuntime, _>(&client, &device, dtype, "wgpu");
            check_softmax::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_gemm_epilogue::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_gemm_epilogue_bwd::<WgpuRuntime, _>(
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
