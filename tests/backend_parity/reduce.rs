// Backend parity tests for ReduceOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::ReduceOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

// ============================================================================
// Test Utilities
// ============================================================================

#[derive(Clone)]
struct ReduceTest {
    data: Vec<f64>,
    shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
}

impl ReduceTest {
    fn new(data: Vec<f64>, shape: Vec<usize>, dims: Vec<usize>, keepdim: bool) -> Self {
        ReduceTest {
            data,
            shape,
            dims,
            keepdim,
        }
    }
}

fn apply_reduce_op<R: Runtime>(
    client: &impl ReduceOps<R>,
    op: &str,
    tensor: &Tensor<R>,
    dims: &[usize],
    keepdim: bool,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "sum" => client.sum(tensor, dims, keepdim),
        "mean" => client.mean(tensor, dims, keepdim),
        "max" => client.max(tensor, dims, keepdim),
        "min" => client.min(tensor, dims, keepdim),
        "prod" => client.prod(tensor, dims, keepdim),
        "any" => client.any(tensor, dims, keepdim),
        "all" => client.all(tensor, dims, keepdim),
        _ => panic!("Unknown reduce op: {}", op),
    }
}

fn test_reduce_parity(op: &str, test_cases: &[ReduceTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let tensor = tensor_from_f64(&tc.data, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_reduce_op(&cpu_client, op, &tensor, &tc.dims, tc.keepdim)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_reduce_op(&cuda_client, op, &tensor, &tc.dims, tc.keepdim)
                    .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_reduce_op(&wgpu_client, op, &tensor, &tc.dims, tc.keepdim)
                    .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

macro_rules! reduce_case {
    ($name:ident, $op:expr, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
                test_reduce_parity($op, $cases, dtype);
            }
        }
    };
}

// ============================================================================
// Reduce Operation Parity Tests
// ============================================================================

reduce_case!(
    test_sum_parity,
    "sum",
    &[
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], true),
        ReduceTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![0],
            false,
        ),
        ReduceTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![1],
            false,
        ),
        ReduceTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
            vec![1],
            false,
        ),
        ReduceTest::new(
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![1, 2],
            false,
        ),
    ]
);

reduce_case!(
    test_mean_parity,
    "mean",
    &[
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![0],
            false,
        ),
        ReduceTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![1],
            false,
        ),
        ReduceTest::new(
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![0, 2],
            true,
        ),
    ]
);

reduce_case!(
    test_max_parity,
    "max",
    &[
        ReduceTest::new(vec![1.0, 4.0, 2.0, 3.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
            vec![2, 3],
            vec![0],
            false,
        ),
        ReduceTest::new(
            vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
            vec![2, 3],
            vec![1],
            false,
        ),
        ReduceTest::new(
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![0, 1],
            false,
        ),
    ]
);

reduce_case!(
    test_min_parity,
    "min",
    &[
        ReduceTest::new(vec![1.0, 4.0, 2.0, 3.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
            vec![2, 3],
            vec![0],
            false,
        ),
        ReduceTest::new(
            vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
            vec![2, 3],
            vec![1],
            false,
        ),
        ReduceTest::new(
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![0, 1],
            false,
        ),
    ]
);

reduce_case!(
    test_prod_parity,
    "prod",
    &[
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![2, 3],
            vec![0],
            false,
        ),
        ReduceTest::new(
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![2, 3],
            vec![1],
            false,
        ),
        ReduceTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 2, 3],
            vec![0, 2],
            false,
        ),
    ]
);

reduce_case!(
    test_any_parity,
    "any",
    &[
        ReduceTest::new(vec![0.0, 0.0, 0.0, 0.0], vec![4], vec![0], false),
        ReduceTest::new(vec![0.0, 1.0, 0.0, 2.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            vec![2, 3],
            vec![0],
            false,
        ),
        ReduceTest::new(
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            vec![2, 3],
            vec![1],
            false,
        ),
        ReduceTest::new(
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![1, 2, 3],
            vec![0, 2],
            false,
        ),
    ]
);

reduce_case!(
    test_all_parity,
    "all",
    &[
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
        ReduceTest::new(vec![1.0, 0.0, 2.0, 3.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0],
            vec![2, 3],
            vec![0],
            false,
        ),
        ReduceTest::new(
            vec![1.0, 2.0, 0.0, 1.0, 2.0, 3.0],
            vec![2, 3],
            vec![1],
            false,
        ),
        ReduceTest::new(
            vec![1.0, 2.0, 3.0, 1.0, 0.0, 3.0],
            vec![1, 2, 3],
            vec![0, 2],
            false,
        ),
    ]
);

// ============================================================================
// CPU Parallelism Config Test (F32-specific, not dtype-parameterized)
// ============================================================================

#[test]
fn test_cpu_reduce_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(64)));

    let shape = [96, 64, 32];
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.013).sin() + (i as f32 * 0.007).cos())
        .collect();
    let boolish_data: Vec<f32> = (0..numel)
        .map(|i| if i % 13 == 0 { 0.0 } else { 1.0 })
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&data, &shape, &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&boolish_data, &shape, &device).unwrap();

    let sum_base: Vec<f32> = default_client.sum(&a, &[1], false).unwrap().to_vec();
    let sum_cfg: Vec<f32> = configured_client.sum(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&sum_base, &sum_cfg, "cpu_reduce_parallelism_sum");

    let mean_base: Vec<f32> = default_client.mean(&a, &[1], false).unwrap().to_vec();
    let mean_cfg: Vec<f32> = configured_client.mean(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&mean_base, &mean_cfg, "cpu_reduce_parallelism_mean");

    let max_base: Vec<f32> = default_client.max(&a, &[1], false).unwrap().to_vec();
    let max_cfg: Vec<f32> = configured_client.max(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&max_base, &max_cfg, "cpu_reduce_parallelism_max");

    let prod_base: Vec<f32> = default_client.prod(&a, &[1], false).unwrap().to_vec();
    let prod_cfg: Vec<f32> = configured_client.prod(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&prod_base, &prod_cfg, "cpu_reduce_parallelism_prod");

    let any_base: Vec<f32> = default_client.any(&b, &[1], false).unwrap().to_vec();
    let any_cfg: Vec<f32> = configured_client.any(&b, &[1], false).unwrap().to_vec();
    assert_parity_f32(&any_base, &any_cfg, "cpu_reduce_parallelism_any");
}

// ============================================================================
// Narrow-float accumulation
//
// A sum held in a float narrower than F32 stops growing once the
// accumulator's spacing exceeds twice the increment. Every backend must
// widen the accumulator, so these pin an absolute value rather than parity:
// two backends that saturate identically would still pass a parity check.
// ============================================================================

#[cfg(any(feature = "f16", feature = "cuda"))]
fn mean_all_as_f32<R: Runtime<DType = DType>>(
    client: &(impl ReduceOps<R> + numr::ops::TypeConversionOps<R>),
    t: &Tensor<R>,
) -> f32 {
    let mean = client.mean(t, &[0, 1], false).expect("mean failed");
    client
        .cast(&mean, DType::F32)
        .expect("cast to F32 failed")
        .item()
        .expect("item failed")
}

/// A BF16 mean must not saturate, on any backend.
///
/// 262144 values of `1.0`. A BF16 accumulator reaches `256` and stops:
/// BF16's spacing at `256` is `2`, and `256 + 1` is a tie that rounds to the
/// even mantissa, back to `256`.
///
/// The reduction length is chosen to expose every backend at once. CPU sums
/// one bucket sequentially, so it stalls at `256` and reports a mean of
/// `0.0009765625`. CUDA splits the reduction across 256 threads and then
/// merges them in a tree, so each thread stalls at `256` and the tree gives
/// `65536`, for a mean of `0.25` — wrong by 4x, and wrong in a way a shorter
/// reduction would hide, since the tree stays exact while each thread's own
/// sum is still short of saturating.
///
/// Widened to F32 both backends give `262144` exactly, a mean of exactly
/// `1.0`, and `262144` is exactly representable in BF16.
#[cfg(feature = "f16")]
#[test]
fn test_bf16_mean_does_not_saturate() {
    let data = vec![1.0f64; 262144];
    let shape = [262144usize, 1];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(&data, &shape, DType::BF16, &cpu_device, &cpu_client)
        .expect("CPU BF16 tensor creation failed");
    let cpu_mean = mean_all_as_f32(&cpu_client, &cpu_tensor);
    assert_eq!(cpu_mean, 1.0, "CPU BF16 mean saturated: {cpu_mean}");

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::BF16) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let tensor = tensor_from_f64(&data, &shape, DType::BF16, &cuda_device, &cuda_client)
                .expect("CUDA BF16 tensor creation failed");
            let got = mean_all_as_f32(&cuda_client, &tensor);
            assert_eq!(got, 1.0, "CUDA BF16 mean saturated: {got}");
        });
    }
}

/// The same guarantee for F16 on CUDA.
///
/// F16 carries 10 mantissa bits, so it saturates later than BF16 and a CUDA
/// thread has to sum more values before its own accumulator stalls. 1048576
/// values of `2^-6`, split 4096 per thread, stall each thread at `32`; the
/// tree then merges them to `8192` instead of `16384`.
///
/// This asserts the sum directly, so the check stays on the accumulator and
/// does not depend on the mean epilogue. CUDA's `mean` is now safe at these
/// lengths too: it promotes a narrow float to F32 and does both the sum and
/// the division there, so neither the total nor the element count is narrowed.
/// The mean-side guarantees are pinned by
/// `test_f16_mean_divisor_does_not_overflow_cuda` and
/// `test_f16_mean_sum_does_not_saturate_cuda` below.
///
/// CPU is not asserted here because this tensor is 2 MiB, above the fused
/// multi-dim threshold, so CPU already reaches the widened SIMD kernel. CPU
/// F16 saturation is pinned by `test_f16_fused_multi_dim_mean_does_not_saturate`
/// in `runtime::cpu::helpers::reduce`.
#[cfg(all(feature = "f16", feature = "cuda"))]
#[test]
fn test_f16_sum_does_not_saturate_cuda() {
    use numr::ops::TypeConversionOps;

    let data = vec![0.015625f64; 1048576];
    let shape = [1048576usize, 1];

    if !is_dtype_supported("cuda", DType::F16) {
        return;
    }

    with_cuda_backend(|cuda_client, cuda_device| {
        let tensor = tensor_from_f64(&data, &shape, DType::F16, &cuda_device, &cuda_client)
            .expect("CUDA F16 tensor creation failed");
        let sum = cuda_client
            .sum(&tensor, &[0, 1], false)
            .expect("CUDA F16 sum failed");
        let got: f32 = cuda_client
            .cast(&sum, DType::F32)
            .expect("cast to F32 failed")
            .item()
            .expect("item failed");
        assert_eq!(got, 16384.0, "CUDA F16 sum saturated: {got}");
    });
}

/// The divisor half of the CUDA F16 mean overflow.
///
/// 65536 values of `0.5`. The true sum is `32768`, exactly representable in
/// F16, so the accumulator is not the problem here — the element count is.
/// Building the mean as `sum` then `div_scalar` narrows the divisor to the
/// tensor's own dtype, and `65536` is above F16's largest finite value of
/// `65504`, so the divisor becomes infinity and `32768 / inf` gives `0.0`.
///
/// Dividing in F32 and narrowing once gives exactly `0.5`, so this asserts
/// equality with no tolerance.
#[cfg(all(feature = "f16", feature = "cuda"))]
#[test]
fn test_f16_mean_divisor_does_not_overflow_cuda() {
    let data = vec![0.5f64; 65536];
    let shape = [65536usize, 1];

    if !is_dtype_supported("cuda", DType::F16) {
        return;
    }

    with_cuda_backend(|cuda_client, cuda_device| {
        let tensor = tensor_from_f64(&data, &shape, DType::F16, &cuda_device, &cuda_client)
            .expect("CUDA F16 tensor creation failed");
        let got = mean_all_as_f32(&cuda_client, &tensor);
        assert_eq!(got, 0.5, "CUDA F16 mean divisor overflowed: {got}");
    });
}

/// The sum half of the CUDA F16 mean overflow.
///
/// 131072 values of `1.0`. The true sum is `131072`, which does NOT fit in
/// F16: `sum` accumulates in F32 but writes its result back in the tensor's
/// dtype, so the total saturates to infinity. The result then comes back as
/// infinity or NaN — never `1.0`.
///
/// This is the case a divisor-only fix would miss, which is why it is separate
/// from `test_f16_mean_divisor_does_not_overflow_cuda`. Summing in F32 and
/// narrowing only the final mean gives exactly `1.0`.
#[cfg(all(feature = "f16", feature = "cuda"))]
#[test]
fn test_f16_mean_sum_does_not_saturate_cuda() {
    let data = vec![1.0f64; 131072];
    let shape = [131072usize, 1];

    if !is_dtype_supported("cuda", DType::F16) {
        return;
    }

    with_cuda_backend(|cuda_client, cuda_device| {
        let tensor = tensor_from_f64(&data, &shape, DType::F16, &cuda_device, &cuda_client)
            .expect("CUDA F16 tensor creation failed");
        let got = mean_all_as_f32(&cuda_client, &tensor);
        assert_eq!(got, 1.0, "CUDA F16 mean sum saturated: {got}");
    });
}

/// F32 `mean` must not be routed through the narrow-float promote path.
///
/// F32 is already the working dtype, so the fix for F16/BF16 must leave it on
/// the direct `sum` + `div_scalar` path. This pins the values so a later
/// refactor that promotes every dtype (or demotes through a narrower one) is
/// caught. Both cases are exact in F32, so equality is asserted with no
/// tolerance.
///
/// Unlike the F16 tests above, this one passes before the fix as well — it is
/// a guard against future regression, not a reproduction of the bug.
#[cfg(feature = "cuda")]
#[test]
fn test_f32_mean_unchanged_cuda() {
    if !is_dtype_supported("cuda", DType::F32) {
        return;
    }

    with_cuda_backend(|cuda_client, cuda_device| {
        // Same length that overflows an F16 divisor; F32 handles it directly.
        let big = vec![0.5f64; 65536];
        let big_shape = [65536usize, 1];
        let tensor = tensor_from_f64(&big, &big_shape, DType::F32, &cuda_device, &cuda_client)
            .expect("CUDA F32 tensor creation failed");
        let got = mean_all_as_f32(&cuda_client, &tensor);
        assert_eq!(got, 0.5, "CUDA F32 mean changed: {got}");

        let small = vec![1.0f64, 2.0, 3.0, 4.0];
        let small_shape = [4usize, 1];
        let tensor = tensor_from_f64(&small, &small_shape, DType::F32, &cuda_device, &cuda_client)
            .expect("CUDA F32 tensor creation failed");
        let got = mean_all_as_f32(&cuda_client, &tensor);
        assert_eq!(got, 2.5, "CUDA F32 mean changed: {got}");
    });
}
