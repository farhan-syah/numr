// Regression tests for the CUDA dim-reduction grid geometry.
//
// The dim-reduction kernels mapped `outer` to blockIdx.x and `inner` to
// blockIdx.y one-to-one. CUDA caps grid dimensions at [2147483647, 65535,
// 65535], so any reduction over a non-last dimension whose trailing dimensions
// multiply past 65535 was rejected at launch with CUDA_ERROR_INVALID_VALUE.
// Summing [1, 32, 4096, 64] over dim 1 (inner = 4096 * 64 = 262144) is an
// ordinary training shape, and it failed.
//
// Both grid axes now stride, so neither `outer` nor `inner` can hit a grid
// limit. These tests assert VALUES against the CPU reference, not merely that
// the call returned Ok: a launch that silently covers only part of the output
// plane must fail here too.
//
// Scoped to CUDA — the defect and the fix are both in the CUDA launch geometry.
// Small-shape value parity across every backend lives in reduce.rs.

#[cfg(feature = "cuda")]
use numr::dtype::DType;
#[cfg(feature = "cuda")]
use numr::ops::{IndexingOps, ReduceOps};

#[cfg(feature = "cuda")]
use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "cuda")]
use crate::common::{assert_tensor_allclose, create_cpu_client, is_dtype_supported};

/// Deterministic dyadic values: every element is a multiple of 0.25 and small,
/// so sums over the reduce dimension are exact in F32 on both backends and a
/// value mismatch means a real coverage or indexing error, never FP drift.
#[cfg(feature = "cuda")]
fn data_for(numel: usize) -> Vec<f64> {
    (0..numel).map(|i| ((i % 13) as f64 - 6.0) * 0.25).collect()
}

/// Print a loud skip marker when the CUDA feature is off, so an absent GPU is
/// never mistaken for a passing regression test.
#[cfg(not(feature = "cuda"))]
fn announce_cuda_skip(test: &str) {
    eprintln!(
        "SKIPPED (not a pass): {test} needs the `cuda` feature; \
         the reduce-dim grid regression was NOT exercised"
    );
}

/// Sum `shape` over `dim` on CPU and CUDA and assert the values match.
#[cfg(feature = "cuda")]
fn assert_sum_dim_parity(shape: &[usize], dim: usize, label: &str) {
    let dtype = DType::F32;
    assert!(
        is_dtype_supported("cuda", dtype),
        "CUDA must support F32 for {label}"
    );

    let numel: usize = shape.iter().product();
    let data = data_for(numel);

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_t = tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor failed for {label}: {e}"));
    let cpu_out = cpu_client
        .sum(&cpu_t, &[dim], false)
        .unwrap_or_else(|e| panic!("CPU sum failed for {label}: {e}"));

    with_cuda_backend(|client, device| {
        let t = tensor_from_f64(&data, shape, dtype, &device, &client)
            .unwrap_or_else(|e| panic!("CUDA tensor failed for {label}: {e}"));
        let out = client
            .sum(&t, &[dim], false)
            .unwrap_or_else(|e| panic!("CUDA sum failed for {label}: {e}"));
        assert_eq!(out.shape(), cpu_out.shape(), "sum shape: {label}");
        assert_tensor_allclose(&out, &cpu_out, dtype, &format!("sum CUDA vs CPU: {label}"));
    });
}

/// The observed failure: inner = 4096 * 64 = 262144, far past the 65535 cap on
/// grid.y. Before the fix the launch is rejected outright.
#[test]
fn test_sum_dim_inner_over_grid_limit() {
    #[cfg(feature = "cuda")]
    assert_sum_dim_parity(&[1, 32, 4096, 64], 1, "[1, 32, 4096, 64] sum dim 1");

    #[cfg(not(feature = "cuda"))]
    announce_cuda_skip("test_sum_dim_inner_over_grid_limit");
}

/// A size-1 reduce dimension with the same oversized inner extent: the launch
/// geometry must not depend on how much there is to reduce.
#[test]
fn test_sum_dim_size_one_reduce_with_large_inner() {
    #[cfg(feature = "cuda")]
    assert_sum_dim_parity(&[1, 1, 4096, 64], 1, "[1, 1, 4096, 64] sum dim 1");

    #[cfg(not(feature = "cuda"))]
    announce_cuda_skip("test_sum_dim_size_one_reduce_with_large_inner");
}

/// The symmetric hazard: outer past 65535. It rides on grid.x today, so this
/// case passes before the fix — it pins the behaviour so a future swap of the
/// axis roles cannot reintroduce the cliff unnoticed.
#[test]
fn test_sum_dim_outer_over_grid_limit() {
    #[cfg(feature = "cuda")]
    assert_sum_dim_parity(&[70_000, 3, 1], 1, "[70000, 3, 1] sum dim 1");

    #[cfg(not(feature = "cuda"))]
    announce_cuda_skip("test_sum_dim_outer_over_grid_limit");
}

/// Small shapes still reduce exactly as before: the stride loops must not
/// perturb results when one block already owns one output element.
#[test]
fn test_sum_dim_small_shapes_unchanged() {
    #[cfg(feature = "cuda")]
    {
        assert_sum_dim_parity(&[2, 3, 4], 1, "[2, 3, 4] sum dim 1");
        assert_sum_dim_parity(&[2, 3, 4], 0, "[2, 3, 4] sum dim 0");
        assert_sum_dim_parity(&[2, 3, 4], 2, "[2, 3, 4] sum dim 2");
        assert_sum_dim_parity(&[1, 5], 1, "[1, 5] sum dim 1");
    }

    #[cfg(not(feature = "cuda"))]
    announce_cuda_skip("test_sum_dim_small_shapes_unchanged");
}

/// max/min/any/all share the launch helper with sum, and argmax/argmin share it
/// too. All of them must survive an inner extent past 65535.
#[test]
fn test_all_dim_reductions_with_large_inner() {
    #[cfg(feature = "cuda")]
    {
        let dtype = DType::F32;
        assert!(
            is_dtype_supported("cuda", dtype),
            "CUDA must support F32 for the dim-reduction grid regression"
        );

        // 4096 * 64 = 262144 inner elements, past the 65535 grid.y cap.
        let shape: &[usize] = &[2, 8, 4096, 64];
        let numel: usize = shape.iter().product();
        let data = data_for(numel);
        let dim = 1;

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_t =
            tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client).expect("CPU tensor");
        let cpu_max = cpu_client.max(&cpu_t, &[dim], false).expect("CPU max");
        let cpu_min = cpu_client.min(&cpu_t, &[dim], false).expect("CPU min");
        let cpu_any = cpu_client.any(&cpu_t, &[dim], false).expect("CPU any");
        let cpu_all = cpu_client.all(&cpu_t, &[dim], false).expect("CPU all");
        let cpu_argmax = cpu_client.argmax(&cpu_t, dim, false).expect("CPU argmax");
        let cpu_argmin = cpu_client.argmin(&cpu_t, dim, false).expect("CPU argmin");

        with_cuda_backend(|client, device| {
            let t = tensor_from_f64(&data, shape, dtype, &device, &client).expect("CUDA tensor");

            let out = client.max(&t, &[dim], false).expect("CUDA max");
            assert_tensor_allclose(&out, &cpu_max, dtype, "max CUDA vs CPU, large inner");

            let out = client.min(&t, &[dim], false).expect("CUDA min");
            assert_tensor_allclose(&out, &cpu_min, dtype, "min CUDA vs CPU, large inner");

            let out = client.any(&t, &[dim], false).expect("CUDA any");
            assert_tensor_allclose(&out, &cpu_any, dtype, "any CUDA vs CPU, large inner");

            let out = client.all(&t, &[dim], false).expect("CUDA all");
            assert_tensor_allclose(&out, &cpu_all, dtype, "all CUDA vs CPU, large inner");

            let out = client.argmax(&t, dim, false).expect("CUDA argmax");
            assert_eq!(
                out.contiguous().expect("argmax contiguous").to_vec::<i64>(),
                cpu_argmax
                    .contiguous()
                    .expect("argmax contiguous")
                    .to_vec::<i64>(),
                "argmax CUDA vs CPU, large inner"
            );

            let out = client.argmin(&t, dim, false).expect("CUDA argmin");
            assert_eq!(
                out.contiguous().expect("argmin contiguous").to_vec::<i64>(),
                cpu_argmin
                    .contiguous()
                    .expect("argmin contiguous")
                    .to_vec::<i64>(),
                "argmin CUDA vs CPU, large inner"
            );
        });
    }

    #[cfg(not(feature = "cuda"))]
    announce_cuda_skip("test_all_dim_reductions_with_large_inner");
}
