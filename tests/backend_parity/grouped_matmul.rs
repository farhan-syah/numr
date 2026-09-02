// Backend parity tests for GroupedMatmulOps
//
// A grouped matmul is one independent GEMM per group, with the row boundaries
// held on device. The CUDA launcher cannot size its grid from a group's own row
// count for that reason, so it covers the total row count for every group and
// each block drops out if its row tile is past its group. The shapes here are
// chosen to put blocks on both sides of that bound.
//
// CPU is the reference. F32, F16 and BF16 all run the same tiled core, which
// accumulates in F32 whatever the storage dtype — so a half run differs from the
// F32 one only by the rounding of the stored operands and result.

use numr::ops::{GemmActivation, GroupedMatmulOps, MatmulOps};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::create_cpu_client;

/// Deterministic, non-repeating values so a mis-strided read changes the result.
fn values(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.017 + phase).sin() * 0.5)
        .collect()
}

/// Offsets tensor for a per-group row count.
fn offsets_from(counts: &[usize]) -> Vec<i32> {
    let mut out = Vec::with_capacity(counts.len() + 1);
    let mut running = 0i32;
    out.push(0);
    for &c in counts {
        running += c as i32;
        out.push(running);
    }
    out
}

/// Runs one grouped shape on CPU and CUDA and asserts they agree.
fn assert_grouped_parity(
    label: &str,
    counts: &[usize],
    k: usize,
    n: usize,
    activation: Option<GemmActivation>,
) {
    let num_groups = counts.len();
    let total_rows: usize = counts.iter().sum();
    let offsets = offsets_from(counts);

    let a_shape = [total_rows, k];
    let b_shape = [num_groups, k, n];
    let o_shape = [num_groups + 1];

    let a_data = values(total_rows * k, 0.0);
    let b_data = values(num_groups * k * n, 1.3);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &a_shape, &cpu_device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &b_shape, &cpu_device).unwrap();
    let o = Tensor::<CpuRuntime>::from_slice(&offsets, &o_shape, &cpu_device).unwrap();

    let cpu_out = match activation {
        Some(act) => cpu_client.grouped_matmul_activation(&a, &b, &o, act),
        None => cpu_client.grouped_matmul(&a, &b, &o),
    }
    .unwrap_or_else(|e| panic!("CPU grouped matmul failed for {label}: {e}"));
    let cpu_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let a_c = Tensor::from_slice(&a_data, &a_shape, &device).unwrap();
        let b_c = Tensor::from_slice(&b_data, &b_shape, &device).unwrap();
        let o_c = Tensor::from_slice(&offsets, &o_shape, &device).unwrap();
        let out = match activation {
            Some(act) => client.grouped_matmul_activation(&a_c, &b_c, &o_c, act),
            None => client.grouped_matmul(&a_c, &b_c, &o_c),
        }
        .unwrap_or_else(|e| panic!("CUDA grouped matmul failed for {label}: {e}"));

        let got = out.to_vec::<f32>();
        assert_eq!(got.len(), cpu_vec.len(), "{label}: length mismatch");
        for (i, (x, y)) in got.iter().zip(cpu_vec.iter()).enumerate() {
            let tol = 1e-4 + 1e-4 * y.abs();
            assert!((x - y).abs() <= tol, "{label} at {i}: CUDA {x} vs CPU {y}");
        }
    });
}

/// Even split wide enough that every group spans several row tiles.
#[test]
fn grouped_matmul_even_split_parity() {
    assert_grouped_parity("grouped_even", &[96, 96, 96, 96], 64, 160, None);
}

/// Uneven split, so the cut-off row tile lands somewhere different per group.
#[test]
fn grouped_matmul_uneven_split_parity() {
    assert_grouped_parity("grouped_uneven", &[5, 70, 33, 120], 64, 160, None);
}

/// A group with no rows at all, between two that have them.
#[test]
fn grouped_matmul_empty_group_parity() {
    assert_grouped_parity("grouped_empty", &[64, 0, 64], 48, 160, None);
}

/// Row counts that are not whole tiles, so the last tile of each group is
/// partly valid rather than wholly in or out.
#[test]
fn grouped_matmul_partial_last_tile_parity() {
    assert_grouped_parity("grouped_partial", &[33, 65, 97], 48, 160, None);
}

/// Narrow output, which selects the small-tile kernel instead of the large one.
#[test]
fn grouped_matmul_narrow_output_parity() {
    assert_grouped_parity("grouped_narrow", &[40, 72], 48, 48, None);
}

/// K not a multiple of the tile depth, so the K loop runs a ragged last step.
#[test]
fn grouped_matmul_ragged_k_parity() {
    assert_grouped_parity("grouped_ragged_k", &[48, 48], 37, 160, None);
}

/// A single group, which is a plain dense matmul routed through the grouped path.
#[test]
fn grouped_matmul_single_group_parity() {
    assert_grouped_parity("grouped_single", &[80], 64, 160, None);
}

/// Fused SiLU epilogue.
#[test]
fn grouped_matmul_silu_parity() {
    assert_grouped_parity(
        "grouped_silu",
        &[33, 65, 97],
        48,
        160,
        Some(GemmActivation::SiLU),
    );
}

/// Fused GELU epilogue.
#[test]
fn grouped_matmul_gelu_parity() {
    assert_grouped_parity(
        "grouped_gelu",
        &[33, 65, 97],
        48,
        160,
        Some(GemmActivation::GELU),
    );
}

/// Fused ReLU epilogue on the small-tile kernel.
#[test]
fn grouped_matmul_relu_narrow_parity() {
    assert_grouped_parity(
        "grouped_relu_narrow",
        &[40, 72],
        48,
        48,
        Some(GemmActivation::ReLU),
    );
}

/// A single group must match a plain dense matmul of the same operands.
#[test]
fn grouped_matmul_single_group_matches_dense_matmul() {
    let (client, device) = create_cpu_client();
    let (rows, k, n) = (80usize, 64usize, 160usize);
    let a_data = values(rows * k, 0.0);
    let b_data = values(k * n, 1.3);

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[rows, k], &device).unwrap();
    let b3 = Tensor::<CpuRuntime>::from_slice(&b_data, &[1, k, n], &device).unwrap();
    let b2 = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &device).unwrap();
    let o = Tensor::<CpuRuntime>::from_slice(&[0i32, rows as i32], &[2], &device).unwrap();

    let grouped = client.grouped_matmul(&a, &b3, &o).unwrap().to_vec::<f32>();
    let dense = client.matmul(&a, &b2).unwrap().to_vec::<f32>();

    assert_eq!(grouped.len(), dense.len());
    for (i, (x, y)) in grouped.iter().zip(dense.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5 + 1e-5 * y.abs(),
            "row-major mismatch at {i}: grouped {x} vs dense {y}"
        );
    }
}

/// Runs one grouped shape in a half dtype on CUDA and compares against the F32
/// CUDA result for the SAME values, rounded through that dtype first.
///
/// F32 is the reference rather than CPU because the core accumulates in F32
/// regardless of storage: comparing against a CPU F32 run would fold in the
/// storage rounding and say nothing about the kernel.
#[cfg(all(feature = "cuda", feature = "f16"))]
fn assert_grouped_half_parity(
    label: &str,
    dtype: numr::dtype::DType,
    counts: &[usize],
    k: usize,
    n: usize,
) {
    use numr::ops::TypeConversionOps;

    let num_groups = counts.len();
    let total_rows: usize = counts.iter().sum();
    let offsets = offsets_from(counts);
    let a_data = values(total_rows * k, 0.0);
    let b_data = values(num_groups * k * n, 1.3);

    with_cuda_backend(|client, device| {
        let a32 = Tensor::from_slice(&a_data, &[total_rows, k], &device).unwrap();
        let b32 = Tensor::from_slice(&b_data, &[num_groups, k, n], &device).unwrap();
        let o = Tensor::from_slice(&offsets, &[num_groups + 1], &device).unwrap();

        // Round-trip first, so both runs see identical values.
        let a_h = client.cast(&a32, dtype).unwrap();
        let b_h = client.cast(&b32, dtype).unwrap();
        let a_ref = client.cast(&a_h, numr::dtype::DType::F32).unwrap();
        let b_ref = client.cast(&b_h, numr::dtype::DType::F32).unwrap();

        let reference = client
            .grouped_matmul(&a_ref, &b_ref, &o)
            .unwrap_or_else(|e| panic!("F32 reference failed for {label}: {e}"))
            .to_vec::<f32>();

        let half = client
            .grouped_matmul(&a_h, &b_h, &o)
            .unwrap_or_else(|e| panic!("half grouped matmul failed for {label}: {e}"));
        let got = client
            .cast(&half, numr::dtype::DType::F32)
            .unwrap()
            .to_vec::<f32>();

        // Only the stored output is rounded; BF16 carries 8 mantissa bits.
        let (rtol, atol) = match dtype {
            numr::dtype::DType::BF16 => (8e-3f32, 1e-3f32),
            _ => (1e-3f32, 1e-4f32),
        };
        assert_eq!(got.len(), reference.len(), "{label}: length mismatch");
        for (i, (x, y)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                (x - y).abs() <= atol + rtol * y.abs(),
                "{label} at {i}: half {x} vs F32 {y}"
            );
        }
    });
}

/// F16 with an uneven split, so tiles fall on both sides of each group's bound.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn grouped_matmul_f16_uneven_split_parity() {
    assert_grouped_half_parity(
        "grouped_f16_uneven",
        numr::dtype::DType::F16,
        &[5, 70, 33, 120],
        64,
        160,
    );
}

/// BF16 with an uneven split.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn grouped_matmul_bf16_uneven_split_parity() {
    assert_grouped_half_parity(
        "grouped_bf16_uneven",
        numr::dtype::DType::BF16,
        &[5, 70, 33, 120],
        64,
        160,
    );
}

/// F16 on the small-tile kernel, which the narrow output selects.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn grouped_matmul_f16_narrow_output_parity() {
    assert_grouped_half_parity(
        "grouped_f16_narrow",
        numr::dtype::DType::F16,
        &[40, 72],
        48,
        48,
    );
}
