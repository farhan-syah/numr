// Backend parity tests for GroupedMatmulOps
//
// A grouped matmul is one independent GEMM per group, with the row boundaries
// held on device. The CUDA launcher cannot size its grid from a group's own row
// count for that reason, so it covers the total row count for every group and
// each block drops out if its row tile is past its group. The shapes here are
// chosen to put blocks on both sides of that bound.
//
// CPU is the reference. F32 only: the kernel reuses the F32 tiled GEMM core.

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
