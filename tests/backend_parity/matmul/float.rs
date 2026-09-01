// Backend parity tests for MatmulOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes (F32, F64, F16, BF16, FP8).
// Tensors are created in f64 then cast to target dtype via tensor_from_f64().
// Comparison reads back in native dtype - no unnecessary f64 conversion.

use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
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

struct MatmulTest {
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
}

impl MatmulTest {
    fn new(a: Vec<f64>, a_shape: Vec<usize>, b: Vec<f64>, b_shape: Vec<usize>) -> Self {
        MatmulTest {
            a,
            a_shape,
            b,
            b_shape,
        }
    }
}

fn test_matmul_parity(test_cases: &[MatmulTest], dtype: DType) {
    // CPU baseline
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            cpu_client
                .matmul(&a, &b)
                .unwrap_or_else(|e| panic!("CPU matmul failed for {dtype:?}: {e}"))
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                let result = cuda_client
                    .matmul(&a, &b)
                    .unwrap_or_else(|e| panic!("CUDA matmul failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("matmul CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                let result = wgpu_client
                    .matmul(&a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU matmul failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("matmul WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

// ============================================================================
// Matmul Parity Tests
// ============================================================================

macro_rules! matmul_case {
    ($name:ident, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
                test_matmul_parity($cases, dtype);
            }
        }
    };
}

matmul_case!(
    test_matmul_2d_parity,
    &[MatmulTest::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![3, 4],
    )]
);

matmul_case!(
    test_matmul_square_parity,
    &[MatmulTest::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        vec![3, 3],
    )]
);

matmul_case!(
    test_matmul_batched_parity,
    &[MatmulTest::new(
        vec![
            // Batch 0: 3x4
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1: 3x4
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        ],
        vec![2, 3, 4],
        vec![
            // Batch 0: 4x2
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Batch 1: 4x2
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ],
        vec![2, 4, 2],
    )]
);

matmul_case!(
    test_matmul_vector_parity,
    &[MatmulTest::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1, 4],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![4, 1],
    )]
);

// K=48 and K=80 are both multiples of 16, the minimum use_wmma requires
// (K % 16 == 0). With BLOCK_K=16 this gives several full K-tiles and no
// partial tile, so these cases exercise multi-K-tile accumulation, not a
// K-tail zero-pad — that branch is unreachable while use_wmma requires
// K % 16 == 0.
matmul_case!(
    test_matmul_wmma_k_tail_parity,
    &[
        MatmulTest::new(
            // M=16, K=48: one full WMMA row-block, K crosses a 32-boundary
            (0..16 * 48).map(|i| (i as f64) * 0.001 + 0.5).collect(),
            vec![16, 48],
            (0..48 * 16).map(|i| (i as f64) * 0.002 - 0.3).collect(),
            vec![48, 16],
        ),
        MatmulTest::new(
            // M=32, K=80: K=80 = 32*2 + 16, two full BLOCK_K tiles + one partial
            (0..32 * 80).map(|i| ((i as f64) * 0.0007).sin()).collect(),
            vec![32, 80],
            (0..80 * 32).map(|i| ((i as f64) * 0.0013).cos()).collect(),
            vec![80, 32],
        ),
    ]
);

// ============================================================================
// WMMA Warp-Tiling Boundary Cases
// ============================================================================
//
// These tests exercise the NEW 128×128 warp-tiled WMMA kernel's boundary
// masking and multi-block-tile coverage.
//
// - 80×96×48 (F16+BF16): dims are multiples of 16 but NOT 128.  M=80 < 128
//   means a single block row that is NOT full; N=96 < 128 same; K=48 = 3 ×
//   BLOCK_K(16) — no K-tail zero-pad needed.
// - 144×176×64 (F16+BF16): spans MORE than one 128-tile in both M and N
//   (ceil(144/128)=2 row blocks, ceil(176/128)=2 col blocks), with the second
//   block being a partial tile.  Verifies boundary masking and multi-tile
//   accumulation.
// - 256×256×256 (F16+BF16): square, all dims exact multiples of 128.  No
//   boundary masking; tests peak-throughput correctness.

#[test]
#[cfg(feature = "f16")]
fn test_matmul_wmma_warp_tile_boundary_f16() {
    // 80×96×48: single partial block tile in M and N, clean K.
    let cases = [
        MatmulTest::new(
            (0..80 * 48)
                .map(|i| ((i as f64) * 0.0011).sin() * 0.6)
                .collect(),
            vec![80, 48],
            (0..48 * 96)
                .map(|i| ((i as f64) * 0.0017).cos() * 0.4)
                .collect(),
            vec![48, 96],
        ),
        // 144×176×64: two block tiles in both M and N.
        MatmulTest::new(
            (0..144 * 64)
                .map(|i| ((i as f64) * 0.0009 - 0.1).sin() * 0.5)
                .collect(),
            vec![144, 64],
            (0..64 * 176)
                .map(|i| ((i as f64) * 0.0013 + 0.2).cos() * 0.5)
                .collect(),
            vec![64, 176],
        ),
        // 256×256×256: clean square, tests multi-block correctness.
        MatmulTest::new(
            (0..256 * 256)
                .map(|i| ((i as f64) * 0.0005).sin() * 0.3)
                .collect(),
            vec![256, 256],
            (0..256 * 256)
                .map(|i| ((i as f64) * 0.0007).cos() * 0.3)
                .collect(),
            vec![256, 256],
        ),
    ];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_wmma_warp_tile_boundary_bf16() {
    // Same shapes as the F16 test, using BF16.
    let cases = [
        MatmulTest::new(
            (0..80 * 48)
                .map(|i| ((i as f64) * 0.0011).sin() * 0.6)
                .collect(),
            vec![80, 48],
            (0..48 * 96)
                .map(|i| ((i as f64) * 0.0017).cos() * 0.4)
                .collect(),
            vec![48, 96],
        ),
        MatmulTest::new(
            (0..144 * 64)
                .map(|i| ((i as f64) * 0.0009 - 0.1).sin() * 0.5)
                .collect(),
            vec![144, 64],
            (0..64 * 176)
                .map(|i| ((i as f64) * 0.0013 + 0.2).cos() * 0.5)
                .collect(),
            vec![64, 176],
        ),
        MatmulTest::new(
            (0..256 * 256)
                .map(|i| ((i as f64) * 0.0005).sin() * 0.3)
                .collect(),
            vec![256, 256],
            (0..256 * 256)
                .map(|i| ((i as f64) * 0.0007).cos() * 0.3)
                .collect(),
            vec![256, 256],
        ),
    ];
    test_matmul_parity(&cases, numr::dtype::DType::BF16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_wmma_aligned_controls_f16() {
    // ALIGNED (all multiples of 16) — isolates whether the WMMA kernel itself is
    // wrong for certain block configs vs the padding wrapper.
    let cases = [
        make_f32_test_data(128, 128, 0.0011, 144, 0.0019), // M=1 block, N=2 blocks
        make_f32_test_data(144, 128, 0.0013, 128, 0.0017), // M=2 blocks, N=1 block
        make_f32_test_data(128, 128, 0.0007, 128, 0.0009), // both exactly 1 block
        make_f32_test_data(128, 80, 0.0005, 128, 0.0015),  // K=80 (5 tiles), 1 block M/N
    ];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

// M unaligned, N & K aligned — THE varlen-embedding case (M=total_tokens unaligned,
// N/K=hidden always 16-aligned). matmul() pads M to a multiple of 16 so WMMA fires.
#[test]
#[cfg(feature = "f16")]
fn test_matmul_wmma_unaligned_m_f16() {
    let cases = [
        make_f32_test_data(130, 64, 0.0013, 128, 0.0017),
        make_f32_test_data(145, 128, 0.0011, 256, 0.0019),
        make_f32_test_data(200, 64, 0.0007, 192, 0.0009),
        make_f32_test_data(17, 32, 0.0021, 48, 0.0008), // just above GEMV threshold
    ];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_wmma_unaligned_m_bf16() {
    let cases = [
        make_f32_test_data(130, 64, 0.0013, 128, 0.0017),
        make_f32_test_data(145, 128, 0.0011, 256, 0.0019),
        make_f32_test_data(200, 64, 0.0007, 192, 0.0009),
    ];
    test_matmul_parity(&cases, numr::dtype::DType::BF16);
}

// Unaligned N/K for F16/BF16. NOTE: the CUDA kernels were ALWAYS correct here;
// the original "failure" was a bug in the CPU REFERENCE matmul — its SIMD packer
// zero-padded a partial N-block to stride NR while the edge microkernel read with
// stride nr_actual, corrupting every kk>0 (fixed in simd/matmul/packing.rs). The
// aligned tests (N=144=9×16) never hit a partial block, so it stayed latent. With
// the CPU reference fixed, CUDA now matches it.
// REGRESSION (§7e): 3D input [1, M, K] with unaligned M through the WMMA
// pad-to-16 wrapper. The padded encoder forward (single query, batch=1) sends a
// 3D [1, seq, hidden] tensor; the wrapper must pad/narrow the LAST TWO dims, not
// dims 0/1 — narrowing dim 0 (the size-1 batch dim) produced a degenerate [0, …].
#[test]
#[cfg(feature = "f16")]
fn test_matmul_3d_unaligned_m_f16() {
    // a: [1, 17, 64], b: [64, 48] → out [1, 17, 48]. M=17 unaligned.
    let a: Vec<f64> = (0..17 * 64)
        .map(|i| ((i as f64) * 0.0013 + 0.2).sin() * 0.5)
        .collect();
    let b: Vec<f64> = (0..64 * 48)
        .map(|i| ((i as f64) * 0.0017 - 0.1).cos() * 0.5)
        .collect();
    let cases = [MatmulTest::new(a, vec![1, 17, 64], b, vec![64, 48])];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
    // Also a larger hidden + K/N aligned, mirroring the real projection shape.
    let a2: Vec<f64> = (0..17 * 128)
        .map(|i| ((i as f64) * 0.0009).sin() * 0.4)
        .collect();
    let b2: Vec<f64> = (0..128 * 256)
        .map(|i| ((i as f64) * 0.0011).cos() * 0.4)
        .collect();
    let cases2 = [MatmulTest::new(a2, vec![1, 17, 128], b2, vec![128, 256])];
    test_matmul_parity(&cases2, numr::dtype::DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_unaligned_nk_f16() {
    let cases = [
        make_f32_test_data(128, 128, 0.0011, 130, 0.0019), // N unaligned (2-wide trailing)
        make_f32_test_data(128, 70, 0.0007, 128, 0.0009),  // K unaligned
        make_f32_test_data(100, 100, 0.0005, 100, 0.0015), // M, N, K all unaligned
    ];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_unaligned_nk_bf16() {
    let cases = [
        make_f32_test_data(128, 128, 0.0011, 130, 0.0019),
        make_f32_test_data(128, 70, 0.0007, 128, 0.0009),
    ];
    test_matmul_parity(&cases, numr::dtype::DType::BF16);
}

// ============================================================================
// WMMA Determinism Test
// ============================================================================
//
// Runs the SAME F16 matmul (144×176×64) 50 times on CUDA and asserts ALL 50
// results are BITWISE identical.  A nondeterministic race (e.g. smem WAR, missing
// __syncthreads, or dangling async copy) fails this because it produces different
// bit patterns on different runs.  Synthetic parity tests with clean shapes can
// miss races; repeated-run bitwise equality is the definitive check.

#[test]
#[cfg(all(feature = "cuda", feature = "f16"))]
fn test_matmul_wmma_determinism_f16() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let m = 144usize;
        let n = 176usize;
        let k = 64usize;
        let dtype = numr::dtype::DType::F16;

        let a_data: Vec<f64> = (0..m * k)
            .map(|i| ((i as f64) * 0.0009 - 0.1).sin() * 0.5)
            .collect();
        let b_data: Vec<f64> = (0..k * n)
            .map(|i| ((i as f64) * 0.0013 + 0.2).cos() * 0.5)
            .collect();

        let a = tensor_from_f64(&a_data, &[m, k], dtype, &cuda_device, &cuda_client)
            .expect("tensor_from_f64 A");
        let b = tensor_from_f64(&b_data, &[k, n], dtype, &cuda_device, &cuda_client)
            .expect("tensor_from_f64 B");

        // First run as reference.
        let reference = cuda_client.matmul(&a, &b).expect("matmul run 0");
        let reference_bits: Vec<u16> = reference
            .to_vec::<half::f16>()
            .iter()
            .map(|x| x.to_bits())
            .collect();

        for run in 1..50usize {
            let result = cuda_client
                .matmul(&a, &b)
                .unwrap_or_else(|e| panic!("matmul run {run} failed: {e}"));
            let bits: Vec<u16> = result
                .to_vec::<half::f16>()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            assert_eq!(
                bits.len(),
                reference_bits.len(),
                "run {run}: result length mismatch"
            );
            for (idx, (got, expected)) in bits.iter().zip(reference_bits.iter()).enumerate() {
                assert_eq!(
                    got, expected,
                    "WMMA determinism: run {run} element {idx} differs: \
                     bits {got:#06x} vs reference {expected:#06x} (nondeterministic race)"
                );
            }
        }
    });
}

// ============================================================================
// F32 Compile-Time-Tiled Kernel Correctness
// ============================================================================
//
// These tests specifically exercise the compile-time-tiled FP32 GEMM path
// (`matmul_f32_tiled_*` extern "C" kernels) added to fix the local-memory
// spill that caused ~61 GFLOP/s on the generic `matmul_f32` kernel.
//
// - 256×192×320: large square, uses the 128×128×8 config.  All dims are
//   multiples of the block tile so no boundary masking is needed.
// - 130×77×141:  non-tile-multiple dims — exercises the bounds-check / zero-pad
//   paths in ct_load_a / ct_load_b and the guarded epilogue writes.
//   With BM=128 the last row-block covers only 2 real rows (130 mod 128 = 2);
//   with BN=128 the last col-block covers only 2 real cols (77 mod 64 ≈ edge).
//   K=141 is not a multiple of BK=8 or BK=32 so the last k-tile is partial.

fn make_f32_test_data(m: usize, k: usize, seed_a: f64, n: usize, seed_b: f64) -> MatmulTest {
    let a: Vec<f64> = (0..m * k)
        .map(|i| ((i as f64) * seed_a + 0.37).sin() * 0.5)
        .collect();
    let b: Vec<f64> = (0..k * n)
        .map(|i| ((i as f64) * seed_b - 0.21).cos() * 0.5)
        .collect();
    MatmulTest::new(a, vec![m, k], b, vec![k, n])
}

// Third independent reference for out[0][128] of 128x128x130: who is right,
// CUDA (~0.087) or the CPU matmul reference (~-0.349)?
#[test]
fn test_matmul_manual_ref_128x128x130() {
    let tc = make_f32_test_data(128, 128, 0.0011, 130, 0.0019); // A[128,128], B[128,130]
    let (_m, k, n) = (128usize, 128usize, 130usize);
    let a = &tc.a; // row-major [m,k] (f64)
    let b = &tc.b; // row-major [k,n] (f64)
    let mut manual = 0.0f64;
    for kk in 0..k {
        manual += a[kk] * b[kk * n + 128];
    }
    // CPU matmul reference value:
    let (cpu_client, cpu_device) = create_cpu_client();
    let at = tensor_from_f64(
        &tc.a,
        &tc.a_shape,
        numr::dtype::DType::F32,
        &cpu_device,
        &cpu_client,
    )
    .unwrap();
    let bt = tensor_from_f64(
        &tc.b,
        &tc.b_shape,
        numr::dtype::DType::F32,
        &cpu_device,
        &cpu_client,
    )
    .unwrap();
    let cpu_out: Vec<f32> = cpu_client.matmul(&at, &bt).unwrap().to_vec();
    println!(
        "MANUAL out[0][128] = {manual:.6}   CPU-matmul out[0][128] = {:.6}",
        cpu_out[128]
    );
    // assert manual matches CPU matmul — if NOT, the CPU kernel is the buggy one.
    assert!(
        (manual as f32 - cpu_out[128]).abs() < 1e-3,
        "CPU matmul disagrees with manual dot product!"
    );
}

// Pure-F32 isolation of the partial sub-16 trailing-block bug. If these FAIL the
// fault is in the compile-time-tiled F32 kernel; if they PASS it is F16/WMMA-only.
#[test]
fn test_matmul_f32_partial_trailing_block() {
    let cases = [
        make_f32_test_data(128, 128, 0.0011, 130, 0.0019), // N=130: 2-wide 2nd N-block
        make_f32_test_data(128, 130, 0.0007, 128, 0.0009), // K=130
        make_f32_test_data(130, 128, 0.0013, 128, 0.0017), // M=130: 2-wide 2nd M-block
        make_f32_test_data(200, 200, 0.0005, 200, 0.0015), // all > one block, partial
    ];
    test_matmul_parity(&cases, numr::dtype::DType::F32);
}

#[test]
fn test_matmul_f32_tiled_large_parity() {
    // 256×192×320 — large, all dims multiples of tile (128, 64).
    // Uses `matmul_f32_tiled_128x128x8_8x8` config.
    let cases = [make_f32_test_data(256, 320, 0.0013, 192, 0.0017)];
    test_matmul_parity(&cases, DType::F32);
}

#[test]
fn test_matmul_f32_tiled_non_tile_multiple_parity() {
    // 130×77×141 — all dims are NOT multiples of any tile parameter.
    // Exercises bounds masking in loads AND the guarded epilogue.
    let cases = [make_f32_test_data(130, 141, 0.0019, 77, 0.0023)];
    test_matmul_parity(&cases, DType::F32);
}

// ============================================================================
// Single-Token Decode (m=1..8) Parity
// ============================================================================
//
// `use_wmma` used to require m > 16, so decode-shaped matmuls (m=1, the
// single-token LLM decode step) stayed on the GEMV path. That gate is gone:
// small m now pads up to 16 and runs on the WMMA kernel. These cases
// bracket m=1 (the target shape), plus m=2 and m=8, to cover the padded
// small-m range rather than probing a single point.

#[test]
#[cfg(feature = "f16")]
fn test_matmul_m1_decode_f16() {
    // m=1: single-token decode; m pads to 16. K, N are 16-multiples so only
    // m needs padding.
    let cases = [make_f32_test_data(1, 64, 0.0013, 128, 0.0017)];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_m1_decode_bf16() {
    let cases = [make_f32_test_data(1, 64, 0.0013, 128, 0.0017)];
    test_matmul_parity(&cases, numr::dtype::DType::BF16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_m1_decode_unaligned_nk_f16() {
    // m=1 with K=70, N=50 (neither a 16-multiple): exercises the full
    // pad-all-dims path, not just the m pad.
    let cases = [make_f32_test_data(1, 70, 0.0011, 50, 0.0019)];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_m2_decode_f16() {
    // m=2: brackets m=1 from above within the padded small-m range.
    let cases = [make_f32_test_data(2, 64, 0.0013, 128, 0.0017)];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn test_matmul_m8_decode_f16() {
    // m=8: upper bracket, still padded (not yet a 16-multiple).
    let cases = [make_f32_test_data(8, 64, 0.0013, 128, 0.0017)];
    test_matmul_parity(&cases, numr::dtype::DType::F16);
}

#[test]
fn test_cpu_matmul_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(1024)));

    // Batched case to exercise batch parallelism path.
    let a_shape = [6, 24, 16];
    let b_shape = [6, 16, 12];
    let a_numel: usize = a_shape.iter().product();
    let b_numel: usize = b_shape.iter().product();
    let a_data: Vec<f32> = (0..a_numel)
        .map(|i| (i as f32 * 0.013).sin() + (i as f32 * 0.007).cos())
        .collect();
    let b_data: Vec<f32> = (0..b_numel)
        .map(|i| (i as f32 * 0.011).cos() - (i as f32 * 0.005).sin())
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &a_shape, &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &b_shape, &device).unwrap();

    let base: Vec<f32> = default_client.matmul(&a, &b).unwrap().to_vec();
    let cfg: Vec<f32> = configured_client.matmul(&a, &b).unwrap().to_vec();

    // Compare with tight tolerance for f32
    assert_eq!(base.len(), cfg.len(), "result length mismatch");
    for (i, (b_val, c_val)) in base.iter().zip(cfg.iter()).enumerate() {
        assert!(
            (b_val - c_val).abs() <= 1e-5,
            "element {} differs: {} vs {} (diff={})",
            i,
            b_val,
            c_val,
            (b_val - c_val).abs()
        );
    }
}
