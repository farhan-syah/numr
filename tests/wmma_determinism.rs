//! Determinism check for the CUDA WMMA GEMM path.
//!
//! Backend parity compares against CPU once per shape, within a tolerance.
//! That cannot catch a timing-dependent race. The race need not fire on a
//! given launch, and a corrupted value can land inside the tolerance.
//!
//! This test runs identical input `REPEATS` times and compares raw bits.
//! One flipped bit fails.
//!
//! Shapes target both tiles. `m < 128 || n < 128` always takes the small tile.
//! The large tile needs a grid past a device-dependent wave threshold, so
//! batching encourages it rather than guaranteeing it.
//!
//! Unaligned shapes drive the pad-then-slice path. Pad copies into a wider
//! buffer, the kernel writes there, then a narrow copies the result back.

// Every test here compares f16/bf16 WMMA output, so the file needs the `f16`
// feature as well as `cuda` — without it `half` is not even a dependency.
#![cfg(all(feature = "cuda", feature = "f16"))]

mod common;

use common::backend_lock::with_cuda_backend;
use numr::dtype::DType;
use numr::ops::{GemmActivation, GemmEpilogueOps, MatmulOps, RandomOps};

/// Repeats per shape. A race is timing-dependent and need not fire on every
/// launch. Lowering this raises the odds a real race goes undetected.
const REPEATS: usize = 50;

/// Seed for every input tensor. Every run uses this same seed, so an output
/// difference can only come from the kernel.
const SEED: u64 = 0x5EED_1234_ABCD_0001;

fn is_bf16_supported() -> bool {
    common::is_dtype_supported("cuda", DType::BF16)
}

/// Runs `f` `REPEATS` times and asserts every run's bits match the first run.
///
/// `f` rebuilds its inputs each run. A reused buffer would only show the
/// kernel is idempotent on memory it already wrote.
fn assert_deterministic_bytes<T, F>(shape: &str, label: &str, f: F)
where
    T: BitsOf,
    F: Fn(usize) -> Vec<T>,
{
    let mut reference: Option<Vec<u16>> = None;
    for run in 0..REPEATS {
        let out = f(run);
        let bits: Vec<u16> = out.iter().map(|v| v.bits()).collect();
        match &reference {
            None => reference = Some(bits),
            Some(reference_bits) => {
                assert_eq!(
                    bits.len(),
                    reference_bits.len(),
                    "wmma determinism [{shape}] {label}: run {run} length mismatch: \
                     {} vs reference {}",
                    bits.len(),
                    reference_bits.len()
                );
                if let Some(idx) = bits
                    .iter()
                    .zip(reference_bits.iter())
                    .position(|(a, b)| a != b)
                {
                    panic!(
                        "wmma determinism [{shape}] {label}: run {run} diverged from the \
                         reference (run 0) at element {idx}: {:#06x} vs reference {:#06x}",
                        bits[idx], reference_bits[idx]
                    );
                }
            }
        }
    }
}

/// Bit pattern of one output element, so the comparison is exact rather than
/// tolerant.
trait BitsOf {
    fn bits(&self) -> u16;
}

impl BitsOf for half::f16 {
    fn bits(&self) -> u16 {
        self.to_bits()
    }
}

impl BitsOf for half::bf16 {
    fn bits(&self) -> u16 {
        self.to_bits()
    }
}

// ---------------------------------------------------------------------------
// Plain matmul
// ---------------------------------------------------------------------------

#[test]
fn matmul_aligned_large_tile_batched_is_deterministic() {
    // 256x256 exact-divides the 128x128 tile; batch=128 pushes the grid
    // (batch * 2 * 2 = 512 blocks) past the wave threshold on any plausible
    // device, so this shape encourages, though it cannot force, the large
    // tile. Determinism must hold under whichever tile the device picks.
    let (batch, m, k, n) = (128usize, 256usize, 64usize, 256usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("aligned_large_tile_batched f16", "matmul", |_run| {
            let a = client
                .randn_seeded(&[batch, m, k], DType::F16, SEED)
                .unwrap();
            let b = client
                .randn_seeded(&[batch, k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            client.matmul(&a, &b).unwrap().to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes("aligned_large_tile_batched bf16", "matmul", |_run| {
                let a = client
                    .randn_seeded(&[batch, m, k], DType::BF16, SEED)
                    .unwrap();
                let b = client
                    .randn_seeded(&[batch, k, n], DType::BF16, SEED ^ 0xBF16)
                    .unwrap();
                client.matmul(&a, &b).unwrap().to_vec::<half::bf16>()
            });
        }
    });
}

#[test]
fn matmul_aligned_small_tile_is_deterministic() {
    // m=n=64 < 128 in both dims: guaranteed small tile on every device.
    let (m, k, n) = (64usize, 64usize, 64usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("aligned_small_tile f16", "matmul", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            client.matmul(&a, &b).unwrap().to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes("aligned_small_tile bf16", "matmul", |_run| {
                let a = client.randn_seeded(&[m, k], DType::BF16, SEED).unwrap();
                let b = client
                    .randn_seeded(&[k, n], DType::BF16, SEED ^ 0xBF16)
                    .unwrap();
                client.matmul(&a, &b).unwrap().to_vec::<half::bf16>()
            });
        }
    });
}

#[test]
fn matmul_unaligned_padded_is_deterministic() {
    // Not a multiple of 16 in any dimension: forces the pad-to-16-multiples,
    // run WMMA, narrow-and-copy-back path in src/ops/cuda/matmul.rs.
    let (m, k, n) = (100usize, 100usize, 100usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("unaligned_padded f16", "matmul", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            client.matmul(&a, &b).unwrap().to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes("unaligned_padded bf16", "matmul", |_run| {
                let a = client.randn_seeded(&[m, k], DType::BF16, SEED).unwrap();
                let b = client
                    .randn_seeded(&[k, n], DType::BF16, SEED ^ 0xBF16)
                    .unwrap();
                client.matmul(&a, &b).unwrap().to_vec::<half::bf16>()
            });
        }
    });
}

#[test]
fn matmul_partial_tile_144_is_deterministic() {
    // 16-aligned (dispatches straight to WMMA, no padding) but not a multiple
    // of 128: a ragged block edge against the large tile, ragged in both
    // directions against the small tile.
    let (m, k, n) = (144usize, 144usize, 144usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("partial_tile_144 f16", "matmul", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            client.matmul(&a, &b).unwrap().to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes("partial_tile_144 bf16", "matmul", |_run| {
                let a = client.randn_seeded(&[m, k], DType::BF16, SEED).unwrap();
                let b = client
                    .randn_seeded(&[k, n], DType::BF16, SEED ^ 0xBF16)
                    .unwrap();
                client.matmul(&a, &b).unwrap().to_vec::<half::bf16>()
            });
        }
    });
}

#[test]
fn matmul_large_k_is_deterministic() {
    // Small M/N with large K: the K-loop cycles the stage ring far more times
    // per launch than any other shape here.
    let (m, k, n) = (32usize, 4096usize, 32usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("large_k f16", "matmul", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            client.matmul(&a, &b).unwrap().to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes("large_k bf16", "matmul", |_run| {
                let a = client.randn_seeded(&[m, k], DType::BF16, SEED).unwrap();
                let b = client
                    .randn_seeded(&[k, n], DType::BF16, SEED ^ 0xBF16)
                    .unwrap();
                client.matmul(&a, &b).unwrap().to_vec::<half::bf16>()
            });
        }
    });
}

#[test]
fn matmul_small_n_is_deterministic() {
    // n=16: N is under one tile in either tile choice, so every block along
    // N discards nothing but also reuses nothing across N-blocks.
    let (m, k, n) = (64usize, 64usize, 16usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("small_n f16", "matmul", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            client.matmul(&a, &b).unwrap().to_vec::<half::f16>()
        });
    });
}

#[test]
fn matmul_batched_is_deterministic() {
    let (batch, m, k, n) = (4usize, 64usize, 32usize, 48usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("batched f16", "matmul", |_run| {
            let a = client
                .randn_seeded(&[batch, m, k], DType::F16, SEED)
                .unwrap();
            let b = client
                .randn_seeded(&[batch, k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            client.matmul(&a, &b).unwrap().to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes("batched bf16", "matmul", |_run| {
                let a = client
                    .randn_seeded(&[batch, m, k], DType::BF16, SEED)
                    .unwrap();
                let b = client
                    .randn_seeded(&[batch, k, n], DType::BF16, SEED ^ 0xBF16)
                    .unwrap();
                client.matmul(&a, &b).unwrap().to_vec::<half::bf16>()
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Fused epilogues: matmul_bias, matmul_bias_activation, matmul_bias_residual
// ---------------------------------------------------------------------------
//
// The epilogue writes into the same shared memory the staging buffers use.
// The plain-matmul cases above never exercise that handover.

#[test]
fn matmul_bias_aligned_is_deterministic() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("bias_aligned f16", "matmul_bias", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            let bias = client
                .randn_seeded(&[n], DType::F16, SEED ^ 0xB1A5)
                .unwrap();
            client
                .matmul_bias(&a, &b, &bias)
                .unwrap()
                .to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes("bias_aligned bf16", "matmul_bias", |_run| {
                let a = client.randn_seeded(&[m, k], DType::BF16, SEED).unwrap();
                let b = client
                    .randn_seeded(&[k, n], DType::BF16, SEED ^ 0xBF16)
                    .unwrap();
                let bias = client
                    .randn_seeded(&[n], DType::BF16, SEED ^ 0xB1A5)
                    .unwrap();
                client
                    .matmul_bias(&a, &b, &bias)
                    .unwrap()
                    .to_vec::<half::bf16>()
            });
        }
    });
}

#[test]
fn matmul_bias_large_k_is_deterministic() {
    let (m, k, n) = (32usize, 4096usize, 32usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("bias_large_k f16", "matmul_bias", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            let bias = client
                .randn_seeded(&[n], DType::F16, SEED ^ 0xB1A5)
                .unwrap();
            client
                .matmul_bias(&a, &b, &bias)
                .unwrap()
                .to_vec::<half::f16>()
        });
    });
}

#[test]
fn matmul_bias_unaligned_padded_is_deterministic() {
    let (m, k, n) = (130usize, 70usize, 50usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("bias_unaligned_padded f16", "matmul_bias", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            let bias = client
                .randn_seeded(&[n], DType::F16, SEED ^ 0xB1A5)
                .unwrap();
            client
                .matmul_bias(&a, &b, &bias)
                .unwrap()
                .to_vec::<half::f16>()
        });
    });
}

#[test]
fn gemm_bias_activation_aligned_is_deterministic() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes(
            "bias_act_aligned f16",
            "gemm_bias_activation(GELU)",
            |_run| {
                let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
                let b = client
                    .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                    .unwrap();
                let bias = client
                    .randn_seeded(&[n], DType::F16, SEED ^ 0xB1A5)
                    .unwrap();
                client
                    .matmul_bias_activation(&a, &b, &bias, GemmActivation::GELU)
                    .unwrap()
                    .to_vec::<half::f16>()
            },
        );
        if is_bf16_supported() {
            assert_deterministic_bytes(
                "bias_act_aligned bf16",
                "gemm_bias_activation(SiLU)",
                |_run| {
                    let a = client.randn_seeded(&[m, k], DType::BF16, SEED).unwrap();
                    let b = client
                        .randn_seeded(&[k, n], DType::BF16, SEED ^ 0xBF16)
                        .unwrap();
                    let bias = client
                        .randn_seeded(&[n], DType::BF16, SEED ^ 0xB1A5)
                        .unwrap();
                    client
                        .matmul_bias_activation(&a, &b, &bias, GemmActivation::SiLU)
                        .unwrap()
                        .to_vec::<half::bf16>()
                },
            );
        }
    });
}

#[test]
fn gemm_bias_residual_aligned_is_deterministic() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes("bias_residual_aligned f16", "gemm_bias_residual", |_run| {
            let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
            let b = client
                .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                .unwrap();
            let bias = client
                .randn_seeded(&[n], DType::F16, SEED ^ 0xB1A5)
                .unwrap();
            let residual = client
                .randn_seeded(&[m, n], DType::F16, SEED ^ 0x9E5D)
                .unwrap();
            client
                .matmul_bias_residual(&a, &b, &bias, &residual)
                .unwrap()
                .to_vec::<half::f16>()
        });
        if is_bf16_supported() {
            assert_deterministic_bytes(
                "bias_residual_aligned bf16",
                "gemm_bias_residual",
                |_run| {
                    let a = client.randn_seeded(&[m, k], DType::BF16, SEED).unwrap();
                    let b = client
                        .randn_seeded(&[k, n], DType::BF16, SEED ^ 0xBF16)
                        .unwrap();
                    let bias = client
                        .randn_seeded(&[n], DType::BF16, SEED ^ 0xB1A5)
                        .unwrap();
                    let residual = client
                        .randn_seeded(&[m, n], DType::BF16, SEED ^ 0x9E5D)
                        .unwrap();
                    client
                        .matmul_bias_residual(&a, &b, &bias, &residual)
                        .unwrap()
                        .to_vec::<half::bf16>()
                },
            );
        }
    });
}

#[test]
fn gemm_bias_residual_unaligned_padded_is_deterministic() {
    // Residual is [M,N]-shaped and padded in 2 dims (unlike bias's 1-D pad),
    // so this exercises the residual-specific padding branch in
    // src/ops/cuda/gemm_epilogue.rs under repetition.
    let (m, k, n) = (130usize, 70usize, 50usize);
    with_cuda_backend(|client, _device| {
        assert_deterministic_bytes(
            "bias_residual_unaligned_padded f16",
            "gemm_bias_residual",
            |_run| {
                let a = client.randn_seeded(&[m, k], DType::F16, SEED).unwrap();
                let b = client
                    .randn_seeded(&[k, n], DType::F16, SEED ^ 0xF16)
                    .unwrap();
                let bias = client
                    .randn_seeded(&[n], DType::F16, SEED ^ 0xB1A5)
                    .unwrap();
                let residual = client
                    .randn_seeded(&[m, n], DType::F16, SEED ^ 0x9E5D)
                    .unwrap();
                client
                    .matmul_bias_residual(&a, &b, &bias, &residual)
                    .unwrap()
                    .to_vec::<half::f16>()
            },
        );
    });
}
