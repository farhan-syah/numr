# Changelog

All notable changes to numr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
numr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file starts at 0.7.0. Releases before it are not covered.

---

## [Unreleased] — 0.7.0

Tensors, linear algebra, FFT, and autograd behind one API on CPU, CUDA, and WebGPU.
Every kernel is written in-house — no cuBLAS, cuSOLVER, or MKL.

### Added

- **Tensors** — `Tensor<R>` generic over the backend. Broadcasting, zero-copy views (`reshape`, `transpose`, `slice`, `permute`), fallible constructors that never panic on allocation failure.
- **Backends** — CPU (`#[target_feature]` AVX-512 / AVX2+FMA / NEON, optional Rayon), CUDA (graph capture, arena allocator, kernel cache), WebGPU (WGSL per dtype). CPU and CUDA cover every dtype; WebGPU is F32 / I32 / U32 / Bool.
- **DTypes** — F64, F32, F16, BF16, FP8E4M3, FP8E5M2, I64, I32, U32, Bool. Narrow dtypes accumulate in a wider type.
- **Element-wise and reductions** — unary, binary, scalar, compare, logical, conditional. Reductions over any axis set, cumulative ops, sorting with a NaN-aware total order, gather/scatter indexing.
- **Matmul** — batched and broadcast, tiled CPU kernel with a transposed-B path, fused GEMM epilogue, FP8, min-plus/max-plus semirings, einsum, 2:4 structured sparsity.
- **Linear algebra** — LU, QR, SVD, Cholesky, Schur, QZ, polar, eigen. Solvers, `lstsq`, inverse, `slogdet`, `cond`, `matrix_rank`. Matrix functions (`expm`, `logm`, `sqrtm`, `signm`, `funm`). Tucker, HOSVD, CP, tensor-train.
- **Iterative solvers** — CG, BiCGSTAB, CGS, GMRES, LGMRES, MINRES, QMR, Jacobi, SOR, sparse eigensolvers, `svds`, AMG.
- **FFT** — 1D / 2D / ND, forward and inverse. Bluestein covers arbitrary sizes on all three backends.
- **Convolution** — `conv1d`, `conv2d`, `depthwise_conv2d`, `conv_transpose1d` with native kernels and autograd on each.
- **Autograd** — reverse mode via `Var<R>` and `GradFn`, forward mode via `DualTensor`, gradient checkpointing. `backward` takes a needed-gradient mask so ops skip gradients the driver discards.
- **Random** — uniform, normal, seeded normal, advanced and multivariate distributions, Sobol sequences, Philox on GPU. Seeds reproduce per backend, not across backends.
- **Statistics and special functions** — quantiles, histograms, correlation, distance metrics, polynomials. Gamma, Bessel, Airy, Fresnel, elliptic, hypergeometric, orthogonal families.
- **Sparse** (`sparse`) — CSR, CSC, COO with conversions, element-wise ops, SpMM, SpMV, and sparse LU / QR with COLAMD ordering.
- **Distributed** (`distributed`, `nccl`) — a `Communicator` trait over NCCL, nexar, hierarchical, and no-op backends, plus process groups.

### Notes

- `tests/backend_parity/` checks CUDA and WebGPU against CPU for every operation, at dtype-appropriate tolerances.
- WebGPU stays 32-bit by design. WGSL has no native F64.
- ROCm is planned.
