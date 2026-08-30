# Changelog

All notable changes to numr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
numr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One entry, covering everything up to and including 0.7.0. Tags `v0.1.0` through
`v0.6.1` predate this file and are folded in here rather than reconstructed, so
nothing below is stated as a delta against an earlier version.

---

## [0.7.0] — 2026-08-30

Tensors, linear algebra, FFT, and autograd behind one API on CPU, CUDA, and WebGPU.
Every kernel is written in-house — no cuBLAS, cuSOLVER, or MKL.

### Added

- **Tensors** — `Tensor<R>` generic over the backend. Broadcasting, zero-copy views (`reshape`, `transpose`, `slice`, `permute`), fallible constructors that never panic on allocation failure.
- **Backends** — CPU (`#[target_feature]` AVX-512 / AVX2+FMA / NEON, optional Rayon), CUDA (graph capture, arena allocator, kernel cache), WebGPU (WGSL per dtype). CPU and CUDA cover every dtype; WebGPU is F32 / I32 / U32 / Bool.
- **DTypes** — F64, F32, F16, BF16, FP8E4M3, FP8E5M2, I64, I32, I16, I8, U64, U32, U16, U8, Bool. Narrow dtypes accumulate in a wider type. WebGPU is F32 / I32 / U32 / Bool.
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

### Semantics

Behaviour a caller must know. These are contracts, not defaults.

- **Output dtype is a function of input dtypes, never input values.** `pow_scalar` on an integer tensor returns F64 unless the exponent is a whole non-negative number. `matmul_output_dtype` is public so a caller can size a bias against it.
- **Integer accumulators saturate** at the dtype's bound: `sum`, `prod`, `mean`, `cumsum`, `cumprod`, `matmul`, `scatter_reduce`.
- **Integer element-wise ops wrap**: `add`, `sub`, `mul`, and the fused forms.
- **Integer division by zero returns 0**, and `INT_MIN / -1` returns `INT_MIN`. Neither panics.
- **I8 matmul widens to I32.** `matmul_bias` on I8 takes an I32 bias and returns I32. The GEMM epilogue ops (`matmul_bias_activation`, `matmul_bias_residual`, and the backward form) reject I8 rather than read a wider bias buffer as I8.

### Notes

- `tests/backend_parity/` checks CUDA and WebGPU against CPU for every operation, at dtype-appropriate tolerances.
- WebGPU stays 32-bit by design. WGSL has no native F64.
- ROCm is planned.
