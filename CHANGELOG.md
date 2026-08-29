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

### Changed

These change results that earlier code could observe. Read them before upgrading.

- `pow_scalar` on an integer tensor returns F64 when the exponent is not a whole non-negative number. A caller reading the result as an integer now fails at runtime. An op's output dtype depends on its input dtypes, never on a value.
- Integer accumulators saturate at the dtype's bound. This covers `sum`, `prod`, `mean`, `cumsum`, `cumprod`, `matmul`, and `scatter_reduce`. They wrapped before.
- Integer element-wise ops wrap. `add`, `sub`, `mul`, and the fused forms panicked on overflow in debug builds.
- Integer division by zero returns 0, and `INT_MIN / -1` returns `INT_MIN`. Both panicked before.
- `matmul_bias` on I8 takes an I32 bias and returns I32, matching `matmul` on I8. It took an I8 bias and saturated the result into I8 before.
- The GEMM epilogue ops (`matmul_bias_activation`, `matmul_bias_residual`, and the backward form) reject I8. They shared `matmul_bias`'s validator, so an I8 call would have read a wider bias buffer as I8.
- `matmul_output_dtype` is public. A caller needs it to size a bias, because I8 widens.

### Fixed

- Multi-dimension `var` computed the variance of the variances. For `[[1, 2], [3, 4]]` it returned 0 where the answer is 1.25. `std` inherited it.
- Multi-dimension integer `mean` returned a different value above 1 MiB or on a non-contiguous input, because only the small contiguous path divided once.
- Integer `matmul_bias` saturated the product and then wrapped the bias into the element type.
- CUDA `max` and `min` on F32 and F64 disagreed with CPU on NaN.
- CUDA `cast` rejected U32 and the narrow integer widths, so no U32 tensor could be built on a GPU.
- CUDA had no integer reduction kernels and no U32 kernels for element-wise, compare, scalar, indexing, sort, or matmul ops.
- WebGPU had no integer matmul, semiring, `topk`, `searchsorted`, `clamp`, `linspace`, or fused element-wise kernels.
- WebGPU integer scatter shaders bound read-only storage against a read-write layout, which failed pipeline creation and lost the device.
- WebGPU integer reductions narrowed to the element type between passes, and `scatter_reduce` wrapped on an overflowing sum.
- WebGPU converted a float to an unsigned integer without a range guard, which WGSL leaves implementation-defined.

### Notes

- `tests/backend_parity/` checks CUDA and WebGPU against CPU for every operation, at dtype-appropriate tolerances.
- WebGPU stays 32-bit by design. WGSL has no native F64.
- ROCm is planned.
