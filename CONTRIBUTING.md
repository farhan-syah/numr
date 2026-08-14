# Contributing

Thanks for contributing to [numr](https://crates.io/crates/numr). This guide covers the architecture conventions and quality gates the project expects.

## Prerequisites

- Rust 1.89 or newer (edition 2024).
- A clean working tree before opening a pull request.
- Optional: a CUDA 12.x toolchain and/or a WebGPU-capable device if you want to run the GPU test paths locally.

## What to contribute

The most valuable contributions are usually **missing primitives** — a tensor operation, dtype, or linear-algebra routine that NumPy has and numr does not — and **backend coverage**, filling in a CUDA or WebGPU implementation for an operation that only runs on CPU today. Bug fixes and numerical-accuracy improvements are equally welcome.

Before writing a non-trivial operation, **open an issue first** describing what you want to add and which crate it belongs in (see below). This avoids duplicated effort and lets us agree on placement and API up front. Small, self-contained fixes can go straight to a pull request.

## Which crate: numr, solvr, or boostr

numr is the bottom layer of a stack, and a contribution only belongs here if it fits this layer. Place new work by what it _is_, not where it's convenient:

- **numr** (this crate) — foundational primitives that everything else builds on: tensor ops, dtypes, the `Runtime`/backend abstraction (and **new backends** themselves), FFT, core linear algebra (matmul, LU/QR/SVD/eigen, `solve`), special functions, and basic descriptive statistics.
- **[solvr](https://github.com/ml-rust/solvr)** — complete scientific/solving algorithms composed from numr primitives: optimization, ODE/DAE/BVP/PDE, interpolation, advanced statistics, signal processing, spatial, clustering.
- **[boostr](https://github.com/ml-rust/boostr)** — AI/ML-specific building blocks: attention, positional encodings, mixture-of-experts, quantization, neural-network layers, and training/inference machinery.

Quick test:

- Is it a building block reused across domains, or does it add/touch a hardware backend? → **numr**.
- Is it a domain solver a scientist or engineer would reach for? → **solvr**.
- Does it only make sense for neural networks / LLMs? → **boostr**.

When in doubt, propose it in an issue and we'll help place it.

## Architecture

numr's central promise is that the same API produces the same numbers on every backend, while each backend is free to reach those numbers its own way.

### Primitive vs composite operations

How you implement an operation depends on which kind it is.

**Primitive** operations are atomic — the kernel _is_ the algorithm (`add`, `exp`, `sum`, `matmul`). There is no shared implementation to factor out: each backend gets its own kernel, and they are expected to differ.

```
ops/traits/unary.rs   trait definition
ops/cpu/unary.rs      impl → calls cpu/kernels/unary.rs   (SIMD)
ops/cuda/unary.rs     impl → launches cuda/kernels/unary.cu (PTX)
ops/wgpu/unary.rs     impl → dispatches wgpu/shaders/unary.wgsl (WGSL)
```

**Composite** operations are built from primitives (`softmax`, `layernorm`, `gelu`). Write these once in `ops/impl_generic/` using primitive ops, and have every backend delegate to it. This is the **default** — reach for it first.

```rust
// ops/impl_generic/activation.rs
pub fn softmax_impl<R, C>(client: &C, input: &Tensor<R>, dim: i64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    let max = client.max(input, &[dim], true)?;
    let shifted = client.sub(input, &max)?;
    let exp = client.exp(&shifted)?;
    let sum = client.sum(&exp, &[dim], true)?;
    client.div(&exp, &sum)
}
```

Replace `impl_generic` with a **fused kernel** only when profiling shows the decomposed version is too slow — typically on a hot path where the generic version makes several passes over memory. A fused kernel does not have to match `impl_generic` bit-for-bit; it has to pass the backend parity tests.

### Backend parity tests are mandatory

Because backends may use different algorithms, correctness is enforced by tests rather than by shared code. **Every operation needs a test in `tests/backend_parity/`** that treats CPU as the reference and compares every other backend against it.

```rust
let cpu_result = cpu_client.softmax(&input_cpu, dim)?;
let cuda_result = cuda_client.softmax(&input_cuda, dim)?;
assert_tensor_allclose(&cuda_result, &cpu_result, dtype, "softmax cuda vs cpu");
```

Cover the edge cases that actually diverge: empty tensors, single-element and size-1 dimensions, very large and very small magnitudes, NaN, and infinities. A parity test that only checks well-behaved values will pass while the operation is wrong.

Note that a test asserting only on _values_ will not catch a divergence in output _shape_ (a `[1]` where CPU returns a scalar `[]`, say). Assert shapes explicitly when the operation is rank-reducing.

### File organization

- `mod.rs` contains **only** `pub mod` and `pub use` — no traits, types, or logic. If a `mod.rs` is more than ~30 lines, something is in the wrong place.
- One operation = one file, with the same file name under `traits/` and each backend directory.
- Backends live in directories (`cpu/`, `cuda/`, `wgpu/`), never as a single `cpu.rs`. Kernels go in `cpu/kernels/`, `cuda/kernels/`, `wgpu/shaders/`.
- Adding an operation should mean adding files, not growing existing ones.

Soft/hard line limits: trait files 100/200, backend impls 200/400, CPU kernels and `.cu` files 300/500, WGSL shaders 150/300, integration tests 400/600. Split at the soft limit if there is a logical boundary.

### No GPU↔CPU transfers

Host/device transfers cost far more than the computation. Inside GPU code paths, never call `tensor.to_vec()`, never loop over device data on the host, and never hide a transfer inside a helper. If a GPU kernel doesn't exist yet, return `Err(Error::Unsupported { .. })` rather than falling back to the CPU.

### No vendor libraries

numr must build and run without cuBLAS, cuDNN, MKL, or any other vendor library. Every kernel is native — PTX for CUDA, WGSL for WebGPU, SIMD intrinsics for CPU. This is what keeps the library portable across NVIDIA, AMD, Intel, and Apple hardware, so please don't add a vendor dependency to close a performance gap.

### CPU SIMD: use `#[target_feature]`

Do not put `is_x86_feature_detected!()` checks inside a hot loop. Detect once, then dispatch to separate functions, each annotated for its ISA, so the compiler can optimize the whole body:

```rust
pub unsafe fn my_kernel(data: *const f32, len: usize, level: SimdLevel) -> f32 {
    match level {
        SimdLevel::Avx512 => my_kernel_avx512(data, len),
        SimdLevel::Avx2Fma => my_kernel_avx2(data, len),
        _ => my_kernel_scalar(data, len),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn my_kernel_avx512(data: *const f32, len: usize) -> f32 { /* ... */ }
```

Use two or more accumulators in FMA loops to hide the 4–5 cycle latency, and always provide a scalar fallback.

### Dtype coverage

Implement every dtype the backend supports, not just `F32`. Use the `dispatch_dtype!` macro for runtime dispatch. WebGPU is intentionally 32-bit only (`F32`, `I32`, `U32`) — return a clear `UnsupportedDType` error there rather than silently degrading precision.

## Building with backends

```bash
cargo build --release                    # CPU (default)
cargo build --release --features cuda    # CUDA (requires a CUDA 12.x toolchain)
cargo build --release --features wgpu    # WebGPU
cargo build --release --features sparse  # sparse tensors
```

CUDA 13.x is not yet supported, as `cudarc` does not support it.

## Testing

- Put unit tests in the same file as the code under test (`#[cfg(test)] mod tests`); they can reach private functions.
- Put public-API tests in `tests/`. Subdirectories need a `mod.rs` so they are not compiled as separate test binaries.
- Test numerical correctness against a reference value, not just that the call returns `Ok`.
- A backend-specific test should skip gracefully when no device is present.

```bash
cargo test --release
cargo test --release --features cuda
cargo test --release --features wgpu
```

If you add a test module under `tests/backend_parity/`, remember to register it in `tests/backend_parity/mod.rs` — an unregistered module is silently never compiled, and the suite will still report green.

## Local quality checks

Run these before submitting. Clippy runs with `-D warnings` in CI, so treat a warning as a failure locally too.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```

If you touch a GPU backend, run clippy and the tests with `--features cuda` and `--features wgpu` as well.

## Pull request guidelines

- Keep pull requests focused and scoped.
- Preserve the module structure described above; add files rather than growing existing ones.
- Include backend parity tests for any new or changed operation.
- Avoid `.unwrap()` and `.expect()` in library code — return a typed error with enough context to act on (operation, shapes, dtype).
- Update docs when public APIs or features change.

## Commit messages

Use Conventional Commits with a clear, imperative summary, for example:

```
feat(linalg): add banded LU factorization for CPU and CUDA
fix(sort): apply a NaN-aware total order across all backends
```
