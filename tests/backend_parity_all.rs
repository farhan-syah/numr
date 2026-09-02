// Most helpers here build a CUDA reference or feed a CUDA-only kernel, so
// without that feature they are legitimately unused. The condition covers the
// wgpu-only build too, not just the no-backend one — otherwise `--features
// wgpu` fails the lint on CUDA-scoped helpers.
#[cfg_attr(
    not(feature = "cuda"),
    allow(unused_imports, unused_variables, dead_code)
)]
mod backend_parity;
mod common;
