//! Params structs for the built-in RNG kernels, and the host-side seed source.
//!
//! `seed` and `seed_hi` together carry a full u64 seed, since WGSL uniforms
//! cannot hold a 64-bit integer.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) seed_hi: u32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandnParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) seed_hi: u32,
    pub(crate) _pad: u32,
}

/// Randint params for signed integer types (`I32`)
/// The `low` field is i32 to properly handle negative bounds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandintParamsI32 {
    pub(crate) numel: u32,
    pub(crate) low: i32, // Signed low bound
    pub(crate) range: u32,
    pub(crate) seed: u32,
}

/// Randint params for unsigned integer types (`U32`)
/// The `low` field is u32 for unsigned bounds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandintParamsU32 {
    pub(crate) numel: u32,
    pub(crate) low: u32, // Unsigned low bound
    pub(crate) range: u32,
    pub(crate) seed: u32,
}

/// Generate a random seed for WebGPU RNG operations.
/// Combines system time with an atomic counter to ensure uniqueness across calls.
pub(crate) fn generate_wgpu_seed() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);

    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(12345u32);
    time_seed.wrapping_add(counter)
}
