//! Params structs for the kernels that fill a fresh tensor: arange, linspace
//! and eye.
//!
//! `LinspaceIntParams` carries a 64-bit exact path split into u32 halves,
//! because WGSL has no 64-bit integer type.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ArangeParams {
    pub(crate) numel: u32,
    pub(crate) start: f32,
    pub(crate) step: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LinspaceParams {
    pub(crate) steps: u32,
    pub(crate) start: f32,
    pub(crate) stop: f32,
}

/// Parameters for an integer `linspace`.
///
/// `exact` selects the 64-bit integer evaluation, which needs `base`
/// (`start * divisor`) and `delta` as signed 64-bit values split into u32
/// halves - WGSL has no 64-bit integer type. `start_f32`/`stop_f32` carry the
/// fallback for fractional bounds. See linspace_i32.wgsl for why both paths
/// exist.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LinspaceIntParams {
    pub(crate) steps: u32,
    pub(crate) exact: u32,
    pub(crate) divisor: u32,
    pub(crate) _pad0: u32,
    pub(crate) base_lo: u32,
    pub(crate) base_hi: u32,
    pub(crate) delta_lo: u32,
    pub(crate) delta_hi: u32,
    pub(crate) start_f32: f32,
    pub(crate) stop_f32: f32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}

impl LinspaceIntParams {
    /// Build params for `steps` integer samples between `start` and `stop`.
    ///
    /// The exact path is taken only when both bounds are whole numbers and every
    /// 64-bit intermediate is provably in range; anything else keeps the f32
    /// evaluation, whose truncation is then far from a boundary anyway.
    pub(crate) fn new(steps: usize, start: f64, stop: f64) -> Self {
        let divisor = (steps - 1) as u64;
        let delta = stop - start;

        // `i64::MAX / 4` leaves room for `base`, `delta * idx`, and their sum
        // without any of the three overflowing.
        let bound = (i64::MAX / 4) as f64;
        let exact = start.fract() == 0.0
            && stop.fract() == 0.0
            && start.abs() < bound
            && delta.abs() * (divisor as f64) < bound
            && start.abs() * (divisor as f64) < bound;

        let (base, delta_i) = if exact {
            ((start as i64) * divisor as i64, delta as i64)
        } else {
            (0, 0)
        };

        Self {
            steps: steps as u32,
            exact: u32::from(exact),
            divisor: divisor as u32,
            _pad0: 0,
            base_lo: base as u32,
            base_hi: (base >> 32) as u32,
            delta_lo: delta_i as u32,
            delta_hi: (delta_i >> 32) as u32,
            start_f32: start as f32,
            stop_f32: stop as f32,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct EyeParams {
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) numel: u32,
}
