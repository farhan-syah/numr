// U32 clamp operation.
//
// The bounds arrive already converted to the element type: `ClampParams::new`
// re-encodes them per dtype, the same trick `ScalarParams` uses, because this
// shader and clamp_f32 read the same two 4-byte fields under different types.
//
// CPU clamps in f64 and converts once (`clamp_scalar` in
// runtime/cpu/kernels/unary/mod.rs). For an integer element that is the same
// answer as clamping against the truncated bounds: no integer sits strictly
// between a bound and its truncation, so the branch a value takes may differ
// while the value it produces does not.

const WORKGROUP_SIZE: u32 = 256u;

struct ClampParams {
    numel: u32,
    min_val: u32,
    max_val: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> clamp_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> clamp_out: array<u32>;
@group(0) @binding(2) var<uniform> clamp_params: ClampParams;

@compute @workgroup_size(256)
fn clamp_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < clamp_params.numel) {
        clamp_out[idx] = min(max(clamp_a[idx], clamp_params.min_val), clamp_params.max_val);
    }
}
