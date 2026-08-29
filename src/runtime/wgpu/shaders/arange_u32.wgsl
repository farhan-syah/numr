// Arange operation for u32.
//
// Concatenated after int_saturate.wgsl and int_from_float.wgsl. The store goes
// through numr_f32_to_u32_sat rather than a bare `u32(value)`: WGSL leaves the
// conversion of an out-of-range float implementation-defined, and a negative
// start or step reaches exactly that case, while CPU's `Element::from_f64`
// clamps to zero.

const WORKGROUP_SIZE: u32 = 256u;

struct ArangeParams {
    numel: u32,
    start: f32,
    step: f32,
}

@group(0) @binding(0) var<storage, read_write> arange_out: array<u32>;
@group(0) @binding(1) var<uniform> arange_params: ArangeParams;

@compute @workgroup_size(256)
fn arange_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arange_params.numel) {
        let value = arange_params.start + arange_params.step * f32(idx);
        arange_out[idx] = numr_f32_to_u32_sat(value);
    }
}
