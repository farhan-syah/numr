// Arange operation for f32

const WORKGROUP_SIZE: u32 = 256u;

struct ArangeParams {
    numel: u32,
    start: f32,
    step: f32,
}

@group(0) @binding(0) var<storage, read_write> arange_out: array<f32>;
@group(0) @binding(1) var<uniform> arange_params: ArangeParams;

@compute @workgroup_size(256)
fn arange_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arange_params.numel) {
        let value = arange_params.start + arange_params.step * f32(idx);
        arange_out[idx] = f32(value);
    }
}
