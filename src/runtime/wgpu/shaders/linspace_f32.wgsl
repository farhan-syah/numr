// Linspace operation for f32

const WORKGROUP_SIZE: u32 = 256u;

struct LinspaceParams {
    steps: u32,
    start: f32,
    stop: f32,
}

@group(0) @binding(0) var<storage, read_write> linspace_out: array<f32>;
@group(0) @binding(1) var<uniform> linspace_params: LinspaceParams;

@compute @workgroup_size(256)
fn linspace_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < linspace_params.steps) {
        let t_val = f32(idx) / f32(linspace_params.steps - 1u);
        let value = linspace_params.start + (linspace_params.stop - linspace_params.start) * t_val;
        linspace_out[idx] = f32(value);
    }
}
