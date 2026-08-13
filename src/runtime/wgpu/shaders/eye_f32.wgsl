// Eye (identity matrix) operation for f32

const WORKGROUP_SIZE: u32 = 256u;

struct EyeParams {
    n: u32,      // rows
    m: u32,      // cols
    numel: u32,  // n * m
}

@group(0) @binding(0) var<storage, read_write> eye_out: array<f32>;
@group(0) @binding(1) var<uniform> eye_params: EyeParams;

@compute @workgroup_size(256)
fn eye_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < eye_params.numel) {
        let row = idx / eye_params.m;
        let col = idx % eye_params.m;
        if (row == col) {
            eye_out[idx] = f32(1.0);
        } else {
            eye_out[idx] = f32(0.0);
        }
    }
}
