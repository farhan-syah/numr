// I32 unary operations

const WORKGROUP_SIZE: u32 = 256u;

struct UnaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> unary_a: array<i32>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<i32>;
@group(0) @binding(2) var<uniform> unary_params: UnaryParams;

@compute @workgroup_size(256)
fn neg_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = -unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn abs_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = abs(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn sign_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let v = unary_a[idx];
        if (v > 0) {
            unary_out[idx] = 1;
        } else if (v < 0) {
            unary_out[idx] = -1;
        } else {
            unary_out[idx] = 0;
        }
    }
}
