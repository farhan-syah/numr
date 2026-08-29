// I32 binary operations

struct BinaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> binary_a: array<i32>;
@group(0) @binding(1) var<storage, read_write> binary_b: array<i32>;
@group(0) @binding(2) var<storage, read_write> binary_out: array<i32>;
@group(0) @binding(3) var<uniform> binary_params: BinaryParams;

@compute @workgroup_size(256)
fn add_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] + binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn sub_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] - binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn mul_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] * binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn div_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] / binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn max_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = max(binary_a[idx], binary_b[idx]);
    }
}

@compute @workgroup_size(256)
fn min_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = min(binary_a[idx], binary_b[idx]);
    }
}

@compute @workgroup_size(256)
fn pow_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = numr_ipow_i32(binary_a[idx], binary_b[idx]);
    }
}
