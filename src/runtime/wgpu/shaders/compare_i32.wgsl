// I32 comparison operations (input I32, output I32: 1=true, 0=false)

const WORKGROUP_SIZE: u32 = 256u;

struct CompareParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> compare_a: array<i32>;
@group(0) @binding(1) var<storage, read_write> compare_b: array<i32>;
@group(0) @binding(2) var<storage, read_write> compare_out: array<i32>;
@group(0) @binding(3) var<uniform> compare_params: CompareParams;

@compute @workgroup_size(256)
fn eq_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0, 1, compare_a[idx] == compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn ne_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0, 1, compare_a[idx] != compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn lt_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0, 1, compare_a[idx] < compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn le_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0, 1, compare_a[idx] <= compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn gt_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0, 1, compare_a[idx] > compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn ge_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0, 1, compare_a[idx] >= compare_b[idx]);
    }
}
