// U32 comparison operations (input U32, output U32: 1u=true, 0u=false)

const WORKGROUP_SIZE: u32 = 256u;

struct CompareParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> compare_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> compare_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> compare_out: array<u32>;
@group(0) @binding(3) var<uniform> compare_params: CompareParams;

@compute @workgroup_size(256)
fn eq_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0u, 1u, compare_a[idx] == compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn ne_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0u, 1u, compare_a[idx] != compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn lt_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0u, 1u, compare_a[idx] < compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn le_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0u, 1u, compare_a[idx] <= compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn gt_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0u, 1u, compare_a[idx] > compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn ge_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0u, 1u, compare_a[idx] >= compare_b[idx]);
    }
}
