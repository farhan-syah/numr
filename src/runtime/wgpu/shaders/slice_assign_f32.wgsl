// slice_assign operations for f32

const WORKGROUP_SIZE: u32 = 256u;

struct SliceAssignParams {
    outer_size: u32,
    dst_dim_size: u32,
    src_dim_size: u32,
    inner_size: u32,
    start: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SliceAssignParams;

@compute @workgroup_size(256)
fn slice_assign_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.outer_size * params.src_dim_size * params.inner_size;
    if (idx >= total) {
        return;
    }

    let inner_idx = idx % params.inner_size;
    let src_dim_idx = (idx / params.inner_size) % params.src_dim_size;
    let outer = idx / (params.src_dim_size * params.inner_size);

    let dst_offset = outer * params.dst_dim_size * params.inner_size + (params.start + src_dim_idx) * params.inner_size + inner_idx;
    output[dst_offset] = src[idx];
}
