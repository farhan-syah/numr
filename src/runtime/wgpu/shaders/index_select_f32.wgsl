// index_select operations for f32

const WORKGROUP_SIZE: u32 = 256u;

struct IndexSelectParams {
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    index_len: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: IndexSelectParams;

@compute @workgroup_size(256)
fn index_select_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.outer_size * params.index_len * params.inner_size;
    if (idx >= total) {
        return;
    }

    let inner = idx % params.inner_size;
    let sel_idx = (idx / params.inner_size) % params.index_len;
    let outer = idx / (params.index_len * params.inner_size);

    let index_val = indices[sel_idx];
    if (index_val < 0 || u32(index_val) >= params.dim_size) {
        output[idx] = 0.0;
        return;
    }

    let src_offset = outer * params.dim_size * params.inner_size + u32(index_val) * params.inner_size + inner;
    output[idx] = input[src_offset];
}
