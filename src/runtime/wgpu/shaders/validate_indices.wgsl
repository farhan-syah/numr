// Index bounds validation kernel

const WORKGROUP_SIZE: u32 = 256u;

struct ValidateIndicesParams {
    index_len: u32,
    dim_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> error_count: atomic<u32>;
@group(0) @binding(2) var<uniform> params: ValidateIndicesParams;

@compute @workgroup_size(256)
fn validate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.index_len) {
        return;
    }

    let index_val = indices[idx];
    if (index_val < 0 || u32(index_val) >= params.dim_size) {
        atomicAdd(&error_count, 1u);
    }
}
