// scatter_reduce_count for mean computation.
//
// Counts contributions per destination slot. It reads only the index tensor and
// writes a u32 counter, so one shader serves every value dtype.

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterReduceParams {
    dim: u32,
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    src_dim_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> scatter_indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> scatter_count: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = scatter_params.outer_size * scatter_params.src_dim_size * scatter_params.inner_size;
    if (idx >= total) {
        return;
    }

    let inner = idx % scatter_params.inner_size;
    let outer = idx / (scatter_params.src_dim_size * scatter_params.inner_size);

    let index_val = scatter_indices[idx];
    if (index_val < 0 || u32(index_val) >= scatter_params.dim_size) {
        return;
    }

    let dst_idx = outer * scatter_params.dim_size * scatter_params.inner_size + u32(index_val) * scatter_params.inner_size + inner;

    atomicAdd(&scatter_count[dst_idx], 1u);
}
