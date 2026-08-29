// scatter_reduce_sum for i32

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

// Every storage binding is declared read_write, including the two this kernel
// only reads. The pipeline cache builds this layout with
// `num_readonly_storage: 0`, and wgpu requires a shader variable's access mode
// to match its layout entry exactly - a `read` variable against a
// `Storage { read_only: false }` entry fails pipeline creation with a bare
// validation error, which then invalidates the device for every later test.
@group(0) @binding(0) var<storage, read_write> scatter_src: array<i32>;
@group(0) @binding(1) var<storage, read_write> scatter_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_dst: array<atomic<i32>>;
@group(0) @binding(3) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_sum_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let src_val = scatter_src[idx];
    let dst_idx = outer * scatter_params.dim_size * scatter_params.inner_size + u32(index_val) * scatter_params.inner_size + inner;

    atomicAdd(&scatter_dst[dst_idx], src_val);
}
