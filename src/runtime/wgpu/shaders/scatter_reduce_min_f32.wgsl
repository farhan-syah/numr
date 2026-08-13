// scatter_reduce_min for f32

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

// Note: All storage buffers use read_write to match the pipeline cache layout.
// The actual access pattern is: src (read), indices (read), dst (read_write).
@group(0) @binding(0) var<storage, read_write> scatter_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> scatter_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_dst: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_min_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // CAS loop for min
    var old_bits: u32;
    var new_bits: u32;
    loop {
        old_bits = atomicLoad(&scatter_dst[dst_idx]);
        let old_val = bitcast<f32>(old_bits);
        let new_val = min(old_val, src_val);
        new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&scatter_dst[dst_idx], old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
    }
}
