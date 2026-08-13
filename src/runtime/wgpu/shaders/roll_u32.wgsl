// Roll operation for u32

const WORKGROUP_SIZE: u32 = 256u;

struct RollParams {
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    shift: u32,
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> roll_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> roll_dst: array<u32>;
@group(0) @binding(2) var<uniform> roll_params: RollParams;

@compute @workgroup_size(256)
fn roll_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= roll_params.total_elements) {
        return;
    }

    // Decompose idx into (outer, dim_coord, inner)
    let inner = idx % roll_params.inner_size;
    let remaining = idx / roll_params.inner_size;
    let dim_coord = remaining % roll_params.dim_size;
    let outer = remaining / roll_params.dim_size;

    // Compute source coordinate with roll (shift goes right, so source is shift positions left)
    let src_dim_coord = (dim_coord + roll_params.dim_size - roll_params.shift) % roll_params.dim_size;

    // Compute source linear index
    let src_idx = outer * roll_params.dim_size * roll_params.inner_size
                + src_dim_coord * roll_params.inner_size
                + inner;

    roll_dst[idx] = roll_src[src_idx];
}
