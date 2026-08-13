// Repeat operation for f32

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

// Use vec4<u32> for 16-byte alignment in uniform buffer
struct RepeatParams {
    ndim: u32,
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    src_shape: array<vec4<u32>, 2>,  // 8 u32 values packed into 2 vec4
    out_shape: array<vec4<u32>, 2>,
}

// Helper to access packed array<vec4<u32>, 2> by index
fn get_packed_value(arr: array<vec4<u32>, 2>, d: i32) -> u32 {
    let vec_idx = u32(d) / 4u;
    let comp_idx = u32(d) % 4u;
    if (vec_idx == 0u) {
        if (comp_idx == 0u) { return arr[0].x; }
        else if (comp_idx == 1u) { return arr[0].y; }
        else if (comp_idx == 2u) { return arr[0].z; }
        else { return arr[0].w; }
    } else {
        if (comp_idx == 0u) { return arr[1].x; }
        else if (comp_idx == 1u) { return arr[1].y; }
        else if (comp_idx == 2u) { return arr[1].z; }
        else { return arr[1].w; }
    }
}

@group(0) @binding(0) var<storage, read_write> repeat_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> repeat_dst: array<f32>;
@group(0) @binding(2) var<uniform> repeat_params: RepeatParams;

@compute @workgroup_size(256)
fn repeat_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= repeat_params.total_elements) {
        return;
    }

    // Decompose idx into multi-dimensional output coordinates
    var remaining = idx;
    var src_idx = 0u;

    // Compute source strides first (row-major)
    var src_strides: array<u32, 8>;
    var stride = 1u;
    for (var d = i32(repeat_params.ndim) - 1; d >= 0; d = d - 1) {
        src_strides[d] = stride;
        stride = stride * get_packed_value(repeat_params.src_shape, d);
    }

    // Process dimensions from last to first
    for (var d = i32(repeat_params.ndim) - 1; d >= 0; d = d - 1) {
        let out_dim = get_packed_value(repeat_params.out_shape, d);
        let coord = remaining % out_dim;
        remaining = remaining / out_dim;

        // Map to source coordinate using modulo
        let src_shape_d = get_packed_value(repeat_params.src_shape, d);
        let src_coord = coord % src_shape_d;
        src_idx = src_idx + src_coord * src_strides[d];
    }

    repeat_dst[idx] = repeat_src[src_idx];
}
