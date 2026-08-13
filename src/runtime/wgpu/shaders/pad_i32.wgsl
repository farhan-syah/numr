// Pad operation for i32

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

// Use vec4<u32> for 16-byte alignment in uniform buffer
struct PadParams {
    ndim: u32,
    total_elements: u32,
    fill_value: i32,
    _pad0: u32,
    src_shape: array<vec4<u32>, 2>,    // 8 u32 values packed into 2 vec4
    out_shape: array<vec4<u32>, 2>,
    pad_before: array<vec4<u32>, 2>,
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

@group(0) @binding(0) var<storage, read_write> pad_src: array<i32>;
@group(0) @binding(1) var<storage, read_write> pad_dst: array<i32>;
@group(0) @binding(2) var<uniform> pad_params: PadParams;

@compute @workgroup_size(256)
fn pad_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= pad_params.total_elements) {
        return;
    }

    // Decompose idx into multi-dimensional output coordinates
    var remaining = idx;
    var coords: array<u32, 8>;
    var in_bounds = true;

    // Process dimensions from last to first
    for (var d = i32(pad_params.ndim) - 1; d >= 0; d = d - 1) {
        let out_dim = get_packed_value(pad_params.out_shape, d);
        coords[d] = remaining % out_dim;
        remaining = remaining / out_dim;

        // Check if coordinate is in original tensor region
        let pb = get_packed_value(pad_params.pad_before, d);
        let ss = get_packed_value(pad_params.src_shape, d);
        if (coords[d] < pb || coords[d] >= pb + ss) {
            in_bounds = false;
        }
    }

    if (in_bounds) {
        // Compute source index
        var src_idx = 0u;
        var src_stride = 1u;
        for (var d = i32(pad_params.ndim) - 1; d >= 0; d = d - 1) {
            let src_coord = coords[d] - get_packed_value(pad_params.pad_before, d);
            src_idx = src_idx + src_coord * src_stride;
            src_stride = src_stride * get_packed_value(pad_params.src_shape, d);
        }
        pad_dst[idx] = pad_src[src_idx];
    } else {
        pad_dst[idx] = pad_params.fill_value;
    }
}
