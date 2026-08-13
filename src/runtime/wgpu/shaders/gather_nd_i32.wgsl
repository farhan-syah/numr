// gather_nd operations for i32

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

struct GatherNdParams {
    num_slices: u32,
    slice_size: u32,
    index_depth: u32,
    ndim: u32,
    input_shape: array<u32, 8>,
    input_strides: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read_write> gather_nd_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> gather_nd_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> gather_nd_output: array<i32>;
@group(0) @binding(3) var<storage, read_write> gather_nd_params: GatherNdParams;

@compute @workgroup_size(256)
fn gather_nd_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = gather_nd_params.num_slices * gather_nd_params.slice_size;
    if (idx >= total) {
        return;
    }

    let slice_idx = idx / gather_nd_params.slice_size;
    let element_in_slice = idx % gather_nd_params.slice_size;

    // Compute input offset from indices
    var input_offset: u32 = 0u;
    let indices_offset = slice_idx * gather_nd_params.index_depth;

    for (var d: u32 = 0u; d < gather_nd_params.index_depth; d = d + 1u) {
        let coord = gather_nd_indices[indices_offset + d];
        if (coord < 0 || u32(coord) >= gather_nd_params.input_shape[d]) {
            gather_nd_output[idx] = 0;
            return;
        }
        input_offset = input_offset + u32(coord) * gather_nd_params.input_strides[d];
    }

    // Add offset for element within slice
    if (gather_nd_params.slice_size > 1u) {
        var remaining = element_in_slice;
        for (var d: u32 = gather_nd_params.index_depth; d < gather_nd_params.ndim; d = d + 1u) {
            let dim_size = gather_nd_params.input_shape[d];
            let coord = remaining / gather_nd_params.input_strides[d];
            remaining = remaining % gather_nd_params.input_strides[d];
            input_offset = input_offset + coord * gather_nd_params.input_strides[d];
        }
    }

    gather_nd_output[idx] = gather_nd_input[input_offset + element_in_slice];
}
