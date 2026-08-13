// gather_2d operation for i32
// Gathers elements from a 2D matrix at (row, col) positions.

const WORKGROUP_SIZE: u32 = 256u;

struct Gather2dParams {
    nrows: u32,
    ncols: u32,
    num_indices: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read> rows: array<i32>;
@group(0) @binding(2) var<storage, read> cols: array<i32>;
@group(0) @binding(3) var<storage, read_write> output: array<i32>;
@group(0) @binding(4) var<uniform> params: Gather2dParams;

@compute @workgroup_size(256)
fn gather_2d_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_indices) {
        return;
    }

    let r = rows[idx];
    let c = cols[idx];

    // Bounds checking
    if (r < 0 || u32(r) >= params.nrows || c < 0 || u32(c) >= params.ncols) {
        output[idx] = 0;
        return;
    }

    // Row-major indexing: input[r, c] = input[r * ncols + c]
    let input_idx = u32(r) * params.ncols + u32(c);
    output[idx] = input[input_idx];
}
