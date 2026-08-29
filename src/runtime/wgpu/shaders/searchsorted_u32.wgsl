// Searchsorted operations for u32.
//
// Integers have one total order and no NaN, so the search compares directly
// rather than through a comparison helper.

const WORKGROUP_SIZE: u32 = 256u;

struct SearchsortedParams {
    seq_len: u32,
    num_values: u32,
    right: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> ss_seq: array<u32>;
@group(0) @binding(1) var<storage, read_write> ss_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> ss_output: array<i32>;
@group(0) @binding(3) var<uniform> ss_params: SearchsortedParams;

@compute @workgroup_size(256)
fn searchsorted_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= ss_params.num_values) {
        return;
    }

    let value = ss_values[idx];
    let seq_len = ss_params.seq_len;
    let right = ss_params.right != 0u;

    // Binary search
    var lo: u32 = 0u;
    var hi: u32 = seq_len;

    while (lo < hi) {
        let mid = lo + ((hi - lo) >> 1u);
        let seq_val = ss_seq[mid];

        let go_right = select(seq_val < value, seq_val <= value, right);

        if (go_right) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    ss_output[idx] = i32(lo);
}
