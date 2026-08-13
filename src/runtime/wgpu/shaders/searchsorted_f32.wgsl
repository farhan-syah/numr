// Searchsorted operations for f32. Ordering helpers are prepended from sort_cmp.rs.

const WORKGROUP_SIZE: u32 = 256u;

struct SearchsortedParams {
    seq_len: u32,
    num_values: u32,
    right: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> ss_seq: array<f32>;
@group(0) @binding(1) var<storage, read_write> ss_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> ss_output: array<i32>;
@group(0) @binding(3) var<uniform> ss_params: SearchsortedParams;

@compute @workgroup_size(256)
fn searchsorted_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
        let mid = lo + (hi - lo) / 2u;
        let seq_val = ss_seq[mid];

        // Same total order the sequence was sorted by, so NaN keys land at the
        // trailing NaN run instead of collapsing to position 0.
        let c = sort_cmp_f32(seq_val, value);
        let go_right = select(c < 0, c <= 0, right);

        if (go_right) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    ss_output[idx] = i32(lo);
}
