// Sort operations for f32. Ordering helpers are prepended from sort_cmp.rs.

const WORKGROUP_SIZE: u32 = 256u;
const MAX_SORT_SIZE: u32 = 512u;

var<workgroup> shared_vals: array<f32, 512>;
var<workgroup> shared_idxs: array<i32, 512>;

struct SortParams {
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    descending: u32,
}

struct TopkParams {
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    k: u32,
    largest: u32,
    sorted: u32,
}

struct SearchsortedParams {
    seq_len: u32,
    num_values: u32,
    right: u32,
    _pad: u32,
}

struct CountParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> sort_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> sort_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> sort_indices: array<i32>;
@group(0) @binding(3) var<uniform> sort_params: SortParams;

// Bitonic compare and swap for sort with indices (stable)
fn bitonic_cas_f32(i: u32, j: u32, ascending_local: bool, descending: bool) {
    let vi = shared_vals[i];
    let vj = shared_vals[j];
    let ii = shared_idxs[i];
    let ij = shared_idxs[j];
    let swap = select(
        sort_rank_less_f32(vi, ii, vj, ij, descending),
        sort_rank_less_f32(vj, ij, vi, ii, descending),
        ascending_local
    );
    if (swap) {
        shared_vals[i] = vj;
        shared_vals[j] = vi;
        shared_idxs[i] = ij;
        shared_idxs[j] = ii;
    }
}

// Bitonic compare and swap for sort values only
fn bitonic_cas_values_f32(i: u32, j: u32, ascending_local: bool, descending: bool) {
    let vi = shared_vals[i];
    let vj = shared_vals[j];
    var c = sort_cmp_f32(vi, vj);
    if (descending) {
        c = -c;
    }
    let swap = select(c < 0, (c > 0), ascending_local);
    if (swap) {
        shared_vals[i] = vj;
        shared_vals[j] = vi;
    }
}

// Sort with indices - returns both sorted values and original indices
@compute @workgroup_size(256)
fn sort_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = sort_params.outer_size;
    let sort_size = sort_params.sort_size;
    let inner_size = sort_params.inner_size;
    let descending = sort_params.descending != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {
        return;
    }

    // Pad to next power of 2
    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {
        p = p << 1u;
    }
    n = min(p, MAX_SORT_SIZE);

    // Load data into shared memory
    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {
        if (i < sort_size) {
            let idx = base_offset + i * inner_size;
            shared_vals[i] = sort_input[idx];
            shared_idxs[i] = i32(i);
        } else {
            // Pad with max/min based on sort direction
            shared_vals[i] = sort_pad_f32(descending);
            shared_idxs[i] = i32(i);
        }
    }
    workgroupBarrier();

    // Bitonic sort
    for (var k: u32 = 2u; k <= n; k = k << 1u) {
        for (var j: u32 = k >> 1u; j > 0u; j = j >> 1u) {
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                let ascending_local = ((ij / k) % 2u == 0u);

                if (ij_pair < n) {
                    bitonic_cas_f32(ij, ij_pair, ascending_local, descending);
                }
            }
            workgroupBarrier();
        }
    }

    // Write sorted values and indices
    for (var i = tid; i < sort_size; i = i + WORKGROUP_SIZE) {
        let out_idx = base_offset + i * inner_size;
        sort_output[out_idx] = shared_vals[i];
        sort_indices[out_idx] = shared_idxs[i];
    }
}

// Sort values only (no indices)
@compute @workgroup_size(256)
fn sort_values_only_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = sort_params.outer_size;
    let sort_size = sort_params.sort_size;
    let inner_size = sort_params.inner_size;
    let descending = sort_params.descending != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {
        return;
    }

    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {
        p = p << 1u;
    }
    n = min(p, MAX_SORT_SIZE);

    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {
        if (i < sort_size) {
            let idx = base_offset + i * inner_size;
            shared_vals[i] = sort_input[idx];
        } else {
            shared_vals[i] = sort_pad_f32(descending);
        }
    }
    workgroupBarrier();

    // Bitonic sort
    for (var k: u32 = 2u; k <= n; k = k << 1u) {
        for (var j: u32 = k >> 1u; j > 0u; j = j >> 1u) {
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                let ascending_local = ((ij / k) % 2u == 0u);

                if (ij_pair < n) {
                    bitonic_cas_values_f32(ij, ij_pair, ascending_local, descending);
                }
            }
            workgroupBarrier();
        }
    }

    for (var i = tid; i < sort_size; i = i + WORKGROUP_SIZE) {
        let out_idx = base_offset + i * inner_size;
        sort_output[out_idx] = shared_vals[i];
    }
}

// Argsort - returns indices only
@compute @workgroup_size(256)
fn argsort_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = sort_params.outer_size;
    let sort_size = sort_params.sort_size;
    let inner_size = sort_params.inner_size;
    let descending = sort_params.descending != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {
        return;
    }

    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {
        p = p << 1u;
    }
    n = min(p, MAX_SORT_SIZE);

    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {
        if (i < sort_size) {
            let idx = base_offset + i * inner_size;
            shared_vals[i] = sort_input[idx];
            shared_idxs[i] = i32(i);
        } else {
            shared_vals[i] = sort_pad_f32(descending);
            shared_idxs[i] = i32(i);
        }
    }
    workgroupBarrier();

    // Bitonic sort
    for (var k: u32 = 2u; k <= n; k = k << 1u) {
        for (var j: u32 = k >> 1u; j > 0u; j = j >> 1u) {
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                let ascending_local = ((ij / k) % 2u == 0u);

                if (ij_pair < n) {
                    bitonic_cas_f32(ij, ij_pair, ascending_local, descending);
                }
            }
            workgroupBarrier();
        }
    }

    // Write indices only
    for (var i = tid; i < sort_size; i = i + WORKGROUP_SIZE) {
        let out_idx = base_offset + i * inner_size;
        sort_indices[out_idx] = shared_idxs[i];
    }
}
