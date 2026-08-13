// unique_with_counts operations for i32

const WORKGROUP_SIZE: u32 = 256u;

struct UniqueCountsParams {
    numel: u32,
    num_unique: u32,
    _pad0: u32,
    _pad1: u32,
}

// Mark boundaries in sorted array (where value changes)
// Output: flags[i] = 1 if sorted[i] != sorted[i-1] (or i=0), else 0
@group(0) @binding(0) var<storage, read> sorted_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> boundary_flags: array<u32>;
@group(0) @binding(2) var<uniform> params: UniqueCountsParams;

@compute @workgroup_size(256)
fn mark_boundaries_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let numel = params.numel;

    if (idx >= numel) {
        return;
    }

    // Mark boundary: first element or different from previous
    if (idx == 0u || sorted_input[idx] != sorted_input[idx - 1u]) {
        boundary_flags[idx] = 1u;
    } else {
        boundary_flags[idx] = 0u;
    }
}

// Scatter unique values and compute counts using prefix sum indices
// prefix_sum[i] contains the output index for element at position i (if it's a boundary)
// We write: unique_values[prefix_sum[i]-1] = sorted[i] when flags[i] == 1
// counts[prefix_sum[i]-1] = (next boundary position - i) computed from adjacent prefix sums
@group(0) @binding(0) var<storage, read> scatter_sorted: array<i32>;
@group(0) @binding(1) var<storage, read> prefix_sum: array<u32>;
@group(0) @binding(2) var<storage, read_write> unique_values: array<i32>;
@group(0) @binding(3) var<storage, read_write> inverse_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> counts: array<i32>;
@group(0) @binding(5) var<uniform> scatter_params: UniqueCountsParams;

@compute @workgroup_size(256)
fn scatter_unique_with_counts_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let numel = scatter_params.numel;
    let num_unique = scatter_params.num_unique;

    if (idx >= numel) {
        return;
    }

    // The prefix sum gives us 1-based output indices
    let out_idx_plus1 = prefix_sum[idx];

    // Check if this is a boundary by comparing with previous prefix sum
    let is_boundary = (idx == 0u) || (prefix_sum[idx] != prefix_sum[idx - 1u]);

    // Write inverse index: which unique element does this sorted element map to
    inverse_indices[idx] = i32(out_idx_plus1 - 1u);

    if (is_boundary) {
        let out_idx = out_idx_plus1 - 1u;
        unique_values[out_idx] = scatter_sorted[idx];

        // Compute count: find next boundary position
        // The count is (next_boundary_position - idx)
        // If we're the last unique, count to numel
        if (out_idx + 1u >= num_unique) {
            // Last unique element
            counts[out_idx] = i32(numel - idx);
        } else {
            // Find next boundary: it's where prefix_sum increases next
            // We need to find the smallest j > idx where prefix_sum[j] > out_idx_plus1
            // Actually, we can compute this differently:
            // The run length is the distance to the next boundary
            // For efficiency, we'll use a second pass or a different approach

            // For now, scan forward (not ideal but correct)
            var run_len: u32 = 1u;
            var j = idx + 1u;
            while (j < numel && prefix_sum[j] == out_idx_plus1) {
                run_len = run_len + 1u;
                j = j + 1u;
            }
            counts[out_idx] = i32(run_len);
        }
    }
}
