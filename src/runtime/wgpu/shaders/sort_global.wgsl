const WORKGROUP_SIZE: u32 = 256u;

struct GlobalSortParams {
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    padded_size: u32,
    segment_count: u32,
    dtype_tag: u32,
    descending: u32,
    k: u32,
    j: u32,
    total_padded: u32,
    _padding_0: u32,
    _padding_1: u32,
}

@group(0) @binding(0) var<storage, read_write> buffer_0: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1: array<u32>;
@group(0) @binding(2) var<storage, read_write> buffer_2: array<u32>;
@group(0) @binding(3) var<storage, read_write> buffer_3: array<u32>;
@group(0) @binding(4) var<uniform> params: GlobalSortParams;

var<workgroup> tile_keys: array<u32, 512>;
var<workgroup> tile_values: array<u32, 512>;
var<workgroup> tile_indices: array<u32, 512>;

fn flat_invocation_index(
    workgroup_id: vec3<u32>,
    local_id: vec3<u32>,
    num_workgroups: vec3<u32>,
    total_items: u32,
) -> u32 {
    let flat_group = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let valid_groups = total_items / WORKGROUP_SIZE
        + select(0u, 1u, total_items % WORKGROUP_SIZE != 0u);
    if (flat_group >= valid_groups) {
        return total_items;
    }
    return flat_group * WORKGROUP_SIZE + local_id.x;
}

fn transformed_key(raw: u32) -> u32 {
    if (params.dtype_tag == 0u) {
        return raw;
    }
    if (params.dtype_tag == 1u) {
        return raw ^ 0x80000000u;
    }

    // Normalize signed zero for comparison while preserving the original bits
    // in the value buffer. Canonicalize every NaN to the largest key.
    let magnitude = raw & 0x7fffffffu;
    if (magnitude == 0u) {
        return 0x80000000u;
    }
    if (magnitude > 0x7f800000u) {
        return 0xffffffffu;
    }
    return select(raw ^ 0x80000000u, ~raw, (raw & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn pack_global_sort(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let packed_index = flat_invocation_index(
        workgroup_id,
        local_id,
        num_workgroups,
        params.total_padded,
    );
    if (packed_index >= params.total_padded) {
        return;
    }

    let segment = packed_index / params.padded_size;
    let axis_index = packed_index % params.padded_size;
    if (segment >= params.segment_count) {
        return;
    }

    if (axis_index < params.sort_size) {
        let outer = segment / params.inner_size;
        let inner = segment % params.inner_size;
        let source_index = outer * params.sort_size * params.inner_size
            + axis_index * params.inner_size + inner;
        let raw = buffer_0[source_index];
        buffer_1[packed_index] = transformed_key(raw);
        buffer_2[packed_index] = raw;
        buffer_3[packed_index] = axis_index;
    } else {
        buffer_1[packed_index] = select(0xffffffffu, 0u, params.descending != 0u);
        buffer_2[packed_index] = 0u;
        buffer_3[packed_index] = axis_index;
    }
}

fn comes_before(key_a: u32, index_a: u32, key_b: u32, index_b: u32) -> bool {
    if (key_a == key_b) {
        return index_a < index_b;
    }
    if (params.descending != 0u) {
        return key_a > key_b;
    }
    return key_a < key_b;
}

fn tile_compare_and_swap(left: u32, right: u32, first_before: bool) {
    let key_left = tile_keys[left];
    let key_right = tile_keys[right];
    let value_left = tile_values[left];
    let value_right = tile_values[right];
    let index_left = tile_indices[left];
    let index_right = tile_indices[right];
    let left_before_right = comes_before(key_left, index_left, key_right, index_right);
    let swap = select(left_before_right, !left_before_right, first_before);
    if (swap) {
        tile_keys[left] = key_right;
        tile_keys[right] = key_left;
        tile_values[left] = value_right;
        tile_values[right] = value_left;
        tile_indices[left] = index_right;
        tile_indices[right] = index_left;
    }
}

// Replaces the first 45 global bitonic stages (k <= 512) with one dispatch.
// Adjacent tiles alternate direction, exactly matching the k=512 network state.
@compute @workgroup_size(256)
fn sort_global_tiles(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let tile = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let tile_count = params.total_padded / 512u;
    if (tile >= tile_count) {
        return;
    }
    let tile_base = tile * 512u;
    let first = local_id.x;
    let second = first + 256u;
    tile_keys[first] = buffer_0[tile_base + first];
    tile_keys[second] = buffer_0[tile_base + second];
    tile_values[first] = buffer_1[tile_base + first];
    tile_values[second] = buffer_1[tile_base + second];
    tile_indices[first] = buffer_2[tile_base + first];
    tile_indices[second] = buffer_2[tile_base + second];
    workgroupBarrier();

    let tiles_per_segment = params.padded_size / 512u;
    let tile_in_segment = tile % tiles_per_segment;
    let tile_before = tile_in_segment % 2u == 0u;
    for (var k = 2u; k <= 512u; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            let pair_left = (local_id.x / j) * 2u * j + (local_id.x % j);
            let pair_right = pair_left + j;
            let stage_before = ((pair_left / k) % 2u == 0u) == tile_before;
            tile_compare_and_swap(pair_left, pair_right, stage_before);
            workgroupBarrier();
        }
    }

    buffer_0[tile_base + first] = tile_keys[first];
    buffer_0[tile_base + second] = tile_keys[second];
    buffer_1[tile_base + first] = tile_values[first];
    buffer_1[tile_base + second] = tile_values[second];
    buffer_2[tile_base + first] = tile_indices[first];
    buffer_2[tile_base + second] = tile_indices[second];
}

@compute @workgroup_size(256)
fn global_bitonic_step(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let packed_index = flat_invocation_index(
        workgroup_id,
        local_id,
        num_workgroups,
        params.total_padded,
    );
    if (packed_index >= params.total_padded) {
        return;
    }

    let axis_index = packed_index % params.padded_size;
    let partner_axis = axis_index ^ params.j;
    if (partner_axis <= axis_index || partner_axis >= params.padded_size) {
        return;
    }

    let segment_base = packed_index - axis_index;
    let partner_index = segment_base + partner_axis;
    let key_a = buffer_0[packed_index];
    let key_b = buffer_0[partner_index];
    let value_a = buffer_1[packed_index];
    let value_b = buffer_1[partner_index];
    let index_a = buffer_2[packed_index];
    let index_b = buffer_2[partner_index];

    let a_before_b = comes_before(key_a, index_a, key_b, index_b);
    let first_half = (axis_index & params.k) == 0u;
    let swap = select(a_before_b, !a_before_b, first_half);
    if (swap) {
        buffer_0[packed_index] = key_b;
        buffer_0[partner_index] = key_a;
        buffer_1[packed_index] = value_b;
        buffer_1[partner_index] = value_a;
        buffer_2[packed_index] = index_b;
        buffer_2[partner_index] = index_a;
    }
}

@compute @workgroup_size(256)
fn scatter_global_sort(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let logical_total = params.segment_count * params.sort_size;
    let output_linear = flat_invocation_index(
        workgroup_id,
        local_id,
        num_workgroups,
        logical_total,
    );
    if (output_linear >= logical_total) {
        return;
    }

    let segment = output_linear / params.sort_size;
    let axis_index = output_linear % params.sort_size;
    let packed_index = segment * params.padded_size + axis_index;
    let outer = segment / params.inner_size;
    let inner = segment % params.inner_size;
    let output_index = outer * params.sort_size * params.inner_size
        + axis_index * params.inner_size + inner;
    buffer_2[output_index] = buffer_0[packed_index];
    buffer_3[output_index] = buffer_1[packed_index];
}
