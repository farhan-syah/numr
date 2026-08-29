// Comparison and predicate reductions for U32.
// Entry points: reduce_max_u32, reduce_min_u32, reduce_any_u32, reduce_all_u32
//
// The shared bindings and workgroup buffer declared here are also used by
// reduce_int_acc_u32.wgsl, which is concatenated after this file.

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> reduce_shared: array<u32, 256>;

struct ReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> reduce_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> reduce_output: array<u32>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

@compute @workgroup_size(256)
fn reduce_max_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var max_val: u32 = 0u;
    var i: u32 = tid;
    while (i < reduce_size) {
        max_val = max(max_val, reduce_input[base_offset + i * inner_size]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_min_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var min_val: u32 = 4294967295u;
    var i: u32 = tid;
    while (i < reduce_size) {
        min_val = min(min_val, reduce_input[base_offset + i * inner_size]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_any_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var found_nonzero: u32 = 0u;
    var i: u32 = tid;
    while (i < reduce_size) {
        if (reduce_input[base_offset + i * inner_size] != 0u) { found_nonzero = 1u; }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = found_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_all_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var all_nonzero: u32 = 1u;
    var i: u32 = tid;
    while (i < reduce_size) {
        if (reduce_input[base_offset + i * inner_size] == 0u) { all_nonzero = 0u; }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = all_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}
