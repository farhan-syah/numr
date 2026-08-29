// Accumulating reductions for U32: sum, prod, mean.
// Entry points: reduce_sum_u32, reduce_prod_u32, reduce_mean_u32
//
// Concatenated after int_saturate.wgsl, int_matmul_acc.wgsl, int_wide_div.wgsl
// and reduce_u32.wgsl, whose bindings and helpers this file uses.
//
// Same convention as the I32 file beside it: accumulate wide, narrow once,
// saturate at the narrow (runtime/cpu/kernels/wide_acc.rs). `sum` could get away
// with a per-step saturating add because an unsigned running total never
// decreases, but `mean` needs the true total before the divide, so both share
// one 64-bit accumulator.

var<workgroup> acc_shared: array<NumrI64, 256>;
var<workgroup> prod_shared: array<u32, 256>;

fn reduce_acc_u32(base_offset: u32, reduce_size: u32, inner_size: u32, tid: u32) -> NumrI64 {
    var acc = NumrI64(0u, 0u);
    var i: u32 = tid;
    while (i < reduce_size) {
        acc = numr_i64_add(acc, numr_u64_from_u32(reduce_input[base_offset + i * inner_size]));
        i = i + WORKGROUP_SIZE;
    }
    return acc;
}

fn reduce_acc_tree_u32(tid: u32, acc: NumrI64) {
    acc_shared[tid] = acc;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            acc_shared[tid] = numr_i64_add(acc_shared[tid], acc_shared[tid + s]);
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn reduce_sum_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
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

    reduce_acc_tree_u32(tid, reduce_acc_u32(base_offset, reduce_size, inner_size, tid));

    if (tid == 0u) { reduce_output[output_idx] = numr_u64_to_u32_sat(acc_shared[0]); }
}

@compute @workgroup_size(256)
fn reduce_mean_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
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

    reduce_acc_tree_u32(tid, reduce_acc_u32(base_offset, reduce_size, inner_size, tid));

    // One division, by the whole reduced count. The sum may leave u32's range
    // while the mean does not.
    if (tid == 0u) {
        let mean = numr_u64_div_u32(acc_shared[0], reduce_size);
        reduce_output[output_idx] = numr_u64_to_u32_sat(mean);
    }
}

// Saturating unsigned multiply. Exact for a running product because there is no
// sign to recover: a factor of 0 pins the answer at 0 whatever came before, and
// every other factor has magnitude at least 1, so a total that has reached
// u32::MAX can never need to come back down.
fn reduce_prod_step_u32(acc: u32, v: u32) -> u32 {
    if (acc == 0u || v == 0u) {
        return 0u;
    }
    if (acc == NUMR_U32_MAX || numr_u32_mul_overflows(acc, v)) {
        return NUMR_U32_MAX;
    }
    return acc * v;
}

@compute @workgroup_size(256)
fn reduce_prod_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
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

    var prod: u32 = 1u;
    var i: u32 = tid;
    while (i < reduce_size) {
        prod = reduce_prod_step_u32(prod, reduce_input[base_offset + i * inner_size]);
        i = i + WORKGROUP_SIZE;
    }

    prod_shared[tid] = prod;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            prod_shared[tid] = reduce_prod_step_u32(prod_shared[tid], prod_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = prod_shared[0]; }
}
