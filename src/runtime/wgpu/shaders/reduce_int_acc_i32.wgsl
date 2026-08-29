// Accumulating reductions for I32: sum, prod, mean.
// Entry points: reduce_sum_i32, reduce_prod_i32, reduce_mean_i32
//
// Concatenated after int_saturate.wgsl, int_matmul_acc.wgsl, int_wide_div.wgsl
// and reduce_i32.wgsl, whose bindings and helpers this file uses.
//
// These three are the reductions that build a running total wider than one
// element, so they follow the accumulator half of the convention in
// runtime/cpu/kernels/wide_acc.rs: accumulate wide, narrow once, saturate at
// the narrow. CPU runs the same three in i128 (`reduce_sum_prod_int_kernel` and
// `reduce_mean_int_kernel` in runtime/cpu/kernels/reduce/int_acc.rs).

var<workgroup> acc_shared: array<NumrI64, 256>;

// Running i32 product state: magnitude, plus the two flags that make a clamped
// product recoverable. See the "Integer cumprod" section of int_saturate.wgsl -
// once the true magnitude leaves the range it can never come back, so the
// clamped answer depends only on the sign from there on.
var<workgroup> prod_mag_shared: array<u32, 256>;
var<workgroup> prod_flags_shared: array<u32, 256>;

const NUMR_PROD_ZERO: u32 = 1u;
const NUMR_PROD_SAT: u32 = 2u;
const NUMR_PROD_NEG: u32 = 4u;

// Sum every element this thread owns into a 64-bit accumulator. The total needs
// at most 31 + log2(reduce_size) bits, and reduce_size is bounded by a storage
// buffer's element count, so 64 bits can never overflow here.
fn reduce_acc_i32(base_offset: u32, reduce_size: u32, inner_size: u32, tid: u32) -> NumrI64 {
    var acc = NumrI64(0u, 0u);
    var i: u32 = tid;
    while (i < reduce_size) {
        acc = numr_i64_add(acc, numr_i64_from_i32(reduce_input[base_offset + i * inner_size]));
        i = i + WORKGROUP_SIZE;
    }
    return acc;
}

// Tree-reduce the per-thread accumulators into `acc_shared[0]`.
fn reduce_acc_tree_i32(tid: u32, acc: NumrI64) {
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
fn reduce_sum_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
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

    reduce_acc_tree_i32(tid, reduce_acc_i32(base_offset, reduce_size, inner_size, tid));

    if (tid == 0u) { reduce_output[output_idx] = numr_i64_to_i32_sat(acc_shared[0]); }
}

@compute @workgroup_size(256)
fn reduce_mean_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
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

    reduce_acc_tree_i32(tid, reduce_acc_i32(base_offset, reduce_size, inner_size, tid));

    // One division, by the whole reduced count, truncating toward zero. The sum
    // may leave i32's range while the mean does not, which is the case a running
    // or per-dimension mean gets wrong.
    if (tid == 0u) {
        let mean = numr_i64_div_u32_trunc(acc_shared[0], reduce_size);
        reduce_output[output_idx] = numr_i64_to_i32_sat(mean);
    }
}

// Fold one factor into a running product state.
fn reduce_prod_step_i32(mag: u32, flags: u32, v: i32) -> vec2<u32> {
    if (v == 0) {
        return vec2<u32>(mag, flags | NUMR_PROD_ZERO);
    }
    var out_flags = flags;
    if (v < 0) {
        out_flags = out_flags ^ NUMR_PROD_NEG;
    }
    let m = numr_i32_magnitude(v);
    if ((out_flags & NUMR_PROD_SAT) != 0u || numr_u32_mul_overflows(mag, m)) {
        return vec2<u32>(mag, out_flags | NUMR_PROD_SAT);
    }
    return vec2<u32>(mag * m, out_flags);
}

// Combine two running product states. Sign parity is the XOR of the two, and a
// magnitude that overflows on the way in is saturated rather than wrapped.
fn reduce_prod_merge_i32(mag_a: u32, flags_a: u32, mag_b: u32, flags_b: u32) -> vec2<u32> {
    let zero = (flags_a | flags_b) & NUMR_PROD_ZERO;
    let neg = (flags_a ^ flags_b) & NUMR_PROD_NEG;
    var sat = (flags_a | flags_b) & NUMR_PROD_SAT;
    if (sat != 0u || numr_u32_mul_overflows(mag_a, mag_b)) {
        return vec2<u32>(mag_a, zero | neg | NUMR_PROD_SAT);
    }
    return vec2<u32>(mag_a * mag_b, zero | neg | sat);
}

@compute @workgroup_size(256)
fn reduce_prod_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
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

    var state = vec2<u32>(1u, 0u);
    var i: u32 = tid;
    while (i < reduce_size) {
        state = reduce_prod_step_i32(state.x, state.y, reduce_input[base_offset + i * inner_size]);
        i = i + WORKGROUP_SIZE;
    }

    prod_mag_shared[tid] = state.x;
    prod_flags_shared[tid] = state.y;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let merged = reduce_prod_merge_i32(
                prod_mag_shared[tid], prod_flags_shared[tid],
                prod_mag_shared[tid + s], prod_flags_shared[tid + s]
            );
            prod_mag_shared[tid] = merged.x;
            prod_flags_shared[tid] = merged.y;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let flags = prod_flags_shared[0];
        reduce_output[output_idx] = numr_i32_product(
            (flags & NUMR_PROD_ZERO) != 0u,
            (flags & NUMR_PROD_SAT) != 0u,
            (flags & NUMR_PROD_NEG) != 0u,
            prod_mag_shared[0]
        );
    }
}
