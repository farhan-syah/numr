// Cumulative sum shader for i32.
//
// Accumulates in `NumrI64` (from int_saturate.wgsl, prepended to this
// module), not `i32`. A signed running total can overflow past i32 and later
// come back into range - CPU's i128 accumulator (`WideAcc` in
// `runtime/cpu/kernels/wide_acc.rs`) does the same, narrowing once per
// element written. A per-step saturating add on i32 would clamp the
// intermediate and never recover.

struct CumsumParams {
    scan_size: u32,
    outer_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<uniform> params: CumsumParams;

@compute @workgroup_size(256)
fn cumsum_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {
        return;
    }

    let base = outer_idx * params.scan_size;
    var acc = NumrI64(0u, 0u);
    for (var i: u32 = 0u; i < params.scan_size; i = i + 1u) {
        acc = numr_i64_add(acc, numr_i64_from_i32(input[base + i]));
        output[base + i] = numr_i64_to_i32_sat(acc);
    }
}
