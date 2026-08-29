// Semiring matmul: max_plus for i32
// C[i,j] = max_k( A[i,k] + B[k,j] )
// Entry points: semiring_matmul_max_plus_i32, batched_semiring_matmul_max_plus_i32
//
// Concatenated after int_saturate.wgsl, which defines NUMR_I32_MAX / NUMR_I32_MIN.
// The reduce identity is i32::MIN: CPU's `reduce_identity` casts -inf to the element
// type, and that cast saturates.

struct SemiringMatmulParams {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read_write> sr_a: array<i32>;
@group(0) @binding(1) var<storage, read_write> sr_b: array<i32>;
@group(0) @binding(2) var<storage, read_write> sr_c: array<i32>;
@group(0) @binding(3) var<uniform> sr_params: SemiringMatmulParams;

// Wrapping add. CPU reaches this through `SemiringOp::combine`'s plain `a + b`
// on the element type, which is NOT the wide, saturating accumulator that
// `matmul` uses - a semiring add is one machine operation on two elements. The
// u32 detour is what makes the wrap defined rather than left to the driver.
fn sr_wrap_add(a: i32, b: i32) -> i32 {
    return bitcast<i32>(bitcast<u32>(a) + bitcast<u32>(b));
}

fn sr_combine(a: i32, b: i32) -> i32 {
    return sr_wrap_add(a, b);
}

fn sr_reduce(acc: i32, val: i32) -> i32 {
    return max(acc, val);
}

@compute @workgroup_size(16, 16, 1)
fn semiring_matmul_max_plus_i32(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = sr_params.M;
    let K = sr_params.K;
    let N = sr_params.N;

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    var acc: i32 = NUMR_I32_MIN;

    for (var kk: u32 = 0u; kk < K; kk = kk + 1u) {
        let a_val = sr_a[row * K + kk];
        let b_val = sr_b[kk * N + col];
        acc = sr_reduce(acc, sr_combine(a_val, b_val));
    }

    sr_c[row * N + col] = acc;
}

@compute @workgroup_size(16, 16, 1)
fn batched_semiring_matmul_max_plus_i32(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = sr_params.M;
    let K = sr_params.K;
    let N = sr_params.N;
    let batch_size = sr_params.batch_size;

    let batch = global_id.z;
    if (batch >= batch_size) {
        return;
    }

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    let a_offset = batch * M * K;
    let b_offset = batch * K * N;
    let c_offset = batch * M * N;

    var acc: i32 = NUMR_I32_MIN;

    for (var kk: u32 = 0u; kk < K; kk = kk + 1u) {
        let a_val = sr_a[a_offset + row * K + kk];
        let b_val = sr_b[b_offset + kk * N + col];
        acc = sr_reduce(acc, sr_combine(a_val, b_val));
    }

    sr_c[c_offset + row * N + col] = acc;
}
