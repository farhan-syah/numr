// Fused matmul+bias operations. U32 only.
// C = A @ B + bias (fused epilogue)
// Entry points: matmul_bias_u32, batched_matmul_bias_u32
//
// Concatenated after int_saturate.wgsl + int_matmul_acc.wgsl. The bias SEEDS the
// accumulator, matching CPU's `matmul_bias_scalar_acc`. With every term
// non-negative the running total is still monotonic, so the per-step saturating
// add of `numr_u32_mul_add_sat` remains exact.

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<u32, 16>, 16>;
var<workgroup> tile_b: array<array<u32, 16>, 16>;

struct MatmulBiasParams {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read_write> matmul_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> matmul_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> matmul_bias: array<u32>;
@group(0) @binding(3) var<storage, read_write> matmul_c: array<u32>;
@group(0) @binding(4) var<uniform> matmul_params: MatmulBiasParams;

@compute @workgroup_size(16, 16, 1)
fn matmul_bias_u32(@builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    // Lanes outside the output still run the barriers below, so the seed read is
    // guarded and their accumulator value is simply never stored.
    var acc: u32 = 0u;
    if (col < N) {
        acc = matmul_bias[col];
    }

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = matmul_a[row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0u;
        }

        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = matmul_b[b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0u;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = numr_u32_mul_add_sat(acc, tile_a[local_id.y][k], tile_b[k][local_id.x]);
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        matmul_c[row * N + col] = acc;
    }
}

@compute @workgroup_size(16, 16, 1)
fn batched_matmul_bias_u32(@builtin(local_invocation_id) local_id: vec3<u32>,
                           @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;
    let batch_size = matmul_params.batch_size;

    let batch = group_id.z;
    if (batch >= batch_size) {
        return;
    }

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    let a_batch_offset = batch * M * K;
    let b_batch_offset = batch * K * N;
    let c_batch_offset = batch * M * N;

    // Same bias for every batch.
    var acc: u32 = 0u;
    if (col < N) {
        acc = matmul_bias[col];
    }

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = matmul_a[a_batch_offset + row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0u;
        }

        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = matmul_b[b_batch_offset + b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0u;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = numr_u32_mul_add_sat(acc, tile_a[local_id.y][k], tile_b[k][local_id.x]);
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        matmul_c[c_batch_offset + row * N + col] = acc;
    }
}
