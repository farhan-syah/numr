// Matrix multiplication operations. U32 only.
// Entry points: matmul_u32, batched_matmul_u32, matmul_simple_u32
//
// Concatenated after int_saturate.wgsl + int_matmul_acc.wgsl, which define
// `numr_u32_mul_add_sat`. Unlike the I32 shaders there is no wide accumulator:
// every term is non-negative, so the running total is monotonic and a per-step
// saturating add gives the same answer as CPU's i128 accumulator narrowed at the
// store. See `numr_u32_mul_add_sat` for the full argument.

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<u32, 16>, 16>;
var<workgroup> tile_b: array<array<u32, 16>, 16>;

struct MatmulParams {
    M: u32,             // Rows of A and C
    K: u32,             // Cols of A, Rows of B
    N: u32,             // Cols of B and C
    batch_size: u32,    // Number of matrices in batch (1 for non-batched)
}

@group(0) @binding(0) var<storage, read_write> matmul_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> matmul_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> matmul_c: array<u32>;
@group(0) @binding(3) var<uniform> matmul_params: MatmulParams;

@compute @workgroup_size(16, 16, 1)
fn matmul_u32(@builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    var acc: u32 = 0u;

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Out-of-range lanes load 0, whose products leave the accumulator alone.
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
fn batched_matmul_u32(@builtin(local_invocation_id) local_id: vec3<u32>,
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

    var acc: u32 = 0u;

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

// Non-tiled variant for small matrices, where the tile loads cost more than the
// redundant global reads they save.
@compute @workgroup_size(256, 1, 1)
fn matmul_simple_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let idx = global_id.x;
    if (idx >= M * N) {
        return;
    }

    // The two divisions stay OUTSIDE the k loop: an integer divide inside a loop
    // crashes the NVIDIA shader compiler (see int_saturate.wgsl).
    let row = idx / N;
    let col = idx % N;

    var acc: u32 = 0u;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        acc = numr_u32_mul_add_sat(acc, matmul_a[row * K + k], matmul_b[k * N + col]);
    }

    matmul_c[idx] = acc;
}
