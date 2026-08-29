// Matrix multiplication operations. I32 only.
// Entry points: matmul_i32, batched_matmul_i32, matmul_simple_i32
//
// Concatenated after int_saturate.wgsl + int_matmul_acc.wgsl, which define the
// 96-bit accumulator `NumrI96` and `numr_i64_mul_i32`. The accumulator is wide
// enough that it never overflows, so the only clamp is the one at the store -
// which is what makes a partial sum that leaves i32's range and returns to it
// report the true value, matching CPU's i128 accumulator.

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<i32, 16>, 16>;
var<workgroup> tile_b: array<array<i32, 16>, 16>;

struct MatmulParams {
    M: u32,             // Rows of A and C
    K: u32,             // Cols of A, Rows of B
    N: u32,             // Cols of B and C
    batch_size: u32,    // Number of matrices in batch (1 for non-batched)
}

@group(0) @binding(0) var<storage, read_write> matmul_a: array<i32>;
@group(0) @binding(1) var<storage, read_write> matmul_b: array<i32>;
@group(0) @binding(2) var<storage, read_write> matmul_c: array<i32>;
@group(0) @binding(3) var<uniform> matmul_params: MatmulParams;

@compute @workgroup_size(16, 16, 1)
fn matmul_i32(@builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    var acc = NumrI96(0u, 0u, 0u);

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Out-of-range lanes load 0, whose products leave the accumulator alone.
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = matmul_a[row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0;
        }

        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = matmul_b[b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = numr_i96_add_i64(
                acc,
                numr_i64_mul_i32(tile_a[local_id.y][k], tile_b[k][local_id.x])
            );
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        matmul_c[row * N + col] = numr_i96_to_i32_sat(acc);
    }
}

@compute @workgroup_size(16, 16, 1)
fn batched_matmul_i32(@builtin(local_invocation_id) local_id: vec3<u32>,
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

    var acc = NumrI96(0u, 0u, 0u);

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = matmul_a[a_batch_offset + row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0;
        }

        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = matmul_b[b_batch_offset + b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = numr_i96_add_i64(
                acc,
                numr_i64_mul_i32(tile_a[local_id.y][k], tile_b[k][local_id.x])
            );
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        matmul_c[c_batch_offset + row * N + col] = numr_i96_to_i32_sat(acc);
    }
}

// Non-tiled variant for small matrices, where the tile loads cost more than the
// redundant global reads they save.
@compute @workgroup_size(256, 1, 1)
fn matmul_simple_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
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

    var acc = NumrI96(0u, 0u, 0u);
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        acc = numr_i96_add_i64(
            acc,
            numr_i64_mul_i32(matmul_a[row * K + k], matmul_b[k * N + col])
        );
    }

    matmul_c[idx] = numr_i96_to_i32_sat(acc);
}
