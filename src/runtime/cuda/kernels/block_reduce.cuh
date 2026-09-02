// Shared block-wide reduction helper for the normalization kernels.
#pragma once

// Tree-sums `shared[0 .. blockDim.x)` and leaves the total in `shared[0]`,
// which every thread may read once the call returns.
//
// The first stride is the power of two at or above blockDim.x, so a block whose
// size is not a power of two still folds in the entries a plain `blockDim.x / 2`
// start would step over. Normalization launchers size the block as
// `min(BLOCK_SIZE, row_or_group_length)`, so a non-power-of-two block is
// ordinary, not exotic.
//
// Every thread must reach this call: the loop carries a __syncthreads().
template <typename T>
__device__ __forceinline__ T block_sum_reduce(T* shared) {
    unsigned int n = blockDim.x;
    unsigned int s = 1;
    while (s < n) s <<= 1;
    for (s >>= 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < n) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    return shared[0];
}
