// Tiled transpose kernel for materializing a permuted view into contiguous memory.
//
// The general strided_copy kernel gives every thread one destination element, so
// on a permuted view a warp's 32 consecutive destination elements map to 32
// scattered source addresses and every warp read touches 32 separate sectors.
//
// Any permutation that materializes to contiguous memory is a 2-D transpose
// between two axes once the others collapse into a batch: the axis with source
// stride 1 and the axis with destination stride 1. The host detects that
// structure (see strided_transpose.rs) and passes it as:
//
//   dst[b][r][c]  ==  src[b * batch_stride + r + c * col_stride]
//
// with dst laid out row-major as [batch, rows, cols]. Reads are coalesced along
// r, writes are coalesced along c, and shared memory does the reorder in between.

// Tile edge. Must match TILE_DIM in strided_transpose.rs, which sizes the grid.
#define STRIDED_TRANSPOSE_TILE_DIM 32

// Tile rows covered per thread iteration. Must match BLOCK_ROWS in
// strided_transpose.rs, which sizes the block. Each thread handles
// STRIDED_TRANSPOSE_TILE_DIM / STRIDED_TRANSPOSE_BLOCK_ROWS rows of the tile.
#define STRIDED_TRANSPOSE_BLOCK_ROWS 8

template <typename T>
__device__ __forceinline__ void strided_transpose_impl(
    const char* __restrict__ src,
    char* __restrict__ dst,
    unsigned int batch,
    unsigned int rows,
    unsigned int cols,
    long long batch_stride,
    long long col_stride,
    unsigned long long src_byte_offset
) {
    // The +1 padding column is required: the write phase walks a tile column,
    // and without it every element of that column lands in one shared-memory bank.
    __shared__ T tile[STRIDED_TRANSPOSE_TILE_DIM][STRIDED_TRANSPOSE_TILE_DIM + 1];

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int row_base = blockIdx.y * STRIDED_TRANSPOSE_TILE_DIM;
    const unsigned int col_base = blockIdx.x * STRIDED_TRANSPOSE_TILE_DIM;

    // Read phase: threadIdx.x walks `rows`, the axis whose source stride is 1,
    // so each warp reads 32 consecutive source elements.
    const unsigned int r_in = row_base + threadIdx.x;
    if (r_in < rows) {
        for (unsigned int k = 0; k < STRIDED_TRANSPOSE_TILE_DIM; k += STRIDED_TRANSPOSE_BLOCK_ROWS) {
            const unsigned int c_in = col_base + threadIdx.y + k;
            if (c_in < cols) {
                const long long elem = (long long)b * batch_stride
                    + (long long)r_in
                    + (long long)c_in * col_stride;
                const unsigned long long addr =
                    src_byte_offset + (unsigned long long)(elem * (long long)sizeof(T));
                tile[threadIdx.y + k][threadIdx.x] = *((const T*)(src + addr));
            }
        }
    }

    __syncthreads();

    // Write phase: threadIdx.x walks `cols`, the destination's contiguous axis.
    const unsigned int c_out = col_base + threadIdx.x;
    if (c_out < cols) {
        for (unsigned int k = 0; k < STRIDED_TRANSPOSE_TILE_DIM; k += STRIDED_TRANSPOSE_BLOCK_ROWS) {
            const unsigned int r_out = row_base + threadIdx.y + k;
            if (r_out < rows) {
                const unsigned long long didx = (unsigned long long)b * rows * cols
                    + (unsigned long long)r_out * cols
                    + (unsigned long long)c_out;
                *((T*)(dst + didx * sizeof(T))) = tile[threadIdx.x][threadIdx.y + k];
            }
        }
    }
}

// The kernel moves raw bytes, so it is instantiated per element width rather
// than per dtype - the same widths strided_copy fast-paths.
#define STRIDED_TRANSPOSE_KERNEL(SUFFIX, T)                                    \
    __global__ void strided_transpose_##SUFFIX(                                \
        const char* __restrict__ src,                                          \
        char* __restrict__ dst,                                                \
        unsigned int batch,                                                    \
        unsigned int rows,                                                     \
        unsigned int cols,                                                     \
        long long batch_stride,                                                \
        long long col_stride,                                                  \
        unsigned long long src_byte_offset                                     \
    ) {                                                                        \
        strided_transpose_impl<T>(                                             \
            src, dst, batch, rows, cols, batch_stride, col_stride,             \
            src_byte_offset);                                                  \
    }

extern "C" {

STRIDED_TRANSPOSE_KERNEL(b1, unsigned char)
STRIDED_TRANSPOSE_KERNEL(b2, unsigned short)
STRIDED_TRANSPOSE_KERNEL(b4, unsigned int)
STRIDED_TRANSPOSE_KERNEL(b8, unsigned long long)

} // extern "C"
