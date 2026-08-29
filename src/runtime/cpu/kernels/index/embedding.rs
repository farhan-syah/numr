//! Embedding-table lookup kernel.
//!
//! Copies whole embedding rows, zeroing rows for out-of-bounds indices.

use crate::dtype::Element;

/// Look up embeddings from an embedding table using indices.
///
/// This is the industry-standard embedding lookup operation used in neural networks
/// for word embeddings, entity embeddings, etc. Optimized for contiguous memory access.
///
/// # Algorithm
/// ```text
/// for i in 0..num_indices:
///     idx = indices[i]
///     if 0 <= idx < vocab_size:
///         output[i * embedding_dim..(i+1) * embedding_dim] = embeddings[idx * embedding_dim..(idx+1) * embedding_dim]
///     else:
///         output[i * embedding_dim..(i+1) * embedding_dim] = 0  // out of bounds
/// ```
///
/// # Arguments
/// * `embeddings` - 2D embedding table pointer [vocab_size, embedding_dim]
/// * `indices` - 1D/ND flattened index tensor pointer (i64 values)
/// * `out` - Output pointer [num_indices, embedding_dim]
/// * `num_indices` - Total number of indices (product of indices.shape())
/// * `vocab_size` - Size of vocabulary (embeddings.shape()[0])
/// * `embedding_dim` - Dimension of each embedding vector (embeddings.shape()[1])
///
/// # Safety
/// - All pointers must be valid for the specified sizes
/// - `out` must have space for `num_indices * embedding_dim` elements
///
/// # Performance
/// - Memory-bound operation - optimized for sequential reads of embedding rows
/// - Uses memcpy for efficient row copying when possible
/// - For large batches, consider using parallel version with Rayon
#[inline]
pub unsafe fn embedding_lookup_kernel<T: Element>(
    embeddings: *const T,
    indices: *const i64,
    out: *mut T,
    num_indices: usize,
    vocab_size: usize,
    embedding_dim: usize,
) {
    if num_indices == 0 || embedding_dim == 0 {
        return;
    }

    let indices_slice = std::slice::from_raw_parts(indices, num_indices);

    for (i, &idx_val) in indices_slice.iter().enumerate() {
        let out_offset = i * embedding_dim;

        // Check bounds
        if idx_val < 0 || idx_val as usize >= vocab_size {
            // Out of bounds - fill with zeros
            let out_slice = std::slice::from_raw_parts_mut(out.add(out_offset), embedding_dim);
            for elem in out_slice {
                *elem = T::zero();
            }
            continue;
        }

        let src_offset = (idx_val as usize) * embedding_dim;

        // Copy the entire embedding row (contiguous memory copy)
        std::ptr::copy_nonoverlapping(
            embeddings.add(src_offset),
            out.add(out_offset),
            embedding_dim,
        );
    }
}
