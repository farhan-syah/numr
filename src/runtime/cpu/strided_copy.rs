//! Strided-to-contiguous copy for the CPU runtime.
//!
//! `Tensor::contiguous()` materializes every permute, reshape, cast and
//! quantized matmul input, so this is one of the hottest code paths in the
//! whole stack. A naive per-element loop recomputes the full N-dimensional
//! source offset for every element and issues a `memcpy` call per element,
//! which for f32 degenerates into a 4-byte `memcpy` — measured at roughly
//! three quarters of all retired instructions in a decode loop.
//!
//! The copy is therefore split into three tiers, from most to least
//! specialized. The destination is always freshly allocated row-major storage,
//! so it can never alias the source and `copy_nonoverlapping` is sound.
//!
//! | Tier | Condition | Cost |
//! | ---- | --------- | ---- |
//! | 1 | the whole layout is row-major contiguous | one `memcpy` |
//! | 2 | a trailing block of dimensions is contiguous | one `memcpy` per outer index |
//! | 3 | anything else | one typed load/store per element |
//!
//! # Why the fast paths are safe
//!
//! Tier 1 and tier 2 are guarded by [`contiguous_suffix_len`], which only ever
//! accepts a dimension whose stride is exactly the product of the shapes inside
//! it. That expected stride is always `>= 1`, so:
//!
//! - **Negative strides** (a reversed view) can never match a positive expected
//!   stride and always fall through to tier 3, which indexes them correctly.
//! - **Stride 0** (a broadcast dimension, where one source element is read
//!   repeatedly) can never match either, so a broadcast is never mistaken for a
//!   contiguous run. Accepting one would silently read `shape[d]` distinct
//!   elements where the layout asks for the same element `shape[d]` times.
//!
//! A dimension of extent 1 is accepted regardless of its stride, because its
//! index is always 0 and the stride is therefore never applied. This is what
//! lets a squeezed or unsqueezed view still take tier 1.

/// Number of trailing dimensions that form one internally contiguous block.
///
/// Walks from the innermost dimension outward, tracking the stride a
/// row-major layout would have there. Returns 0 when even the innermost
/// dimension is not unit-strided.
///
/// Callers must have already rejected `numel == 0`; a zero-extent dimension
/// would make the expected stride collapse to 0 and could then match a
/// broadcast stride.
fn contiguous_suffix_len(shape: &[usize], strides: &[isize]) -> usize {
    let mut expected: isize = 1;
    let mut count = 0;

    for d in (0..shape.len()).rev() {
        if shape[d] == 1 {
            // Index is always 0 here, so the stride is never applied and the
            // run of contiguous memory is unbroken.
            count += 1;
            continue;
        }
        if strides[d] != expected {
            break;
        }
        count += 1;
        expected *= shape[d] as isize;
    }

    count
}

/// Copy a strided source region into freshly allocated row-major destination
/// storage.
///
/// `strides` are in elements and may be negative or zero. `src` must already
/// have the source byte offset applied. `dst` must have room for
/// `shape.iter().product::<usize>() * elem_size` bytes.
///
/// # Safety
///
/// Every offset reachable from `shape`/`strides` must be in bounds of the
/// source allocation, `dst` must be valid for the full destination length, and
/// the two regions must not overlap.
pub(super) unsafe fn copy_strided_impl(
    src: *const u8,
    dst: *mut u8,
    shape: &[usize],
    strides: &[isize],
    elem_size: usize,
) {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    let suffix = contiguous_suffix_len(shape, strides);

    // Tier 1: the entire view is row-major. This also catches a view that is
    // contiguous at a NON-ZERO offset — every `narrow()` result, used here for
    // KV-cache slices and row extraction — because the offset is already folded
    // into `src`.
    if suffix == ndim {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, numel * elem_size) };
        return;
    }

    // Tier 2: the innermost dimensions form a contiguous block. A `[B, H, S, D]`
    // permute typically leaves `D` or `S * D` intact, so this replaces one
    // memcpy per element with one per outer index.
    let block_elems: usize = shape[ndim - suffix..].iter().product();
    if suffix > 0 && block_elems > 1 {
        unsafe { copy_blocks(src, dst, shape, strides, elem_size, suffix, block_elems) };
        return;
    }

    // Tier 3: fully general. Specialized by element width so the common f32/f16
    // cases become a plain load/store instead of a `memcpy` call.
    match elem_size {
        1 => unsafe { copy_elements::<u8>(src, dst, shape, strides, numel) },
        2 => unsafe { copy_elements::<u16>(src, dst, shape, strides, numel) },
        4 => unsafe { copy_elements::<u32>(src, dst, shape, strides, numel) },
        8 => unsafe { copy_elements::<u64>(src, dst, shape, strides, numel) },
        _ => unsafe { copy_elements_bytes(src, dst, shape, strides, numel, elem_size) },
    }
}

/// Advance a row-major odometer over `shape[..dims]`, carrying the source
/// element offset incrementally.
///
/// Incrementing dimension `d` adds `strides[d]`; wrapping it back to 0 subtracts
/// `strides[d] * (shape[d] - 1)`, undoing exactly what the completed sweep
/// added. The full offset is therefore never recomputed from scratch.
#[inline(always)]
fn advance(idx: &mut [usize], off: &mut isize, shape: &[usize], strides: &[isize], dims: usize) {
    for d in (0..dims).rev() {
        idx[d] += 1;
        if idx[d] < shape[d] {
            *off += strides[d];
            return;
        }
        idx[d] = 0;
        *off -= strides[d] * (shape[d] as isize - 1);
    }
}

/// Tier 2 driver: one `memcpy` of `block_elems` per outer index.
///
/// # Safety
///
/// See [`copy_strided_impl`]. `suffix` must be a valid contiguous suffix length
/// as reported by [`contiguous_suffix_len`].
unsafe fn copy_blocks(
    src: *const u8,
    dst: *mut u8,
    shape: &[usize],
    strides: &[isize],
    elem_size: usize,
    suffix: usize,
    block_elems: usize,
) {
    let outer_dims = shape.len() - suffix;
    let outer_count: usize = shape[..outer_dims].iter().product();
    let block_bytes = block_elems * elem_size;

    let mut idx = vec![0usize; outer_dims];
    let mut off: isize = 0;

    for b in 0..outer_count {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.offset(off * elem_size as isize),
                dst.add(b * block_bytes),
                block_bytes,
            );
        }
        advance(&mut idx, &mut off, shape, strides, outer_dims);
    }
}

/// Tier 3 driver for a known element width.
///
/// Reads and writes are unaligned: the source offset can land on any element
/// boundary and `T` is only a width stand-in, never the real dtype. On the
/// targets numr supports this still lowers to a single load and store.
///
/// # Safety
///
/// See [`copy_strided_impl`]. `size_of::<T>()` must equal the element size.
unsafe fn copy_elements<T: Copy>(
    src: *const u8,
    dst: *mut u8,
    shape: &[usize],
    strides: &[isize],
    numel: usize,
) {
    let src = src as *const T;
    let dst = dst as *mut T;
    let ndim = shape.len();

    let mut idx = vec![0usize; ndim];
    let mut off: isize = 0;

    for i in 0..numel {
        unsafe {
            let v = std::ptr::read_unaligned(src.offset(off));
            std::ptr::write_unaligned(dst.add(i), v);
        }
        advance(&mut idx, &mut off, shape, strides, ndim);
    }
}

/// Tier 3 driver for element widths with no integer stand-in.
///
/// No dtype in numr currently lands here; it exists so an odd width stays
/// correct rather than unreachable.
///
/// # Safety
///
/// See [`copy_strided_impl`].
unsafe fn copy_elements_bytes(
    src: *const u8,
    dst: *mut u8,
    shape: &[usize],
    strides: &[isize],
    numel: usize,
    elem_size: usize,
) {
    let ndim = shape.len();
    let mut idx = vec![0usize; ndim];
    let mut off: isize = 0;

    for i in 0..numel {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.offset(off * elem_size as isize),
                dst.add(i * elem_size),
                elem_size,
            );
        }
        advance(&mut idx, &mut off, shape, strides, ndim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The original element-by-element implementation, kept verbatim as the
    /// correctness oracle. A fast path that takes the wrong branch produces
    /// different bytes than this and fails loudly.
    unsafe fn naive(
        src: *const u8,
        dst: *mut u8,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
    ) {
        let numel: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];

        for dst_offset in 0..numel {
            let mut src_elem_offset: isize = 0;
            for (i, &idx) in indices.iter().enumerate() {
                src_elem_offset += (idx as isize) * strides[i];
            }

            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.offset(src_elem_offset * elem_size as isize),
                    dst.add(dst_offset * elem_size),
                    elem_size,
                );
            }

            for dim in (0..shape.len()).rev() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    /// Run both implementations over the same source region and compare bytes.
    fn check(
        name: &str,
        shape: &[usize],
        strides: &[isize],
        offset_elems: usize,
        elem_size: usize,
        src_elems: usize,
    ) {
        let src: Vec<u8> = (0..src_elems * elem_size)
            .map(|i| (i % 251) as u8)
            .collect();
        let numel: usize = shape.iter().product();

        let mut expected = vec![0u8; numel * elem_size];
        let mut actual = vec![0u8; numel * elem_size];

        unsafe {
            let base = src.as_ptr().add(offset_elems * elem_size);
            naive(base, expected.as_mut_ptr(), shape, strides, elem_size);
            copy_strided_impl(base, actual.as_mut_ptr(), shape, strides, elem_size);
        }

        assert_eq!(actual, expected, "{name} (elem_size={elem_size})");
    }

    #[test]
    fn test_fully_contiguous() {
        check("contiguous", &[2, 3, 4], &[12, 4, 1], 0, 4, 24);
    }

    #[test]
    fn test_contiguous_at_nonzero_offset() {
        // The narrow() case: contiguous, but starting partway into the storage.
        check("narrowed", &[2, 3, 4], &[12, 4, 1], 7, 4, 31);
    }

    #[test]
    fn test_2d_transpose() {
        // A [4, 3] row-major tensor viewed as its [3, 4] transpose.
        check("transpose", &[3, 4], &[1, 3], 0, 4, 12);
    }

    #[test]
    fn test_4d_permute() {
        // [2, 3, 4, 5] row-major (strides [60, 20, 5, 1]) permuted to
        // [2, 4, 3, 5] — the attention layout case.
        check("permute", &[2, 4, 3, 5], &[60, 5, 20, 1], 0, 4, 120);
    }

    #[test]
    fn test_negative_stride() {
        // A reversed view: start at the last element and walk backwards.
        check("reversed_1d", &[6], &[-1], 5, 4, 6);
        check("reversed_2d_rows", &[3, 4], &[-4, 1], 8, 4, 12);
    }

    #[test]
    fn test_broadcast() {
        // Stride 0 must never be read as a contiguous run.
        check("broadcast_outer", &[3, 4], &[0, 1], 0, 4, 4);
        check("broadcast_inner", &[3, 4], &[1, 0], 0, 4, 3);
        check("broadcast_middle", &[2, 3, 4], &[4, 0, 1], 0, 4, 8);
        check("broadcast_all", &[2, 3], &[0, 0], 0, 4, 1);
    }

    #[test]
    fn test_contiguous_inner_block() {
        // The tier 2 path: gapped outer dimension, contiguous [3, 4] block.
        check("gapped_outer", &[2, 3, 4], &[100, 4, 1], 0, 4, 300);
        // Only the innermost dimension is contiguous.
        check("gapped_middle", &[2, 3, 4], &[60, 9, 1], 0, 4, 120);
    }

    #[test]
    fn test_unit_extent_dimensions() {
        // Extent-1 dimensions carry arbitrary strides that are never applied.
        check("unit_middle", &[4, 1, 3], &[3, 999, 1], 0, 4, 12);
        check("unit_broadcast", &[4, 1, 3], &[3, 0, 1], 0, 4, 12);
    }

    #[test]
    fn test_all_element_sizes() {
        for elem_size in [1usize, 2, 4, 8] {
            check("permute", &[2, 4, 3, 5], &[60, 5, 20, 1], 0, elem_size, 120);
            check("contiguous", &[2, 3, 4], &[12, 4, 1], 3, elem_size, 27);
            check("transpose", &[3, 4], &[1, 3], 0, elem_size, 12);
            check("reversed", &[6], &[-1], 5, elem_size, 6);
            check("broadcast", &[3, 4], &[0, 1], 0, elem_size, 4);
            check("blocks", &[2, 3, 4], &[100, 4, 1], 0, elem_size, 300);
        }
    }

    #[test]
    fn test_suffix_detection() {
        assert_eq!(contiguous_suffix_len(&[2, 3, 4], &[12, 4, 1]), 3);
        assert_eq!(contiguous_suffix_len(&[2, 3, 4], &[100, 4, 1]), 2);
        assert_eq!(contiguous_suffix_len(&[2, 3, 4], &[60, 9, 1]), 1);
        assert_eq!(contiguous_suffix_len(&[3, 4], &[1, 3]), 0);
        // Broadcast and negative strides must never be reported as contiguous.
        assert_eq!(contiguous_suffix_len(&[3, 4], &[0, 1]), 1);
        assert_eq!(contiguous_suffix_len(&[3, 4], &[1, 0]), 0);
        assert_eq!(contiguous_suffix_len(&[6], &[-1]), 0);
    }
}
