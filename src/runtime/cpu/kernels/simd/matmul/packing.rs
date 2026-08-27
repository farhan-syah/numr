//! Matrix packing functions for microkernel consumption
//!
//! These functions reorder matrix data into a layout optimized for the
//! 6×NR microkernels. Packing improves cache utilization by ensuring
//! sequential memory access in the innermost loop.

use super::MR;

/// Generate pack_a function for a given type
macro_rules! define_pack_a {
    ($name:ident, $ty:ty) => {
        /// Pack A matrix panel for microkernel consumption
        ///
        /// Layout: For each MR-row block, for each k: MR consecutive elements
        /// `[a[0,0], a[1,0], ..., a[MR-1,0], a[0,1], a[1,1], ..., a[MR-1,1], ...]`
        ///
        /// # Safety
        /// - `a` must be valid for reading `mc * kc` elements with stride `lda`
        /// - `packed` must be valid for writing `(mc.div_ceil(MR) * MR) * kc` elements
        #[inline]
        pub unsafe fn $name(a: *const $ty, packed: *mut $ty, mc: usize, kc: usize, lda: usize) {
            let mut p = 0;
            for ir in (0..mc).step_by(MR) {
                let mr_actual = (mc - ir).min(MR);
                if mr_actual == MR {
                    // Full MR block - no padding needed
                    for k in 0..kc {
                        for i in 0..MR {
                            *packed.add(p) = *a.add((ir + i) * lda + k);
                            p += 1;
                        }
                    }
                } else {
                    // Partial block - pad with zeros
                    for k in 0..kc {
                        for i in 0..mr_actual {
                            *packed.add(p) = *a.add((ir + i) * lda + k);
                            p += 1;
                        }
                        for _ in mr_actual..MR {
                            *packed.add(p) = 0.0;
                            p += 1;
                        }
                    }
                }
            }
        }
    };
}

/// Generate pack_b function for a given type
macro_rules! define_pack_b {
    ($name:ident, $ty:ty) => {
        /// Pack B matrix panel for microkernel consumption
        ///
        /// Layout: For each NR-column block, for each k: NR consecutive elements.
        /// Uses bulk copy for full NR blocks since B is row-major.
        ///
        /// # Safety
        /// - `b` must be valid for reading `kc * nc` elements with stride `ldb`
        /// - `packed` must be valid for writing `(nc.div_ceil(NR) * NR) * kc` elements
        #[inline]
        pub unsafe fn $name<const NR: usize>(
            b: *const $ty,
            packed: *mut $ty,
            nc: usize,
            kc: usize,
            ldb: usize,
        ) {
            let mut p = 0;
            for jr in (0..nc).step_by(NR) {
                let nr_actual = (nc - jr).min(NR);
                if nr_actual == NR {
                    // Full NR block: B elements are contiguous in each row
                    for k in 0..kc {
                        std::ptr::copy_nonoverlapping(b.add(k * ldb + jr), packed.add(p), NR);
                        p += NR;
                    }
                } else {
                    // Partial (or half) block — pack CONTIGUOUSLY with stride
                    // `nr_actual`, NOT padded to NR. The consuming microkernels
                    // (microkernel_edge / the single-width half kernel) index this
                    // block with stride `nr_actual` (`b.add(kk * nr + j)`), so the
                    // packed stride MUST be nr_actual; zero-padding to NR here would
                    // make every kk>0 read into the previous row's pad → wrong dot
                    // product. The partial block is always the LAST jr-block, so its
                    // (smaller) size does not shift any later block's offset.
                    for k in 0..kc {
                        for j in 0..nr_actual {
                            *packed.add(p) = *b.add(k * ldb + jr + j);
                            p += 1;
                        }
                    }
                }
            }
        }
    };
}

/// Generate the transposed-source pack_b function for a given type
macro_rules! define_pack_b_t {
    ($name:ident, $ty:ty) => {
        /// Pack a B panel whose source is the transpose of a contiguous `[N, K]`
        /// buffer, without materializing the `[K, N]` matrix first.
        ///
        /// Logical element `B[p][j]` (contraction index `p`, output column `j`)
        /// lives at `b.add(j * ldb_t + p)`, where `ldb_t` is the contraction
        /// extent `K`. The panel produced is byte-identical to what the row-major
        /// packer produces from a materialized `[K, N]` buffer: same block order,
        /// same element order, same contiguous (unpadded) stride for a partial
        /// trailing block. Only the address each element is read from changes.
        ///
        /// # Why this exists
        ///
        /// A linear layer holds its weights as a contiguous `[N, K]` buffer and
        /// multiplies against the `[K, N]` view with strides `[1, K]`. Making that
        /// view contiguous copies the whole weight matrix on every call. A profiled
        /// VoxCPM2 decode moved ~50 GB through `copy_strided` over four generated
        /// patches, and a `perf record -e instructions` call graph put 41% of all
        /// program instructions under `Tensor::contiguous` on that path. Packing is
        /// a strided gather either way, so reading the `[N, K]` source directly
        /// costs nothing extra and removes the copy entirely.
        ///
        /// # Safety
        /// - `b` must be valid for reading element `j * ldb_t + p` for every
        ///   `p < kc` and `j < nc`
        /// - `packed` must be valid for writing `(nc.div_ceil(NR) * NR) * kc` elements
        #[inline]
        pub unsafe fn $name<const NR: usize>(
            b: *const $ty,
            packed: *mut $ty,
            nc: usize,
            kc: usize,
            ldb_t: usize,
        ) {
            let mut p = 0;
            for jr in (0..nc).step_by(NR) {
                let nr_actual = (nc - jr).min(NR);
                // No bulk-copy branch: with a transposed source the elements of one
                // k-row are `ldb_t` apart, so both the full and the partial block
                // gather element by element. The write order is the packer's, so
                // the resulting panel matches byte for byte.
                for k in 0..kc {
                    for j in 0..nr_actual {
                        *packed.add(p) = *b.add((jr + j) * ldb_t + k);
                        p += 1;
                    }
                }
            }
        }
    };
}

define_pack_a!(pack_a_f32, f32);
define_pack_a!(pack_a_f64, f64);
define_pack_b!(pack_b_f32, f32);
define_pack_b!(pack_b_f64, f64);
define_pack_b_t!(pack_b_t_f32, f32);
define_pack_b_t!(pack_b_t_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `[k_total, n_total]` row-major matrix and its `[n_total, k_total]`
    /// transpose, both holding the same logical matrix.
    fn matrix_pair(k_total: usize, n_total: usize) -> (Vec<f32>, Vec<f32>) {
        let row_major: Vec<f32> = (0..k_total * n_total)
            .map(|i| (i % 251) as f32 * 0.125 - 7.0)
            .collect();
        let mut transposed = vec![0.0f32; k_total * n_total];
        for p in 0..k_total {
            for j in 0..n_total {
                transposed[j * k_total + p] = row_major[p * n_total + j];
            }
        }
        (row_major, transposed)
    }

    /// The transposed packer must be byte-identical to the row-major packer:
    /// packing order and values are the correctness property, only the source
    /// address changes.
    fn assert_panels_identical<const NR: usize>(
        k_total: usize,
        n_total: usize,
        pc: usize,
        jc: usize,
        kc: usize,
        nc: usize,
    ) {
        let (row_major, transposed) = matrix_pair(k_total, n_total);
        let packed_len = nc.div_ceil(NR) * NR * kc;
        let mut from_row_major = vec![f32::NAN; packed_len];
        let mut from_transposed = vec![f32::NAN; packed_len];

        unsafe {
            pack_b_f32::<NR>(
                row_major.as_ptr().add(pc * n_total + jc),
                from_row_major.as_mut_ptr(),
                nc,
                kc,
                n_total,
            );
            pack_b_t_f32::<NR>(
                transposed.as_ptr().add(jc * k_total + pc),
                from_transposed.as_mut_ptr(),
                nc,
                kc,
                k_total,
            );
        }

        let lhs: Vec<u32> = from_row_major.iter().map(|v| v.to_bits()).collect();
        let rhs: Vec<u32> = from_transposed.iter().map(|v| v.to_bits()).collect();
        assert_eq!(lhs, rhs, "packed panels differ (NR={NR}, kc={kc}, nc={nc})");
    }

    #[test]
    fn test_pack_b_t_matches_pack_b_full_blocks() {
        assert_panels_identical::<16>(64, 64, 0, 0, 64, 64);
        assert_panels_identical::<8>(64, 64, 0, 0, 64, 64);
    }

    #[test]
    fn test_pack_b_t_matches_pack_b_partial_block() {
        // nc is not a multiple of NR: exercises the contiguous partial-block stride.
        assert_panels_identical::<16>(37, 53, 0, 0, 37, 53);
        assert_panels_identical::<8>(37, 53, 0, 0, 37, 53);
    }

    #[test]
    fn test_pack_b_t_matches_pack_b_offset_panel() {
        // A panel taken from the middle of a larger matrix, as the tiled loop does.
        assert_panels_identical::<16>(100, 130, 24, 17, 40, 61);
    }
}
