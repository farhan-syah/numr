//! Shared `group_norm` group-count validation.
//!
//! Every backend splits `channels` into `num_groups` groups and then divides by
//! `num_groups`. `channels.is_multiple_of(num_groups)` is NOT a sufficient guard
//! on its own: `usize::is_multiple_of(0)` is `true` when `channels == 0`, so a
//! zero group count slips past it and the division that follows panics. The
//! check lives here so CPU, CUDA, WebGPU and the autograd backward all reject
//! the same inputs with the same error instead of each keeping its own copy.

use crate::error::{Error, Result};

/// Channels per group for `group_norm`, or an error naming both values.
///
/// Rejects `num_groups == 0` before dividing, and rejects a `channels` that the
/// group count does not divide evenly.
pub fn group_norm_channels_per_group(channels: usize, num_groups: usize) -> Result<usize> {
    // Order matters: `channels.is_multiple_of(0)` answers `true` for
    // `channels == 0`, so the zero check must come first or the divide below
    // panics. Do not fold these two branches together.
    if num_groups == 0 {
        return Err(Error::InvalidArgument {
            arg: "num_groups",
            reason: format!(
                "group_norm requires num_groups > 0, got num_groups {num_groups} \
                 with channels {channels}"
            ),
        });
    }

    if !channels.is_multiple_of(num_groups) {
        return Err(Error::InvalidArgument {
            arg: "num_groups",
            reason: format!(
                "group_norm requires channels divisible by num_groups, \
                 got channels {channels} and num_groups {num_groups}"
            ),
        });
    }

    Ok(channels / num_groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_groups_is_rejected_not_a_panic() {
        // `0usize.is_multiple_of(0)` is true, which is exactly how a zero group
        // count used to reach the division.
        assert!(0usize.is_multiple_of(0));
        for channels in [0usize, 1, 6] {
            let err = group_norm_channels_per_group(channels, 0)
                .expect_err("num_groups 0 must be rejected");
            match err {
                Error::InvalidArgument { arg, reason } => {
                    assert_eq!(arg, "num_groups");
                    assert!(reason.contains("group_norm"), "reason: {reason}");
                    assert!(reason.contains(&channels.to_string()), "reason: {reason}");
                }
                other => panic!("want InvalidArgument, got {other:?}"),
            }
        }
    }

    #[test]
    fn uneven_split_is_rejected() {
        let err =
            group_norm_channels_per_group(6, 4).expect_err("6 channels over 4 groups is uneven");
        assert!(matches!(err, Error::InvalidArgument { .. }));
    }

    #[test]
    fn even_split_returns_channels_per_group() {
        assert_eq!(group_norm_channels_per_group(6, 3).expect("6 / 3"), 2);
        assert_eq!(group_norm_channels_per_group(0, 3).expect("0 / 3"), 0);
        assert_eq!(group_norm_channels_per_group(6, 6).expect("6 / 6"), 1);
    }
}
