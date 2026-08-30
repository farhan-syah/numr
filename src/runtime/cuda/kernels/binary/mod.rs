//! Binary operation CUDA kernels: element-wise and broadcasting launchers.
//!
//! - `launchers` - kernel launch entry points (plain and broadcast element-wise ops)
//! - `broadcast_strides` - stride and magic-divisor helpers backing the broadcast launcher

mod broadcast_strides;
mod launchers;

pub(crate) use broadcast_strides::compute_broadcast_strides;
pub use broadcast_strides::{
    MAX_BROADCAST_DIMS, compute_magic_divisor, detect_fast_trailing_broadcast,
};
pub use launchers::{
    launch_binary_op, launch_broadcast_binary_op, launch_logical_and_op, launch_logical_or_op,
    launch_logical_xor_op,
};
