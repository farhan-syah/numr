//! Normalization operation implementations for WebGPU.

pub(crate) mod fused_add;
pub(crate) mod norm;

pub(crate) use fused_add::{
    native_fused_add_layer_norm, native_fused_add_layer_norm_bwd, native_fused_add_rms_norm,
    native_fused_add_rms_norm_bwd,
};
pub(crate) use norm::{native_group_norm, native_layer_norm, native_rms_norm};
