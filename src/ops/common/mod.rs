//! Common utilities and validation logic shared across operation backends.

pub mod complex_validation;
pub mod group_norm_validation;
pub mod quasirandom;

pub use complex_validation::{validate_complex_real_inputs, validate_make_complex_inputs};
#[cfg(feature = "wgpu")]
pub use complex_validation::{
    validate_complex_real_inputs_f32_only, validate_make_complex_inputs_f32_only,
};
pub use group_norm_validation::group_norm_channels_per_group;
