//! Shape, creation and sampling WGSL kernel launchers.
//!
//! All operations run entirely on GPU with no CPU fallback. split and chunk
//! need no kernel: they are zero-copy views built from narrow.

mod creation;
mod data_movement;
mod multinomial;
mod random;
mod shader_registry;

pub use creation::{launch_arange, launch_eye, launch_linspace};
pub use data_movement::{launch_cat_copy, launch_pad, launch_repeat, launch_roll};
pub use multinomial::{
    launch_multinomial_with_replacement, launch_multinomial_without_replacement,
};
pub use random::{launch_rand, launch_randint, launch_randn};
