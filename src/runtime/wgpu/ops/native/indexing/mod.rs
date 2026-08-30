//! Indexing operation implementations for WebGPU.

mod gather;
mod index_put;
mod index_select;
mod scatter;
mod slice_assign;

pub(crate) use gather::native_gather;
pub(crate) use index_put::native_index_put;
pub(crate) use index_select::native_index_select;
pub(crate) use scatter::native_scatter;
pub(crate) use slice_assign::native_slice_assign;
