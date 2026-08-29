//! WGSL source table for the index, gather and scatter kernels.
//!
//! One `(op, dtype)` lookup returns the shader source, its module cache key and
//! the entry point to dispatch. Several ops share a module and differ only by
//! entry point (`copy` rides the scatter module, `masked_count` and
//! `masked_prefix_sum` ride the masked_select module), which is why the table
//! lives in one place rather than beside each launcher.

use crate::dtype::DType;
use crate::error::{Error, Result};

const INDEX_SELECT_SHADER_F32: &str = include_str!("../index_select_f32.wgsl");
const INDEX_SELECT_SHADER_I32: &str = include_str!("../index_select_i32.wgsl");
const INDEX_SELECT_SHADER_U32: &str = include_str!("../index_select_u32.wgsl");

const INDEX_PUT_SHADER_F32: &str = include_str!("../index_put_f32.wgsl");
const INDEX_PUT_SHADER_I32: &str = include_str!("../index_put_i32.wgsl");
const INDEX_PUT_SHADER_U32: &str = include_str!("../index_put_u32.wgsl");

const GATHER_SHADER_F32: &str = include_str!("../gather_f32.wgsl");
const GATHER_SHADER_I32: &str = include_str!("../gather_i32.wgsl");
const GATHER_SHADER_U32: &str = include_str!("../gather_u32.wgsl");

const SCATTER_SHADER_F32: &str = include_str!("../scatter_f32.wgsl");
const SCATTER_SHADER_I32: &str = include_str!("../scatter_i32.wgsl");
const SCATTER_SHADER_U32: &str = include_str!("../scatter_u32.wgsl");

const MASKED_FILL_SHADER_F32: &str = include_str!("../masked_fill_f32.wgsl");
const MASKED_FILL_SHADER_I32: &str = include_str!("../masked_fill_i32.wgsl");
const MASKED_FILL_SHADER_U32: &str = include_str!("../masked_fill_u32.wgsl");

const MASKED_SELECT_SHADER_F32: &str = include_str!("../masked_select_f32.wgsl");
const MASKED_SELECT_SHADER_I32: &str = include_str!("../masked_select_i32.wgsl");
const MASKED_SELECT_SHADER_U32: &str = include_str!("../masked_select_u32.wgsl");

const EMBEDDING_LOOKUP_SHADER_F32: &str = include_str!("../embedding_lookup_f32.wgsl");
const EMBEDDING_LOOKUP_SHADER_I32: &str = include_str!("../embedding_lookup_i32.wgsl");
const EMBEDDING_LOOKUP_SHADER_U32: &str = include_str!("../embedding_lookup_u32.wgsl");

const GATHER_ND_SHADER_F32: &str = include_str!("../gather_nd_f32.wgsl");
const GATHER_ND_SHADER_I32: &str = include_str!("../gather_nd_i32.wgsl");
const GATHER_ND_SHADER_U32: &str = include_str!("../gather_nd_u32.wgsl");

const SCATTER_REDUCE_SUM_SHADER_F32: &str = include_str!("../scatter_reduce_sum_f32.wgsl");
const SCATTER_REDUCE_SUM_SHADER_I32: &str = include_str!("../scatter_reduce_sum_i32.wgsl");
const SCATTER_REDUCE_SUM_SHADER_U32: &str = include_str!("../scatter_reduce_sum_u32.wgsl");

const SCATTER_REDUCE_MAX_SHADER_F32: &str = include_str!("../scatter_reduce_max_f32.wgsl");
const SCATTER_REDUCE_MAX_SHADER_I32: &str = include_str!("../scatter_reduce_max_i32.wgsl");
const SCATTER_REDUCE_MAX_SHADER_U32: &str = include_str!("../scatter_reduce_max_u32.wgsl");

const SCATTER_REDUCE_MIN_SHADER_F32: &str = include_str!("../scatter_reduce_min_f32.wgsl");
const SCATTER_REDUCE_MIN_SHADER_I32: &str = include_str!("../scatter_reduce_min_i32.wgsl");
const SCATTER_REDUCE_MIN_SHADER_U32: &str = include_str!("../scatter_reduce_min_u32.wgsl");

const SCATTER_REDUCE_PROD_SHADER_F32: &str = include_str!("../scatter_reduce_prod_f32.wgsl");

// The integer product saturates, so it needs the shared saturating helpers.
// WGSL has no include and no forward declarations, so the order is load-bearing.
const SCATTER_REDUCE_PROD_SHADER_I32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../scatter_reduce_prod_i32.wgsl"),
);
const SCATTER_REDUCE_PROD_SHADER_U32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../scatter_reduce_prod_u32.wgsl"),
);

const SCATTER_REDUCE_COUNT_SHADER: &str = include_str!("../scatter_reduce_count.wgsl");
const SCATTER_REDUCE_MEAN_DIV_SHADER_F32: &str =
    include_str!("../scatter_reduce_mean_div_f32.wgsl");

const SLICE_ASSIGN_SHADER_F32: &str = include_str!("../slice_assign_f32.wgsl");
const SLICE_ASSIGN_SHADER_I32: &str = include_str!("../slice_assign_i32.wgsl");
const SLICE_ASSIGN_SHADER_U32: &str = include_str!("../slice_assign_u32.wgsl");

const GATHER_2D_SHADER_F32: &str = include_str!("../gather_2d_f32.wgsl");
const GATHER_2D_SHADER_I32: &str = include_str!("../gather_2d_i32.wgsl");
const GATHER_2D_SHADER_U32: &str = include_str!("../gather_2d_u32.wgsl");

/// Returns (shader, module_key, entry_point) for standard index/scatter/gather ops.
pub(super) fn shader_info(
    op: &'static str,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    Ok(match (op, dtype) {
        ("index_select", DType::F32) => (
            INDEX_SELECT_SHADER_F32,
            "index_select_f32",
            "index_select_f32",
        ),
        ("index_select", DType::I32) => (
            INDEX_SELECT_SHADER_I32,
            "index_select_i32",
            "index_select_i32",
        ),
        ("index_select", DType::U32) => (
            INDEX_SELECT_SHADER_U32,
            "index_select_u32",
            "index_select_u32",
        ),
        ("index_put", DType::F32) => (INDEX_PUT_SHADER_F32, "index_put_f32", "index_put_f32"),
        ("index_put", DType::I32) => (INDEX_PUT_SHADER_I32, "index_put_i32", "index_put_i32"),
        ("index_put", DType::U32) => (INDEX_PUT_SHADER_U32, "index_put_u32", "index_put_u32"),
        ("gather", DType::F32) => (GATHER_SHADER_F32, "gather_f32", "gather_f32"),
        ("gather", DType::I32) => (GATHER_SHADER_I32, "gather_i32", "gather_i32"),
        ("gather", DType::U32) => (GATHER_SHADER_U32, "gather_u32", "gather_u32"),
        ("scatter", DType::F32) => (SCATTER_SHADER_F32, "scatter_f32", "scatter_f32"),
        ("scatter", DType::I32) => (SCATTER_SHADER_I32, "scatter_i32", "scatter_i32"),
        ("scatter", DType::U32) => (SCATTER_SHADER_U32, "scatter_u32", "scatter_u32"),
        // copy shares the scatter shader module but uses a different entry point
        ("copy", DType::F32) => (SCATTER_SHADER_F32, "scatter_f32", "copy_f32"),
        ("copy", DType::I32) => (SCATTER_SHADER_I32, "scatter_i32", "copy_i32"),
        ("copy", DType::U32) => (SCATTER_SHADER_U32, "scatter_u32", "copy_u32"),
        ("masked_fill", DType::F32) => {
            (MASKED_FILL_SHADER_F32, "masked_fill_f32", "masked_fill_f32")
        }
        ("masked_fill", DType::I32) => {
            (MASKED_FILL_SHADER_I32, "masked_fill_i32", "masked_fill_i32")
        }
        ("masked_fill", DType::U32) => {
            (MASKED_FILL_SHADER_U32, "masked_fill_u32", "masked_fill_u32")
        }
        ("masked_select", DType::F32) => (
            MASKED_SELECT_SHADER_F32,
            "masked_select_f32",
            "masked_select_f32",
        ),
        ("masked_select", DType::I32) => (
            MASKED_SELECT_SHADER_I32,
            "masked_select_i32",
            "masked_select_i32",
        ),
        ("masked_select", DType::U32) => (
            MASKED_SELECT_SHADER_U32,
            "masked_select_u32",
            "masked_select_u32",
        ),
        // masked_count and masked_prefix_sum share the masked_select shader module
        ("masked_count", DType::F32) => (
            MASKED_SELECT_SHADER_F32,
            "masked_select_f32",
            "masked_count",
        ),
        ("masked_count", DType::I32) => (
            MASKED_SELECT_SHADER_I32,
            "masked_select_i32",
            "masked_count",
        ),
        ("masked_count", DType::U32) => (
            MASKED_SELECT_SHADER_U32,
            "masked_select_u32",
            "masked_count",
        ),
        ("masked_prefix_sum", DType::F32) => (
            MASKED_SELECT_SHADER_F32,
            "masked_select_f32",
            "masked_prefix_sum",
        ),
        ("masked_prefix_sum", DType::I32) => (
            MASKED_SELECT_SHADER_I32,
            "masked_select_i32",
            "masked_prefix_sum",
        ),
        ("masked_prefix_sum", DType::U32) => (
            MASKED_SELECT_SHADER_U32,
            "masked_select_u32",
            "masked_prefix_sum",
        ),
        ("embedding_lookup", DType::F32) => (
            EMBEDDING_LOOKUP_SHADER_F32,
            "embedding_lookup_f32",
            "embedding_lookup_f32",
        ),
        ("embedding_lookup", DType::I32) => (
            EMBEDDING_LOOKUP_SHADER_I32,
            "embedding_lookup_i32",
            "embedding_lookup_i32",
        ),
        ("embedding_lookup", DType::U32) => (
            EMBEDDING_LOOKUP_SHADER_U32,
            "embedding_lookup_u32",
            "embedding_lookup_u32",
        ),
        ("gather_nd", DType::F32) => (GATHER_ND_SHADER_F32, "gather_nd_f32", "gather_nd_f32"),
        ("gather_nd", DType::I32) => (GATHER_ND_SHADER_I32, "gather_nd_i32", "gather_nd_i32"),
        ("gather_nd", DType::U32) => (GATHER_ND_SHADER_U32, "gather_nd_u32", "gather_nd_u32"),
        ("scatter_reduce_sum", DType::F32) => (
            SCATTER_REDUCE_SUM_SHADER_F32,
            "scatter_reduce_sum_f32",
            "scatter_reduce_sum_f32",
        ),
        ("scatter_reduce_sum", DType::I32) => (
            SCATTER_REDUCE_SUM_SHADER_I32,
            "scatter_reduce_sum_i32",
            "scatter_reduce_sum_i32",
        ),
        ("scatter_reduce_sum", DType::U32) => (
            SCATTER_REDUCE_SUM_SHADER_U32,
            "scatter_reduce_sum_u32",
            "scatter_reduce_sum_u32",
        ),
        ("scatter_reduce_max", DType::F32) => (
            SCATTER_REDUCE_MAX_SHADER_F32,
            "scatter_reduce_max_f32",
            "scatter_reduce_max_f32",
        ),
        ("scatter_reduce_max", DType::I32) => (
            SCATTER_REDUCE_MAX_SHADER_I32,
            "scatter_reduce_max_i32",
            "scatter_reduce_max_i32",
        ),
        ("scatter_reduce_max", DType::U32) => (
            SCATTER_REDUCE_MAX_SHADER_U32,
            "scatter_reduce_max_u32",
            "scatter_reduce_max_u32",
        ),
        ("scatter_reduce_min", DType::F32) => (
            SCATTER_REDUCE_MIN_SHADER_F32,
            "scatter_reduce_min_f32",
            "scatter_reduce_min_f32",
        ),
        ("scatter_reduce_min", DType::I32) => (
            SCATTER_REDUCE_MIN_SHADER_I32,
            "scatter_reduce_min_i32",
            "scatter_reduce_min_i32",
        ),
        ("scatter_reduce_min", DType::U32) => (
            SCATTER_REDUCE_MIN_SHADER_U32,
            "scatter_reduce_min_u32",
            "scatter_reduce_min_u32",
        ),
        ("scatter_reduce_prod", DType::F32) => (
            SCATTER_REDUCE_PROD_SHADER_F32,
            "scatter_reduce_prod_f32",
            "scatter_reduce_prod_f32",
        ),
        ("scatter_reduce_prod", DType::I32) => (
            SCATTER_REDUCE_PROD_SHADER_I32,
            "scatter_reduce_prod_i32",
            "scatter_reduce_prod_i32",
        ),
        ("scatter_reduce_prod", DType::U32) => (
            SCATTER_REDUCE_PROD_SHADER_U32,
            "scatter_reduce_prod_u32",
            "scatter_reduce_prod_u32",
        ),
        // One count kernel for every value dtype: it reads only the index tensor.
        ("scatter_reduce_count", DType::F32 | DType::I32 | DType::U32) => (
            SCATTER_REDUCE_COUNT_SHADER,
            "scatter_reduce_count",
            "scatter_reduce_count",
        ),
        ("scatter_reduce_mean_div", DType::F32) => (
            SCATTER_REDUCE_MEAN_DIV_SHADER_F32,
            "scatter_reduce_mean_div_f32",
            "scatter_reduce_mean_div_f32",
        ),
        ("slice_assign", DType::F32) => (
            SLICE_ASSIGN_SHADER_F32,
            "slice_assign_f32",
            "slice_assign_f32",
        ),
        ("slice_assign", DType::I32) => (
            SLICE_ASSIGN_SHADER_I32,
            "slice_assign_i32",
            "slice_assign_i32",
        ),
        ("slice_assign", DType::U32) => (
            SLICE_ASSIGN_SHADER_U32,
            "slice_assign_u32",
            "slice_assign_u32",
        ),
        ("gather_2d", DType::F32) => (GATHER_2D_SHADER_F32, "gather_2d_f32", "gather_2d_f32"),
        ("gather_2d", DType::I32) => (GATHER_2D_SHADER_I32, "gather_2d_i32", "gather_2d_i32"),
        ("gather_2d", DType::U32) => (GATHER_2D_SHADER_U32, "gather_2d_u32", "gather_2d_u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    })
}
