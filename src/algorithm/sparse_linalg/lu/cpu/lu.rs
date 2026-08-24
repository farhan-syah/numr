//! CPU implementation of sparse LU factorization
//!
//! Gilbert-Peierls left-looking algorithm with partial pivoting.

use crate::algorithm::sparse_linalg::lu::types::{
    LuFactors, LuMetrics, LuOptions, LuSymbolic, LuSymbolicSimple, LuWorkspace,
};
use crate::algorithm::sparse_linalg::traits::validate_square_sparse;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cpu::kernels::sparse as kernels;
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// Sparse LU factorization with full symbolic information (CPU)
///
/// Uses Gilbert-Peierls left-looking algorithm with partial pivoting.
pub fn sparse_lu_cpu<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    symbolic: &LuSymbolic,
    options: &LuOptions,
) -> Result<LuFactors<R>> {
    let (factors, _metrics) = sparse_lu_cpu_with_metrics(a, symbolic, options)?;
    Ok(factors)
}

/// Sparse LU factorization with metrics (CPU)
pub fn sparse_lu_cpu_with_metrics<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    symbolic: &LuSymbolic,
    options: &LuOptions,
) -> Result<(LuFactors<R>, LuMetrics)> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    // Extract CSC components
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
    let row_indices: Vec<i64> = a.row_indices().to_vec();
    let values: Vec<f64> = extract_values_f64(a)?;

    // Run factorization
    let result = gilbert_peierls_lu(n, &col_ptrs, &row_indices, &values, symbolic, options)?;

    // Create output tensors
    let device = a.values().device();
    let (l, u) = create_lu_tensors::<R>(
        n,
        &result.l_col_ptrs,
        &result.l_row_indices,
        &result.l_values,
        &result.u_col_ptrs,
        &result.u_row_indices,
        &result.u_values,
        dtype,
        device,
    )?;

    let factors = LuFactors {
        l,
        u,
        row_perm: result.row_perm,
        row_perm_inv: result.row_perm_inv,
    };

    let metrics = LuMetrics {
        original_nnz: values.len(),
        l_nnz: result.l_values.len(),
        u_nnz: result.u_values.len(),
        fill_ratio: (result.l_values.len() + result.u_values.len()) as f64 / values.len() as f64,
        small_pivots: result.small_pivots,
        row_swaps: result.row_swaps,
        pivot_growth: result.pivot_growth,
    };

    Ok((factors, metrics))
}

/// Sparse LU factorization with workspace reuse (CPU)
///
/// This is the allocation-free variant for repeated factorizations with the same
/// sparsity pattern. Ideal for Newton iterations in ODE/DAE solvers.
///
/// # Arguments
///
/// * `a` - Input matrix in CSC format
/// * `symbolic` - Precomputed symbolic factorization
/// * `options` - Factorization options
/// * `workspace` - Pre-allocated workspace buffers (will be cleared and reused)
///
/// # Errors
///
/// Returns error if:
/// - Matrix dimensions don't match symbolic structure
/// - Workspace dimension doesn't match matrix
/// - Zero pivot encountered (unless diagonal shift is enabled)
pub fn sparse_lu_cpu_with_workspace<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    symbolic: &LuSymbolic,
    options: &LuOptions,
    workspace: &mut LuWorkspace,
) -> Result<LuFactors<R>> {
    let (factors, _metrics) =
        sparse_lu_cpu_with_workspace_and_metrics(a, symbolic, options, workspace)?;
    Ok(factors)
}

/// Sparse LU factorization with workspace reuse and metrics (CPU)
pub fn sparse_lu_cpu_with_workspace_and_metrics<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    symbolic: &LuSymbolic,
    options: &LuOptions,
    workspace: &mut LuWorkspace,
) -> Result<(LuFactors<R>, LuMetrics)> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    if !workspace.is_compatible(n) {
        return Err(Error::Internal(format!(
            "Workspace dimension {} doesn't match matrix dimension {}",
            workspace.dimension(),
            n
        )));
    }

    // Clear workspace for reuse
    workspace.clear();

    // Extract CSC components
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
    let row_indices: Vec<i64> = a.row_indices().to_vec();
    let values: Vec<f64> = extract_values_f64(a)?;

    // Run factorization with workspace
    let result = gilbert_peierls_lu_with_workspace(
        n,
        &col_ptrs,
        &row_indices,
        &values,
        symbolic,
        options,
        workspace,
    )?;

    // Create output tensors
    let device = a.values().device();
    let (l, u) = create_lu_tensors::<R>(
        n,
        &result.l_col_ptrs,
        &result.l_row_indices,
        &result.l_values,
        &result.u_col_ptrs,
        &result.u_row_indices,
        &result.u_values,
        dtype,
        device,
    )?;

    let factors = LuFactors {
        l,
        u,
        row_perm: result.row_perm,
        row_perm_inv: result.row_perm_inv,
    };

    let metrics = LuMetrics {
        original_nnz: values.len(),
        l_nnz: result.l_values.len(),
        u_nnz: result.u_values.len(),
        fill_ratio: (result.l_values.len() + result.u_values.len()) as f64 / values.len() as f64,
        small_pivots: result.small_pivots,
        row_swaps: result.row_swaps,
        pivot_growth: result.pivot_growth,
    };

    Ok((factors, metrics))
}

/// Sparse LU factorization with simple symbolic structure (CPU)
///
/// This version doesn't require full symbolic analysis from solvr.
/// Fill-in is discovered dynamically, which is less efficient.
pub fn sparse_lu_simple_cpu<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    options: &LuOptions,
) -> Result<LuFactors<R>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();

    // Extract CSC components
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
    let row_indices: Vec<i64> = a.row_indices().to_vec();
    let values: Vec<f64> = extract_values_f64(a)?;

    // Create simple symbolic structure
    let simple = LuSymbolicSimple::from_pattern(&col_ptrs, &row_indices, n);

    // Run factorization with dynamic fill discovery
    let result = gilbert_peierls_lu_simple(n, &col_ptrs, &row_indices, &values, &simple, options)?;

    // Create output tensors
    let device = a.values().device();
    let (l, u) = create_lu_tensors::<R>(
        n,
        &result.l_col_ptrs,
        &result.l_row_indices,
        &result.l_values,
        &result.u_col_ptrs,
        &result.u_row_indices,
        &result.u_values,
        dtype,
        device,
    )?;

    Ok(LuFactors {
        l,
        u,
        row_perm: result.row_perm,
        row_perm_inv: result.row_perm_inv,
    })
}

/// Solve Ax = b using precomputed LU factors (CPU)
///
/// Solves by: x = U⁻¹ L⁻¹ P b
pub fn sparse_lu_solve_cpu<R: Runtime<DType = DType>>(
    factors: &LuFactors<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>> {
    let n = factors.row_perm.len();
    let b_shape = b.shape();

    if b_shape.is_empty() || b_shape[0] != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: b_shape.to_vec(),
        });
    }

    let nrhs = if b_shape.len() == 1 { 1 } else { b_shape[1] };
    let dtype = b.dtype();

    // Extract L and U components
    let l_col_ptrs: Vec<i64> = factors.l.col_ptrs().to_vec();
    let l_row_indices: Vec<i64> = factors.l.row_indices().to_vec();
    let l_values: Vec<f64> = extract_values_f64_tensor(factors.l.values())?;

    let u_col_ptrs: Vec<i64> = factors.u.col_ptrs().to_vec();
    let u_row_indices: Vec<i64> = factors.u.row_indices().to_vec();
    let u_values: Vec<f64> = extract_values_f64_tensor(factors.u.values())?;

    // Extract b
    let b_vec: Vec<f64> = extract_values_f64_tensor(b)?;

    // Solve for each right-hand side
    let mut x_vec = vec![0.0f64; n * nrhs];

    for rhs in 0..nrhs {
        let b_col = &b_vec[rhs * n..(rhs + 1) * n];
        let x_col = &mut x_vec[rhs * n..(rhs + 1) * n];

        // Apply permutation: y = P * b
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            y[i] = b_col[factors.row_perm[i]];
        }

        // Forward substitution: L * z = y (L is unit lower triangular)
        let mut z = y;
        for j in 0..n {
            let col_start = l_col_ptrs[j] as usize;
            let col_end = l_col_ptrs[j + 1] as usize;

            let z_j = z[j];
            for idx in col_start..col_end {
                let i = l_row_indices[idx] as usize;
                z[i] -= l_values[idx] * z_j;
            }
        }

        // Backward substitution: U * x = z (CSC format)
        // For each column j from right to left:
        //   1. x[j] = z[j] / U[j,j]
        //   2. Update z[i] -= U[i,j] * x[j] for i < j
        for j in (0..n).rev() {
            let col_start = u_col_ptrs[j] as usize;
            let col_end = u_col_ptrs[j + 1] as usize;

            // Find diagonal element and compute x[j]
            let mut diag_val = 0.0;
            let mut diag_found = false;

            for idx in col_start..col_end {
                let i = u_row_indices[idx] as usize;
                if i == j {
                    diag_val = u_values[idx];
                    diag_found = true;
                    break;
                }
            }

            if !diag_found || diag_val.abs() < 1e-15 {
                return Err(Error::Internal(format!(
                    "Zero diagonal in U at column {}",
                    j
                )));
            }

            x_col[j] = z[j] / diag_val;

            // Update z[i] for i < j
            for idx in col_start..col_end {
                let i = u_row_indices[idx] as usize;
                if i < j {
                    z[i] -= u_values[idx] * x_col[j];
                }
            }
        }
    }

    // Create output tensor
    let device = b.device();
    let x_tensor = match dtype {
        DType::F32 => {
            let x_f32: Vec<f32> = x_vec.iter().map(|&x| x as f32).collect();
            if nrhs == 1 {
                Tensor::<R>::try_from_slice(&x_f32, &[n], device)?
            } else {
                Tensor::<R>::try_from_slice(&x_f32, &[n, nrhs], device)?
            }
        }
        DType::F64 => {
            if nrhs == 1 {
                Tensor::<R>::try_from_slice(&x_vec, &[n], device)?
            } else {
                Tensor::<R>::try_from_slice(&x_vec, &[n, nrhs], device)?
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_lu_solve",
            });
        }
    };

    Ok(x_tensor)
}

// ============================================================================
// Internal Implementation
// ============================================================================

/// Result of numeric LU factorization
struct LuNumericResult {
    l_col_ptrs: Vec<i64>,
    l_row_indices: Vec<i64>,
    l_values: Vec<f64>,
    u_col_ptrs: Vec<i64>,
    u_row_indices: Vec<i64>,
    u_values: Vec<f64>,
    row_perm: Vec<usize>,
    row_perm_inv: Vec<usize>,
    small_pivots: usize,
    row_swaps: usize,
    pivot_growth: f64,
}

/// Gilbert-Peierls LU with full symbolic information
fn gilbert_peierls_lu(
    n: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    values: &[f64],
    symbolic: &LuSymbolic,
    options: &LuOptions,
) -> Result<LuNumericResult> {
    // Allocate work vector and permutation
    let mut work = vec![0.0f64; n];
    let mut row_perm: Vec<usize> = (0..n).collect();
    let mut row_perm_inv: Vec<usize> = (0..n).collect();

    // Storage for L and U factors (will grow with fill-in)
    let mut l_col_ptrs = vec![0i64; n + 1];
    let mut l_row_indices = Vec::with_capacity(symbolic.l_nnz());
    let mut l_values = Vec::with_capacity(symbolic.l_nnz());

    let mut u_col_ptrs = vec![0i64; n + 1];
    let mut u_row_indices = Vec::with_capacity(symbolic.u_nnz());
    let mut u_values = Vec::with_capacity(symbolic.u_nnz());

    // Track metrics
    let mut small_pivots = 0usize;
    let mut row_swaps = 0usize;
    let max_a = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut max_u_diag = 0.0f64;

    // Process columns in post-order (or just sequential if not available)
    for k in 0..n {
        // Step 1: Scatter column k of A into work vector
        let a_col_start = col_ptrs[k] as usize;
        let a_col_end = col_ptrs[k + 1] as usize;

        kernels::scatter_column(
            &values[a_col_start..a_col_end],
            &row_indices[a_col_start..a_col_end],
            &mut work,
        );

        // Step 2: Sparse triangular solve using reach information
        // For each j in reach(k), solve L[j,j] * x[j] = b[j] - sum(L[i,j] * x[i])
        for &j in &symbolic.reach[k] {
            if j >= k {
                continue;
            }

            let l_col_start = l_col_ptrs[j] as usize;
            let l_col_end = l_col_ptrs[j + 1] as usize;

            // x[j] is already in work[j] from previous iterations
            let x_j = work[row_perm_inv[j]];

            // Update: work[i] -= L[i,j] * x[j] for i > j
            kernels::sparse_axpy(
                x_j,
                &l_values[l_col_start..l_col_end],
                &l_row_indices[l_col_start..l_col_end],
                &mut work,
            );
        }

        // Step 3: Find pivot (partial pivoting)
        let (pivot_row, pivot_abs) = kernels::find_pivot(&work, k, n);

        // Check pivot
        if pivot_abs < options.pivot_threshold {
            if options.diagonal_shift > 0.0 {
                work[pivot_row] = options.diagonal_shift.copysign(work[pivot_row]);
                small_pivots += 1;
            } else if options.check_zeros {
                return Err(Error::Internal(format!(
                    "Zero pivot at column {} (value: {})",
                    k, pivot_abs
                )));
            }
        }

        // Step 4: Swap rows if needed
        if pivot_row != k {
            kernels::swap_rows(&mut work, &mut row_perm, k, pivot_row);

            // Update inverse permutation
            row_perm_inv[row_perm[k]] = k;
            row_perm_inv[row_perm[pivot_row]] = pivot_row;

            row_swaps += 1;
        }

        let pivot = work[k];
        max_u_diag = max_u_diag.max(pivot.abs());

        // Step 5: Store column k of L and U

        // U[0:k+1, k] - upper triangular part (includes diagonal)
        for i in 0..=k {
            let val = work[i];
            if val.abs() > 1e-15 {
                u_row_indices.push(i as i64);
                u_values.push(val);
            }
        }
        u_col_ptrs[k + 1] = u_row_indices.len() as i64;

        // L[k+1:n, k] = work[k+1:n] / pivot - lower triangular part
        let inv_pivot = 1.0 / pivot;
        for i in (k + 1)..n {
            let val = work[i] * inv_pivot;
            if val.abs() > 1e-15 {
                l_row_indices.push(i as i64);
                l_values.push(val);
            }
        }
        l_col_ptrs[k + 1] = l_row_indices.len() as i64;

        // Clear work vector for next iteration
        for i in 0..n {
            work[i] = 0.0;
        }
    }

    let pivot_growth = if max_a > 0.0 { max_u_diag / max_a } else { 1.0 };

    Ok(LuNumericResult {
        l_col_ptrs,
        l_row_indices,
        l_values,
        u_col_ptrs,
        u_row_indices,
        u_values,
        row_perm,
        row_perm_inv,
        small_pivots,
        row_swaps,
        pivot_growth,
    })
}

/// Gilbert-Peierls LU with workspace reuse (no allocations)
fn gilbert_peierls_lu_with_workspace(
    n: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    values: &[f64],
    symbolic: &LuSymbolic,
    options: &LuOptions,
    workspace: &mut LuWorkspace,
) -> Result<LuNumericResult> {
    // Use workspace buffers (already cleared by caller)
    let work = &mut workspace.work;
    let row_perm = &mut workspace.row_perm;
    let row_perm_inv = &mut workspace.row_perm_inv;

    // Storage for L and U factors - use workspace buffers
    let mut l_col_ptrs = vec![0i64; n + 1];
    workspace.l_row_indices_buffer.clear();
    workspace.l_values_buffer.clear();

    let mut u_col_ptrs = vec![0i64; n + 1];
    workspace.u_row_indices_buffer.clear();
    workspace.u_values_buffer.clear();

    // Track metrics
    let mut small_pivots = 0usize;
    let mut row_swaps = 0usize;
    let max_a = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut max_u_diag = 0.0f64;

    // Process columns
    for k in 0..n {
        // Step 1: Scatter column k of A into work vector
        let a_col_start = col_ptrs[k] as usize;
        let a_col_end = col_ptrs[k + 1] as usize;

        kernels::scatter_column(
            &values[a_col_start..a_col_end],
            &row_indices[a_col_start..a_col_end],
            work,
        );

        // Step 2: Sparse triangular solve using reach information
        for &j in &symbolic.reach[k] {
            if j >= k {
                continue;
            }

            let l_col_start = l_col_ptrs[j] as usize;
            let l_col_end = l_col_ptrs[j + 1] as usize;

            let x_j = work[row_perm_inv[j]];

            kernels::sparse_axpy(
                x_j,
                &workspace.l_values_buffer[l_col_start..l_col_end],
                &workspace.l_row_indices_buffer[l_col_start..l_col_end],
                work,
            );
        }

        // Step 3: Find pivot (partial pivoting)
        let (pivot_row, pivot_abs) = kernels::find_pivot(work, k, n);

        // Check pivot
        if pivot_abs < options.pivot_threshold {
            if options.diagonal_shift > 0.0 {
                work[pivot_row] = options.diagonal_shift.copysign(work[pivot_row]);
                small_pivots += 1;
            } else if options.check_zeros {
                return Err(Error::Internal(format!(
                    "Zero pivot at column {} (value: {})",
                    k, pivot_abs
                )));
            }
        }

        // Step 4: Swap rows if needed
        if pivot_row != k {
            kernels::swap_rows(work, row_perm, k, pivot_row);
            row_perm_inv[row_perm[k]] = k;
            row_perm_inv[row_perm[pivot_row]] = pivot_row;
            row_swaps += 1;
        }

        let pivot = work[k];
        max_u_diag = max_u_diag.max(pivot.abs());

        // Step 5: Store column k of L and U (into workspace buffers)

        // U[0:k+1, k]
        for i in 0..=k {
            let val = work[i];
            if val.abs() > 1e-15 {
                workspace.u_row_indices_buffer.push(i as i64);
                workspace.u_values_buffer.push(val);
            }
        }
        u_col_ptrs[k + 1] = workspace.u_row_indices_buffer.len() as i64;

        // L[k+1:n, k]
        let inv_pivot = 1.0 / pivot;
        for i in (k + 1)..n {
            let val = work[i] * inv_pivot;
            if val.abs() > 1e-15 {
                workspace.l_row_indices_buffer.push(i as i64);
                workspace.l_values_buffer.push(val);
            }
        }
        l_col_ptrs[k + 1] = workspace.l_row_indices_buffer.len() as i64;

        // Clear work vector for next iteration
        for i in 0..n {
            work[i] = 0.0;
        }
    }

    let pivot_growth = if max_a > 0.0 { max_u_diag / max_a } else { 1.0 };

    // Clone results from workspace (we need to return owned data)
    Ok(LuNumericResult {
        l_col_ptrs,
        l_row_indices: workspace.l_row_indices_buffer.clone(),
        l_values: workspace.l_values_buffer.clone(),
        u_col_ptrs,
        u_row_indices: workspace.u_row_indices_buffer.clone(),
        u_values: workspace.u_values_buffer.clone(),
        row_perm: row_perm.clone(),
        row_perm_inv: row_perm_inv.clone(),
        small_pivots,
        row_swaps,
        pivot_growth,
    })
}

/// Gilbert-Peierls LU with simple symbolic structure (dynamic fill discovery)
fn gilbert_peierls_lu_simple(
    n: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    values: &[f64],
    _simple: &LuSymbolicSimple,
    options: &LuOptions,
) -> Result<LuNumericResult> {
    // For simple version, we compute reach dynamically using DFS
    // This is less efficient but works without solvr's symbolic analysis

    // Allocate work vector and permutation
    let mut work = vec![0.0f64; n];
    let mut row_perm: Vec<usize> = (0..n).collect();
    let mut row_perm_inv: Vec<usize> = (0..n).collect();

    // Build adjacency structure for reach computation
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for col in 0..n {
        let start = col_ptrs[col] as usize;
        let end = col_ptrs[col + 1] as usize;
        for idx in start..end {
            let row = row_indices[idx] as usize;
            if row != col {
                adj[col].push(row);
            }
        }
    }

    // Storage for L and U factors
    let mut l_col_ptrs = vec![0i64; n + 1];
    let mut l_row_indices = Vec::new();
    let mut l_values = Vec::new();

    let mut u_col_ptrs = vec![0i64; n + 1];
    let mut u_row_indices = Vec::new();
    let mut u_values = Vec::new();

    // Track L column structure for reach computation
    let mut l_adj: Vec<Vec<usize>> = vec![vec![]; n];

    // Track metrics
    let mut small_pivots = 0usize;
    let mut row_swaps = 0usize;
    let max_a = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut max_u_diag = 0.0f64;

    // Workspace for DFS
    let mut visited = vec![false; n];
    let mut reach_stack = Vec::with_capacity(n);

    for k in 0..n {
        // Step 1: Scatter column k of A into work vector
        let a_col_start = col_ptrs[k] as usize;
        let a_col_end = col_ptrs[k + 1] as usize;

        kernels::scatter_column(
            &values[a_col_start..a_col_end],
            &row_indices[a_col_start..a_col_end],
            &mut work,
        );

        // Step 2: Compute reach(k) using DFS on L structure
        // reach(k) = all j < k such that L[k, j] might be nonzero
        reach_stack.clear();
        for idx in a_col_start..a_col_end {
            let row = row_indices[idx] as usize;
            if row < k && !visited[row] {
                dfs_reach(row, k, &l_adj, &mut visited, &mut reach_stack);
            }
        }

        // Sort reach in topological order (just by index for now)
        reach_stack.sort_unstable();

        // Step 2b: Sparse triangular solve using computed reach
        for &j in &reach_stack {
            let l_col_start = l_col_ptrs[j] as usize;
            let l_col_end = l_col_ptrs[j + 1] as usize;

            let x_j = work[j];

            kernels::sparse_axpy(
                x_j,
                &l_values[l_col_start..l_col_end],
                &l_row_indices[l_col_start..l_col_end],
                &mut work,
            );
        }

        // Clear visited flags
        for &j in &reach_stack {
            visited[j] = false;
        }
        for idx in a_col_start..a_col_end {
            let row = row_indices[idx] as usize;
            visited[row] = false;
        }

        // Step 3: Find pivot
        let (pivot_row, pivot_abs) = kernels::find_pivot(&work, k, n);

        if pivot_abs < options.pivot_threshold {
            if options.diagonal_shift > 0.0 {
                work[pivot_row] = options.diagonal_shift.copysign(work[pivot_row]);
                small_pivots += 1;
            } else if options.check_zeros {
                return Err(Error::Internal(format!(
                    "Zero pivot at column {} (value: {})",
                    k, pivot_abs
                )));
            }
        }

        // Step 4: Swap rows if needed
        if pivot_row != k {
            kernels::swap_rows(&mut work, &mut row_perm, k, pivot_row);
            row_perm_inv[row_perm[k]] = k;
            row_perm_inv[row_perm[pivot_row]] = pivot_row;
            row_swaps += 1;
        }

        let pivot = work[k];
        max_u_diag = max_u_diag.max(pivot.abs());

        // Step 5: Store column k of L and U

        // U[0:k+1, k]
        for i in 0..=k {
            let val = work[i];
            if val.abs() > 1e-15 {
                u_row_indices.push(i as i64);
                u_values.push(val);
            }
        }
        u_col_ptrs[k + 1] = u_row_indices.len() as i64;

        // L[k+1:n, k] = work[k+1:n] / pivot
        let inv_pivot = 1.0 / pivot;
        for i in (k + 1)..n {
            let val = work[i] * inv_pivot;
            if val.abs() > 1e-15 {
                l_row_indices.push(i as i64);
                l_values.push(val);
                l_adj[k].push(i); // Track for reach computation
            }
        }
        l_col_ptrs[k + 1] = l_row_indices.len() as i64;

        // Clear work vector
        for i in 0..n {
            work[i] = 0.0;
        }
    }

    let pivot_growth = if max_a > 0.0 { max_u_diag / max_a } else { 1.0 };

    Ok(LuNumericResult {
        l_col_ptrs,
        l_row_indices,
        l_values,
        u_col_ptrs,
        u_row_indices,
        u_values,
        row_perm,
        row_perm_inv,
        small_pivots,
        row_swaps,
        pivot_growth,
    })
}

/// DFS to compute reach set
fn dfs_reach(
    start: usize,
    limit: usize,
    l_adj: &[Vec<usize>],
    visited: &mut [bool],
    reach: &mut Vec<usize>,
) {
    if visited[start] || start >= limit {
        return;
    }

    visited[start] = true;

    // Visit children first (for topological order)
    for &child in &l_adj[start] {
        if child < limit {
            dfs_reach(child, limit, l_adj, visited, reach);
        }
    }

    reach.push(start);
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract values as f64 from CSC matrix
fn extract_values_f64<R: Runtime<DType = DType>>(a: &CscData<R>) -> Result<Vec<f64>> {
    let dtype = a.values().dtype();
    match dtype {
        DType::F32 => Ok(a
            .values()
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect()),
        DType::F64 => Ok(a.values().to_vec()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_lu",
        }),
    }
}

/// Extract values as f64 from tensor
fn extract_values_f64_tensor<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Result<Vec<f64>> {
    let dtype = t.dtype();
    match dtype {
        DType::F32 => Ok(t.to_vec::<f32>().iter().map(|&x| x as f64).collect()),
        DType::F64 => Ok(t.to_vec()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_lu",
        }),
    }
}

/// Create L and U tensors from computed values
fn create_lu_tensors<R: Runtime<DType = DType>>(
    n: usize,
    l_col_ptrs: &[i64],
    l_row_indices: &[i64],
    l_values: &[f64],
    u_col_ptrs: &[i64],
    u_row_indices: &[i64],
    u_values: &[f64],
    dtype: DType,
    device: &R::Device,
) -> Result<(CscData<R>, CscData<R>)> {
    let l = match dtype {
        DType::F32 => {
            let l_values_f32: Vec<f32> = l_values.iter().map(|&x| x as f32).collect();
            CscData::<R>::from_slices(l_col_ptrs, l_row_indices, &l_values_f32, [n, n], device)?
        }
        DType::F64 => {
            CscData::<R>::from_slices(l_col_ptrs, l_row_indices, l_values, [n, n], device)?
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_lu",
            });
        }
    };

    let u = match dtype {
        DType::F32 => {
            let u_values_f32: Vec<f32> = u_values.iter().map(|&x| x as f32).collect();
            CscData::<R>::from_slices(u_col_ptrs, u_row_indices, &u_values_f32, [n, n], device)?
        }
        DType::F64 => {
            CscData::<R>::from_slices(u_col_ptrs, u_row_indices, u_values, [n, n], device)?
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_lu",
            });
        }
    };

    Ok((l, u))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuRuntime;

    fn create_test_matrix() -> CscData<CpuRuntime> {
        // Simple 4x4 matrix:
        // [4, 1, 0, 0]
        // [1, 4, 1, 0]
        // [0, 1, 4, 1]
        // [0, 0, 1, 4]
        let device = <CpuRuntime as crate::runtime::Runtime>::Device::default();

        // CSC format: column by column
        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];

        CscData::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &device).unwrap()
    }

    #[test]
    fn test_sparse_lu_simple() {
        let a = create_test_matrix();
        let options = LuOptions::default();

        let factors = sparse_lu_simple_cpu(&a, &options).unwrap();

        assert_eq!(factors.row_perm.len(), 4);
        assert_eq!(factors.row_perm_inv.len(), 4);

        // Verify L is lower triangular (stored in CSC)
        let l_col_ptrs: Vec<i64> = factors.l.col_ptrs().to_vec();
        for col in 0..4 {
            let start = l_col_ptrs[col] as usize;
            let end = l_col_ptrs[col + 1] as usize;
            let l_row_indices: Vec<i64> = factors.l.row_indices().to_vec();
            for idx in start..end {
                let row = l_row_indices[idx] as usize;
                assert!(row > col, "L should be strictly lower triangular");
            }
        }
    }

    #[test]
    fn test_sparse_lu_solve() {
        let a = create_test_matrix();
        let options = LuOptions::default();

        let factors = sparse_lu_simple_cpu(&a, &options).unwrap();

        // Create test RHS: b = A * [1, 2, 3, 4]^T
        // b = [4*1+1*2, 1*1+4*2+1*3, 1*2+4*3+1*4, 1*3+4*4] = [6, 12, 18, 19]
        let device = <CpuRuntime as crate::runtime::Runtime>::Device::default();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[6.0f64, 12.0, 18.0, 19.0], &[4], &device)
            .unwrap();

        let x = sparse_lu_solve_cpu(&factors, &b).unwrap();

        let x_vec: Vec<f64> = x.to_vec();
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        for i in 0..4 {
            assert!(
                (x_vec[i] - expected[i]).abs() < 1e-10,
                "x[{}] = {}, expected {}",
                i,
                x_vec[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_sparse_lu_with_symbolic() {
        let a = create_test_matrix();

        // Create a simple symbolic structure
        let symbolic = LuSymbolic {
            n: 4,
            etree: vec![1, 2, 3, 4],
            post_order: vec![0, 1, 2, 3],
            reach: vec![vec![], vec![0], vec![0, 1], vec![0, 1, 2]],
            l_col_ptrs: vec![0, 1, 2, 3, 3],
            l_row_indices: vec![1, 2, 3],
            u_col_ptrs: vec![0, 1, 3, 6, 10],
            u_row_indices: vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            workspace_size: 4,
        };

        let options = LuOptions::default();
        let (_factors, metrics) = sparse_lu_cpu_with_metrics(&a, &symbolic, &options).unwrap();

        assert_eq!(metrics.original_nnz, 10);
        assert!(metrics.fill_ratio >= 1.0);
    }

    #[test]
    fn test_sparse_lu_with_workspace() {
        let a = create_test_matrix();

        // Create a simple symbolic structure
        let symbolic = LuSymbolic {
            n: 4,
            etree: vec![1, 2, 3, 4],
            post_order: vec![0, 1, 2, 3],
            reach: vec![vec![], vec![0], vec![0, 1], vec![0, 1, 2]],
            l_col_ptrs: vec![0, 1, 2, 3, 3],
            l_row_indices: vec![1, 2, 3],
            u_col_ptrs: vec![0, 1, 3, 6, 10],
            u_row_indices: vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            workspace_size: 4,
        };

        let options = LuOptions::default();

        // Create workspace
        let mut workspace = LuWorkspace::new(4, &symbolic);

        // First factorization
        let factors1 =
            sparse_lu_cpu_with_workspace(&a, &symbolic, &options, &mut workspace).unwrap();

        // Verify factorization works
        assert_eq!(factors1.row_perm.len(), 4);

        // Create a modified matrix with same pattern but different values
        let device = <CpuRuntime as crate::runtime::Runtime>::Device::default();
        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        // Different values: multiply all by 2
        let values = vec![8.0f64, 2.0, 2.0, 8.0, 2.0, 2.0, 8.0, 2.0, 2.0, 8.0];
        let a2 =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &device)
                .unwrap();

        // Second factorization - reuses workspace (no allocations!)
        let factors2 =
            sparse_lu_cpu_with_workspace(&a2, &symbolic, &options, &mut workspace).unwrap();

        // Verify second factorization also works
        assert_eq!(factors2.row_perm.len(), 4);

        // Verify workspace dimension check
        let mut wrong_workspace = LuWorkspace::new(3, &LuSymbolic::identity(3));
        let result = sparse_lu_cpu_with_workspace(&a, &symbolic, &options, &mut wrong_workspace);
        assert!(
            result.is_err(),
            "Should fail with wrong workspace dimension"
        );
    }

    #[test]
    fn test_sparse_lu_workspace_solve() {
        let a = create_test_matrix();

        // Create a simple symbolic structure
        let symbolic = LuSymbolic {
            n: 4,
            etree: vec![1, 2, 3, 4],
            post_order: vec![0, 1, 2, 3],
            reach: vec![vec![], vec![0], vec![0, 1], vec![0, 1, 2]],
            l_col_ptrs: vec![0, 1, 2, 3, 3],
            l_row_indices: vec![1, 2, 3],
            u_col_ptrs: vec![0, 1, 3, 6, 10],
            u_row_indices: vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            workspace_size: 4,
        };

        let options = LuOptions::default();
        let mut workspace = LuWorkspace::new(4, &symbolic);

        let factors =
            sparse_lu_cpu_with_workspace(&a, &symbolic, &options, &mut workspace).unwrap();

        // Solve: b = A * [1, 2, 3, 4]^T = [6, 12, 18, 19]
        let device = <CpuRuntime as crate::runtime::Runtime>::Device::default();
        let b = Tensor::<CpuRuntime>::try_from_slice(&[6.0f64, 12.0, 18.0, 19.0], &[4], &device)
            .unwrap();

        let x = sparse_lu_solve_cpu(&factors, &b).unwrap();

        let x_vec: Vec<f64> = x.to_vec();
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        for i in 0..4 {
            assert!(
                (x_vec[i] - expected[i]).abs() < 1e-10,
                "x[{}] = {}, expected {}",
                i,
                x_vec[i],
                expected[i]
            );
        }
    }
}
