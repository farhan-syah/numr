//! Sparse Tensor Workflows (COO, CSR, SpMV)
//!
//! Demonstrates numr's sparse tensor support:
//! - Building a sparse matrix in COO (coordinate) format
//! - Converting to CSR (compressed sparse row) for efficient operations
//! - Sparse matrix-vector multiplication (SpMV)
//! - Converting back to dense for verification
//!
//! Requires the `sparse` feature:
//! ```sh
//! cargo run --example sparse_coo_csr_workflow --features sparse
//! ```

#[cfg(feature = "sparse")]
fn main() -> numr::error::Result<()> {
    use numr::prelude::*;
    use numr::sparse::SparseTensor;

    let device = CpuDevice::new();
    let _client = CpuRuntime::default_client(&device);

    // -----------------------------------------------------------------------
    // 1. Build a sparse matrix in COO format
    // -----------------------------------------------------------------------
    // Represent a 4×4 matrix with 5 non-zero entries:
    //
    //   [ 2  0  0  1 ]
    //   [ 0  3  0  0 ]
    //   [ 0  0  0  0 ]
    //   [ 4  0  5  0 ]

    let rows = [0i64, 0, 1, 3, 3];
    let cols = [0i64, 3, 1, 0, 2];
    let vals = [2.0f32, 1.0, 3.0, 4.0, 5.0];

    let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
        &rows,
        &cols,
        &vals,
        [4, 4], // shape
        &device,
    )?;

    println!("Created COO sparse matrix (4×4, {} non-zeros)", vals.len());

    // -----------------------------------------------------------------------
    // 2. Convert COO → CSR
    // -----------------------------------------------------------------------
    // CSR is the go-to format for row-oriented access and SpMV.
    let csr = sparse.to_csr()?;
    println!("Converted to CSR format");

    // -----------------------------------------------------------------------
    // 3. Sparse matrix-vector multiplication (SpMV)
    // -----------------------------------------------------------------------
    // y = A · x
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)?;
    let y = csr.spmv(&x)?;
    let y_vec: Vec<f32> = y.to_vec();

    println!("\nSpMV: A · [1, 2, 3, 4]");
    println!("Result: {y_vec:?}");
    // Expected:
    //   row 0: 2*1 + 1*4 = 6
    //   row 1: 3*2 = 6
    //   row 2: 0
    //   row 3: 4*1 + 5*3 = 19
    println!("Expected: [6.0, 6.0, 0.0, 19.0]");

    // -----------------------------------------------------------------------
    // 4. Convert sparse → dense for visual inspection
    // -----------------------------------------------------------------------
    let dense = sparse.to_dense(&device)?;
    let dense_data: Vec<f32> = dense.to_vec();
    println!("\nDense representation:");
    for row in 0..4 {
        let start = row * 4;
        println!("  {:?}", &dense_data[start..start + 4]);
    }

    // -----------------------------------------------------------------------
    // 5. Sparse algebra via the client trait
    // -----------------------------------------------------------------------
    // SparseTensor also supports sparse × dense matrix multiplication.
    let x2 = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &device,
    )?;
    let y2 = csr.spmm(&x2)?;
    println!("\nSpMM: A · B result (shape {:?}):", y2.shape());
    println!("{:?}", y2.to_vec::<f32>());

    println!("\nSparse workflow example completed successfully!");
    Ok(())
}

#[cfg(not(feature = "sparse"))]
fn main() {
    eprintln!("This example requires the `sparse` feature.");
    eprintln!("Run with: cargo run --example sparse_coo_csr_workflow --features sparse");
    std::process::exit(1);
}
