//! Basic Tensor Operations
//!
//! This example demonstrates core numr tensor operations on the CPU backend:
//! creating tensors, element-wise arithmetic, reductions, matmul, shape
//! manipulation, and type conversions.
//!
//! Run with:
//! ```sh
//! cargo run --example basic_tensor_ops
//! ```

use numr::prelude::*;

fn main() -> Result<()> {
    // -----------------------------------------------------------------------
    // 1. Obtain a backend client
    // -----------------------------------------------------------------------
    // numr's operations live on a *client* tied to a device.  For the CPU
    // backend the device is simply `CpuDevice::new()`.
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // -----------------------------------------------------------------------
    // 2. Create tensors
    // -----------------------------------------------------------------------

    // From a slice – you provide data and the desired shape.
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    println!("a (2×3):\n{:?}", a.to_vec::<f32>());

    // Convenience constructors.
    let zeros = Tensor::<CpuRuntime>::zeros(&[2, 3], DType::F32, &device)?;
    let ones = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device)?;
    let filled = Tensor::<CpuRuntime>::full_scalar(&[2, 3], DType::F32, 7.0, &device)?;
    println!("zeros: {:?}", zeros.to_vec::<f32>());
    println!("ones:  {:?}", ones.to_vec::<f32>());
    println!("filled:{:?}", filled.to_vec::<f32>());

    // Random tensors (uniform [0,1) and standard normal).
    let uniform = client.rand(&[3, 3], DType::F32)?;
    let normal = client.randn(&[3, 3], DType::F32)?;
    println!("uniform: {:?}", uniform.to_vec::<f32>());
    println!("normal:  {:?}", normal.to_vec::<f32>());

    // -----------------------------------------------------------------------
    // 3. Tensor properties
    // -----------------------------------------------------------------------
    println!(
        "\na: shape={:?}, ndim={}, numel={}, dtype={:?}, contiguous={}",
        a.shape(),
        a.ndim(),
        a.numel(),
        a.dtype(),
        a.is_contiguous(),
    );

    // -----------------------------------------------------------------------
    // 4. Element-wise arithmetic
    // -----------------------------------------------------------------------
    // All operations go through the client, not operator overloading.

    let b = Tensor::<CpuRuntime>::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &device,
    )?;

    let sum = client.add(&a, &b)?;
    let diff = client.sub(&a, &b)?;
    let prod = client.mul(&a, &b)?;
    let quot = client.div(&a, &b)?;

    println!("\na + b = {:?}", sum.to_vec::<f32>());
    println!("a - b = {:?}", diff.to_vec::<f32>());
    println!("a * b = {:?}", prod.to_vec::<f32>());
    println!("a / b = {:?}", quot.to_vec::<f32>());

    // Scalar operations.
    let scaled = client.mul_scalar(&a, 100.0)?;
    println!("a * 100 = {:?}", scaled.to_vec::<f32>());

    // -----------------------------------------------------------------------
    // 5. Unary math functions
    // -----------------------------------------------------------------------
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4], &device)?;
    println!("\nexp(x) = {:?}", client.exp(&x)?.to_vec::<f32>());
    println!("sqrt(x) = {:?}", client.sqrt(&x)?.to_vec::<f32>());
    println!("sin(x) = {:?}", client.sin(&x)?.to_vec::<f32>());

    // Activations.
    let logits = Tensor::<CpuRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device)?;
    println!(
        "relu(logits)    = {:?}",
        client.relu(&logits)?.to_vec::<f32>()
    );
    println!(
        "sigmoid(logits) = {:?}",
        client.sigmoid(&logits)?.to_vec::<f32>()
    );

    // -----------------------------------------------------------------------
    // 6. Reductions
    // -----------------------------------------------------------------------
    // `dims` selects which axes to reduce; `keepdim` controls whether
    // reduced dimensions are retained as size-1.

    let m = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    let row_sum = client.sum(&m, &[1], false)?; // sum across columns
    let col_mean = client.mean(&m, &[0], false)?; // mean down rows
    let global_max = client.max(&m, &[0, 1], false)?;

    println!("\nrow sums  = {:?}", row_sum.to_vec::<f32>());
    println!("col means = {:?}", col_mean.to_vec::<f32>());
    println!("global max= {:?}", global_max.to_vec::<f32>());

    // -----------------------------------------------------------------------
    // 7. Matrix multiplication
    // -----------------------------------------------------------------------
    // matmul follows standard linear-algebra rules: (M,K) @ (K,N) → (M,N).

    let lhs =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    let rhs =
        Tensor::<CpuRuntime>::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], &device)?;
    let matmul_result = client.matmul(&lhs, &rhs)?;
    println!(
        "\n(2×3) @ (3×2) = {:?}  (shape {:?})",
        matmul_result.to_vec::<f32>(),
        matmul_result.shape(),
    );

    // -----------------------------------------------------------------------
    // 8. Shape manipulation (zero-copy views)
    // -----------------------------------------------------------------------
    // These operations create a *view* sharing the same underlying storage.

    let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

    let reshaped = t.reshape(&[3, 2])?;
    println!("\nreshaped (3×2): {:?}", reshaped.to_vec::<f32>());

    let transposed = t.transpose(0, 1)?;
    println!(
        "transposed (3×2): {:?}",
        transposed.contiguous().unwrap().to_vec::<f32>()
    );

    let unsqueezed = t.unsqueeze(0)?; // [1, 2, 3]
    println!("unsqueeze(0) shape: {:?}", unsqueezed.shape());

    // Broadcasting: [2, 1] + [1, 3] → [2, 3]
    let col = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[2, 1], &device)?;
    let row = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device)?;
    let broadcast_sum = client.add(&col, &row)?;
    println!(
        "\nbroadcast [2,1]+[1,3] = {:?}  (shape {:?})",
        broadcast_sum.to_vec::<f32>(),
        broadcast_sum.shape(),
    );

    // -----------------------------------------------------------------------
    // 9. Extracting scalar values
    // -----------------------------------------------------------------------
    let scalar = Tensor::<CpuRuntime>::from_slice(&[42.0f32], &[], &device)?;
    let value: f32 = scalar.item()?;
    println!("\nscalar item = {value}");

    // -----------------------------------------------------------------------
    // 10. Comparison operations
    // -----------------------------------------------------------------------
    let p = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0], &[3], &device)?;
    let q = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 5.0, 1.0], &[3], &device)?;
    let eq_mask = client.eq(&p, &q)?;
    let gt_mask = client.gt(&p, &q)?;
    // Comparison results use the same dtype (1.0 = true, 0.0 = false).
    println!("\np == q: {:?}", eq_mask.to_vec::<f32>());
    println!("p >  q: {:?}", gt_mask.to_vec::<f32>());

    println!("\nAll basic tensor operations completed successfully!");
    Ok(())
}
