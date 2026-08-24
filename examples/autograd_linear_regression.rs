//! Autograd: Training a Linear Regression Model
//!
//! This example shows how to use numr's reverse-mode automatic differentiation
//! to train a simple linear model `y = W·x + b` via gradient descent.
//!
//! Key concepts demonstrated:
//! - `Var` wraps a tensor for gradient tracking
//! - `var_*` functions build a computation graph
//! - `backward()` computes gradients for all leaf variables
//! - Gradients are used to manually update parameters (SGD)
//!
//! Run with:
//! ```sh
//! cargo run --example autograd_linear_regression
//! ```

use numr::autograd::{Var, backward, var_add, var_matmul, var_mean, var_mul, var_sub};
use numr::prelude::*;

fn main() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // -----------------------------------------------------------------------
    // 1. Generate synthetic data: y = 3·x₁ + 2·x₂ + 1 (with noise)
    // -----------------------------------------------------------------------
    let n_samples = 64;
    let n_features = 2;

    // Input features: (n_samples, n_features)
    let x_data = client.randn(&[n_samples, n_features], DType::F32)?;

    // True weights [3.0, 2.0] and bias 1.0
    let true_w = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 2.0], &[n_features, 1], &device)?;
    let true_b = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device)?;

    // y = X @ W_true + b_true + noise
    let noise = client.randn(&[n_samples, 1], DType::F32)?;
    let noise_scaled = client.mul_scalar(&noise, 0.1)?; // small noise
    let xw = client.matmul(&x_data, &true_w)?;
    let y_clean = client.add(&xw, &true_b)?;
    let y_data = client.add(&y_clean, &noise_scaled)?;

    // -----------------------------------------------------------------------
    // 2. Initialize learnable parameters
    // -----------------------------------------------------------------------
    // `Var::new(tensor, requires_grad)` marks tensors as leaves of the
    // computation graph whose gradients we want to compute.

    let mut w = Var::new(
        client.randn(&[n_features, 1], DType::F32)?,
        true, // requires_grad
    );
    let mut b = Var::new(
        Tensor::<CpuRuntime>::zeros(&[1], DType::F32, &device)?,
        true,
    );

    // Wrap immutable inputs as Var with requires_grad=false.
    let x_var = Var::new(x_data.clone(), false);
    let y_var = Var::new(y_data.clone(), false);

    // -----------------------------------------------------------------------
    // 3. Training loop
    // -----------------------------------------------------------------------
    let lr: f64 = 0.01;
    let n_epochs = 200;

    for epoch in 0..n_epochs {
        // Forward pass: predictions = X @ W + b
        let pred = var_matmul(&x_var, &w, &client)?;
        let pred = var_add(&pred, &b, &client)?;

        // Loss: MSE = mean((pred - y)²)
        let residual = var_sub(&pred, &y_var, &client)?;
        let sq = var_mul(&residual, &residual, &client)?;
        let loss = var_mean(&sq, &[0, 1], false, &client)?;

        // Backward pass – computes dL/dW and dL/db.
        let grads = backward(&loss, &client)?;

        // Print loss every 50 epochs.
        let loss_val: f32 = loss.tensor().item()?;
        if epoch % 50 == 0 || epoch == n_epochs - 1 {
            println!("epoch {epoch:>4}: loss = {loss_val:.6}");
        }

        // Manual SGD update: param = param - lr * grad
        // We extract the gradient tensors, compute the update, and create
        // new Var instances for the next iteration.
        let grad_w = grads.get(w.id()).expect("gradient for w");
        let grad_b = grads.get(b.id()).expect("gradient for b");

        let w_update = client.mul_scalar(grad_w, lr)?;
        let new_w_tensor = client.sub(w.tensor(), &w_update)?;
        let b_update = client.mul_scalar(grad_b, lr)?;
        let new_b_tensor = client.sub(b.tensor(), &b_update)?;

        // Rebind: create new Var nodes for the next forward pass.
        // This detaches from the old graph (no gradient accumulation).
        w = Var::new(new_w_tensor, true);
        b = Var::new(new_b_tensor, true);
    }

    // -----------------------------------------------------------------------
    // 4. Inspect learned parameters
    // -----------------------------------------------------------------------
    let learned_w: Vec<f32> = w.tensor().to_vec();
    let learned_b: Vec<f32> = b.tensor().to_vec();
    println!("\nLearned weights: {learned_w:?}  (true: [3.0, 2.0])");
    println!("Learned bias:    {learned_b:?}  (true: [1.0])");

    println!("\nLinear regression training completed!");
    Ok(())
}
