//! Backend Portability: CPU ↔ WebGPU
//!
//! Demonstrates writing backend-agnostic code that runs identically on CPU
//! and WebGPU.  The same generic function performs matmul + softmax + reduce,
//! and both backends produce matching results.
//!
//! Run CPU-only (default):
//! ```sh
//! cargo run --example backend_switch_cpu_wgpu
//! ```
//!
//! Run with WebGPU comparison:
//! ```sh
//! cargo run --example backend_switch_cpu_wgpu --features wgpu
//! ```

use numr::prelude::*;

/// A backend-agnostic computation: softmax of a matrix product, then row sums.
///
/// This function works on *any* runtime (CPU, CUDA, WebGPU) because it only
/// requires the standard operation traits.
fn compute<R: Runtime>(a: &Tensor<R>, b: &Tensor<R>, client: &R::Client) -> Result<Tensor<R>>
where
    R::Client: MatmulOps<R> + ActivationOps<R> + ReduceOps<R>,
{
    // Step 1: Matrix multiply
    let product = client.matmul(a, b)?;

    // Step 2: Softmax along last dimension
    let softmax = client.softmax(&product, -1)?;

    // Step 3: Sum each row (reduce dim 1)
    let row_sums = client.sum(&softmax, &[1], false)?;

    Ok(row_sums)
}

fn main() -> Result<()> {
    // -----------------------------------------------------------------------
    // CPU computation
    // -----------------------------------------------------------------------
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let a_cpu =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu_device)?;
    let b_cpu =
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2], &cpu_device)?;

    let cpu_result = compute(&a_cpu, &b_cpu, &cpu_client)?;
    let cpu_vec: Vec<f32> = cpu_result.to_vec();
    println!("CPU result:  {cpu_vec:?}");
    // Each row of softmax sums to 1.0, so row sums should all be 1.0.

    // -----------------------------------------------------------------------
    // WebGPU computation (feature-gated)
    // -----------------------------------------------------------------------
    #[cfg(feature = "wgpu")]
    {
        let wgpu_device = WgpuDevice::new(0);
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        // Create the same data on the WebGPU device.
        let a_wgpu = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &wgpu_device,
        )?;
        let b_wgpu = Tensor::<WgpuRuntime>::from_slice(
            &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[3, 2],
            &wgpu_device,
        )?;

        let wgpu_result = compute(&a_wgpu, &b_wgpu, &wgpu_client)?;
        let wgpu_vec: Vec<f32> = wgpu_result.to_vec();
        println!("WGPU result: {wgpu_vec:?}");

        // Verify parity.
        let max_diff: f32 = cpu_vec
            .iter()
            .zip(wgpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Max CPU–WGPU difference: {max_diff:.2e}");
        assert!(
            max_diff < 1e-4,
            "CPU and WebGPU results should match within FP tolerance"
        );
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("\n(WebGPU comparison skipped — enable with --features wgpu)");
    }

    println!("\nBackend switch example completed successfully!");
    Ok(())
}
