//! Convolution via Unfold (im2col) and Direct conv2d
//!
//! Demonstrates two approaches to 2D convolution in numr:
//!
//! 1. **Direct**: `client.conv2d()` – the standard high-level API.
//! 2. **Manual im2col**: Use `unfold` to extract sliding patches, reshape the
//!    kernel, and express convolution as a matrix multiplication.  This is
//!    the classic im2col trick used by many frameworks internally.
//!
//! Run with:
//! ```sh
//! cargo run --example conv_unfold_im2col
//! ```

use numr::prelude::*;

fn main() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // -----------------------------------------------------------------------
    // 1. Create a small input image and kernel
    // -----------------------------------------------------------------------
    // Input: batch=1, channels=1, height=4, width=4
    #[rustfmt::skip]
    let input_data: &[f32] = &[
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
       13.0, 14.0, 15.0, 16.0,
    ];
    let input = Tensor::<CpuRuntime>::from_slice(input_data, &[1, 1, 4, 4], &device)?;

    // Kernel: out_channels=1, in_channels=1, kH=3, kW=3
    #[rustfmt::skip]
    let kernel_data: &[f32] = &[
        1.0, 0.0, -1.0,
        1.0, 0.0, -1.0,
        1.0, 0.0, -1.0,
    ];
    let kernel = Tensor::<CpuRuntime>::from_slice(kernel_data, &[1, 1, 3, 3], &device)?;

    // -----------------------------------------------------------------------
    // 2. Direct conv2d (stride=1, no padding, dilation=1, groups=1)
    // -----------------------------------------------------------------------
    let direct_out = client.conv2d(
        &input,
        &kernel,
        None,               // no bias
        (1, 1),             // stride (h, w)
        PaddingMode::Valid, // no padding
        (1, 1),             // dilation
        1,                  // groups
    )?;
    println!("Direct conv2d output (shape {:?}):", direct_out.shape());
    println!("{:?}\n", direct_out.to_vec::<f32>());

    // -----------------------------------------------------------------------
    // 3. Manual im2col via unfold + matmul
    // -----------------------------------------------------------------------
    // The idea: unfold extracts overlapping patches along a dimension.
    // For 2D convolution we unfold along H then W to get columns of patches,
    // then reshape into a matrix and multiply by the flattened kernel.

    // Step 3a: Unfold along height (dim=2), window=3, step=1
    let unfolded_h = client.unfold(&input, 2, 3, 1)?;
    // Shape: [1, 1, 2, 4, 3]  (batch, C, out_h, W, kH)

    // Step 3b: Unfold along width (dim=3), window=3, step=1
    let unfolded_hw = client.unfold(&unfolded_h, 3, 3, 1)?;
    // Shape: [1, 1, 2, 2, 3, 3]  (batch, C, out_h, out_w, kH, kW)

    println!("Unfolded patches shape: {:?}", unfolded_hw.shape());

    // Step 3c: Reshape patches to (out_h*out_w, kH*kW) for matmul.
    let out_h = unfolded_hw.shape()[2];
    let out_w = unfolded_hw.shape()[3];
    let k_h = unfolded_hw.shape()[4];
    let k_w = unfolded_hw.shape()[5];
    let patches = unfolded_hw
        .contiguous()?
        .reshape(&[out_h * out_w, k_h * k_w])?;

    // Step 3d: Flatten kernel to (kH*kW, out_channels=1).
    let kernel_flat = kernel.reshape(&[1, k_h * k_w])?;
    let kernel_col = kernel_flat.transpose(0, 1)?;

    // Step 3e: matmul → (out_h*out_w, 1)
    let im2col_flat = client.matmul(&patches, &kernel_col.contiguous()?)?;
    let im2col_out = im2col_flat.reshape(&[1, 1, out_h, out_w])?;

    println!("im2col conv output (shape {:?}):", im2col_out.shape());
    println!("{:?}", im2col_out.to_vec::<f32>());

    // -----------------------------------------------------------------------
    // 4. Verify both approaches match
    // -----------------------------------------------------------------------
    let direct_vec: Vec<f32> = direct_out.to_vec();
    let im2col_vec: Vec<f32> = im2col_out.to_vec();
    let max_diff: f32 = direct_vec
        .iter()
        .zip(im2col_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\nMax difference between direct and im2col: {max_diff:.6e}");
    assert!(max_diff < 1e-5, "Results should match within FP tolerance");

    println!("\nConv/unfold im2col example completed successfully!");
    Ok(())
}
