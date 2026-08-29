//! Build script for numr
//!
//! Compiles CUDA kernels to PTX when the cuda feature is enabled.
//!
//! # Requirements
//!
//! - CUDA Toolkit (nvcc compiler)
//! - Compute Capability 7.5+ (Turing architecture, sm_75)
//!
//! # Environment Variables
//!
//! - `CUDA_PATH`: Custom CUDA installation path (optional)
//!
//! # Troubleshooting
//!
//! If nvcc is not found:
//! 1. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
//! 2. Ensure nvcc is in your PATH, or set CUDA_PATH environment variable
//! 3. Common paths: /usr/local/cuda, /opt/cuda, C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y

fn main() {
    // Only compile CUDA kernels when the cuda feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = PathBuf::from("src/runtime/cuda/kernels");

    // List of kernel files to compile
    #[allow(unused_mut)]
    let mut kernel_files = vec![
        "activation.cu",
        "softmax.cu",
        "advanced_random.cu",
        "binary.cu",
        "cast.cu",
        "compare.cu",
        "complex.cu",
        "conv.cu",
        "cumulative.cu",
        "distance.cu",
        "distributions.cu",
        "fft.cu",
        "fft_bluestein.cu",
        "fused_activation_mul.cu",
        "fused_activation_mul_bwd.cu",
        "fused_add_norm.cu",
        "fused_elementwise.cu",
        "index.cu",
        "linalg_advanced.cu",
        "linalg_banded.cu",
        "linalg_basic.cu",
        "linalg_decomp.cu",
        "linalg_eigen.cu",
        "linalg_eigen_general.cu",
        "linalg_matrix_funcs.cu",
        "linalg_qz.cu",
        "linalg_schur.cu",
        "linalg_solvers.cu",
        "linalg_svd.cu",
        "fp8_matmul.cu",
        "gemv.cu",
        "gemv_int.cu",
        "matmul.cu",
        "matmul_int.cu",
        "norm.cu",
        "semiring_matmul.cu",
        "quasirandom.cu",
        "reduce.cu",
        "scalar.cu",
        "shape.cu",
        "sort.cu",
        "special.cu",
        "statistics.cu",
        "strided_copy.cu",
        "ternary.cu",
        "unary.cu",
        "utility.cu",
        "gemm_epilogue.cu",
        "gemm_epilogue_bwd.cu",
        "matmul_wmma.cu",
    ];

    // Add sparse kernels if sparse feature is enabled
    #[cfg(feature = "sparse")]
    {
        kernel_files.push("sparse_24.cu");
        kernel_files.push("sparse_spmv.cu");
        kernel_files.push("sparse_merge.cu");
        kernel_files.push("sparse_convert.cu");
        kernel_files.push("sparse_coo.cu");
        kernel_files.push("sparse_utils.cu");
        kernel_files.push("spgemm.cu");
        kernel_files.push("scan.cu");
        kernel_files.push("dsmm.cu");
        kernel_files.push("sparse_linalg.cu");
        kernel_files.push("sparse_levels.cu");
    }

    // Find nvcc with helpful error message
    let nvcc = find_nvcc().unwrap_or_else(|| {
        eprintln!();
        eprintln!("=== CUDA COMPILATION ERROR ===");
        eprintln!();
        eprintln!("Could not find nvcc (NVIDIA CUDA Compiler).");
        eprintln!();
        eprintln!("To fix this:");
        eprintln!("  1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads");
        eprintln!("  2. Add nvcc to your PATH, or set CUDA_PATH environment variable");
        eprintln!();
        eprintln!("Common installation paths:");
        eprintln!("  - Linux: /usr/local/cuda/bin/nvcc");
        eprintln!("  - macOS: /usr/local/cuda/bin/nvcc");
        eprintln!("  - Windows: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\bin\\nvcc.exe");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  export CUDA_PATH=/usr/local/cuda");
        eprintln!("  # or");
        eprintln!("  export PATH=$PATH:/usr/local/cuda/bin");
        eprintln!();
        panic!("nvcc not found - CUDA Toolkit must be installed for the 'cuda' feature");
    });

    // Determine compute capability from NUMR_CUDA_ARCH env var, default sm_80 (Ampere)
    // sm_80 enables tensor cores for F16/BF16, async copy, and other Ampere features
    //
    // Accept the bare capability too (`86`), not just `sm_86`/`compute_86`: `86`
    // is the form `nvidia-smi --query-gpu=compute_cap` reports (as `8.6`), so it
    // is the obvious thing to pass, and forwarding it raw makes nvcc die with a
    // bare "Unsupported gpu architecture '86'" that points at the kernel rather
    // than at the env var.
    let cuda_arch = env::var("NUMR_CUDA_ARCH")
        .map(|v| {
            let v = v.trim().to_string();
            let bare = v.strip_prefix("sm_").or_else(|| v.strip_prefix("compute_"));
            match bare {
                // Already prefixed — pass through, digits validated below.
                Some(digits) => (digits.to_string(), v),
                // Bare: accept `86` and `8.6` alike.
                None => (v.replace('.', ""), format!("sm_{}", v.replace('.', ""))),
            }
        })
        .map(|(digits, arch)| {
            assert!(
                !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()),
                "NUMR_CUDA_ARCH must be a compute capability such as `86`, `8.6`, \
                 `sm_86`, or `compute_86` — got {arch:?}"
            );
            arch
        })
        .unwrap_or_else(|_| "sm_80".to_string());
    println!(
        "cargo:warning=numr: compiling {} CUDA kernels for {cuda_arch} (set NUMR_CUDA_ARCH to override)",
        kernel_files.len()
    );

    // Shared headers are pulled in by `#include`, so nvcc never sees them as
    // inputs cargo tracks. Without these the PTX goes stale when a header
    // changes and only the .cu files are watched.
    for header in [
        "dtype_traits.cuh",
        "activation_deriv.cuh",
        "ipow.cuh",
        "numr128.cuh",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            kernels_dir.join(header).display()
        );
    }

    for kernel_file in kernel_files {
        let cu_path = kernels_dir.join(kernel_file);
        let ptx_name = kernel_file.replace(".cu", ".ptx");
        let ptx_path = out_dir.join(&ptx_name);

        // Rerun if source changes
        println!("cargo:rerun-if-changed={}", cu_path.display());

        // Verify source file exists
        if !cu_path.exists() {
            panic!(
                "CUDA kernel source not found: {}\n\
                 Ensure kernel files exist in src/runtime/cuda/kernels/",
                cu_path.display()
            );
        }

        let output = Command::new(&nvcc)
            .args([
                "-ptx",
                "-O3",
                "--use_fast_math",
                "--ftz=false",
                &format!("-arch={cuda_arch}"),
                "-o",
                ptx_path.to_str().unwrap(),
                cu_path.to_str().unwrap(),
            ])
            .output();

        match output {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    eprintln!();
                    eprintln!("=== CUDA COMPILATION FAILED ===");
                    eprintln!();
                    eprintln!("Failed to compile: {}", kernel_file);
                    eprintln!();
                    if !stdout.is_empty() {
                        eprintln!("stdout:");
                        eprintln!("{}", stdout);
                    }
                    if !stderr.is_empty() {
                        eprintln!("stderr:");
                        eprintln!("{}", stderr);
                    }
                    eprintln!();
                    eprintln!("Possible causes:");
                    eprintln!("  - Syntax error in CUDA kernel code");
                    eprintln!("  - Incompatible CUDA version");
                    eprintln!("  - Missing CUDA headers");
                    eprintln!();
                    panic!("nvcc compilation failed for {}", kernel_file);
                }
            }
            Err(e) => {
                eprintln!();
                eprintln!("=== NVCC EXECUTION ERROR ===");
                eprintln!();
                eprintln!("Failed to execute nvcc: {}", e);
                eprintln!("nvcc path: {}", nvcc);
                eprintln!();
                eprintln!("This may indicate:");
                eprintln!("  - nvcc exists but is not executable");
                eprintln!("  - Missing library dependencies");
                eprintln!("  - Permissions issue");
                eprintln!();
                panic!("Failed to execute nvcc: {}", e);
            }
        }
    }

    // Export the OUT_DIR for the Rust code to find PTX files
    println!("cargo:rustc-env=CUDA_KERNEL_DIR={}", out_dir.display());
}

#[cfg(feature = "cuda")]
fn find_nvcc() -> Option<String> {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    // Check CUDA_PATH environment variable first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc.to_string_lossy().to_string());
        }
        // Also try with .exe extension on Windows
        let nvcc_exe = PathBuf::from(&cuda_path).join("bin").join("nvcc.exe");
        if nvcc_exe.exists() {
            return Some(nvcc_exe.to_string_lossy().to_string());
        }
    }

    // Check common CUDA installation paths
    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        // Add more common paths as needed
    ];

    for path in common_paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    // Try to find nvcc in PATH by running it
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }

    None
}
