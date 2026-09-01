//! Build script for numr
//!
//! Compiles CUDA kernels to a multi-arch fatbin when the cuda feature is enabled.
//!
//! # Requirements
//!
//! - CUDA Toolkit (nvcc compiler)
//! - Compute Capability 7.5+ (Turing architecture, sm_75)
//!
//! # Environment Variables
//!
//! - `CUDA_PATH`: Custom CUDA installation path (optional)
//! - `NUMR_CUDA_ARCH`: Controls which compute capabilities get real SASS
//!   cubins in the fatbin — a comma-separated list (`86`, `sm_86`, `8.6`,
//!   `86,89,90`), `all`/`portable` for every supported arch, or unset to
//!   auto-detect the local GPU(s) via `nvidia-smi` (falls back to
//!   `all`/`portable` when no GPU is detected). Every mode also embeds
//!   `compute_75`/`compute_120` PTX as a JIT floor/ceiling.
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
        "cumulative_int.cu",
        "distance.cu",
        "distributions.cu",
        "fft.cu",
        "fft_bluestein.cu",
        "fused_activation_mul.cu",
        "fused_activation_mul_bwd.cu",
        "fused_add_norm.cu",
        "fused_elementwise.cu",
        "im2col.cu",
        "index.cu",
        "index_nd.cu",
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
        "matmul_fp8.cu",
        "matmul_int.cu",
        "norm.cu",
        "semiring_matmul.cu",
        "quasirandom.cu",
        "reduce.cu",
        "reduce_int.cu",
        "scalar.cu",
        "scatter_reduce.cu",
        "shape.cu",
        "sort.cu",
        "special.cu",
        "statistics.cu",
        "strided_copy.cu",
        "ternary.cu",
        "unary.cu",
        "unary_int.cu",
        "utility.cu",
        "utility_random.cu",
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

    // Re-run when the override changes — otherwise a stale fatbin survives
    // an `export NUMR_CUDA_ARCH=...` until something else invalidates OUT_DIR.
    println!("cargo:rerun-if-env-changed=NUMR_CUDA_ARCH");

    // Real SASS cubins for hardware people actually own: sm_75 (Turing),
    // sm_80 (Ampere A100), sm_86 (Ampere consumer), sm_89 (Ada), sm_90
    // (Hopper), sm_100 (Blackwell datacenter), sm_120 (Blackwell consumer).
    const REAL_ARCHES: &[&str] = &["75", "80", "86", "89", "90", "100", "120"];

    // Four modes, selected by NUMR_CUDA_ARCH and local GPU detection:
    //
    //   1. Unset + GPU(s) detected (the default): build cubins for exactly
    //      the distinct compute capabilities present on this machine. Cheap
    //      dev/deploy iteration on the box that will also run the binary.
    //   2. Set to a comma-separated list (`86`, `sm_86`, `8.6`,
    //      `86,89,90`, ...): build exactly those archs.
    //   3. Set to `all`/`portable` (case-insensitive): build every arch in
    //      REAL_ARCHES. Use for releases, Docker images, and anything built
    //      on one machine to run on another.
    //   4. Unset + no GPU detected: same output as mode 3, plus a warning —
    //      an unset var with no GPU present is almost always CI or a
    //      container, which must get the portable artifact.
    let requested = env::var("NUMR_CUDA_ARCH").ok();
    let (selected_arches, mode_desc): (Vec<String>, String) = match requested.as_deref() {
        Some(v)
            if v.trim().eq_ignore_ascii_case("all")
                || v.trim().eq_ignore_ascii_case("portable") =>
        {
            (
                REAL_ARCHES.iter().map(|a| a.to_string()).collect(),
                format!(
                    "all {} portable archs (NUMR_CUDA_ARCH={v})",
                    REAL_ARCHES.len()
                ),
            )
        }
        Some(v) => {
            let mut archs: Vec<String> = v.split(',').map(parse_arch).collect();
            archs.dedup();
            let desc = format!(
                "{} arch(es) requested via NUMR_CUDA_ARCH={v}: {}",
                archs.len(),
                describe_archs(&archs)
            );
            (archs, desc)
        }
        None => match detect_local_gpu_arches() {
            Some(archs) if !archs.is_empty() => {
                let desc = format!(
                    "{} arch(es) detected locally: {}",
                    archs.len(),
                    describe_archs(&archs)
                );
                (archs, desc)
            }
            _ => {
                println!(
                    "cargo:warning=numr: no GPU detected (nvidia-smi missing, failed, or \
                     reported nothing) — building a portable fatbin for all {} archs; set \
                     NUMR_CUDA_ARCH to the local arch(s) to skip this cost",
                    REAL_ARCHES.len()
                );
                (
                    REAL_ARCHES.iter().map(|a| a.to_string()).collect(),
                    format!("all {} portable archs (no GPU detected)", REAL_ARCHES.len()),
                )
            }
        },
    };

    // Two virtual PTX entries are REQUIRED in every mode, on top of the
    // real-SASS cubins for `selected_arches`:
    //
    //   compute_75/compute_75 — the floor. PTX only JIT-compiles FORWARD,
    //   never backward, so any arch >= 75 with no matching cubin above
    //   (sm_87, sm_88, sm_103, sm_110, sm_121, ...) falls through to this
    //   entry and JITs at load time. It looks redundant next to an sm_75
    //   cubin — it is not: delete it and every one of those devices fails
    //   to load the module outright. It also covers detect-mode staleness:
    //   Cargo has no "rerun-if-GPU-changed" hook, so if a card is swapped
    //   after this build ran, this entry is what lets the new card JIT and
    //   run (slower) instead of failing to load.
    //
    //   compute_120/compute_120 — forward JIT for hardware newer than this
    //   toolkit's SASS targets (nvcc 13.2 tops out emitting cubins at
    //   sm_120/121).
    let mut gencode_flags: Vec<String> = selected_arches
        .iter()
        .map(|a| format!("arch=compute_{a},code=sm_{a}"))
        .collect();
    gencode_flags.push("arch=compute_75,code=compute_75".to_string());
    gencode_flags.push("arch=compute_120,code=compute_120".to_string());

    println!(
        "cargo:warning=numr: compiling {} CUDA kernels into a fatbin for {} \
         (+ compute_75/compute_120 JIT floor/ceiling)",
        kernel_files.len(),
        mode_desc
    );

    // Shared headers are pulled in by `#include`, so nvcc never sees them as
    // inputs cargo tracks. Without these the compiled fatbins go stale when
    // a header changes and only the .cu files are watched.
    for header in [
        "dtype_traits.cuh",
        "narrow_f64.cuh",
        "activation_deriv.cuh",
        "binary_ops.cuh",
        "cumulative_ops.cuh",
        "ipow.cuh",
        "numr128.cuh",
        "scalar_ops.cuh",
        "index_ops.cuh",
        "index_nd_ops.cuh",
        "rng_xorshift.cuh",
        "gemm_activation.cuh",
        "matmul_f32_tiled.cuh",
        "matmul_wmma.cuh",
        "matmul_wmma_stage.cuh",
        "semiring_matmul_ops.cuh",
        "sort_bitonic.cuh",
        "sort_compare.cuh",
        "sort_scan.cuh",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            kernels_dir.join(header).display()
        );
    }

    // Serial pre-pass: verify sources exist and register rerun triggers.
    // Cheap (no nvcc involved), so no benefit to parallelizing it, and it
    // keeps the rerun-if-changed emission independent of worker scheduling.
    for kernel_file in &kernel_files {
        let cu_path = kernels_dir.join(kernel_file);
        println!("cargo:rerun-if-changed={}", cu_path.display());
        if !cu_path.exists() {
            panic!(
                "CUDA kernel source not found: {}\n\
                 Ensure kernel files exist in src/runtime/cuda/kernels/",
                cu_path.display()
            );
        }
    }

    // Compiling every kernel to a multi-arch fatbin multiplies nvcc's work
    // ~7x over the old single-arch PTX build, so the loop is parallelized
    // with a bounded worker pool. Each kernel's output path is independent
    // (no shared-state hazard), but nvcc is memory-hungry, so concurrency
    // is capped rather than spawning one thread per file.
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);

    struct KernelOutcome {
        file: String,
        success: bool,
        stdout: String,
        stderr: String,
        exec_error: Option<String>,
    }

    let work_queue = std::sync::Mutex::new(kernel_files.to_vec());
    let outcomes = std::sync::Mutex::new(Vec::<KernelOutcome>::new());

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let kernel_file: &str = {
                        let mut queue = work_queue.lock().unwrap();
                        match queue.pop() {
                            Some(f) => f,
                            None => break,
                        }
                    };

                    let cu_path = kernels_dir.join(kernel_file);
                    let fatbin_name = kernel_file.replace(".cu", ".fatbin");
                    let fatbin_path = out_dir.join(&fatbin_name);

                    let mut args: Vec<String> = vec![
                        "-fatbin".to_string(),
                        "-O3".to_string(),
                        "--use_fast_math".to_string(),
                        "--ftz=false".to_string(),
                    ];
                    for gc in &gencode_flags {
                        args.push("-gencode".to_string());
                        args.push(gc.clone());
                    }
                    args.push("-o".to_string());
                    args.push(fatbin_path.to_str().unwrap().to_string());
                    args.push(cu_path.to_str().unwrap().to_string());

                    let outcome = match Command::new(&nvcc).args(&args).output() {
                        Ok(output) => KernelOutcome {
                            file: kernel_file.to_string(),
                            success: output.status.success(),
                            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                            exec_error: None,
                        },
                        Err(e) => KernelOutcome {
                            file: kernel_file.to_string(),
                            success: false,
                            stdout: String::new(),
                            stderr: String::new(),
                            exec_error: Some(e.to_string()),
                        },
                    };

                    outcomes.lock().unwrap().push(outcome);
                }
            });
        }
    });

    // All workers joined: emit any failures (collected, not interleaved)
    // and fail the build if there were any.
    let outcomes = outcomes.into_inner().unwrap();
    let failed: Vec<&KernelOutcome> = outcomes.iter().filter(|o| !o.success).collect();
    if !failed.is_empty() {
        for outcome in &failed {
            eprintln!();
            if let Some(e) = &outcome.exec_error {
                eprintln!("=== NVCC EXECUTION ERROR ===");
                eprintln!();
                eprintln!("Failed to execute nvcc for: {}", outcome.file);
                eprintln!("Error: {}", e);
                eprintln!("nvcc path: {}", nvcc);
                eprintln!();
                eprintln!("This may indicate:");
                eprintln!("  - nvcc exists but is not executable");
                eprintln!("  - Missing library dependencies");
                eprintln!("  - Permissions issue");
            } else {
                eprintln!("=== CUDA COMPILATION FAILED ===");
                eprintln!();
                eprintln!("Failed to compile: {}", outcome.file);
                eprintln!();
                if !outcome.stdout.is_empty() {
                    eprintln!("stdout:");
                    eprintln!("{}", outcome.stdout);
                }
                if !outcome.stderr.is_empty() {
                    eprintln!("stderr:");
                    eprintln!("{}", outcome.stderr);
                }
                eprintln!();
                eprintln!("Possible causes:");
                eprintln!("  - Syntax error in CUDA kernel code");
                eprintln!("  - Incompatible CUDA version");
                eprintln!("  - Missing CUDA headers");
            }
        }
        eprintln!();
        let failed_names: Vec<&str> = failed.iter().map(|o| o.file.as_str()).collect();
        panic!("nvcc compilation failed for: {}", failed_names.join(", "));
    }

    // Export the OUT_DIR for the Rust code to find the compiled fatbins
    println!("cargo:rustc-env=CUDA_KERNEL_DIR={}", out_dir.display());
}

// Join bare-digit arches into a human-readable `sm_75,sm_86,...` list for
// the diagnostic warning/mode description. Formatting-only, kept in one
// place so the `sm_` prefix convention doesn't drift between call sites.
#[cfg(feature = "cuda")]
fn describe_archs(archs: &[String]) -> String {
    archs
        .iter()
        .map(|a| format!("sm_{a}"))
        .collect::<Vec<_>>()
        .join(",")
}

// Parse one NUMR_CUDA_ARCH list entry into bare digits (`86`), validated to
// be within the supported range. Accepts every spelling nvidia-smi and a
// human might type: `86`, `8.6`, `sm_86`, `compute_86`. Bare-form support
// matters because `86` is exactly what `nvidia-smi --query-gpu=compute_cap`
// reports (as `8.6`), so it is the obvious thing to paste in, and forwarding
// it raw to nvcc dies with a bare "Unsupported gpu architecture '86'" that
// points at the kernel rather than at the env var.
#[cfg(feature = "cuda")]
fn parse_arch(entry: &str) -> String {
    let v = entry.trim();
    let bare = v.strip_prefix("sm_").or_else(|| v.strip_prefix("compute_"));
    let digits = match bare {
        // Already prefixed — pass through, digits validated below.
        Some(digits) => digits.to_string(),
        // Bare: accept `86` and `8.6` alike.
        None => v.replace('.', ""),
    };
    assert!(
        !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()),
        "NUMR_CUDA_ARCH entry must be a compute capability such as `86`, `8.6`, \
         `sm_86`, `compute_86`, or `all`/`portable` — got {v:?}"
    );
    validate_arch_range(&digits, v);
    digits
}

// Shared range check for both NUMR_CUDA_ARCH entries and nvidia-smi-detected
// arches: below the toolkit's floor or above its ceiling is a config error,
// named explicitly, never a silent drop from the build.
#[cfg(feature = "cuda")]
fn validate_arch_range(digits: &str, original: &str) {
    let n: u32 = digits.parse().expect("digits already validated numeric");
    assert!(
        (75..=120).contains(&n),
        "compute capability {original:?} (compute_{digits}) is outside the range this \
         build supports: compute_75 (Turing) to compute_120 (Blackwell consumer)"
    );
}

// Query locally installed GPUs for their compute capabilities via nvidia-smi
// (subprocess only — never links or calls the CUDA driver API from the build
// script). Returns `None` on any detection failure: missing binary,
// non-zero exit, empty output, or output that doesn't parse — the caller
// falls back to the portable multi-arch build rather than panicking, since
// "no GPU visible during build" is a normal and expected condition (CI,
// containers, cross-compilation).
#[cfg(feature = "cuda")]
fn detect_local_gpu_arches() -> Option<Vec<String>> {
    use std::process::Command;

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut arches: Vec<String> = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // nvidia-smi prints e.g. "8.6" — strip the dot to match REAL_ARCHES.
        let digits: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            continue;
        }
        validate_arch_range(&digits, line);
        arches.push(digits);
    }
    if arches.is_empty() {
        return None;
    }

    // Deduplicate (a box with four identical GPUs must yield one arch) and
    // sort (deterministic -gencode flag order for reproducible builds).
    arches.sort();
    arches.dedup();
    Some(arches)
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
