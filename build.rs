//! Build script for Octane
//!
//! Compiles native C code with NEON SIMD optimizations for Apple Silicon
//! and Metal shaders for GPU compute.

use std::env;
use std::path::PathBuf;

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Only compile SIMD code when the 'simd' feature is enabled
    if env::var("CARGO_FEATURE_SIMD").is_ok() {
        compile_simd_code(&target_arch);
    }

    // Compile Metal shaders on macOS
    #[cfg(target_os = "macos")]
    if target_os == "macos" {
        compile_metal_shaders();
    }

    // Re-run build script if these files change
    println!("cargo:rerun-if-changed=src/simd/gae_neon.c");
    println!("cargo:rerun-if-changed=src/simd/gaussian_neon.c");
    println!("cargo:rerun-if-changed=src/simd/categorical_neon.c");
    println!("cargo:rerun-if-changed=src/simd/buffer_ops_neon.c");
    println!("cargo:rerun-if-changed=src/metal/rl_kernels.metal");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Compile C code with NEON SIMD optimizations
fn compile_simd_code(target_arch: &str) {
    let mut build = cc::Build::new();

    // Common flags
    build
        .opt_level(3)
        .flag("-ffast-math")
        .flag("-fno-exceptions")
        .flag("-Wall")
        .flag("-Wextra");

    // Architecture-specific flags
    if target_arch == "aarch64" {
        // For AArch64, NEON is always available, no -mfpu flag needed
        build
            .flag("-DUSE_NEON=1");

        // Apple Silicon specific tuning (optional, may not be supported by all compilers)
        // build.flag("-mcpu=apple-m1");
    } else if target_arch == "x86_64" {
        // Fallback for x86_64 with SSE/AVX
        build
            .flag("-march=native")
            .flag("-DUSE_SSE=1");
    }

    // Source files
    let simd_sources = [
        "src/simd/gae_neon.c",
        "src/simd/gaussian_neon.c",
        "src/simd/categorical_neon.c",
        "src/simd/buffer_ops_neon.c",
    ];

    // Only add files that exist
    for source in &simd_sources {
        let path = PathBuf::from(source);
        if path.exists() {
            build.file(source);
            println!("cargo:rerun-if-changed={}", source);
        } else {
            println!("cargo:warning=SIMD source file not found: {}", source);
        }
    }

    // Include directory for headers
    build.include("src/simd");

    // Compile static library
    build.compile("octane_simd");

    // Link the library
    println!("cargo:rustc-link-lib=static=octane_simd");
}

/// Compile Metal shaders on macOS
#[cfg(target_os = "macos")]
fn compile_metal_shaders() {
    use std::process::Command;

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let shader_source = "src/metal/rl_kernels.metal";
    let shader_air = format!("{}/shaders.air", out_dir);
    let shader_lib = format!("{}/shaders.metallib", out_dir);

    // Check if shader source exists
    if !std::path::Path::new(shader_source).exists() {
        println!("cargo:warning=Metal shader source not found: {}", shader_source);
        return;
    }

    // Compile Metal shader to AIR (Apple Intermediate Representation)
    let status = Command::new("xcrun")
        .args([
            "-sdk", "macosx",
            "metal",
            "-c", shader_source,
            "-o", &shader_air,
            "-O3",                      // Optimize
            "-ffast-math",              // Fast math
            "-std=metal3.0",            // Metal 3.0 for M-series
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:info=Compiled Metal shader to AIR");
        }
        Ok(s) => {
            println!("cargo:warning=Metal shader compilation failed with status: {}", s);
            return;
        }
        Err(e) => {
            println!("cargo:warning=Failed to run xcrun metal: {}", e);
            return;
        }
    }

    // Link AIR to metallib
    let status = Command::new("xcrun")
        .args([
            "-sdk", "macosx",
            "metallib",
            &shader_air,
            "-o", &shader_lib,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:info=Created Metal library at {}", shader_lib);
            // Set environment variable for runtime loading
            println!("cargo:rustc-env=METAL_LIB_PATH={}", shader_lib);
        }
        Ok(s) => {
            println!("cargo:warning=Metal library creation failed with status: {}", s);
        }
        Err(e) => {
            println!("cargo:warning=Failed to run xcrun metallib: {}", e);
        }
    }

    println!("cargo:rerun-if-changed={}", shader_source);
}

#[cfg(not(target_os = "macos"))]
fn compile_metal_shaders() {
    // Metal is only available on macOS
    println!("cargo:warning=Metal shaders are only compiled on macOS");
}
