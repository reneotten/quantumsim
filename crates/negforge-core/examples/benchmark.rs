//! Performance audit: dense vs. recursive NEGF, and parallel vs. serial
//! energy-loop evaluation.
//!
//! Run with `cargo run --release --example benchmark -p negforge-core`.
//! Numbers from this example are quoted in the top-level README's
//! "Performance" section.

use std::time::Instant;

use negforge_core::negf::{green_function_sweep, GreenFunctionAlgorithm, GreenOutputs};
use negforge_core::{Device, DeviceParams};

fn device_with_n_sites(target_n: usize) -> Device {
    // l_ch/l_ds/a chosen so the resulting grid has (very close to) target_n
    // sites, holding geometry proportions roughly fixed.
    let a = 1.0;
    let l_ch = (target_n as f64 - 1.0) * a / 3.0;
    let mut dev = Device::new(DeviceParams {
        a,
        l_ch,
        l_ds: l_ch,
        auto_size_contacts: false,
        v_ds: 0.3,
        v_g: 0.2,
        ..Default::default()
    });
    dev.calc_potential();
    dev
}

fn time_sweep(
    device: &Device,
    n_energies: usize,
    algorithm: GreenFunctionAlgorithm,
    outputs: GreenOutputs,
) -> f64 {
    let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
    let energies: Vec<f64> = (0..n_energies).map(|k| e_min + k as f64 * 0.005).collect();
    let start = Instant::now();
    let _ = green_function_sweep(device, &energies, 0.005, algorithm, outputs);
    start.elapsed().as_secs_f64()
}

fn main() {
    const N_ENERGIES: usize = 100;
    println!(
        "=== Dense (O(N^3)) vs. Recursive (O(N)) NEGF, {N_ENERGIES} energy points per sweep ==="
    );
    println!(
        "{:>6} {:>14} {:>14} {:>10}",
        "N", "dense (s)", "recursive (s)", "speedup"
    );
    for &n in &[21usize, 51, 101, 201, 351] {
        let device = device_with_n_sites(n);
        let actual_n = device.n;
        let dense_t = time_sweep(
            &device,
            N_ENERGIES,
            GreenFunctionAlgorithm::Dense,
            GreenOutputs::Full,
        );
        let recursive_t = time_sweep(
            &device,
            N_ENERGIES,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::Full,
        );
        println!(
            "{:>6} {:>14.4} {:>14.4} {:>9.0}x",
            actual_n,
            dense_t,
            recursive_t,
            dense_t / recursive_t
        );
    }

    println!();
    println!("=== Parallel vs. serial energy loop (Recursive, default-scale device N=561, 900 energy points) ===");
    let mut device = Device::new(DeviceParams {
        v_ds: 0.3,
        v_g: 0.2,
        ..Default::default()
    });
    device.calc_potential();
    println!("N = {}", device.n);

    let n_energies = 900;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let serial_t = serial_pool.install(|| {
        time_sweep(
            &device,
            n_energies,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::Full,
        )
    });

    // Default global pool: uses all available cores.
    let parallel_t = time_sweep(
        &device,
        n_energies,
        GreenFunctionAlgorithm::Recursive,
        GreenOutputs::Full,
    );

    println!("available cores (per std::thread::available_parallelism): {cores}");
    println!("1 thread:            {serial_t:.4} s");
    println!("all cores ({cores}):        {parallel_t:.4} s");
    println!("speedup:             {:.2}x", serial_t / parallel_t);

    println!();
    println!(
        "=== Output selection (Recursive, N={}, {n_energies} energy points, 1 thread) ===",
        device.n
    );
    let full_t = serial_pool.install(|| {
        time_sweep(
            &device,
            n_energies,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::Full,
        )
    });
    let columns_t = serial_pool.install(|| {
        time_sweep(
            &device,
            n_energies,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::BoundaryColumns,
        )
    });
    println!("LDOS + columns (Full):        {full_t:.4} s");
    println!("columns only (self-consistent loop): {columns_t:.4} s");
    println!("saving:                       {:.2}x", full_t / columns_t);
}
