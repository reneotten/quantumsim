//! Demonstrates the planar/FinFET/nanoribbon presets and the textbook
//! result they're expected to reproduce: at a fixed (short) channel
//! length, more gates means better electrostatic control and a
//! subthreshold swing closer to the ideal thermal limit
//! (`ln(10)*kT/e = 59.6 mV/decade` at 300 K).
//!
//! Run with `cargo run --release --example geometry_comparison -p
//! negforge-core`. Numbers from this example are quoted in the top-level
//! README's "Planar, FinFET and nanoribbon devices" section.

use negforge_core::sweep::{subthreshold_swing, sweep_v_g};
use negforge_core::{Device, DeviceParams};

fn report(name: &str, mut params: DeviceParams, l_ch: f64) {
    params.l_ch = l_ch;
    params.v_ds = 0.3;
    let dev = Device::new(params);
    let points = sweep_v_g(&dev, 0.0, 0.4, 0.02, None).unwrap();
    let s = subthreshold_swing(&points, 0.0, 0.4);
    println!(
        "{name:<11} lambda={:>6.3} nm   S={:>6.1} mV/decade",
        dev.lambda,
        s * 1000.0
    );
}

fn main() {
    println!("(ideal thermal limit at 300 K: 59.6 mV/decade)\n");
    for l_ch in [10.0, 20.0, 40.0] {
        println!("=== l_ch = {l_ch} nm ===");
        report("planar", DeviceParams::planar(), l_ch);
        report("fin_fet", DeviceParams::fin_fet(), l_ch);
        report("nanoribbon", DeviceParams::nanoribbon(), l_ch);
        println!();
    }
}
