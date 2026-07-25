//! Integration test at the model's default (non-toy) device scale — the
//! same geometry a notebook user gets from `Device::new(DeviceParams
//!::default())`. Confirms the self-consistent loop converges reliably and
//! produces the expected ballistic-MOSFET trend (current increasing with
//! gate bias) rather than just checking the tiny unit-test-scale device in
//! `selfconsistent.rs`.

use quantumsim_core::selfconsistent::solve_self_consistent;
use quantumsim_core::{Device, DeviceParams, SelfConsistentOptions};

#[test]
fn self_consistent_current_increases_with_gate_bias_at_default_scale() {
    let opts = SelfConsistentOptions::default();

    let mut low_vg = Device::new(DeviceParams {
        v_ds: 0.3,
        v_g: 0.0,
        ..Default::default()
    });
    let r_low = solve_self_consistent(&mut low_vg, &opts).expect("should converge at V_g=0");
    assert!(r_low.residual < opts.tolerance);
    let i_low = low_vg.calc_current();

    let mut high_vg = Device::new(DeviceParams {
        v_ds: 0.3,
        v_g: 0.3,
        ..Default::default()
    });
    let r_high = solve_self_consistent(&mut high_vg, &opts).expect("should converge at V_g=0.3");
    assert!(r_high.residual < opts.tolerance);
    let i_high = high_vg.calc_current();

    assert!(
        i_high > i_low,
        "self-consistent current should increase with gate bias: {i_low:e} -> {i_high:e}"
    );
}
