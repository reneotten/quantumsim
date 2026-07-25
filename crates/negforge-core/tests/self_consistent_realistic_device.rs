//! Integration test at the model's default (non-toy) device scale — the
//! same geometry a notebook user gets from `Device::new(DeviceParams
//!::default())`. Confirms the self-consistent loop converges reliably and
//! produces the expected ballistic-MOSFET trend (current increasing with
//! gate bias) rather than just checking the tiny unit-test-scale device in
//! `selfconsistent.rs`.

use negforge_core::negf::{
    green_function_sweep, negf_current, GreenFunctionAlgorithm, GreenOutputs, DEFAULT_ETA,
};
use negforge_core::selfconsistent::solve_self_consistent;
use negforge_core::{Device, DeviceParams, SelfConsistentOptions};

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

#[test]
fn local_density_of_states_is_positive_at_default_scale() {
    // The retarded sign convention has to hold on the real device too, not
    // just the toy unit-test geometry: every LDOS entry positive, over the
    // full energy grid the notebook plots.
    let device = Device::new(DeviceParams {
        v_ds: 0.3,
        v_g: 0.2,
        ..Default::default()
    });
    let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
    let d_e = device.params.d_e;
    let steps = ((0.7 * device.e_max - e_min) / d_e) as usize;
    let energies: Vec<f64> = (0..=steps).map(|k| e_min + k as f64 * d_e).collect();

    let green = green_function_sweep(
        &device,
        &energies,
        DEFAULT_ETA,
        GreenFunctionAlgorithm::Recursive,
        GreenOutputs::Full,
    );
    let worst = green.ldos.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        worst >= 0.0 && worst.is_finite(),
        "LDOS must be positive everywhere at default scale, worst value {worst:e}"
    );
}

#[test]
fn negf_and_ballistic_currents_agree_in_the_on_state() {
    // Above threshold the barrier is transparent, so the NEGF transmission is
    // ~1 across the conducting window and the two current models should meet
    // — a check that the Caroli prefactor (2e/h, Gamma_s Gamma_d |G|^2) is
    // normalized consistently with the analytic Landauer integral.
    //
    // They part company in the subthreshold regime, and correctly so: the
    // analytic formula assumes a parabolic band of infinite extent, while the
    // discretized chain has a finite bandwidth `4t` that a large `V_ds` can
    // push the drain band out of. See `negf_current`'s docs.
    let device = Device::new(DeviceParams {
        v_ds: 0.3,
        v_g: 0.4,
        ..Default::default()
    });

    let ballistic = device.calc_current();
    let negf = negf_current(&device, DEFAULT_ETA, GreenFunctionAlgorithm::Recursive);
    let ratio = negf / ballistic;
    assert!(
        (0.9..=1.0).contains(&ratio),
        "on-state currents should agree to ~10%, with NEGF slightly lower from \
         quantum reflection: ballistic {ballistic:e}, NEGF {negf:e} (ratio {ratio})"
    );
}
