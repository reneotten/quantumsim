//! Physical sanity checks for the planar/FinFET/nanoribbon presets — the
//! "make sure the solver works for these architectures" checks.
//!
//! Background: `lambda = sqrt(k_si/k_ox * d_ch * d_ox / geo)` (the
//! "generalized scale length" model) is how this 1D model distinguishes
//! device architectures — `geo` is the number of gates. See the
//! `device.rs` module docs.

use negforge_core::selfconsistent::{solve_self_consistent, SelfConsistentOptions};
use negforge_core::sweep::{subthreshold_swing, sweep_v_g};
use negforge_core::{Device, DeviceParams, GateGeometry};

#[test]
fn presets_have_the_expected_gate_counts() {
    assert_eq!(DeviceParams::planar().geo, 1.0);
    assert_eq!(DeviceParams::fin_fet().geo, 2.0);
    assert_eq!(DeviceParams::nanoribbon().geo, 4.0);
}

#[test]
fn with_gate_geometry_only_changes_geo() {
    let base = DeviceParams::planar();
    let tri_gate = base.with_gate_geometry(GateGeometry::TriGate);
    assert_eq!(tri_gate.geo, 3.0);
    // Everything else should be untouched.
    assert_eq!(tri_gate.d_ch, base.d_ch);
    assert_eq!(tri_gate.d_ox, base.d_ox);
    assert_eq!(tri_gate.a, base.a);
}

#[test]
fn more_gates_means_a_shorter_natural_length() {
    // lambda ~ 1/sqrt(geo), so more gates -> tighter electrostatic
    // control, for otherwise-comparable body/oxide thickness.
    let planar = Device::new(DeviceParams::planar());
    let fin_fet = Device::new(DeviceParams::fin_fet());
    let nanoribbon = Device::new(DeviceParams::nanoribbon());

    assert!(
        planar.lambda > fin_fet.lambda,
        "planar lambda ({}) should exceed FinFET lambda ({})",
        planar.lambda,
        fin_fet.lambda
    );
    assert!(
        fin_fet.lambda > nanoribbon.lambda,
        "FinFET lambda ({}) should exceed nanoribbon lambda ({})",
        fin_fet.lambda,
        nanoribbon.lambda
    );
}

#[test]
fn all_three_architectures_solve_without_producing_nan_or_panicking() {
    for params in [
        DeviceParams::planar(),
        DeviceParams::fin_fet(),
        DeviceParams::nanoribbon(),
    ] {
        let mut dev = Device::new(DeviceParams {
            v_ds: 0.3,
            v_g: 0.2,
            ..params
        });
        dev.calc_potential();
        assert!(dev.psi_f.iter().all(|v| v.is_finite()));
        let current = dev.calc_current();
        assert!(current.is_finite() && current >= 0.0);
    }
}

#[test]
fn all_three_architectures_converge_self_consistently() {
    // Small devices (short l_ch, coarse grid) to keep the test fast; see
    // geometry_comparison.rs for a realistic-scale comparison.
    for params in [
        DeviceParams::planar(),
        DeviceParams::fin_fet(),
        DeviceParams::nanoribbon(),
    ] {
        let mut dev = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 15.0,
            auto_size_contacts: false,
            v_ds: 0.3,
            v_g: 0.2,
            d_e: 0.01,
            ..params
        });
        // `eta` now defaults to 5 * d_e, which for this device's d_e = 0.01
        // is the 0.05 this test used to pass explicitly.
        let opts = SelfConsistentOptions::default();
        let result = solve_self_consistent(&mut dev, &opts)
            .unwrap_or_else(|e| panic!("geo={} should converge: {e}", dev.params.geo));
        assert!(result.residual < opts.tolerance);
        assert!(dev.psi_f.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn more_gates_gives_better_subthreshold_swing_at_short_channel_length() {
    // The textbook multi-gate result: at a channel length short relative
    // to the planar device's natural length, planar suffers a
    // short-channel-effect-degraded (larger) subthreshold swing, while
    // FinFET (more gates) and especially gate-all-around nanoribbon (most
    // gates) stay much closer to the ideal thermal limit
    // (ln(10)*kT/e = 59.6 mV/decade at 300 K).
    let swing_for = |params: DeviceParams| {
        let dev = Device::new(DeviceParams {
            l_ch: 10.0,
            v_ds: 0.3,
            ..params
        });
        let points = sweep_v_g(&dev, 0.0, 0.4, 0.02, None).unwrap();
        subthreshold_swing(&points, 0.0, 0.4).expect("swing fit should be well posed")
    };

    let s_planar = swing_for(DeviceParams::planar());
    let s_fin_fet = swing_for(DeviceParams::fin_fet());
    let s_nanoribbon = swing_for(DeviceParams::nanoribbon());

    assert!(
        s_planar > s_fin_fet,
        "planar swing ({:.1} mV/dec) should be worse than FinFET ({:.1} mV/dec)",
        s_planar * 1000.0,
        s_fin_fet * 1000.0
    );
    assert!(
        s_fin_fet > s_nanoribbon,
        "FinFET swing ({:.1} mV/dec) should be worse than nanoribbon ({:.1} mV/dec)",
        s_fin_fet * 1000.0,
        s_nanoribbon * 1000.0
    );
    // Gate-all-around should be close to the ideal limit even at this
    // short channel length.
    assert!(
        s_nanoribbon * 1000.0 < 70.0,
        "nanoribbon swing ({:.1} mV/dec) should be close to the 59.6 mV/dec ideal limit",
        s_nanoribbon * 1000.0
    );
}
