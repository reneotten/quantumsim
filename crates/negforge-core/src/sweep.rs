//! Bias sweeps and derived figures of merit (subthreshold swing), mirroring
//! `plot_Vg_I`, `plot_Vds_I` and `plot_S` in `legacy_matlab/quantumsim.m`.
//!
//! Each bias point in a sweep is fully independent: `set_v_g`/`set_v_ds`
//! reset `psi_g`, `psi_bi` and `rho` from scratch (see `Device::init_vectors`),
//! so nothing carries over from one point to the next. That makes a sweep
//! embarrassingly parallel — each point runs on its own cloned `Device` via
//! `rayon`, which matters most for self-consistent sweeps (each point is a
//! full Poisson<->NEGF iteration, the most expensive operation in this
//! crate). The sweep functions therefore take `&Device` (a template whose
//! bias is varied) rather than `&mut Device`: the input device's own state
//! is left untouched.

use rayon::prelude::*;

use crate::device::Device;
use crate::error::Result;
use crate::selfconsistent::{self, SelfConsistentOptions};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvPoint {
    pub voltage: f64,
    pub current: f64,
}

/// If `Some`, each sweep point solves the full self-consistent Poisson<->NEGF
/// loop; if `None`, each point uses the decoupled electrostatic solve only
/// (matching the original scripts' `calc_potential` + `calc_current`).
pub type SelfConsistency<'a> = Option<&'a SelfConsistentOptions>;

fn inclusive_range(min: f64, max: f64, step: f64) -> Vec<f64> {
    let steps = ((max - min) / step).round().max(0.0) as usize;
    (0..=steps).map(|i| min + i as f64 * step).collect()
}

fn one_point(
    device: &Device,
    voltage: f64,
    self_consistency: SelfConsistency,
    set_bias: impl Fn(&mut Device, f64),
) -> Result<IvPoint> {
    let mut dev = device.clone();
    set_bias(&mut dev, voltage);
    match self_consistency {
        Some(opts) => {
            selfconsistent::solve_self_consistent(&mut dev, opts)?;
        }
        None => dev.calc_potential(),
    }
    Ok(IvPoint {
        voltage,
        current: dev.calc_current(),
    })
}

/// Gate-voltage sweep at fixed drain bias. Mirrors `plot_Vg_I`. Points are
/// evaluated in parallel (see module docs).
pub fn sweep_v_g(
    device: &Device,
    v_min: f64,
    v_max: f64,
    step: f64,
    self_consistency: SelfConsistency,
) -> Result<Vec<IvPoint>> {
    inclusive_range(v_min, v_max, step)
        .into_par_iter()
        .map(|v| one_point(device, v, self_consistency, Device::set_v_g))
        .collect()
}

/// Drain-voltage sweep at fixed gate bias. Mirrors `plot_Vds_I`. Points are
/// evaluated in parallel (see module docs).
pub fn sweep_v_ds(
    device: &Device,
    v_min: f64,
    v_max: f64,
    step: f64,
    self_consistency: SelfConsistency,
) -> Result<Vec<IvPoint>> {
    inclusive_range(v_min, v_max, step)
        .into_par_iter()
        .map(|v| one_point(device, v, self_consistency, Device::set_v_ds))
        .collect()
}

/// Ordinary least-squares fit `y = slope * x + intercept`.
fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        num += (xi - mean_x) * (yi - mean_y);
        den += (xi - mean_x).powi(2);
    }
    let slope = num / den;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

/// Subthreshold swing (mV/decade equivalent, in the original's V/decade
/// units) from a gate-voltage sweep, computed as `1 / slope` of a
/// `log10(I)` vs `V_g` linear fit over `[fit_v_min, fit_v_max]`. Mirrors
/// the `polyfit` step inside `plot_Vg_I`.
pub fn subthreshold_swing(points: &[IvPoint], fit_v_min: f64, fit_v_max: f64) -> f64 {
    let (xs, ys): (Vec<f64>, Vec<f64>) = points
        .iter()
        .filter(|p| p.voltage >= fit_v_min && p.voltage <= fit_v_max)
        .map(|p| (p.voltage, p.current.log10()))
        .unzip();
    let (slope, _intercept) = linear_fit(&xs, &ys);
    1.0 / slope
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceParams;

    #[test]
    fn v_g_sweep_produces_monotonic_current_for_ballistic_mosfet() {
        let device = Device::new(DeviceParams {
            v_ds: 0.3,
            ..Default::default()
        });
        let points = sweep_v_g(&device, 0.0, 0.4, 0.05, None).unwrap();
        assert_eq!(points.len(), 9);
        for w in points.windows(2) {
            assert!(
                w[1].current >= w[0].current,
                "current should be monotonic in V_g"
            );
        }
    }

    #[test]
    fn subthreshold_swing_is_positive_for_a_reasonable_device() {
        let device = Device::new(DeviceParams {
            v_ds: 0.3,
            ..Default::default()
        });
        let points = sweep_v_g(&device, 0.0, 0.4, 0.02, None).unwrap();
        let s = subthreshold_swing(&points, 0.0, 0.4);
        assert!(s > 0.0 && s.is_finite());
    }

    #[test]
    fn sweep_does_not_mutate_the_input_device() {
        let device = Device::new(DeviceParams {
            v_ds: 0.3,
            ..Default::default()
        });
        let psi_f_before = device.psi_f.clone();
        let v_g_before = device.params.v_g;
        let _ = sweep_v_g(&device, 0.0, 0.4, 0.05, None).unwrap();
        assert_eq!(device.psi_f, psi_f_before);
        assert_eq!(device.params.v_g, v_g_before);
    }

    #[test]
    fn linear_fit_recovers_known_line() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let (slope, intercept) = linear_fit(&x, &y);
        assert!((slope - 2.0).abs() < 1e-9);
        assert!((intercept - 1.0).abs() < 1e-9);
    }
}
