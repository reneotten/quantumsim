//! Bias sweeps and derived figures of merit (subthreshold swing), mirroring
//! `plot_Vg_I`, `plot_Vds_I` and `plot_S` in `legacy_matlab/quantumsim.m`.
//!
//! Each bias point in a sweep is fully independent: `set_v_g`/`set_v_ds`
//! reset `psi_g`, `psi_bi` and `rho` from scratch (see `Device::init_vectors`)
//! and re-solve `psi_f` for the new bias, so nothing carries over from one
//! point to the next. That makes a sweep
//! embarrassingly parallel — each point runs on its own cloned `Device` via
//! `rayon`, which matters most for self-consistent sweeps (each point is a
//! full Poisson<->NEGF iteration, the most expensive operation in this
//! crate). The sweep functions therefore take `&Device` (a template whose
//! bias is varied) rather than `&mut Device`: the input device's own state
//! is left untouched.

use rayon::prelude::*;

use crate::device::Device;
use crate::error::{NegForgeError, Result};
use crate::selfconsistent::{self, SelfConsistentOptions};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvPoint {
    pub voltage: f64,
    /// Current in amperes, or `NaN` if this point's self-consistent solve
    /// failed to converge (see [`IvPoint::converged`]).
    pub current: f64,
    /// Whether this point's self-consistent solve converged. Always `true`
    /// for a decoupled sweep. A single stubborn bias point shouldn't discard
    /// an otherwise good sweep, so non-convergence is reported per point
    /// rather than failing the whole call.
    pub converged: bool,
}

/// If `Some`, each sweep point solves the full self-consistent Poisson<->NEGF
/// loop; if `None`, each point uses the decoupled electrostatic solve only
/// (matching the original scripts' `calc_potential` + `calc_current`).
pub type SelfConsistency<'a> = Option<&'a SelfConsistentOptions>;

/// Grid of bias points from `min` to at most `max`, spaced by `step`.
///
/// The endpoint is included only when it actually lands on the grid. The
/// number of steps is truncated rather than rounded, so a range whose span
/// is not a whole multiple of `step` (`0.0..0.4` by `0.15`) stops at the
/// last point *inside* `[min, max]` instead of overshooting past `max`. The
/// truncation snaps up first when the ratio is within rounding noise of an
/// integer, so exactly-representable-looking ranges such as `0.0..0.4` by
/// `0.05` (whose ratio evaluates to 7.999999999999999 in binary floating
/// point) still include their endpoint.
fn inclusive_range(min: f64, max: f64, step: f64) -> Result<Vec<f64>> {
    if !step.is_finite() || step <= 0.0 {
        return Err(NegForgeError::InvalidParameter(format!(
            "sweep step must be finite and positive, got {step}"
        )));
    }
    if !min.is_finite() || !max.is_finite() {
        return Err(NegForgeError::InvalidParameter(format!(
            "sweep bounds must be finite, got [{min}, {max}]"
        )));
    }
    if max < min {
        return Err(NegForgeError::InvalidParameter(format!(
            "sweep upper bound {max} is below the lower bound {min}"
        )));
    }

    let ratio = (max - min) / step;
    let rounded = ratio.round();
    let steps = if (ratio - rounded).abs() <= 1e-9 * ratio.abs().max(1.0) {
        rounded
    } else {
        ratio.floor()
    } as usize;
    Ok((0..=steps).map(|i| min + i as f64 * step).collect())
}

fn one_point(
    device: &Device,
    voltage: f64,
    self_consistency: SelfConsistency,
    set_bias: impl Fn(&mut Device, f64),
) -> Result<IvPoint> {
    let mut dev = device.clone();
    // The bias setters re-solve the decoupled electrostatic potential
    // themselves (see `Device::set_v_ds`), which is exactly what the
    // non-self-consistent path wants; the self-consistent path then
    // iterates Poisson<->NEGF on top of it.
    set_bias(&mut dev, voltage);
    if let Some(opts) = self_consistency {
        match selfconsistent::solve_self_consistent(&mut dev, opts) {
            Ok(_) => {}
            // A bias point that won't converge is a fact about that point,
            // not a reason to throw away the rest of the sweep.
            Err(NegForgeError::NotConverged { .. }) => {
                return Ok(IvPoint {
                    voltage,
                    current: f64::NAN,
                    converged: false,
                })
            }
            Err(other) => return Err(other),
        }
    }
    Ok(IvPoint {
        voltage,
        current: dev.calc_current(),
        converged: true,
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
    inclusive_range(v_min, v_max, step)?
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
    inclusive_range(v_min, v_max, step)?
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
///
/// The fit is only meaningful for strictly positive, finite currents: a
/// zero or negative current (numerical noise deep in the off state, or a
/// bias range where the ballistic window collapses) turns `log10` into
/// `-inf`/`NaN` and would silently poison the fit. Those cases are
/// reported as errors rather than returned as a meaningless number.
pub fn subthreshold_swing(points: &[IvPoint], fit_v_min: f64, fit_v_max: f64) -> Result<f64> {
    let window: Vec<&IvPoint> = points
        .iter()
        .filter(|p| p.voltage >= fit_v_min && p.voltage <= fit_v_max)
        .collect();

    if window.len() < 2 {
        return Err(NegForgeError::InvalidParameter(format!(
            "subthreshold-swing fit needs at least 2 sweep points in [{fit_v_min}, {fit_v_max}], got {}",
            window.len()
        )));
    }
    if let Some(bad) = window
        .iter()
        .find(|p| !p.current.is_finite() || p.current <= 0.0)
    {
        return Err(NegForgeError::InvalidParameter(format!(
            "subthreshold-swing fit needs strictly positive currents (log10 scale), \
             but V = {} has I = {:e}",
            bad.voltage, bad.current
        )));
    }

    let (xs, ys): (Vec<f64>, Vec<f64>) = window
        .iter()
        .map(|p| (p.voltage, p.current.log10()))
        .unzip();
    let (slope, _intercept) = linear_fit(&xs, &ys);
    if !slope.is_finite() || slope == 0.0 {
        return Err(NegForgeError::InvalidParameter(format!(
            "subthreshold-swing fit produced a degenerate slope ({slope}); \
             the current does not vary with gate bias over [{fit_v_min}, {fit_v_max}]"
        )));
    }
    Ok(1.0 / slope)
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
        let s = subthreshold_swing(&points, 0.0, 0.4).unwrap();
        assert!(s > 0.0 && s.is_finite());
    }

    #[test]
    fn subthreshold_swing_rejects_non_positive_currents() {
        let points = vec![
            IvPoint {
                voltage: 0.0,
                current: 0.0,
                converged: true,
            },
            IvPoint {
                voltage: 0.1,
                current: 1e-9,
                converged: true,
            },
            IvPoint {
                voltage: 0.2,
                current: f64::NAN,
                converged: false,
            },
        ];
        let err = subthreshold_swing(&points, 0.0, 0.4).unwrap_err();
        assert!(
            err.to_string().contains("strictly positive"),
            "unexpected error: {err}"
        );

        // Too few points in the fit window is an error too, rather than a
        // division by a zero-variance slope.
        assert!(subthreshold_swing(&points, 0.0, 0.0).is_err());
    }

    #[test]
    fn inclusive_range_never_overshoots_the_upper_bound() {
        // Span is not a whole multiple of the step: the last point must stay
        // inside the range instead of rounding up past it.
        let vs = inclusive_range(0.0, 0.4, 0.15).unwrap();
        assert_eq!(vs.len(), 3);
        assert!(vs.iter().all(|&v| v <= 0.4 + 1e-12), "{vs:?}");

        // Span *is* a whole multiple, modulo binary floating-point noise
        // (0.4 / 0.05 = 7.999999999999999) — the endpoint must survive.
        let vs = inclusive_range(0.0, 0.4, 0.05).unwrap();
        assert_eq!(vs.len(), 9);
        assert!((vs[8] - 0.4).abs() < 1e-12);

        assert!(inclusive_range(0.0, 0.4, 0.0).is_err());
        assert!(inclusive_range(0.0, 0.4, -0.05).is_err());
        assert!(inclusive_range(0.4, 0.0, 0.05).is_err());
        assert!(inclusive_range(0.0, f64::NAN, 0.05).is_err());
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
