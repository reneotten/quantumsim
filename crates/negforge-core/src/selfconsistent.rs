//! Self-consistent Poisson <-> NEGF loop.
//!
//! **This module has no equivalent in the original MATLAB code.** The
//! legacy `quantumsim.m` computed the electrostatic potential once with
//! `rho = 0` (`calc_potential`) and, separately, an NEGF charge density
//! (`calc_n`) from that potential — but never fed the computed charge back
//! into another electrostatic solve. This module closes that loop: solve
//! electrostatics, compute the NEGF electron density from the result, turn
//! it into a charge density, damp/mix it into `rho`, and re-solve, until
//! the potential profile stops changing (or a caller-supplied iteration
//! budget is exhausted, which is reported as an error rather than silently
//! returning an unconverged result).
//!
//! ## Charge-density units
//!
//! `calc_potential`'s RHS divides `rho + N_dot` by `EPS_0 * k_si`, with
//! `EPS_0` in SI units (F/m). For that division to land back in the same
//! eV/nm^2 ballpark as the other RHS term (`(Psi_g + Psi_bi) / lambda^2`),
//! `rho` has to actually be a charge density in C/m^3 — the original code
//! never exercised this (`rho` was always zero) so it was never validated.
//! This loop therefore converts the NEGF electron density to a proper
//! volume charge density before assigning it to `rho`:
//!
//! 1. [`charge::electron_density`] returns a linear density in nm^-1
//!    (natural units for this model's nm-scaled grid).
//! 2. Convert to m^-1 (`* 1e9`) and treat the 1D chain as having an
//!    implicit unit (1 m^2) cross-section, so the linear density doubles as
//!    a volume density.
//! 3. Multiply by `-e` (electrons are negatively charged) to get `rho` in
//!    C/m^3: `rho(x) = -e * n_electron(x) * 1e9`.
//!
//! `N_dot` (fixed dopant charge) is left as-is (default zero); if it is
//! used with a nonzero value it should be supplied in the same C/m^3 units
//! for consistency.
//!
//! ## Validity: mixing, tolerance, and convergence are numerical, not physical
//!
//! `mixing` (linear/Picard mixing of the charge-density update) and `eta`
//! are numerical stabilization knobs, not physical parameters: they affect
//! *whether and how fast* the iteration converges, not what it converges
//! to (a true fixed point of Poisson<->NEGF is independent of both). `eta`
//! in particular still needs to be several times the energy-grid spacing
//! `d_e` even with a correctly-signed contact self-energy (see
//! `negf::DEFAULT_ETA`'s docs) — that requirement comes from resolving
//! sharp resonances smoothly across iterations, not from compensating for
//! any sign error. A run that fails to converge (`NotConverged`) is not
//! evidence the device physics is invalid — it may just need more
//! `max_iterations`, a smaller `mixing`, or a larger `eta`. Conversely,
//! convergence is a necessary but not sufficient condition for physical
//! correctness: a converged fixed point still inherits every approximation
//! described in `device.rs`, `negf.rs` and `charge.rs` (ballistic
//! transport, natural-length electrostatics, no explicit spin-degeneracy
//! factor in the charge density, etc.).

use crate::charge;
use crate::constants::E as ELEMENTARY_CHARGE;
use crate::device::Device;
use crate::error::{NegForgeError, Result};
use crate::negf::{self, GreenFunctionResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfConsistentOptions {
    /// Maximum number of Poisson<->NEGF iterations before giving up.
    pub max_iterations: usize,
    /// Converged when the largest change in `psi_f` between iterations
    /// (eV) drops below this.
    pub tolerance: f64,
    /// Linear mixing factor in `(0, 1]` applied to the charge-density
    /// update (simple damping to stabilize the fixed-point iteration).
    pub mixing: f64,
    /// Upper bound of the NEGF energy grid, as a fraction of `device.e_max`
    /// (matches the `0.7 * E_max` hardcoded in the original `calc_green`).
    pub green_energy_fraction: f64,
    /// Imaginary broadening used for the NEGF evaluations inside the loop.
    /// Should be several times the energy-grid step (`device.params.d_e`)
    /// so that resonances are resolved smoothly across iterations instead
    /// of spiking whenever a grid point happens to land near a pole — see
    /// the module docs and [`crate::negf::DEFAULT_ETA`]. Increase this if
    /// the loop fails to converge; decrease for sharper resonance
    /// resolution (at the cost of a noisier, potentially non-converging
    /// iteration).
    pub eta: f64,
}

impl Default for SelfConsistentOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            mixing: 0.3,
            green_energy_fraction: 0.7,
            eta: 0.08,
        }
    }
}

pub struct SelfConsistentResult {
    pub iterations: usize,
    pub residual: f64,
    pub green: GreenFunctionResult,
    pub electron_density: Vec<f64>,
}

fn negf_energy_grid(device: &Device, fraction: f64) -> Vec<f64> {
    let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
    let e_max = fraction * device.e_max;
    let d_e = device.params.d_e;
    let steps = ((e_max - e_min) / d_e).floor().max(0.0) as usize;
    (0..=steps).map(|k| e_min + k as f64 * d_e).collect()
}

/// Run the self-consistent Poisson<->NEGF loop on `device`, mutating its
/// `rho` and `psi_f` in place. Returns diagnostics plus the final NEGF
/// sweep and electron density on success.
pub fn solve_self_consistent(
    device: &mut Device,
    opts: &SelfConsistentOptions,
) -> Result<SelfConsistentResult> {
    if !(0.0 < opts.mixing && opts.mixing <= 1.0) {
        return Err(NegForgeError::InvalidParameter(format!(
            "mixing must be in (0, 1], got {}",
            opts.mixing
        )));
    }

    device.calc_potential();

    let mut last_residual = f64::INFINITY;
    for iteration in 1..=opts.max_iterations {
        let energies = negf_energy_grid(device, opts.green_energy_fraction);
        let green = negf::green_function_sweep(device, &energies, opts.eta);
        let n_electron = charge::electron_density(
            &green,
            device.t_hop,
            device.params.a,
            device.psi_f[0],
            device.psi_f[device.n - 1],
            device.params.e_fs,
            device.e_fd,
            device.params.t,
            device.params.d_e,
        );

        let mixing = opts.mixing;
        for (rho_i, &n_i) in device.rho.iter_mut().zip(n_electron.iter()) {
            let target_rho = -ELEMENTARY_CHARGE * n_i * 1e9;
            *rho_i += mixing * (target_rho - *rho_i);
        }

        let psi_before = device.psi_f.clone();
        device.calc_potential();

        let residual = psi_before
            .iter()
            .zip(device.psi_f.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        last_residual = residual;

        if residual < opts.tolerance {
            return Ok(SelfConsistentResult {
                iterations: iteration,
                residual,
                green,
                electron_density: n_electron,
            });
        }
    }

    Err(NegForgeError::NotConverged {
        iterations: opts.max_iterations,
        residual: last_residual,
        tolerance: opts.tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceParams;

    #[test]
    fn converges_for_a_small_device() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            d_e: 0.01,
            ..Default::default()
        });

        let opts = SelfConsistentOptions {
            max_iterations: 100,
            tolerance: 1e-5,
            mixing: 0.3,
            green_energy_fraction: 0.7,
            eta: 0.04,
        };
        let result = solve_self_consistent(&mut device, &opts).expect("should converge");
        assert!(result.residual < opts.tolerance);
        assert_eq!(result.electron_density.len(), device.n);
        // Structurally guaranteed (electron_density is a sum of |G|^2-weighted
        // terms, see charge.rs), independent of the contact self-energy sign
        // question discussed in negf.rs — worth locking in as a regression
        // check since it's the one physical positivity requirement the loop
        // actually depends on.
        assert!(result
            .electron_density
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
        assert!(device.psi_f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_invalid_mixing() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        let opts = SelfConsistentOptions {
            mixing: 0.0,
            ..Default::default()
        };
        assert!(solve_self_consistent(&mut device, &opts).is_err());
    }
}
