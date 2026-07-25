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
//! [`charge::electron_density`] returns a *linear* density in nm^-1, while
//! `Device::calc_potential` needs a volume charge density. The conversion
//! needs the channel cross-section and a unit rescaling, both of which are
//! easy to get silently wrong:
//!
//! 1. Divide by the cross-section [`Device::cross_section_nm2`] to get a
//!    volume density in nm^-3, then multiply by 1e27 for m^-3.
//! 2. Multiply by `-e` (electrons are negatively charged) for C/m^3.
//! 3. Multiply by [`device::RHO_SI_TO_MODEL`] (1e-18), because the
//!    electrostatic solve's Laplacian is built in nm^-2 — see the
//!    `device` module docs.
//!
//! Steps 1-3 collapse to `rho(x) = -e * n(x) * 1e9 / cross_section_nm2`.
//!
//! The cross-section matters: it sets how strongly the charge talks back to
//! the potential. For the default geometry the charge term reaches roughly
//! 10% of the gate/built-in term, so self-consistency is a real but
//! perturbative correction. Dropping the cross-section entirely (equivalent
//! to assuming 1 nm^2) makes it the *dominant* term, which is how the
//! original scaling behaved.

use crate::charge;
use crate::constants::E as ELEMENTARY_CHARGE;
use crate::device::{self, Device};
use crate::error::{NegForgeError, Result};
use crate::negf::{self, GreenFunctionAlgorithm, GreenFunctionResult, GreenOutputs};

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
    /// `None` (the default) uses [`ETA_GRID_FACTOR`] times the device's
    /// energy-grid step `d_e`.
    ///
    /// This is a numerical-integration parameter, not physics: it broadens
    /// every spectral feature by `2*eta`. It only has to be large enough that
    /// a resonance is resolved over several energy-grid points — otherwise
    /// the charge estimate jumps around with grid alignment between
    /// iterations and the fixed point never settles — so it should scale with
    /// `d_e`, not be a fixed number. Increase it if the loop fails to
    /// converge; decrease for sharper resonances.
    ///
    /// Note this is ~1-2 orders of magnitude below the value the loop needed
    /// before the contact self-energy sign was corrected: with the wrong
    /// sign, `eta` partially cancelled the contact broadening, and only a
    /// large `eta` kept the iteration stable (see the `negf` module docs).
    pub eta: Option<f64>,
    /// Which NEGF algorithm to use for the sweep inside each iteration.
    /// Defaults to `Recursive` (O(N)) — see `negf.rs` module docs for the
    /// dense-vs-recursive tradeoff. `Dense` works too, just far slower per
    /// iteration; mainly useful for cross-validating a suspicious result.
    pub algorithm: GreenFunctionAlgorithm,
}

/// Default [`SelfConsistentOptions::eta`], as a multiple of the device's
/// energy-grid step `d_e`. Empirically the smallest factor that converges
/// robustly across device scales; 1x `d_e` does not converge at all.
pub const ETA_GRID_FACTOR: f64 = 5.0;

impl Default for SelfConsistentOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            mixing: 0.3,
            green_energy_fraction: 0.7,
            eta: None,
            algorithm: GreenFunctionAlgorithm::Recursive,
        }
    }
}

pub struct SelfConsistentResult {
    pub iterations: usize,
    pub residual: f64,
    /// The converged sweep. Computed with
    /// [`GreenOutputs::BoundaryColumns`], so its `ldos` is empty — re-run
    /// [`negf::green_function_sweep`] on the converged device if you want
    /// the local density of states.
    pub green: GreenFunctionResult,
    /// Converged electron density, nm^-1.
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

    let eta = opts.eta.unwrap_or(ETA_GRID_FACTOR * device.params.d_e);
    if eta <= 0.0 || !eta.is_finite() {
        return Err(NegForgeError::InvalidParameter(format!(
            "eta must be positive, got {eta}"
        )));
    }

    let mut last_residual = f64::INFINITY;
    for iteration in 1..=opts.max_iterations {
        let energies = negf_energy_grid(device, opts.green_energy_fraction);
        // The charge density only reads the boundary columns, so the LDOS
        // diagonal (~30% of the recursive sweep, and the largest output
        // array) is never computed here.
        let green = negf::green_function_sweep(
            device,
            &energies,
            eta,
            opts.algorithm,
            GreenOutputs::BoundaryColumns,
        );
        let n_electron = charge::electron_density(
            &green,
            device.params.a,
            device.params.e_fs,
            device.e_fd,
            device.params.t,
            device.params.d_e,
        );

        // nm^-1 -> C/m^3 -> model units; see the module docs.
        let rho_scale =
            -ELEMENTARY_CHARGE * 1e27 * device::RHO_SI_TO_MODEL / device.cross_section_nm2;
        let mixing = opts.mixing;
        for (rho_i, &n_i) in device.rho.iter_mut().zip(n_electron.iter()) {
            let target_rho = rho_scale * n_i;
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
            eta: Some(0.04),
            algorithm: GreenFunctionAlgorithm::Recursive,
        };
        let result = solve_self_consistent(&mut device, &opts).expect("should converge");
        assert!(result.residual < opts.tolerance);
        assert_eq!(result.electron_density.len(), device.n);
        assert!(result.electron_density.iter().all(|v| v.is_finite()));
        assert!(device.psi_f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dense_and_recursive_converge_to_the_same_potential() {
        let make_device = || {
            Device::new(DeviceParams {
                a: 1.0,
                l_ch: 10.0,
                l_ds: 10.0,
                auto_size_contacts: false,
                d_e: 0.01,
                ..Default::default()
            })
        };
        let base_opts = SelfConsistentOptions {
            max_iterations: 100,
            tolerance: 1e-5,
            mixing: 0.3,
            green_energy_fraction: 0.7,
            eta: Some(0.04),
            algorithm: GreenFunctionAlgorithm::Recursive,
        };

        let mut recursive_device = make_device();
        solve_self_consistent(&mut recursive_device, &base_opts)
            .expect("recursive should converge");

        let mut dense_device = make_device();
        let dense_opts = SelfConsistentOptions {
            algorithm: GreenFunctionAlgorithm::Dense,
            ..base_opts
        };
        solve_self_consistent(&mut dense_device, &dense_opts).expect("dense should converge");

        for (a, b) in recursive_device.psi_f.iter().zip(dense_device.psi_f.iter()) {
            assert!((a - b).abs() < 1e-4, "psi_f mismatch: {a} vs {b}");
        }
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
