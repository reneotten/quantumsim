//! NEGF-derived electron density, feeding the self-consistent electrostatic
//! loop.
//!
//! The density is the standard contact-injection expression: each contact
//! fills the states it couples to, weighted by its own Fermi function,
//!
//! ```text
//! n(x_i) = (g_s / 2pi) * integral dE [ Gamma_s(E) |G_{i,0}(E)|^2 f_s(E)
//!                                    + Gamma_d(E) |G_{i,N-1}(E)|^2 f_d(E) ] / a
//! ```
//!
//! where `Gamma = -2 Im(Sigma^r)` is the contact broadening, `[G Gamma_s G^†]_ii`
//! collapses to `Gamma_s |G_{i,0}|^2` because `Gamma_s` has a single nonzero
//! element, and `g_s = 2` is the spin degeneracy — the same factor 2 that
//! appears in the `2e/h` prefactor of the current. Dividing by the grid
//! spacing `a` turns a per-site occupancy into a linear density in nm^-1.
//!
//! This differs from `calc_n()` in `legacy_matlab/quantumsim.m` in four ways,
//! all corrections rather than ports:
//!
//! 1. The original multiplies the whole energy sum by a single scalar
//!    `f(E_fs)` / `f(E_fd)` (the Fermi function evaluated once, outside the
//!    sum) instead of the energy-dependent occupation `f_s(E)` / `f_d(E)`
//!    used everywhere else in the model. That collapses the energy dependence
//!    of the injected charge to a single number.
//! 2. The original reuses one `mask = E > Psi_f(1)` (the *source* band edge)
//!    for both contact terms. Each contact's broadening now follows from its
//!    own self-energy, which vanishes outside its own band automatically — no
//!    mask needed.
//! 3. The original omits spin degeneracy here while including it in the
//!    current, so the charge fed back into Poisson was a factor 2 light.
//! 4. `Gamma`, not `Gamma / 2`, is the correct weight against the `1/2pi`
//!    prefactor. Together with (3) this leaves the numerical prefactor at
//!    `1/pi` — the same constant the original used, now for the right reason.
//!
//! The result is a *linear* density in nm^-1, the model's native length unit.
//! Converting it to the volume charge density that `Device::rho` expects
//! (which needs the device cross-section) is the caller's job — see
//! `selfconsistent.rs`.

use crate::constants::{E, K_B};
use crate::negf::GreenFunctionResult;

/// Spin degeneracy, matching the factor 2 in the `2e/h` current prefactor.
const SPIN_DEGENERACY: f64 = 2.0;

/// Compute the electron number density `n(x)` in nm^-1 from a Green's-function
/// sweep. `green` only has to carry the boundary columns
/// ([`crate::negf::GreenOutputs::BoundaryColumns`] is enough).
pub fn electron_density(
    green: &GreenFunctionResult,
    a: f64,
    e_fs: f64,
    e_fd: f64,
    temperature: f64,
    d_e: f64,
) -> Vec<f64> {
    let n_sites = green.n_sites;
    let mut n = vec![0.0; n_sites];

    let thermal = K_B * temperature / E;
    let f_s = |energy: f64| 1.0 / (((energy - e_fs) / thermal).exp() + 1.0);
    let f_d = |energy: f64| 1.0 / (((energy - e_fd) / thermal).exp() + 1.0);

    for (k, &energy) in green.energies.iter().enumerate() {
        // Gamma is zero outside each contact's band, so evanescent energies
        // contribute nothing without an explicit band-edge mask.
        let weight_s = green.gamma_source(k) * f_s(energy);
        let weight_d = green.gamma_drain(k) * f_d(energy);
        if weight_s == 0.0 && weight_d == 0.0 {
            continue;
        }

        let col_s = green.col_source_row(k);
        let col_d = green.col_drain_row(k);
        for (i, n_i) in n.iter_mut().enumerate() {
            *n_i += weight_s * col_s[i] + weight_d * col_d[i];
        }
    }

    // g_s / 2pi, times the energy-grid spacing, per unit length.
    let prefactor = SPIN_DEGENERACY * d_e / (2.0 * std::f64::consts::PI * a);
    for v in n.iter_mut() {
        *v *= prefactor;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceParams};
    use crate::negf::{self, GreenFunctionAlgorithm, GreenOutputs, DEFAULT_ETA};

    #[test]
    fn density_is_positive_and_falls_off_where_the_barrier_is_highest() {
        let device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            v_ds: 0.1,
            v_g: 0.0,
            d_e: 0.01,
            ..Default::default()
        });
        let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
        let energies: Vec<f64> = (0..80).map(|k| e_min + k as f64 * 0.01).collect();
        let green = negf::green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::BoundaryColumns,
        );
        let n = electron_density(
            &green,
            device.params.a,
            device.params.e_fs,
            device.e_fd,
            device.params.t,
            device.params.d_e,
        );

        assert_eq!(n.len(), device.n);
        assert!(n.iter().all(|v| v.is_finite() && *v >= 0.0));
        // Contacts are the reservoirs; the gated channel sits behind a barrier
        // and must hold less charge than the source contact does.
        let barrier_site = device.n / 2;
        assert!(
            n[0] > n[barrier_site],
            "expected contact density {} to exceed channel density {}",
            n[0],
            n[barrier_site]
        );
    }
}
