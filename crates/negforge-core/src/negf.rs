//! Non-Equilibrium Green's Function (NEGF) transport calculation.
//!
//! Port of `calc_green` / `calc_n` in `legacy_matlab/quantumsim.m`: builds
//! the tight-binding Hamiltonian for the potential profile `psi_f`, attaches
//! open-boundary self-energies at the source/drain contacts, and computes
//! the retarded Green's function `G^r(E) = [(E + i*eta) I - H]^-1` for a
//! grid of energies.
//!
//! Only the diagonal (local density of states) and the first/last columns
//! (needed for the injected charge density) are needed, so rather than a
//! dense O(N^3) matrix inversion (what the original MATLAB `inv()` call
//! did) this uses:
//!
//! - a left-connected recursive Green's function sweep (O(N)) for the full
//!   diagonal, and
//! - a direct tridiagonal solve (O(N), via [`tridiag::solve_complex`]) for
//!   each of the two boundary columns, sidestepping the (easy to get wrong)
//!   off-diagonal recursive Green's function formulas entirely.
//!
//! Both pieces are cross-checked against a dense reference solver in the
//! test module below (that check validates the O(N) algorithms reproduce
//! whatever matrix is actually constructed; it does not by itself validate
//! that the matrix's physics — the self-energy formula, sign convention,
//! etc. — is right, which is discussed below).
//!
//! ## Validity of the tight-binding discretization
//!
//! The on-site/hopping parameters (`2*t_hop + Psi_f`, `-t_hop`, with
//! `t_hop = hbar^2 / (2 m_eff a^2)`) are the standard finite-difference
//! discretization of the effective-mass Schrodinger equation, and reproduce
//! the continuum parabolic dispersion `E = hbar^2 k^2 / (2 m_eff)` only for
//! small `k*a` (long wavelength compared to the grid spacing). A
//! tight-binding chain's true dispersion is `E = 2*t_hop*(1 - cos(k*a))`,
//! bounded above by a bandwidth of `4*t_hop`; states populated near or above
//! that bandwidth are lattice artifacts with no continuum-model
//! counterpart, not real physics. In practice this means `a` must be chosen
//! fine enough that the energy window actually swept (`psi_0` up to
//! `e_max`) stays well below `4*t_hop` — worth checking (e.g. by halving
//! `a` and confirming the results are unchanged) before trusting results at
//! a new device scale.
//!
//! ## Contact self-energies: derivation, and a corrected sign
//!
//! The source and drain are modeled as semi-infinite continuations of the
//! same tight-binding chain (on-site `2*t_hop + Psi_f(contact)`, hopping
//! `-t_hop`), held in local equilibrium at their own Fermi level. Attaching
//! a semi-infinite lead to a boundary site is equivalent to adding a
//! *self-energy* `Sigma(E)` to that site's on-site energy, replacing the
//! (infinite) lead with a finite, energy-dependent boundary condition. For
//! this 1D chain the surface Green's function of the lead is known in
//! closed form, and the standard result (e.g. Datta, *Quantum Transport*,
//! §8.2–8.3) is
//!
//! ```text
//! Sigma(E) = -t_hop * exp(i * k(E) * a)
//! ```
//!
//! where `k(E)` solves the lead's own bulk dispersion `E = 2*t_hop*(1 -
//! cos(k*a)) + Psi_f(contact)`, taking the branch `k*a in [0, pi]` for
//! energies inside the propagating band (`Psi_f(contact) <= E <=
//! Psi_f(contact) + 4*t_hop`). The code below instead computes
//!
//! ```text
//! ka_s = acos(-(2*t_hop - E + Psi_f(contact)) / (2*t_hop))
//! ```
//!
//! which is *not* `k(E)*a` from that dispersion relation — algebraically it
//! is `pi - k(E)*a`, the reflection of the correct branch. Composed with a
//! `+i` in the exponent (as `calc_green()`'s original `t*exp(1i*k_sa)`
//! did), that reflection flips the sign of the self-energy's real part
//! *and* gives it the wrong-signed imaginary part; composed with `-i`
//! instead, the `pi` phase and the branch reflection cancel out exactly:
//!
//! ```text
//! t_hop * exp(-i * ka_s) = t_hop * exp(-i*(pi - k(E)*a))
//!                         = t_hop * exp(-i*pi) * exp(i*k(E)*a)
//!                         = -t_hop * exp(i*k(E)*a)   =  Sigma(E)
//! ```
//!
//! So `sigma_l = t_hop * exp(-i * ka_s)` (implemented below) is exactly the
//! textbook self-energy, without needing to touch the `ka_s`/`ka_d`
//! formulas or any other consumer of them — `sin(ka_s)` (used for the
//! broadening factor in `crate::charge::electron_density`) is unchanged by
//! the reflection (`sin(pi - x) = sin(x)`), and `k_sa`/`k_da` are still
//! reported in the code's original (reflected) branch for anyone consuming
//! them directly.
//!
//! **Why this matters, concretely.** A causal, absorbing (as opposed to
//! amplifying) contact requires `Im(Sigma) <= 0`, so that the broadening
//! `Gamma = -2*Im(Sigma) >= 0`. The original `+i` formula gave `Im(Sigma) =
//! +t_hop*sin(k*a) > 0` for every propagating mode — backwards. This wasn't
//! just a relabeling that washed out elsewhere: with the wrong sign, the
//! effective damping at the contacts was negative, and `g_diag` (the
//! `-Im(G_ii)/a` local-density-of-states proxy) could go materially
//! negative — confirmed empirically before this fix, at exactly the
//! broadening used inside the self-consistent loop — which is unphysical
//! for a real density of states. `crate::charge::electron_density` was
//! shielded from producing a negative *charge* density purely because it
//! only ever uses squared magnitudes `|G|^2`, but its quantitative values
//! (resonance heights/widths) were still computed from a non-causal Green's
//! function. With the corrected sign, `g_diag` is non-negative (see
//! `tests::g_diag_is_non_negative_at_self_consistent_loop_broadening`), as
//! required for a properly causal system.
//!
//! This bug was inherited unchanged from `quantumsim.m`'s `calc_green()`
//! (not introduced by this rewrite), and is now fixed here — a deliberate,
//! documented deviation from a byte-for-byte port, on top of the ones
//! listed in the top-level README.

use num_complex::Complex64;

use crate::device::Device;
use crate::tridiag;

/// Default imaginary broadening added to the energy, `E + i*eta`, standing
/// in for an infinitesimal escape rate. Matches `eta = 1i*1e-8` in the
/// original code; suitable for a one-off local-density-of-states plot at a
/// fixed potential.
///
/// This value is deliberately *not* used inside the self-consistent loop
/// (see `selfconsistent.rs`): with such a small broadening, a bound-state
/// resonance whose energy happens to land within `eta` of an energy-grid
/// point produces a `|G|^2` spike many orders of magnitude larger than
/// neighboring grid points (the finite chain's true poles are only
/// infinitesimally broadened). Feeding that spiky, grid-alignment-dependent
/// charge estimate back into the electrostatic solve makes the fixed-point
/// iteration diverge. The self-consistent loop instead uses a broadening
/// comparable to the energy-grid spacing, so resonances are resolved over
/// several grid points and the charge estimate varies smoothly between
/// iterations. This is standard NEGF numerical practice (`eta` is already
/// an artificial regularization in the model, not a physical dephasing
/// rate) rather than a physics change.
pub const DEFAULT_ETA: f64 = 1e-8;

/// Result of a [`green_function_sweep`] call: retarded-Green's-function
/// quantities on an energy grid, one entry per energy point.
pub struct GreenFunctionResult {
    /// Energy grid, eV.
    pub energies: Vec<f64>,
    /// `-Im(G_ii(E)) / a` for every site `i`, indexed `[energy][site]`.
    /// Proportional to the local density of states (`(a / pi)` times this is
    /// the textbook LDOS `-Im(G_ii)/pi`; the original MATLAB `G_r_diag`
    /// carried neither the minus sign nor the `1/pi`, see the module docs'
    /// "Contact self-energies" section for why the sign is corrected here).
    pub g_diag: Vec<Vec<f64>>,
    /// `|G_{i,0}(E)|^2` for every site `i`, indexed `[energy][site]`
    /// (matches `G_r_1N(:,:,1)`, the source-injected column).
    pub g_col_source: Vec<Vec<f64>>,
    /// `|G_{i,N-1}(E)|^2` for every site `i`, indexed `[energy][site]`
    /// (matches `G_r_1N(:,:,2)`, the drain-injected column).
    pub g_col_drain: Vec<Vec<f64>>,
    /// Source contact wavevector at each energy (complex: evanescent below
    /// the contact band edge). Matches `k_sa`.
    pub k_sa: Vec<Complex64>,
    /// Drain contact wavevector at each energy. Matches `k_da`.
    pub k_da: Vec<Complex64>,
}

/// Run the NEGF sweep over `energies` for the device's current `psi_f`,
/// with imaginary broadening `eta` (see [`DEFAULT_ETA`] for guidance on
/// choosing it). Mirrors `calc_green()`. The caller chooses the energy grid
/// (the original hardcoded `min(Psi_f) : dE : 0.7*E_max`).
pub fn green_function_sweep(device: &Device, energies: &[f64], eta: f64) -> GreenFunctionResult {
    let n = device.n;
    let t = device.t_hop;
    let a = device.params.a;
    let psi_0s = device.psi_f[0];
    let psi_0d = device.psi_f[n - 1];

    // Bulk on-site energies (before contact self-energy is applied).
    let bulk_diag: Vec<f64> = device.psi_f.iter().map(|p| 2.0 * t + p).collect();
    let c1 = bulk_diag[0];
    let c_n = bulk_diag[n - 1];
    let off_diag = vec![t; n - 1]; // constant real off-diagonal of (E+i eta)I - H

    let mut g_diag = Vec::with_capacity(energies.len());
    let mut g_col_source = Vec::with_capacity(energies.len());
    let mut g_col_drain = Vec::with_capacity(energies.len());
    let mut k_sa = Vec::with_capacity(energies.len());
    let mut k_da = Vec::with_capacity(energies.len());

    for &e in energies {
        let ka_s = Complex64::new(-(2.0 * t - e + psi_0s) / (2.0 * t), 0.0).acos();
        let ka_d = Complex64::new(-(2.0 * t - e + psi_0d) / (2.0 * t), 0.0).acos();
        k_sa.push(ka_s);
        k_da.push(ka_d);

        let mut diag: Vec<Complex64> = bulk_diag
            .iter()
            .map(|&d| Complex64::new(e, eta) - Complex64::new(d, 0.0))
            .collect();

        if e >= psi_0s {
            let sigma_l = t * (-(Complex64::i() * ka_s)).exp();
            diag[0] = Complex64::new(e, eta) - (Complex64::new(c1, 0.0) + sigma_l);
        }
        if e >= psi_0d {
            let sigma_r = t * (-(Complex64::i() * ka_d)).exp();
            diag[n - 1] = Complex64::new(e, eta) - (Complex64::new(c_n, 0.0) + sigma_r);
        }

        let full_diag = full_diagonal(&off_diag, &diag);

        let mut e_source = vec![Complex64::new(0.0, 0.0); n];
        e_source[0] = Complex64::new(1.0, 0.0);
        let col_source = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e_source);

        let mut e_drain = vec![Complex64::new(0.0, 0.0); n];
        e_drain[n - 1] = Complex64::new(1.0, 0.0);
        let col_drain = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e_drain);

        g_diag.push(full_diag.iter().map(|g| -g.im / a).collect());
        g_col_source.push(col_source.iter().map(|g| g.norm_sqr()).collect());
        g_col_drain.push(col_drain.iter().map(|g| g.norm_sqr()).collect());
    }

    GreenFunctionResult {
        energies: energies.to_vec(),
        g_diag,
        g_col_source,
        g_col_drain,
        k_sa,
        k_da,
    }
}

/// Full diagonal of `A^-1` for a tridiagonal `A` with constant real
/// off-diagonal `t` (i.e. `A[i][i+1] = A[i+1][i] = t` for all `i`) and
/// complex diagonal `diag`, via the standard left-connected recursive
/// Green's function sweep.
fn full_diagonal(off_diag: &[f64], diag: &[Complex64]) -> Vec<Complex64> {
    let n = diag.len();
    let mut g_left = vec![Complex64::new(0.0, 0.0); n];
    g_left[0] = Complex64::new(1.0, 0.0) / diag[0];
    for i in 1..n {
        let t = off_diag[i - 1];
        g_left[i] = Complex64::new(1.0, 0.0) / (diag[i] - g_left[i - 1] * (t * t));
    }

    let mut full = vec![Complex64::new(0.0, 0.0); n];
    full[n - 1] = g_left[n - 1];
    for i in (0..n - 1).rev() {
        let t = off_diag[i];
        full[i] = g_left[i] + g_left[i] * g_left[i] * (t * t) * full[i + 1];
    }
    full
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceParams};
    use approx::assert_relative_eq;

    /// Dense reference: build the full complex tridiagonal matrix and
    /// invert it via Gauss-Jordan elimination, for cross-checking the O(N)
    /// recursive/solve-based implementation above on small systems.
    fn dense_inverse(off_diag: &[f64], diag: &[Complex64]) -> Vec<Vec<Complex64>> {
        let n = diag.len();
        let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            a[i][i] = diag[i];
            if i > 0 {
                a[i][i - 1] = Complex64::new(off_diag[i - 1], 0.0);
            }
            if i < n - 1 {
                a[i][i + 1] = Complex64::new(off_diag[i], 0.0);
            }
        }
        let mut inv = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            inv[i][i] = Complex64::new(1.0, 0.0);
        }
        for col in 0..n {
            let mut pivot = col;
            for row in (col + 1)..n {
                if a[row][col].norm() > a[pivot][col].norm() {
                    pivot = row;
                }
            }
            a.swap(col, pivot);
            inv.swap(col, pivot);
            let pivot_val = a[col][col];
            for k in 0..n {
                a[col][k] /= pivot_val;
                inv[col][k] /= pivot_val;
            }
            let pivot_row_a = a[col].clone();
            let pivot_row_inv = inv[col].clone();
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = a[row][col];
                for k in 0..n {
                    a[row][k] -= factor * pivot_row_a[k];
                    inv[row][k] -= factor * pivot_row_inv[k];
                }
            }
        }
        inv
    }

    #[test]
    fn recursive_diagonal_matches_dense_inverse() {
        let n = 9;
        let off_diag: Vec<f64> = vec![0.37; n - 1];
        let diag: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(-1.5 - 0.3 * i as f64, 1e-6))
            .collect();

        let recursive = full_diagonal(&off_diag, &diag);
        let dense = dense_inverse(&off_diag, &diag);

        for i in 0..n {
            assert_relative_eq!(recursive[i].re, dense[i][i].re, epsilon = 1e-8);
            assert_relative_eq!(recursive[i].im, dense[i][i].im, epsilon = 1e-8);
        }
    }

    #[test]
    fn boundary_columns_match_dense_inverse() {
        let n = 9;
        let off_diag: Vec<f64> = vec![0.37; n - 1];
        let diag: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(-1.5 - 0.3 * i as f64, 1e-6))
            .collect();

        let mut e0 = vec![Complex64::new(0.0, 0.0); n];
        e0[0] = Complex64::new(1.0, 0.0);
        let col0 = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e0);

        let mut e_last = vec![Complex64::new(0.0, 0.0); n];
        e_last[n - 1] = Complex64::new(1.0, 0.0);
        let col_last = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e_last);

        let dense = dense_inverse(&off_diag, &diag);

        for i in 0..n {
            assert_relative_eq!(col0[i].re, dense[i][0].re, epsilon = 1e-8);
            assert_relative_eq!(col0[i].im, dense[i][0].im, epsilon = 1e-8);
            assert_relative_eq!(col_last[i].re, dense[i][n - 1].re, epsilon = 1e-8);
            assert_relative_eq!(col_last[i].im, dense[i][n - 1].im, epsilon = 1e-8);
        }
    }

    #[test]
    fn green_function_sweep_runs_and_is_finite() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        device.calc_potential();

        let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
        let energies: Vec<f64> = (0..20).map(|i| e_min + i as f64 * 0.01).collect();
        let result = green_function_sweep(&device, &energies, DEFAULT_ETA);

        assert_eq!(result.g_diag.len(), energies.len());
        for row in &result.g_diag {
            assert_eq!(row.len(), device.n);
            assert!(row.iter().all(|v| v.is_finite()));
        }
        for row in &result.g_col_source {
            assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
    }

    /// Larger `eta` must produce a smoother, better-resolved spectrum on a
    /// fixed energy grid. This is the property the Python `eta` argument on
    /// `local_density_of_states()` exists to expose: at the historical
    /// `DEFAULT_ETA` (1e-8 eV) the Lorentzian width is ~10^5 times narrower
    /// than a typical 1 meV grid step, so resonances fall between grid points
    /// and the sampled LDOS is a sparse set of spikes rather than a spectrum.
    #[test]
    fn larger_eta_resolves_more_of_the_spectrum_on_a_fixed_grid() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            v_g: 0.3,
            ..Default::default()
        });
        device.calc_potential();

        let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
        let energies: Vec<f64> = (0..400).map(|i| e_min + i as f64 * 0.001).collect();

        // Fraction of (energy, site) samples carrying non-negligible weight.
        let occupancy = |eta: f64| -> f64 {
            let result = green_function_sweep(&device, &energies, eta);
            let max = result
                .g_diag
                .iter()
                .flat_map(|row| row.iter())
                .fold(0.0f64, |acc, v| acc.max(v.abs()));
            assert!(max > 0.0, "sweep produced an all-zero LDOS");
            let total: usize = result.g_diag.iter().map(|row| row.len()).sum();
            let filled = result
                .g_diag
                .iter()
                .flat_map(|row| row.iter())
                .filter(|v| v.abs() > max * 1e-6)
                .count();
            filled as f64 / total as f64
        };

        let sharp = occupancy(DEFAULT_ETA);
        let broadened = occupancy(5e-3);

        assert!(
            broadened > sharp,
            "broadening should resolve more of the spectrum: \
             sharp(eta={DEFAULT_ETA:e})={sharp:.4}, broadened(eta=5e-3)={broadened:.4}"
        );
        assert!(
            broadened > 0.5,
            "eta a few times the grid step should fill most of the map, got {broadened:.4}"
        );
    }

    /// Documents the corrected sign claim from the module docs' "Contact
    /// self-energies" section: for the branch of `ka_s`/`ka_d` this module
    /// actually computes (`ka in (0, pi)`, the reflected branch), the
    /// self-energy `sigma = t_hop * exp(-i*ka)` — as constructed in
    /// [`green_function_sweep`] — has `Im(sigma) <= 0`, as required for a
    /// causal, absorbing (not amplifying) contact.
    #[test]
    fn contact_self_energy_has_non_positive_imaginary_part_for_propagating_modes() {
        let t = 1.69_f64;
        for ka in [0.2, 0.7_f64, std::f64::consts::FRAC_PI_2, 2.5, 3.0] {
            let sigma = t * (-(Complex64::i() * Complex64::new(ka, 0.0))).exp();
            assert!(
                sigma.im <= 0.0,
                "expected Im(sigma) <= 0 at ka={ka}, got {}",
                sigma.im
            );
            assert!((sigma.im + t * ka.sin()).abs() < 1e-12);
        }
    }

    /// With the corrected self-energy sign, `g_diag` (the `-Im(G_ii)/a`
    /// LDOS proxy) should be non-negative — the defining property of a
    /// proper (causal) local density of states, and the concrete symptom
    /// that was violated before the sign fix. Checked at the same
    /// device/energy range and broadening (`eta = 0.08`, the value used
    /// inside the self-consistent loop) that previously exhibited a clearly
    /// negative dip, as a regression guard.
    #[test]
    fn g_diag_is_non_negative_at_self_consistent_loop_broadening() {
        let mut device = Device::new(DeviceParams {
            a: 0.5,
            l_ch: 20.0,
            l_ds: 20.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        device.calc_potential();

        let energies: Vec<f64> = (0..200)
            .map(|i| device.psi_0 + i as f64 * 0.005)
            .collect();
        let result = green_function_sweep(&device, &energies, 0.08);

        let worst = result
            .g_diag
            .iter()
            .flat_map(|row| row.iter().cloned())
            .fold(f64::INFINITY, f64::min);
        assert!(worst >= -1e-9, "expected non-negative g_diag, got worst={worst}");
    }
}
