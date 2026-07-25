//! Non-Equilibrium Green's Function (NEGF) transport calculation.
//!
//! Port of `calc_green` / `calc_n` in `legacy_matlab/quantumsim.m`: builds
//! the tight-binding Hamiltonian for the potential profile `psi_f`, attaches
//! open-boundary self-energies at the source/drain contacts, and computes
//! the retarded Green's function `G^r(E) = [(E + i*eta) I - H]^-1` for a
//! grid of energies. Only the diagonal (local density of states) and the
//! first/last columns (needed for the injected charge density) are needed.
//!
//! ## Dense vs. recursive: the tradeoff
//!
//! Two algorithms compute those same outputs ([`GreenFunctionAlgorithm`]):
//!
//! - **[`GreenFunctionAlgorithm::Dense`]**: build the full `N x N` complex
//!   matrix and invert it (Gauss-Jordan elimination), then read off the
//!   diagonal and the two boundary columns. This is what the original
//!   MATLAB `inv()` call did. It's O(N^3) time and O(N^2) memory per energy
//!   point. The appeal is that it's *obviously* correct — a direct
//!   definition-level matrix inversion with no tridiagonal-structure
//!   assumptions and no recursion formulas to get subtly wrong — which is
//!   exactly why it exists here too, as a trusted reference to validate the
//!   fast path against (see the tests below) and as a fallback that keeps
//!   working if the Hamiltonian ever stops being purely tridiagonal (e.g. a
//!   future extension adding longer-range hopping or a non-nearest-neighbor
//!   coupling).
//! - **[`GreenFunctionAlgorithm::Recursive`]** (default): a left-connected
//!   recursive Green's-function sweep (O(N)) for the diagonal, plus a
//!   direct tridiagonal solve (O(N), via [`tridiag::solve_complex`]) for
//!   each boundary column — sidestepping the (easy to get wrong)
//!   off-diagonal recursive Green's-function formulas entirely. This is
//!   the one to use for anything performance-sensitive: a single sweep at
//!   `N = 561` (the model's default device size) is already ~200x fewer
//!   floating-point operations than dense, and the gap grows with N. It's
//!   also the only one fast enough to call repeatedly inside the
//!   self-consistent loop or a bias sweep — see `selfconsistent.rs`.
//!
//! Both are exercised and cross-checked against each other in the test
//! module below; `crates/negforge-core/examples/benchmark.rs` has wall-clock
//! numbers.
//!
//! ## Parallelism
//!
//! Every energy point is an independent Green's-function evaluation (they
//! only share the read-only Hamiltonian built from the device's current
//! `psi_f`), so the energy loop is embarrassingly parallel. It's run
//! through `rayon`'s `par_iter`, which spreads energy points across all
//! available CPU cores automatically (respects `RAYON_NUM_THREADS` if you
//! want to cap it). This benefits both algorithms, but matters far more for
//! `Dense`, where each energy point is individually expensive.

use num_complex::Complex64;
use rayon::prelude::*;

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

/// Which algorithm [`green_function_sweep`] uses to evaluate the retarded
/// Green's function at each energy point. See the module docs for the
/// tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreenFunctionAlgorithm {
    /// O(N) recursive Green's-function sweep + tridiagonal solves. Use this
    /// for anything performance-sensitive (self-consistent loop, bias
    /// sweeps, large devices). Default.
    #[default]
    Recursive,
    /// O(N^3) full dense matrix inversion, matching the original MATLAB
    /// `inv()` call. Simple and easy to trust, but slow — mainly useful as
    /// a reference for cross-validation, for small devices, or if you
    /// don't trust the recursive path for some reason.
    Dense,
}

/// Result of a [`green_function_sweep`] call: retarded-Green's-function
/// quantities on an energy grid, one entry per energy point.
pub struct GreenFunctionResult {
    /// Energy grid, eV.
    pub energies: Vec<f64>,
    /// `Im(G_ii(E)) / a` for every site `i`, indexed `[energy][site]`.
    /// Proportional to the local density of states (matches `G_r_diag`).
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

/// One energy point's worth of Green's-function output: `(k_sa, k_da,
/// diag_row, col_source_row, col_drain_row)`.
type EnergyPointResult = (Complex64, Complex64, Vec<f64>, Vec<f64>, Vec<f64>);

/// Run the NEGF sweep over `energies` for the device's current `psi_f`,
/// with imaginary broadening `eta` (see [`DEFAULT_ETA`] for guidance on
/// choosing it) and the given [`GreenFunctionAlgorithm`]. Mirrors
/// `calc_green()`. The caller chooses the energy grid (the original
/// hardcoded `min(Psi_f) : dE : 0.7*E_max`).
///
/// Energy points are evaluated in parallel across available CPU cores (see
/// the module docs).
pub fn green_function_sweep(
    device: &Device,
    energies: &[f64],
    eta: f64,
    algorithm: GreenFunctionAlgorithm,
) -> GreenFunctionResult {
    let n = device.n;
    let t = device.t_hop;
    let a = device.params.a;
    let psi_0s = device.psi_f[0];
    let psi_0d = device.psi_f[n - 1];

    // Bulk on-site energies (before contact self-energy is applied), and
    // the constant real off-diagonal — shared read-only state across all
    // (parallel) energy-point evaluations.
    let bulk_diag: Vec<f64> = device.psi_f.iter().map(|p| 2.0 * t + p).collect();
    let c1 = bulk_diag[0];
    let c_n = bulk_diag[n - 1];
    let off_diag = vec![t; n - 1];

    let per_energy: Vec<EnergyPointResult> = energies
        .par_iter()
        .map(|&e| {
            energy_point(
                e, eta, t, a, psi_0s, psi_0d, c1, c_n, &bulk_diag, &off_diag, algorithm,
            )
        })
        .collect();

    let mut k_sa = Vec::with_capacity(energies.len());
    let mut k_da = Vec::with_capacity(energies.len());
    let mut g_diag = Vec::with_capacity(energies.len());
    let mut g_col_source = Vec::with_capacity(energies.len());
    let mut g_col_drain = Vec::with_capacity(energies.len());
    for (ka_s, ka_d, diag_row, col_source_row, col_drain_row) in per_energy {
        k_sa.push(ka_s);
        k_da.push(ka_d);
        g_diag.push(diag_row);
        g_col_source.push(col_source_row);
        g_col_drain.push(col_drain_row);
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

#[allow(clippy::too_many_arguments)]
fn energy_point(
    e: f64,
    eta: f64,
    t: f64,
    a: f64,
    psi_0s: f64,
    psi_0d: f64,
    c1: f64,
    c_n: f64,
    bulk_diag: &[f64],
    off_diag: &[f64],
    algorithm: GreenFunctionAlgorithm,
) -> EnergyPointResult {
    let n = bulk_diag.len();
    let ka_s = Complex64::new(-(2.0 * t - e + psi_0s) / (2.0 * t), 0.0).acos();
    let ka_d = Complex64::new(-(2.0 * t - e + psi_0d) / (2.0 * t), 0.0).acos();

    let mut diag: Vec<Complex64> = bulk_diag
        .iter()
        .map(|&d| Complex64::new(e, eta) - Complex64::new(d, 0.0))
        .collect();

    if e >= psi_0s {
        let sigma_l = t * (Complex64::i() * ka_s).exp();
        diag[0] = Complex64::new(e, eta) - (Complex64::new(c1, 0.0) + sigma_l);
    }
    if e >= psi_0d {
        let sigma_r = t * (Complex64::i() * ka_d).exp();
        diag[n - 1] = Complex64::new(e, eta) - (Complex64::new(c_n, 0.0) + sigma_r);
    }

    let (diag_row, col_source, col_drain) = match algorithm {
        GreenFunctionAlgorithm::Recursive => {
            let full_diag = full_diagonal(off_diag, &diag);

            let mut e_source = vec![Complex64::new(0.0, 0.0); n];
            e_source[0] = Complex64::new(1.0, 0.0);
            let col_source = tridiag::solve_complex(off_diag, &diag, off_diag, &e_source);

            let mut e_drain = vec![Complex64::new(0.0, 0.0); n];
            e_drain[n - 1] = Complex64::new(1.0, 0.0);
            let col_drain = tridiag::solve_complex(off_diag, &diag, off_diag, &e_drain);

            (full_diag, col_source, col_drain)
        }
        GreenFunctionAlgorithm::Dense => {
            let inverse = dense_full_inverse(off_diag, &diag);
            let full_diag: Vec<Complex64> = (0..n).map(|i| inverse[i][i]).collect();
            let col_source: Vec<Complex64> = (0..n).map(|i| inverse[i][0]).collect();
            let col_drain: Vec<Complex64> = (0..n).map(|i| inverse[i][n - 1]).collect();
            (full_diag, col_source, col_drain)
        }
    };

    (
        ka_s,
        ka_d,
        diag_row.iter().map(|g| g.im / a).collect(),
        col_source.iter().map(|g| g.norm_sqr()).collect(),
        col_drain.iter().map(|g| g.norm_sqr()).collect(),
    )
}

/// Full diagonal of `A^-1` for a tridiagonal `A` with constant real
/// off-diagonal `t` (i.e. `A[i][i+1] = A[i+1][i] = t` for all `i`) and
/// complex diagonal `diag`, via the standard left-connected recursive
/// Green's function sweep. O(N).
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

/// Full `A^-1` for a tridiagonal `A` with constant real off-diagonal `t`
/// and complex diagonal `diag`, via Gauss-Jordan elimination with partial
/// pivoting on the dense `N x N` matrix. O(N^3). This is the
/// [`GreenFunctionAlgorithm::Dense`] backend, and also serves as the
/// reference implementation the recursive path is validated against in
/// tests.
#[allow(clippy::needless_range_loop)]
fn dense_full_inverse(off_diag: &[f64], diag: &[Complex64]) -> Vec<Vec<Complex64>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceParams};
    use approx::assert_relative_eq;

    #[test]
    fn recursive_diagonal_matches_dense_inverse() {
        let n = 9;
        let off_diag: Vec<f64> = vec![0.37; n - 1];
        let diag: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(-1.5 - 0.3 * i as f64, 1e-6))
            .collect();

        let recursive = full_diagonal(&off_diag, &diag);
        let dense = dense_full_inverse(&off_diag, &diag);

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

        let dense = dense_full_inverse(&off_diag, &diag);

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
        let result = green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Recursive,
        );

        assert_eq!(result.g_diag.len(), energies.len());
        for row in &result.g_diag {
            assert_eq!(row.len(), device.n);
            assert!(row.iter().all(|v| v.is_finite()));
        }
        for row in &result.g_col_source {
            assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
    }

    #[test]
    fn dense_and_recursive_algorithms_agree_end_to_end() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        device.calc_potential();

        let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
        let energies: Vec<f64> = (0..15).map(|i| e_min + i as f64 * 0.02).collect();

        let recursive = green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Recursive,
        );
        let dense = green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Dense,
        );

        for k in 0..energies.len() {
            for i in 0..device.n {
                assert_relative_eq!(recursive.g_diag[k][i], dense.g_diag[k][i], epsilon = 1e-6);
                assert_relative_eq!(
                    recursive.g_col_source[k][i],
                    dense.g_col_source[k][i],
                    epsilon = 1e-6
                );
                assert_relative_eq!(
                    recursive.g_col_drain[k][i],
                    dense.g_col_drain[k][i],
                    epsilon = 1e-6
                );
            }
        }
    }
}
