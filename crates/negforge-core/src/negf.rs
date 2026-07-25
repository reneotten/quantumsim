//! Non-Equilibrium Green's Function (NEGF) transport calculation.
//!
//! Builds the tight-binding Hamiltonian for the potential profile `psi_f`,
//! attaches open-boundary self-energies at the source/drain contacts, and
//! computes the retarded Green's function `G^r(E) = [(E + i*eta) I - H -
//! Sigma^r(E)]^-1` on a grid of energies. Only the diagonal (local density of
//! states) and the first/last columns (contact-injected charge, and the
//! transmission) are needed, so the full inverse is never formed on the fast
//! path.
//!
//! ## Contact self-energy sign convention
//!
//! The lead dispersion implied by [`Device::t_hop`] is
//! `E = psi_contact + 2t + 2t*cos(k a)`, so the band runs from `psi_contact`
//! (at `ka = pi`) to `psi_contact + 4t` (at `ka = 0`). The retarded
//! self-energy of a semi-infinite lead in that convention is
//!
//! ```text
//! Sigma^r = t * exp(-i k a)      =>   Im(Sigma^r) = -t sin(ka) <= 0
//! ```
//!
//! **This is a deliberate correction to the original MATLAB code**, which
//! used `+t*exp(+i k a)` (`quantumsim.m`, the `H(1)`/`H(end)` assignments).
//! With that sign, `Im(Sigma) > 0` and the contact term *subtracts* from the
//! `+i*eta` regularization instead of adding to it:
//!
//! - the resulting `G` is retarded on sites reached only by `eta` and
//!   advanced on contact-coupled sites, so `Im(G_ii)` is not sign-definite
//!   and the "LDOS" it produces is negative over part of the grid (measured:
//!   32% of the entries for the default device);
//! - inside the self-consistent loop, where `eta` is comparable to
//!   `t sin(ka)`, the two nearly cancel, inflating `|G|^2` at the contacts by
//!   5-14x and feeding a badly distorted charge density back into Poisson.
//!
//! With the correct sign, `Im[(E + i*eta) I - H - Sigma^r]` is positive
//! definite for any `eta > 0`, so `Im(G_ii) < 0` everywhere and the local
//! density of states `-Im(G_ii) / (pi a)` is positive by construction — the
//! tests below assert exactly that.
//!
//! The self-energy is attached at *every* energy, not only inside the contact
//! band. Outside it the closed form below is real (a level shift with no
//! broadening, `Gamma = 0`), which is the physically correct evanescent
//! behavior; the original dropped that shift by leaving the bare on-site
//! energy in place below the band edge.
//!
//! ## Dense vs. recursive: the tradeoff
//!
//! Two algorithms compute the same outputs ([`GreenFunctionAlgorithm`]):
//!
//! - **[`GreenFunctionAlgorithm::Dense`]**: build the full `N x N` complex
//!   matrix and eliminate on it directly. This is what the original MATLAB
//!   `inv()` call did. It's O(N^3) time and O(N^2) memory per energy point.
//!   The appeal is that it's *obviously* correct — no tridiagonal-structure
//!   assumptions and no recursion formulas to get subtly wrong — which is
//!   exactly why it exists here too, as a trusted reference to validate the
//!   fast path against (see the tests below) and as a fallback that keeps
//!   working if the Hamiltonian ever stops being purely tridiagonal (e.g. a
//!   future extension adding longer-range hopping).
//! - **[`GreenFunctionAlgorithm::Recursive`]** (default): the O(N)
//!   left-connected recursive Green's-function sweep for the diagonal, plus a
//!   single shared tridiagonal factorization for both boundary columns (see
//!   [`tridiag::boundary_columns_complex`]). This is the one to use for
//!   anything performance-sensitive: a sweep at `N = 561` (the model's
//!   default device size) is ~4 orders of magnitude cheaper than dense, and
//!   the gap grows with N.
//!
//! Both are cross-checked against each other in the test module below;
//! `crates/negforge-core/examples/benchmark.rs` has wall-clock numbers.
//!
//! ## Cost control
//!
//! [`GreenOutputs`] selects what the sweep computes. The self-consistent loop
//! only ever reads the boundary columns, so it asks for
//! [`GreenOutputs::BoundaryColumns`] and skips the diagonal recursion — about
//! a third of the recursive work, and the largest output array. Under `Dense`
//! the saving is larger still: two right-hand sides instead of a full inverse.
//!
//! ## Parallelism
//!
//! Every energy point is an independent evaluation (they only share the
//! read-only Hamiltonian built from the device's current `psi_f`), so the
//! energy loop is embarrassingly parallel and runs through `rayon`
//! (respecting `RAYON_NUM_THREADS`). Results are written straight into flat,
//! preallocated output buffers, and each worker keeps one reusable scratch
//! workspace, so the loop allocates nothing per energy point.
//!
//! A parallel bias sweep nests this inside its own `par_iter`. Rayon's
//! work-stealing copes with that: forcing the inner loop to run serially when
//! nested was measured *slower* on a 4-core box (2.85 s vs 2.58-2.64 s over
//! three runs, for a 21-point self-consistent sweep), so the nesting is
//! simply left alone.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::constants::{E as ELEMENTARY_CHARGE, H as PLANCK, K_B};
use crate::device::Device;
use crate::tridiag;

/// Default imaginary broadening added to the energy, `E + i*eta`, standing
/// in for an infinitesimal escape rate.
///
/// With the retarded self-energy sign convention (see the module docs), the
/// contacts already broaden every state that couples to them, so `eta` only
/// has to regularize states that don't — and can stay small. This is the
/// value used for one-off local-density-of-states plots and for transmission,
/// where the sharp resonances of the finite chain are what you want to see.
///
/// [`crate::selfconsistent::SelfConsistentOptions::eta`] uses a larger value:
/// inside a feedback loop, a resonance that lands between two energy-grid
/// points produces a charge estimate that jumps around with grid alignment,
/// and smoothing it over a few grid spacings makes the fixed-point iteration
/// well behaved. That is a numerical-integration concern, not physics.
pub const DEFAULT_ETA: f64 = 1e-8;

/// Which algorithm [`green_function_sweep`] uses at each energy point. See
/// the module docs for the tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreenFunctionAlgorithm {
    /// O(N) recursive Green's-function sweep + one shared tridiagonal
    /// factorization. Use this for anything performance-sensitive
    /// (self-consistent loop, bias sweeps, large devices). Default.
    #[default]
    Recursive,
    /// O(N^3) dense elimination, matching the original MATLAB `inv()` call.
    /// Simple and easy to trust, but slow — mainly useful as a reference for
    /// cross-validation and for small devices.
    Dense,
}

/// Which quantities [`green_function_sweep`] should produce. Computing the
/// diagonal costs about a third of the recursive sweep (much more under
/// `Dense`) and is pure waste for callers that only need the charge density
/// or the transmission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreenOutputs {
    /// Local density of states *and* both boundary columns.
    #[default]
    Full,
    /// Boundary columns only — everything the charge density and the
    /// Landauer-Caroli transmission need. [`GreenFunctionResult::ldos`] comes
    /// back empty.
    BoundaryColumns,
}

/// Result of a [`green_function_sweep`]: retarded-Green's-function quantities
/// on an energy grid.
///
/// The per-site arrays are flat, row-major `[energy][site]` buffers of length
/// `energies.len() * n_sites`; use [`GreenFunctionResult::ldos_row`] and
/// friends rather than indexing by hand.
pub struct GreenFunctionResult {
    /// Energy grid, eV.
    pub energies: Vec<f64>,
    /// Number of grid sites (the row length of the flat arrays below).
    pub n_sites: usize,
    /// Local density of states `-Im(G_ii) / (pi a)`, in states/(eV nm).
    /// Positive by construction. Empty when the sweep was run with
    /// [`GreenOutputs::BoundaryColumns`].
    pub ldos: Vec<f64>,
    /// `|G_{i,0}(E)|^2`, the source-injected column.
    pub col_source: Vec<f64>,
    /// `|G_{i,N-1}(E)|^2`, the drain-injected column.
    pub col_drain: Vec<f64>,
    /// Retarded source-contact self-energy at each energy.
    pub sigma_source: Vec<Complex64>,
    /// Retarded drain-contact self-energy at each energy.
    pub sigma_drain: Vec<Complex64>,
}

impl GreenFunctionResult {
    fn row<'a>(&self, data: &'a [f64], k: usize) -> &'a [f64] {
        &data[k * self.n_sites..(k + 1) * self.n_sites]
    }

    /// Local density of states across all sites at energy index `k`. Panics
    /// if the sweep was run without [`GreenOutputs::Full`].
    pub fn ldos_row(&self, k: usize) -> &[f64] {
        assert!(
            !self.ldos.is_empty(),
            "this sweep was run with GreenOutputs::BoundaryColumns; no diagonal was computed"
        );
        self.row(&self.ldos, k)
    }

    /// `|G_{i,0}|^2` across all sites at energy index `k`.
    pub fn col_source_row(&self, k: usize) -> &[f64] {
        self.row(&self.col_source, k)
    }

    /// `|G_{i,N-1}|^2` across all sites at energy index `k`.
    pub fn col_drain_row(&self, k: usize) -> &[f64] {
        self.row(&self.col_drain, k)
    }

    /// Source-contact broadening `Gamma_s = -2 Im(Sigma^r_s)`, eV. Zero
    /// outside the contact band, where the self-energy is purely real.
    pub fn gamma_source(&self, k: usize) -> f64 {
        -2.0 * self.sigma_source[k].im
    }

    /// Drain-contact broadening `Gamma_d = -2 Im(Sigma^r_d)`, eV.
    pub fn gamma_drain(&self, k: usize) -> f64 {
        -2.0 * self.sigma_drain[k].im
    }

    /// Landauer-Caroli transmission at energy index `k`,
    /// `T(E) = Tr[Gamma_s G^r Gamma_d G^a] = Gamma_s Gamma_d |G_{N-1,0}|^2`,
    /// which for this two-terminal chain needs only the source column already
    /// computed here. Zero unless the energy lies in *both* contact bands.
    pub fn transmission(&self, k: usize) -> f64 {
        self.gamma_source(k) * self.gamma_drain(k) * self.col_source_row(k)[self.n_sites - 1]
    }
}

/// Retarded self-energy of a semi-infinite lead whose band bottom sits at
/// `psi_contact`, for the dispersion `E = psi_contact + 2t + 2t cos(ka)`.
///
/// Inside the band this is `t exp(-i k a)`, with `Im < 0` as a retarded
/// self-energy must have. Outside it, the decaying root is selected, giving a
/// real level shift with `|Sigma| < t` and no broadening. Writing it in
/// closed form (rather than via a complex `acos`) keeps the branch choice
/// explicit instead of relying on the principal value landing on the
/// physical sheet.
fn contact_self_energy(e: f64, t: f64, psi_contact: f64) -> Complex64 {
    let z = (e - psi_contact - 2.0 * t) / (2.0 * t);
    if z.abs() <= 1.0 {
        // Propagating: exp(-ika) = cos(ka) - i sin(ka).
        Complex64::new(t * z, -t * (1.0 - z * z).sqrt())
    } else {
        // Evanescent: of the two real roots, take the one with |exp(-ika)| < 1
        // so the lead's influence decays into the device.
        Complex64::new(t * (z - z.signum() * (z * z - 1.0).sqrt()), 0.0)
    }
}

/// Per-worker scratch for the recursive path, so the energy loop allocates
/// nothing.
struct Scratch {
    diag: Vec<Complex64>,
    work: tridiag::ComplexWorkspace,
    buf_a: Vec<Complex64>,
    buf_b: Vec<Complex64>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            diag: vec![Complex64::new(0.0, 0.0); n],
            work: tridiag::ComplexWorkspace::new(n),
            buf_a: vec![Complex64::new(0.0, 0.0); n],
            buf_b: vec![Complex64::new(0.0, 0.0); n],
        }
    }
}

/// Run the NEGF sweep over `energies` for the device's current `psi_f`, with
/// imaginary broadening `eta` (see [`DEFAULT_ETA`]), the given
/// [`GreenFunctionAlgorithm`], and the given [`GreenOutputs`].
///
/// Energy points are evaluated in parallel across available CPU cores.
pub fn green_function_sweep(
    device: &Device,
    energies: &[f64],
    eta: f64,
    algorithm: GreenFunctionAlgorithm,
    outputs: GreenOutputs,
) -> GreenFunctionResult {
    let n = device.n;
    let t = device.t_hop;
    let a = device.params.a;
    let psi_source = device.psi_f[0];
    let psi_drain = device.psi_f[n - 1];
    let n_energies = energies.len();

    // On-site energies of the bare chain, shared read-only across all
    // (parallel) energy points: H_ii = 2t + psi_f(i), H_i,i+1 = -t.
    let bulk_diag: Vec<f64> = device.psi_f.iter().map(|p| 2.0 * t + p).collect();

    let full = outputs == GreenOutputs::Full;
    // One dummy slot per energy point when the diagonal isn't wanted, so a
    // single zipped pipeline covers both cases (chunk sizes must be nonzero).
    let ldos_stride = if full { n } else { 1 };
    let mut ldos = vec![0.0; n_energies * ldos_stride];
    let mut col_source = vec![0.0; n_energies * n];
    let mut col_drain = vec![0.0; n_energies * n];
    let mut sigma_source = vec![Complex64::new(0.0, 0.0); n_energies];
    let mut sigma_drain = vec![Complex64::new(0.0, 0.0); n_energies];

    energies
        .par_iter()
        .zip(ldos.par_chunks_mut(ldos_stride))
        .zip(col_source.par_chunks_mut(n))
        .zip(col_drain.par_chunks_mut(n))
        .zip(sigma_source.par_iter_mut())
        .zip(sigma_drain.par_iter_mut())
        .for_each_init(
            || Scratch::new(n),
            |scratch, (((((&e, ldos_row), col_s_row), col_d_row), sigma_s), sigma_d)| {
                *sigma_s = contact_self_energy(e, t, psi_source);
                *sigma_d = contact_self_energy(e, t, psi_drain);

                let diag = &mut scratch.diag;
                for (d, &bulk) in diag.iter_mut().zip(bulk_diag.iter()) {
                    *d = Complex64::new(e - bulk, eta);
                }
                diag[0] -= *sigma_s;
                diag[n - 1] -= *sigma_d;

                match algorithm {
                    GreenFunctionAlgorithm::Recursive => {
                        if full {
                            tridiag::inverse_diagonal_complex(
                                t,
                                diag,
                                &mut scratch.work,
                                &mut scratch.buf_a,
                            );
                            write_ldos(&scratch.buf_a, a, ldos_row);
                        }
                        tridiag::boundary_columns_complex(
                            t,
                            diag,
                            &mut scratch.work,
                            &mut scratch.buf_a,
                            &mut scratch.buf_b,
                        );
                        write_norm_sqr(&scratch.buf_a, col_s_row);
                        write_norm_sqr(&scratch.buf_b, col_d_row);
                    }
                    GreenFunctionAlgorithm::Dense => {
                        if full {
                            let inverse = dense_full_inverse(t, diag);
                            let diagonal: Vec<Complex64> = (0..n).map(|i| inverse[i][i]).collect();
                            write_ldos(&diagonal, a, ldos_row);
                            let col_s: Vec<Complex64> = (0..n).map(|i| inverse[i][0]).collect();
                            let col_d: Vec<Complex64> = (0..n).map(|i| inverse[i][n - 1]).collect();
                            write_norm_sqr(&col_s, col_s_row);
                            write_norm_sqr(&col_d, col_d_row);
                        } else {
                            let (col_s, col_d) = dense_boundary_columns(t, diag);
                            write_norm_sqr(&col_s, col_s_row);
                            write_norm_sqr(&col_d, col_d_row);
                        }
                    }
                }
            },
        );

    if !full {
        ldos = Vec::new();
    }

    GreenFunctionResult {
        energies: energies.to_vec(),
        n_sites: n,
        ldos,
        col_source,
        col_drain,
        sigma_source,
        sigma_drain,
    }
}

/// Local density of states from the Green's-function diagonal:
/// `-Im(G_ii) / (pi a)`, in states/(eV nm).
fn write_ldos(diagonal: &[Complex64], a: f64, out: &mut [f64]) {
    for (o, g) in out.iter_mut().zip(diagonal.iter()) {
        *o = -g.im / (std::f64::consts::PI * a);
    }
}

fn write_norm_sqr(column: &[Complex64], out: &mut [f64]) {
    for (o, g) in out.iter_mut().zip(column.iter()) {
        *o = g.norm_sqr();
    }
}

/// Ballistic current from the NEGF transmission, via the Landauer formula
/// with the Caroli expression for `T(E)`:
///
/// ```text
/// I = (2e/h) * integral T(E) [f_s(E) - f_d(E)] dE
/// ```
///
/// The factor 2 is spin degeneracy, matching [`Device::calc_current`]. Unlike
/// that function — which assumes perfect transmission above the barrier top —
/// this one uses the actual transmission of the potential profile, so it
/// includes tunneling through the barrier and quantum reflection above it.
///
/// The energy grid spans the whole contact band (`min(psi_f)` up to
/// `device.e_max`, beyond which `f_s - f_d` is negligible) in steps of
/// `params.d_e`. Returns amperes, per conducting subband.
///
/// # Discretization caveat
///
/// This is the transmission of the *discretized* chain, whose band has finite
/// width `4t = 2 hbar^2 / (m a^2)` — 0.68 eV at the default `a = 0.5 nm`. The
/// tight-binding dispersion only matches the parabolic band that the analytic
/// [`Device::calc_current`] assumes near the band bottom, so the two agree in
/// the on-state but can diverge sharply when transport happens near the band
/// top: a large `V_ds` slides the drain band down, and any conduction window
/// above the drain band top is closed here while the analytic formula still
/// counts it. Shrinking `a` widens the band (`4t ~ 1/a^2`) and is the way to
/// check whether a given result is discretization-limited.
pub fn negf_current(device: &Device, eta: f64, algorithm: GreenFunctionAlgorithm) -> f64 {
    let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
    let d_e = device.params.d_e;
    let steps = ((device.e_max - e_min) / d_e).floor().max(0.0) as usize;
    let energies: Vec<f64> = (0..=steps).map(|k| e_min + k as f64 * d_e).collect();

    let green = green_function_sweep(
        device,
        &energies,
        eta,
        algorithm,
        GreenOutputs::BoundaryColumns,
    );

    let thermal = K_B * device.params.t / ELEMENTARY_CHARGE;
    let f_s = |e: f64| 1.0 / (((e - device.params.e_fs) / thermal).exp() + 1.0);
    let f_d = |e: f64| 1.0 / (((e - device.e_fd) / thermal).exp() + 1.0);

    let integral: f64 = energies
        .iter()
        .enumerate()
        .map(|(k, &e)| green.transmission(k) * (f_s(e) - f_d(e)))
        .sum::<f64>()
        * d_e;

    2.0 * ELEMENTARY_CHARGE / PLANCK * integral * ELEMENTARY_CHARGE
}

/// Dense `N x N` matrix `(E + i*eta) I - H - Sigma`, from its constant real
/// off-diagonal `t` and complex diagonal.
fn dense_matrix(t: f64, diag: &[Complex64]) -> Vec<Vec<Complex64>> {
    let n = diag.len();
    let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = diag[i];
        if i > 0 {
            row[i - 1] = Complex64::new(t, 0.0);
        }
        if i < n - 1 {
            row[i + 1] = Complex64::new(t, 0.0);
        }
    }
    a
}

/// Full `A^-1` via Gauss-Jordan elimination with partial pivoting. O(N^3).
/// The [`GreenFunctionAlgorithm::Dense`] backend when the diagonal is wanted,
/// and the reference the recursive path is validated against in tests.
#[allow(clippy::needless_range_loop)]
fn dense_full_inverse(t: f64, diag: &[Complex64]) -> Vec<Vec<Complex64>> {
    let n = diag.len();
    let mut a = dense_matrix(t, diag);
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

/// Columns `0` and `N-1` of `A^-1`, via Gaussian elimination with partial
/// pivoting on the two unit right-hand sides. Same O(N^3) elimination as
/// [`dense_full_inverse`] but with 2 right-hand sides instead of N, which is
/// all the charge density and the transmission need.
#[allow(clippy::needless_range_loop)]
fn dense_boundary_columns(t: f64, diag: &[Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
    let n = diag.len();
    let mut a = dense_matrix(t, diag);
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    // rhs[row] = [e_0, e_{N-1}] as two columns.
    let mut rhs = vec![[zero; 2]; n];
    rhs[0][0] = one;
    rhs[n - 1][1] = one;

    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if a[row][col].norm() > a[pivot][col].norm() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        rhs.swap(col, pivot);
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            if factor == zero {
                continue;
            }
            for k in col..n {
                let update = factor * a[col][k];
                a[row][k] -= update;
            }
            for j in 0..2 {
                let update = factor * rhs[col][j];
                rhs[row][j] -= update;
            }
        }
    }

    let mut x = vec![[zero; 2]; n];
    for i in (0..n).rev() {
        for j in 0..2 {
            let mut s = rhs[i][j];
            for k in (i + 1)..n {
                s -= a[i][k] * x[k][j];
            }
            x[i][j] = s / a[i][i];
        }
    }

    (
        x.iter().map(|row| row[0]).collect(),
        x.iter().map(|row| row[1]).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceParams};
    use approx::assert_relative_eq;

    fn small_device() -> Device {
        Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            v_ds: 0.3,
            v_g: 0.2,
            ..Default::default()
        })
    }

    /// A barrier-free device: no band gap and no gate barrier, so the source
    /// and drain bands overlap and transmission is close to 1 in the window
    /// between them.
    fn flat_device() -> Device {
        Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            e_f: 0.0,
            e_g: 0.0,
            v_ds: 0.1,
            v_g: 0.0,
            ..Default::default()
        })
    }

    fn energy_grid(device: &Device, count: usize, step: f64) -> Vec<f64> {
        let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
        (0..count).map(|i| e_min + i as f64 * step).collect()
    }

    #[test]
    fn contact_self_energy_is_retarded_inside_the_band_and_decaying_outside() {
        let t = 0.17;
        let psi = -0.1;
        // Band bottom, middle, top.
        for &z in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let e = psi + 2.0 * t + 2.0 * t * z;
            let sigma = contact_self_energy(e, t, psi);
            assert!(
                sigma.im <= 0.0,
                "retarded self-energy must have Im <= 0, got {sigma} at z = {z}"
            );
            assert_relative_eq!(sigma.norm(), t, epsilon = 1e-12);
        }
        // Outside the band: real, decaying, and continuous with the edges.
        for &e in &[
            psi - 0.5,
            psi - 1e-9,
            psi + 4.0 * t + 1e-9,
            psi + 4.0 * t + 0.5,
        ] {
            let sigma = contact_self_energy(e, t, psi);
            assert_eq!(sigma.im, 0.0);
            assert!(
                sigma.norm() <= t + 1e-12,
                "evanescent |Sigma| must not grow: {sigma}"
            );
        }
        let edge = contact_self_energy(psi - 1e-12, t, psi);
        assert_relative_eq!(edge.re, -t, epsilon = 1e-6);
    }

    #[test]
    fn local_density_of_states_is_positive_everywhere() {
        // The whole point of the retarded sign convention: Im[(E+i eta) I - H
        // - Sigma] is positive definite, so the LDOS cannot come out negative.
        let device = small_device();
        let energies = energy_grid(&device, 60, 0.02);
        for algorithm in [
            GreenFunctionAlgorithm::Recursive,
            GreenFunctionAlgorithm::Dense,
        ] {
            let green = green_function_sweep(
                &device,
                &energies,
                DEFAULT_ETA,
                algorithm,
                GreenOutputs::Full,
            );
            for (k, &energy) in energies.iter().enumerate() {
                for (i, &value) in green.ldos_row(k).iter().enumerate() {
                    assert!(
                        value >= 0.0 && value.is_finite(),
                        "{algorithm:?}: LDOS must be positive and finite, got {value} at \
                         E = {energy}, site {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn broadening_vanishes_outside_the_contact_band() {
        let device = small_device();
        let t = device.t_hop;
        let psi_source = device.psi_f[0];
        let below = psi_source - 0.05;
        let above = psi_source + 4.0 * t + 0.05;
        let inside = psi_source + 2.0 * t;
        let green = green_function_sweep(
            &device,
            &[below, inside, above],
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::BoundaryColumns,
        );
        assert_eq!(green.gamma_source(0), 0.0);
        assert!(green.gamma_source(1) > 0.0);
        assert_eq!(green.gamma_source(2), 0.0);
    }

    #[test]
    fn dense_and_recursive_algorithms_agree_end_to_end() {
        let device = small_device();
        let energies = energy_grid(&device, 15, 0.02);

        let recursive = green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::Full,
        );
        let dense = green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Dense,
            GreenOutputs::Full,
        );

        for k in 0..energies.len() {
            for i in 0..device.n {
                assert_relative_eq!(
                    recursive.ldos_row(k)[i],
                    dense.ldos_row(k)[i],
                    epsilon = 1e-6
                );
                assert_relative_eq!(
                    recursive.col_source_row(k)[i],
                    dense.col_source_row(k)[i],
                    epsilon = 1e-6
                );
                assert_relative_eq!(
                    recursive.col_drain_row(k)[i],
                    dense.col_drain_row(k)[i],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn boundary_columns_only_matches_the_full_sweep() {
        // Skipping the diagonal must not change the columns, under either
        // algorithm (the dense path takes a different elimination route).
        let device = small_device();
        let energies = energy_grid(&device, 12, 0.03);
        for algorithm in [
            GreenFunctionAlgorithm::Recursive,
            GreenFunctionAlgorithm::Dense,
        ] {
            let full = green_function_sweep(
                &device,
                &energies,
                DEFAULT_ETA,
                algorithm,
                GreenOutputs::Full,
            );
            let columns = green_function_sweep(
                &device,
                &energies,
                DEFAULT_ETA,
                algorithm,
                GreenOutputs::BoundaryColumns,
            );
            assert!(columns.ldos.is_empty());
            for k in 0..energies.len() {
                for i in 0..device.n {
                    assert_relative_eq!(
                        full.col_source_row(k)[i],
                        columns.col_source_row(k)[i],
                        epsilon = 1e-9
                    );
                    assert_relative_eq!(
                        full.col_drain_row(k)[i],
                        columns.col_drain_row(k)[i],
                        epsilon = 1e-9
                    );
                }
            }
        }
    }

    #[test]
    fn transmission_is_bounded_by_the_single_channel_limit() {
        // A single-mode chain can transmit at most one quantum of conductance,
        // so 0 <= T(E) <= 1 — a sharp check on the Gamma/|G|^2 normalization.
        // Uses the barrier-free device: with the default gate barrier the
        // channel is opaque and every T(E) would trivially be ~0.
        let device = flat_device();
        let energies = energy_grid(&device, 200, 0.001);
        let green = green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::BoundaryColumns,
        );
        let mut max_t = 0.0_f64;
        for (k, &energy) in energies.iter().enumerate() {
            let transmission = green.transmission(k);
            assert!(
                (-1e-12..=1.0 + 1e-9).contains(&transmission),
                "T = {transmission} out of range at E = {energy}"
            );
            max_t = max_t.max(transmission);
        }
        assert!(
            max_t > 0.5,
            "expected a well-transmitting window, got {max_t}"
        );
    }

    #[test]
    fn negf_current_approaches_the_ballistic_limit_for_a_flat_potential() {
        // With no barrier, transmission is ~1 across the band and the NEGF
        // current should track the analytic ballistic (T = 1) formula. They
        // differ only where the analytic model is wrong: quantum reflection
        // near the band edges, and the finite band top the analytic formula
        // ignores.
        let device = flat_device();
        let ballistic = device.calc_current();
        let negf = negf_current(&device, DEFAULT_ETA, GreenFunctionAlgorithm::Recursive);
        assert!(negf > 0.0, "NEGF current should be positive, got {negf}");
        assert!(
            negf < ballistic,
            "reflection and the finite band top can only reduce the current: \
             NEGF {negf:e} vs ballistic {ballistic:e}"
        );
        assert!(
            negf > 0.2 * ballistic,
            "NEGF current should be the same order as ballistic: {negf:e} vs {ballistic:e}"
        );
    }

    #[test]
    fn green_function_sweep_runs_and_is_finite() {
        let device = small_device();
        let energies = energy_grid(&device, 20, 0.01);
        let result = green_function_sweep(
            &device,
            &energies,
            DEFAULT_ETA,
            GreenFunctionAlgorithm::Recursive,
            GreenOutputs::Full,
        );

        assert_eq!(result.energies.len(), energies.len());
        assert_eq!(result.ldos.len(), energies.len() * device.n);
        for k in 0..energies.len() {
            assert!(result.ldos_row(k).iter().all(|v| v.is_finite()));
            assert!(result
                .col_source_row(k)
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0));
        }
    }
}
