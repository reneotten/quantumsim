//! NEGF-derived charge density, feeding the self-consistent electrostatic
//! loop.
//!
//! This is a port of `calc_n()` in `legacy_matlab/quantumsim.m`, with two
//! deliberate bug fixes documented here and in the top-level README:
//!
//! 1. The original multiplies the whole energy sum by a single scalar
//!    `f(E_fs)` / `f(E_fd)` (the Fermi function evaluated once, outside the
//!    sum) instead of the energy-dependent occupation `f_s(E)` / `f_d(E)`
//!    used everywhere else in the model (e.g. in `calc_current`). That
//!    collapses the energy dependence of the injected charge to a single
//!    number and made the original `calc_n` effectively decorative — it was
//!    never fed back into `calc_potential`. Here the proper per-energy
//!    Fermi weight is used, matching `calc_current`.
//! 2. The original reuses one `mask = E > Psi_f(1)` (the *source* band
//!    edge) for both the source and drain contact terms. Each contact's
//!    contribution is now masked by its own band edge, consistent with how
//!    `calc_green` already conditions each contact's self-energy on its own
//!    band edge.
//!
//! **Third deviation, required for the self-consistent loop to be
//! numerically meaningful at all:** the original additionally multiplied by
//! `1e9` to convert the `1/a`-scaled sum from nm^-1 to m^-1 — sensible for
//! plotting a density on a real-world axis, but `a` and `lambda` (the two
//! length scales that actually drive `calc_potential`) are expressed in nm
//! *everywhere else* in this model. Feeding a value that's ~1e9x too large
//! back into `rho` (which shares scale with `Psi_g/lambda^2` etc., all
//! still in nm) overwhelms the electrostatic solve and the fixed-point
//! iteration diverges. [`electron_density`] therefore returns the density
//! in nm^-1, matching the model's own native length unit. The caller
//! (`selfconsistent.rs`) is responsible for converting to a proper volume
//! charge density (m^-1 -> m^-3 assuming unit cross-section, times `-e`)
//! before assigning it to `rho` — see that module's docs.
//!
//! ## Validity: normalization, sign, and open questions
//!
//! Every term summed into `n` is `(non-negative broadening) * (Fermi
//! occupation in (0,1)) * |G|^2`, so `n(x) >= 0` always holds by
//! construction — this does *not* depend on the contact self-energy sign
//! question discussed in `negf.rs` (which affects `g_diag`, not the `|G|^2`
//! columns used here).
//!
//! What *is* inherited from that sign question: `gamma_s`/`gamma_d` here
//! use `+t_hop * sin(k)` together with a `1/pi` prefactor (rather than the
//! more commonly quoted `Gamma = -2*Im(Sigma)` broadening with a `1/(2*pi)`
//! prefactor); the two conventions happen to give the same magnitude
//! (`t_hop*sin(k)/pi == (2*t_hop*sin(k))/(2*pi)`), so this reproduces the
//! standard formula's *magnitude* even though the sign story behind
//! `gamma_s`/`gamma_d` individually is the same non-causal one described in
//! `negf.rs`. Also note there is no explicit spin-degeneracy factor of two
//! here (unlike `calc_current`'s explicit `2*e/h`) — carried over unchanged
//! from the original `calc_n`, which never had this checked against a
//! reference since it was never fed back into anything before this rewrite.
//! Treat absolute magnitudes of the self-consistent charge density as
//! illustrative rather than quantitatively validated; the qualitative
//! behavior (current increasing with gate/drain bias, convergence to a
//! stable fixed point) is covered by the integration tests.

use crate::constants::{E, K_B};
use crate::negf::GreenFunctionResult;

/// Compute the electron number density `n(x)` from a Green's-function
/// sweep, using the same convention as `legacy_matlab/quantumsim.m`
/// (see module docs for the two bug fixes applied).
#[allow(clippy::too_many_arguments)]
pub fn electron_density(
    green: &GreenFunctionResult,
    t_hop: f64,
    a: f64,
    psi_f_source: f64,
    psi_f_drain: f64,
    e_fs: f64,
    e_fd: f64,
    temperature: f64,
    d_e: f64,
) -> Vec<f64> {
    let n_sites = green.g_col_source[0].len();
    let mut n = vec![0.0; n_sites];

    let f_s = |energy: f64| 1.0 / (((energy - e_fs) * E / (K_B * temperature)).exp() + 1.0);
    let f_d = |energy: f64| 1.0 / (((energy - e_fd) * E / (K_B * temperature)).exp() + 1.0);

    for (k, &energy) in green.energies.iter().enumerate() {
        let gamma_s = if energy > psi_f_source {
            t_hop * green.k_sa[k].re.sin()
        } else {
            0.0
        };
        let gamma_d = if energy > psi_f_drain {
            t_hop * green.k_da[k].re.sin()
        } else {
            0.0
        };

        let weight_s = gamma_s * f_s(energy);
        let weight_d = gamma_d * f_d(energy);

        for (i, n_i) in n.iter_mut().enumerate().take(n_sites) {
            *n_i += weight_s * green.g_col_source[k][i] + weight_d * green.g_col_drain[k][i];
        }
    }

    for v in n.iter_mut() {
        *v = *v / std::f64::consts::PI * d_e / a;
    }
    n
}
