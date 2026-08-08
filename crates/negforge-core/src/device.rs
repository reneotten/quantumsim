//! Device geometry, electrostatics and ballistic-current model.
//!
//! This is a port of `legacy_matlab/quantumsim.m`. It models a 1D
//! source-channel-drain chain (source and drain treated as long, heavily
//! screened contact regions) with:
//!
//! - an electrostatic potential solved from a 1D "natural length" (`lambda`)
//!   approximation rather than a full 2D/3D Poisson equation — the same
//!   simplification the original coursework code used, common in compact
//!   models for thin-body/double-gate MOSFETs;
//! - a ballistic (Landauer) drain current from Fermi-Dirac occupation of the
//!   source and drain reservoirs.
//!
//! Units follow the original code: lengths in nm, energies/potentials in eV,
//! temperature in Kelvin. `Psi_g`/`Psi_bi`/`Psi_f` store *potential energy*
//! (already in eV), not electrostatic potential in volts — consistent with
//! `E_f`/`E_g` also being in eV.
//!
//! ## Validity of the approximations
//!
//! - **Natural-length electrostatics** (see [`Device::compute_lambda`]) is a
//!   thin-body/double-gate compact-model reduction of the full 2D/3D Poisson
//!   equation. It assumes the transverse potential profile is parabolic
//!   (valid when the body is thin compared to the channel length) and breaks
//!   down for bulk (thick-body) MOSFETs, where the transverse charge
//!   distribution is not well approximated by a single decay length. It is
//!   also a *classical* charge model in the transverse direction — it does
//!   not resolve transverse-mode (subband) quantization from confinement in
//!   `d_ch`, unlike the longitudinal NEGF treatment in `negf.rs`.
//! - **Ballistic (Landauer) current** ([`Device::calc_current`]) assumes
//!   perfect transmission (`T = 1`) for every state above the barrier top
//!   (`psi_0`) and zero transmission below it — i.e. no scattering
//!   (phonon/impurity/surface-roughness) anywhere in the channel, and no
//!   sub-barrier tunneling. This is the "top-of-the-barrier" thermionic
//!   -emission limit; it is a good approximation only when the channel is
//!   short compared to the carrier mean free path, and it overestimates
//!   current (and underestimates subthreshold current, since tunneling is
//!   excluded) for longer or more heavily doped channels.
//! - **Effective-mass, single-valley, parabolic band** (`m_eff`, used in
//!   `t_hop` here and in the NEGF hopping parameter). Real silicon has
//!   multiple equivalent conduction-band valleys (a valley-degeneracy factor
//!   this model does not include) and a non-parabolic dispersion away from
//!   the band edge; `m_eff` is a single fitting parameter standing in for
//!   both effects, so results should be read as illustrating trends, not as
//!   quantitatively predicting absolute currents for a specific real device.
//!   Spin degeneracy *is* included, via the explicit factor of 2 in
//!   `calc_current`'s `2*e/h` prefactor (the standard single-mode Landauer
//!   conductance quantum).

use crate::constants::{E, EPS_0, H_BAR, K_B, M_E};
use crate::tridiag;

/// Configuration for a [`Device`]. Field defaults match the original
/// `quantumsim.m` `properties` block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceParams {
    /// Grid spacing, nm.
    pub a: f64,
    /// Fermi energy, eV.
    pub e_f: f64,
    /// Band gap, eV.
    pub e_g: f64,
    /// Drain-source voltage, V.
    pub v_ds: f64,
    /// Gate voltage (Psi_g = -e*V_g), V.
    pub v_g: f64,
    /// Oxide thickness, nm.
    pub d_ox: f64,
    /// Channel (body) thickness, nm.
    pub d_ch: f64,
    /// Relative permittivity of the channel (silicon by default).
    pub k_si: f64,
    /// Relative permittivity of the oxide.
    pub k_ox: f64,
    /// Gate geometry factor (number of gates).
    pub geo: f64,
    /// Channel length, nm.
    pub l_ch: f64,
    /// If `true` (default, matching the original constructor), the
    /// source/drain contact length is auto-sized to `floor(lambda) * 15`
    /// instead of using `l_ds` directly.
    pub auto_size_contacts: bool,
    /// Source/drain contact region length, nm. Only used verbatim when
    /// `auto_size_contacts` is `false`.
    pub l_ds: f64,
    /// Fixed dopant charge term added to the mobile charge density in the
    /// electrostatic solve.
    pub n_dot: f64,
    /// Fermi-function tolerance used to bound the ballistic energy window.
    pub epsilon: f64,
    /// Source Fermi level, eV.
    pub e_fs: f64,
    /// Temperature, K.
    pub t: f64,
    /// Energy integration step, eV.
    pub d_e: f64,
    /// Effective mass, kg.
    pub m_eff: f64,
}

impl Default for DeviceParams {
    fn default() -> Self {
        Self {
            a: 0.5,
            e_f: 0.15,
            e_g: 1.0,
            v_ds: 0.0,
            v_g: 0.0,
            d_ox: 5.0,
            d_ch: 5.0,
            k_si: 11.2,
            k_ox: 3.9,
            geo: 1.0,
            l_ch: 40.0,
            auto_size_contacts: true,
            l_ds: 40.0,
            n_dot: 0.0,
            epsilon: 10e-15,
            e_fs: 0.05,
            t: 300.0,
            d_e: 0.001,
            m_eff: 0.9 * M_E,
        }
    }
}

/// A 1D ballistic-MOSFET device: geometry, electrostatics and derived
/// transport quantities.
#[derive(Debug, Clone)]
pub struct Device {
    pub params: DeviceParams,

    /// Natural (screening) length, nm.
    pub lambda: f64,
    /// Number of grid points along the channel.
    pub n: usize,
    /// Last grid index (0-based) belonging to the source region.
    pub n_left: usize,
    /// First grid index (0-based) belonging to the drain region.
    pub n_right: usize,

    /// Gate potential energy profile, eV.
    pub psi_g: Vec<f64>,
    /// Built-in potential energy profile, eV.
    pub psi_bi: Vec<f64>,
    /// Mobile charge density term entering the electrostatic solve. Zero
    /// until a self-consistent (NEGF-fed) solve updates it.
    pub rho: Vec<f64>,
    /// Solved electrostatic potential energy profile, eV.
    pub psi_f: Vec<f64>,
    /// `max(psi_f)`, used as the lower bound of the ballistic energy window.
    pub psi_0: f64,

    /// Drain Fermi level, eV (`-V_ds + 0.05`, matching the original code).
    pub e_fd: f64,
    /// Upper bound of the ballistic energy integration window, eV.
    pub e_max: f64,
    /// Nearest-neighbor hopping parameter, eV.
    pub t_hop: f64,
}

impl Device {
    /// Build a new device, replicating the `quantumsim()` MATLAB
    /// constructor: `lambda` is computed first, then (if
    /// `auto_size_contacts`) `l_ds` is overwritten from it before the grid
    /// is sized.
    pub fn new(mut params: DeviceParams) -> Self {
        let lambda = Self::compute_lambda(&params);
        if params.auto_size_contacts {
            params.l_ds = (lambda.floor()) * 15.0;
        }

        let (n, n_left, n_right) = Self::compute_grid(&params);
        let e_fd = -params.v_ds + 0.05;
        let e_max = params.e_fs - K_B * params.t * params.epsilon.ln() / E;
        let t_hop = H_BAR.powi(2) / (2.0 * params.m_eff * (params.a * 1e-9).powi(2) * E);

        let mut device = Self {
            params,
            lambda,
            n,
            n_left,
            n_right,
            psi_g: vec![0.0; n],
            psi_bi: vec![0.0; n],
            rho: vec![0.0; n],
            psi_f: vec![0.0; n],
            psi_0: 0.0,
            e_fd,
            e_max,
            t_hop,
        };
        device.init_vectors();
        device
    }

    /// `lambda = sqrt(k_si/k_ox * d_ch * d_ox / geo)`, the scale length over
    /// which source/drain potentials leak into the channel (see module
    /// docs). Only meaningful in the thin-body regime this formula was
    /// derived for; not a substitute for a full Poisson solve when `d_ch` is
    /// not small compared to `l_ch`.
    fn compute_lambda(params: &DeviceParams) -> f64 {
        (params.k_si / params.k_ox * params.d_ch * params.d_ox / params.geo).sqrt()
    }

    fn compute_grid(params: &DeviceParams) -> (usize, usize, usize) {
        let l_g = 2.0 * params.l_ds + params.l_ch;
        let n = (l_g / params.a).floor() as usize + 1;
        let n_left = (params.l_ds / params.a).floor() as usize;
        let n_right = n - n_left;
        (n, n_left, n_right)
    }

    /// Rebuild `psi_g`, `psi_bi` and reset `rho` from the current
    /// parameters and region boundaries. Mirrors `init_vectors()`.
    pub fn init_vectors(&mut self) {
        self.psi_g = vec![0.0; self.n];
        self.psi_bi = vec![0.0; self.n];
        self.rho = vec![0.0; self.n];

        // Channel: indices n_left..n_right-1 (0-based, matching MATLAB's
        // 1-based `n_left+1 : n_right-1`).
        for i in self.n_left..self.n_right.saturating_sub(1) {
            self.psi_g[i] = -self.params.v_g;
            self.psi_bi[i] = self.params.e_f + self.params.e_g / 2.0;
        }
        // Drain: indices n_right-1..N (0-based, matching `n_right:end`).
        for i in (self.n_right.saturating_sub(1))..self.n {
            self.psi_g[i] = 0.0;
            self.psi_bi[i] = -self.params.v_ds;
        }
        // Source region (0..n_left) stays at zero, as in the original.
    }

    /// Build the tridiagonal second-difference operator with the
    /// reflective (Neumann-like) boundary doubling used for the
    /// electrostatic solve, scaled by `1/a^2`. Returns `(sub, diag, sup)`.
    fn laplacian(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n;
        let inv_a2 = 1.0 / self.params.a.powi(2);
        let diag = vec![-2.0 * inv_a2; n];
        let mut sup = vec![inv_a2; n - 1];
        let mut sub = vec![inv_a2; n - 1];
        sup[0] = 2.0 * inv_a2;
        sub[n - 2] = 2.0 * inv_a2;
        (sub, diag, sup)
    }

    /// Solve for the self-consistent-length electrostatic potential energy
    /// profile given the current `rho`, storing the result in `psi_f` and
    /// updating `psi_0 = max(psi_f)`. Mirrors `calc_potential()`.
    pub fn calc_potential(&mut self) {
        let (sub, mut diag, sup) = self.laplacian();
        let inv_lambda2 = 1.0 / self.lambda.powi(2);
        for d in diag.iter_mut() {
            *d -= inv_lambda2;
        }

        let rhs: Vec<f64> = (0..self.n)
            .map(|i| {
                (self.rho[i] + self.params.n_dot) / (EPS_0 * self.params.k_si)
                    - inv_lambda2 * (self.psi_g[i] + self.psi_bi[i])
            })
            .collect();

        self.psi_f = tridiag::solve_real(&sub, &diag, &sup, &rhs);
        self.psi_0 = self.psi_f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }

    /// Ballistic Landauer current for the current potential profile.
    /// Mirrors `calc_current()`. Units match the original scaling
    /// convention (see `legacy_matlab/README.md`).
    ///
    /// Integrates `T(E) * [f_s(E) - f_d(E)]` over `E` with `T(E)` implicitly
    /// `1` for `E >= psi_0` (the barrier top) and `0` below it — see the
    /// "Ballistic (Landauer) current" note in the module docs for when this
    /// no-scattering, no-tunneling approximation is (and isn't) reasonable.
    pub fn calc_current(&self) -> f64 {
        let f_s = |e: f64| 1.0 / (((e - self.params.e_fs) * E / (K_B * self.params.t)).exp() + 1.0);
        let f_d = |e: f64| 1.0 / (((e - self.e_fd) * E / (K_B * self.params.t)).exp() + 1.0);

        let n_steps = ((self.e_max - self.psi_0) / self.params.d_e)
            .floor()
            .max(0.0) as usize;
        let mut sum = 0.0;
        for k in 0..=n_steps {
            let energy = self.psi_0 + k as f64 * self.params.d_e;
            sum += f_s(energy) - f_d(energy);
        }
        2.0 * E / crate::constants::H * sum * self.params.d_e * E * 1e-3
    }

    pub fn set_v_ds(&mut self, v: f64) {
        self.params.v_ds = v;
        self.e_fd = -v + 0.05;
        self.init_vectors();
    }

    pub fn set_v_g(&mut self, v: f64) {
        self.params.v_g = v;
        self.init_vectors();
    }

    pub fn set_l_ch(&mut self, l: f64) {
        self.params.l_ch = l;
        let (n, n_left, n_right) = Self::compute_grid(&self.params);
        self.n = n;
        self.n_left = n_left;
        self.n_right = n_right;
        self.init_vectors();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `m_eff` sets the tight-binding hopping `t_hop = hbar^2/(2 m_eff a^2)`,
    /// so a lighter carrier must raise `t_hop`. It deliberately does *not*
    /// affect `calc_current`, whose Landauer integral assumes unit
    /// transmission above the barrier and carries no mass prefactor -- pinned
    /// here so the asymmetry is a documented property rather than a surprise.
    #[test]
    fn m_eff_scales_hopping_but_not_ballistic_current() {
        let heavy_params = DeviceParams {
            v_ds: 0.1,
            v_g: 0.4,
            ..Default::default()
        };
        let light_params = DeviceParams {
            m_eff: heavy_params.m_eff / 4.0,
            ..heavy_params
        };

        let mut heavy = Device::new(heavy_params);
        let mut light = Device::new(light_params);
        heavy.calc_potential();
        light.calc_potential();

        // t_hop ~ 1/m_eff: quartering the mass quadruples the hopping.
        assert_relative_eq!(light.t_hop, 4.0 * heavy.t_hop, max_relative = 1e-12);

        // ...while the ballistic current is mass-independent by construction.
        assert_relative_eq!(
            light.calc_current(),
            heavy.calc_current(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn lambda_matches_closed_form() {
        let params = DeviceParams::default();
        let dev = Device::new(params);
        let expected = (params.k_si / params.k_ox * params.d_ch * params.d_ox / params.geo).sqrt();
        assert_relative_eq!(dev.lambda, expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_bias_zero_gate_gives_symmetric_flat_contacts() {
        // With V_g = V_ds = 0 and no charge, psi_bi is zero everywhere
        // outside the channel and the channel barrier is E_f + E_g/2 — the
        // solved potential should be flat (zero) far into the contacts,
        // since there is no driving term there.
        let dev = Device::new(DeviceParams::default());
        assert_relative_eq!(dev.psi_g[0], 0.0);
        assert_relative_eq!(dev.psi_bi[0], 0.0);
        assert_relative_eq!(dev.psi_bi[dev.n - 1], 0.0);
    }

    #[test]
    fn calc_potential_is_finite_and_bounded_by_source_terms() {
        let mut dev = Device::new(DeviceParams::default());
        dev.calc_potential();
        assert_eq!(dev.psi_f.len(), dev.n);
        assert!(dev.psi_f.iter().all(|v| v.is_finite()));
        // The channel barrier term is E_f + E_g/2; the solved potential
        // (a screened, smoothed version of the driving terms) should not
        // wildly overshoot it.
        let max_drive = dev.params.e_f + dev.params.e_g / 2.0;
        assert!(dev.psi_0 <= max_drive * 1.01);
    }

    #[test]
    fn increasing_v_ds_increases_current() {
        let mut dev = Device::new(DeviceParams::default());
        dev.calc_potential();
        let i0 = dev.calc_current();

        dev.set_v_ds(0.3);
        dev.calc_potential();
        let i1 = dev.calc_current();

        assert!(
            i1 > i0,
            "current should increase with drain bias: {i0} -> {i1}"
        );
    }

    #[test]
    fn increasing_v_g_increases_current_ballistic_mosfet() {
        let mut dev = Device::new(DeviceParams::default());
        dev.set_v_ds(0.3);
        dev.calc_potential();
        let i0 = dev.calc_current();

        dev.set_v_g(0.3);
        dev.calc_potential();
        let i1 = dev.calc_current();

        assert!(
            i1 > i0,
            "current should increase with gate bias: {i0} -> {i1}"
        );
    }
}
