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
//! ## Planar, FinFET and nanoribbon devices
//!
//! `lambda = sqrt(k_si/k_ox * d_ch * d_ox / geo)` is the standard
//! "generalized scale length" (natural length) model used throughout the
//! multi-gate MOSFET literature (Auth & Plummer 1997; see also the
//! Ferain/Colinge/Colinge review of multigate transistors). `geo` is the
//! number of gates electrostatically controlling the channel, and is the
//! one parameter that actually distinguishes a planar, FinFET, or
//! gate-all-around (nanoribbon) architecture in this model — more gates
//! means a smaller natural length, i.e. tighter electrostatic control and
//! better short-channel-effect immunity, for the same body thickness. See
//! [`GateGeometry`] and the `DeviceParams::planar`/`fin_fet`/`nanoribbon`
//! constructors below, and
//! `crates/negforge-core/tests/multigate_geometry.rs` for the physical
//! sanity checks (natural length ordering, subthreshold swing improving
//! with more gates).
//!
//! This is still a 1D model along the transport direction: `d_ch` is a
//! single effective body-thickness number (fin width, film thickness, or
//! ribbon diameter, depending on architecture) rather than an actual
//! resolved cross-section, and there's no transverse-mode/subband
//! quantization. See the top-level README's "Extending to a real 3D
//! solver" section for what a genuine cross-section-resolved model would
//! require.
//!
//! ## Charge-density units in the electrostatic solve
//!
//! Because the discrete Laplacian is built with `a` in nm, every term of
//! `calc_potential`'s right-hand side is an energy per nm^2 (eV/nm^2). The
//! charge term is `rho / (EPS_0 * k_si)`, and `EPS_0` is in SI units (F/m),
//! so a charge density in C/m^3 would give V/m^2 there — off by 1e18 from the
//! rest of the equation. [`Device::rho`] and [`DeviceParams::n_dot`] are
//! therefore expressed in *model units*: an SI volume charge density scaled
//! by [`RHO_SI_TO_MODEL`]. [`crate::selfconsistent`] does that conversion for
//! the NEGF charge; anyone setting `rho`/`n_dot` by hand must do the same.

use crate::constants::{E, EPS_0, H_BAR, K_B, M_E};
use crate::tridiag;

/// Number of gates electrostatically controlling the channel, used to pick
/// `DeviceParams::geo` for a given device architecture. See the module
/// docs for the underlying scale-length model and its literature basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateGeometry {
    /// One gate (planar bulk MOSFET, or a single-gate SOI film).
    SingleGate,
    /// Two gates (planar double-gate SOI, or a FinFET with an inactive —
    /// e.g. hard-mask-capped — fin top, so only the two sidewalls conduct).
    DoubleGate,
    /// Three active gates (a FinFET with an active top gate as well as
    /// both sidewalls).
    TriGate,
    /// The gate fully wraps the channel (gate-all-around nanowire /
    /// nanoribbon), idealized as a rectangular cross-section with all four
    /// sides gated.
    GateAllAround,
}

impl GateGeometry {
    /// The `geo` (number-of-gates) factor for [`DeviceParams`].
    pub fn geo_factor(self) -> f64 {
        match self {
            GateGeometry::SingleGate => 1.0,
            GateGeometry::DoubleGate => 2.0,
            GateGeometry::TriGate => 3.0,
            GateGeometry::GateAllAround => 4.0,
        }
    }
}

/// Multiply an SI volume charge density (C/m^3) by this to get the model
/// units [`Device::rho`] and [`DeviceParams::n_dot`] use.
///
/// The factor is `(1 nm / 1 m)^2 = 1e-18`, the same conversion that takes the
/// electrostatic solve's `V/m^2` charge term into the `eV/nm^2` its Laplacian
/// works in. See the module docs.
pub const RHO_SI_TO_MODEL: f64 = 1e-18;

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
    /// Gate geometry factor (number of gates). See [`GateGeometry`] and
    /// the module docs for how this distinguishes planar/FinFET/nanoribbon
    /// architectures.
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
    /// Fixed dopant charge density added to the mobile charge density in the
    /// electrostatic solve, in model units (SI C/m^3 times
    /// [`RHO_SI_TO_MODEL`] — see the module docs).
    pub n_dot: f64,
    /// Cross-sectional area of the conducting channel, nm^2. The transport
    /// model is one-dimensional, so it yields a *linear* electron density;
    /// turning that into the volume charge density the electrostatic solve
    /// needs requires an area, and leaving it implicit is how a model quietly
    /// acquires an arbitrary charge-density scale. `None` (the default) uses
    /// `d_ch^2`, i.e. a square channel of the model's own body thickness.
    pub cross_section_nm2: Option<f64>,
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
            cross_section_nm2: None,
            epsilon: 10e-15,
            e_fs: 0.05,
            t: 300.0,
            d_e: 0.001,
            m_eff: 0.9 * M_E,
        }
    }
}

impl DeviceParams {
    /// Set `geo` from a [`GateGeometry`], leaving every other field
    /// unchanged. Note this only changes the *number of gates*; a real
    /// FinFET or nanoribbon also has a much thinner body than a planar
    /// device — see `fin_fet()`/`nanoribbon()` below for presets that set
    /// realistic dimensions too.
    pub fn with_gate_geometry(mut self, geometry: GateGeometry) -> Self {
        self.geo = geometry.geo_factor();
        self
    }

    /// Typical parameters for a planar, single-gate (bulk or SOI) MOSFET.
    /// Identical to `Default::default()` — planar/single-gate is this
    /// model's baseline architecture.
    pub fn planar() -> Self {
        Self::default().with_gate_geometry(GateGeometry::SingleGate)
    }

    /// Typical parameters for a double-gate FinFET: a thin fin (`d_ch` is
    /// the fin width) with two active gates on the sidewalls (see
    /// [`GateGeometry::DoubleGate`]; use
    /// `.with_gate_geometry(GateGeometry::TriGate)` on the result if the
    /// fin top is also gated) and a thin oxide, both narrower than the
    /// planar preset for realistic electrostatic control.
    pub fn fin_fet() -> Self {
        Self {
            d_ch: 8.0,
            d_ox: 1.5,
            ..Self::default()
        }
        .with_gate_geometry(GateGeometry::DoubleGate)
    }

    /// Typical parameters for a gate-all-around nanoribbon/nanowire FET:
    /// a narrow body (`d_ch` is the ribbon width/diameter) fully wrapped
    /// by the gate ([`GateGeometry::GateAllAround`]), with a thin oxide.
    pub fn nanoribbon() -> Self {
        Self {
            d_ch: 5.0,
            d_ox: 1.0,
            ..Self::default()
        }
        .with_gate_geometry(GateGeometry::GateAllAround)
    }
}

/// A 1D ballistic-MOSFET device: geometry, electrostatics and derived
/// transport quantities.
#[derive(Debug, Clone)]
pub struct Device {
    pub params: DeviceParams,

    /// Natural (screening) length, nm.
    pub lambda: f64,
    /// Resolved channel cross-section, nm^2 (see
    /// [`DeviceParams::cross_section_nm2`]).
    pub cross_section_nm2: f64,
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
    /// Mobile charge density entering the electrostatic solve, in model units
    /// (SI C/m^3 times [`RHO_SI_TO_MODEL`] — see the module docs). Zero until
    /// a self-consistent (NEGF-fed) solve updates it.
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
    ///
    /// # Panics
    ///
    /// Panics if `params` fails [`Self::try_new`]'s validation — see
    /// there for what that covers. Use `try_new` instead if you're
    /// constructing a device from values you don't already trust (e.g.
    /// user-supplied parameters).
    pub fn new(params: DeviceParams) -> Self {
        Self::try_new(params).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible version of [`Self::new`]. Returns
    /// `Err(NegForgeError::InvalidParameter)` if any of the
    /// geometry/material parameters that the electrostatic solve divides
    /// by (`geo`, `d_ch`, `d_ox`, `k_si`, `k_ox`, `a`) or the grid sizing
    /// (`l_ch`, and `l_ds` when `auto_size_contacts` is `false`) are
    /// non-positive or non-finite. Silently letting one of these through
    /// would produce an infinite or NaN natural length and propagate
    /// garbage through the rest of the solve instead of failing at the
    /// point of misconfiguration — a real risk once `geo` is something a
    /// caller sets explicitly (e.g. via [`GateGeometry`]) rather than only
    /// ever the hardcoded default.
    pub fn try_new(mut params: DeviceParams) -> crate::error::Result<Self> {
        Self::validate_params(&params)?;
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
            cross_section_nm2: params
                .cross_section_nm2
                .unwrap_or(params.d_ch * params.d_ch),
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
        // Leave the device in a self-consistent-with-its-parameters state:
        // `psi_f`/`psi_0` always reflect the current bias, so `calc_current`
        // and the NEGF entry points can never read a potential that belongs
        // to a different device configuration. Same contract as the
        // `set_*` bias setters below.
        device.calc_potential();
        Ok(device)
    }

    fn validate_params(params: &DeviceParams) -> crate::error::Result<()> {
        let positive = [
            ("geo", params.geo),
            ("d_ch", params.d_ch),
            ("d_ox", params.d_ox),
            ("k_si", params.k_si),
            ("k_ox", params.k_ox),
            ("a", params.a),
            ("l_ch", params.l_ch),
        ];
        for (name, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(crate::error::NegForgeError::InvalidParameter(format!(
                    "{name} must be finite and positive, got {value}"
                )));
            }
        }
        let l_ds_ok = params.l_ds.is_finite() && params.l_ds > 0.0;
        if !params.auto_size_contacts && !l_ds_ok {
            return Err(crate::error::NegForgeError::InvalidParameter(format!(
                "l_ds must be finite and positive when auto_size_contacts is false, got {}",
                params.l_ds
            )));
        }
        Ok(())
    }

    fn compute_lambda(params: &DeviceParams) -> f64 {
        (params.k_si / params.k_ox * params.d_ch * params.d_ox / params.geo).sqrt()
    }

    fn compute_grid(params: &DeviceParams) -> (usize, usize, usize) {
        let l_g = 2.0 * params.l_ds + params.l_ch;
        assert!(
            params.a > 0.0 && l_g.is_finite() && l_g >= params.a,
            "grid spacing a = {} nm is too coarse for a device of total length {} nm: \
             the discretization needs at least two grid points",
            params.a,
            l_g
        );
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

    /// Ballistic Landauer current for the current potential profile, in
    /// **amperes** per conducting subband:
    ///
    /// ```text
    /// I = (2e/h) * integral_{Psi_0}^{E_max} [f_s(E) - f_d(E)] dE
    /// ```
    ///
    /// The factor 2 is spin degeneracy; the lower limit `Psi_0 = max(Psi_f)`
    /// is the top of the barrier. This is the classic ballistic-MOSFET model:
    /// transmission is assumed to be exactly 1 above the barrier and 0 below
    /// it, so there is no tunneling and no quantum reflection. For the actual
    /// transmission of the potential profile, use
    /// [`crate::negf::negf_current`], which evaluates the Landauer integral
    /// with the NEGF `T(E)` instead.
    ///
    /// The original MATLAB multiplied this by an unexplained `1e-3`, which is
    /// dropped here: the formula above is the physical current, and the extra
    /// factor only made absolute values wrong. Ratios and the subthreshold
    /// swing (a log-slope) are unaffected by the change.
    pub fn calc_current(&self) -> f64 {
        let thermal = K_B * self.params.t / E;
        let f_s = |e: f64| 1.0 / (((e - self.params.e_fs) / thermal).exp() + 1.0);
        let f_d = |e: f64| 1.0 / (((e - self.e_fd) / thermal).exp() + 1.0);

        let n_steps = ((self.e_max - self.psi_0) / self.params.d_e)
            .floor()
            .max(0.0) as usize;
        let mut sum = 0.0;
        for k in 0..=n_steps {
            let energy = self.psi_0 + k as f64 * self.params.d_e;
            sum += f_s(energy) - f_d(energy);
        }
        // `sum * d_e` is in eV; the trailing `E` converts it to joules.
        2.0 * E / crate::constants::H * sum * self.params.d_e * E
    }

    /// Set the drain-source bias.
    ///
    /// Like the other setters, this rebuilds the drive terms *and* re-solves
    /// the electrostatic potential, so `psi_f`/`psi_0` are never left over
    /// from the previous bias point — `calc_current` and the NEGF entry
    /// points are safe to call immediately afterwards. Note that this resets
    /// `rho` to zero (see `init_vectors`), so the re-solve is the decoupled
    /// one; a self-consistent result has to be re-established by calling
    /// [`crate::selfconsistent::solve_self_consistent`] again.
    pub fn set_v_ds(&mut self, v: f64) {
        self.params.v_ds = v;
        self.e_fd = -v + 0.05;
        self.init_vectors();
        self.calc_potential();
    }

    /// Set the gate bias. Same re-solve contract as [`Device::set_v_ds`].
    pub fn set_v_g(&mut self, v: f64) {
        self.params.v_g = v;
        self.init_vectors();
        self.calc_potential();
    }

    /// Set the channel length, re-deriving the grid. Same re-solve contract
    /// as [`Device::set_v_ds`].
    pub fn set_l_ch(&mut self, l: f64) {
        self.params.l_ch = l;
        let (n, n_left, n_right) = Self::compute_grid(&self.params);
        self.n = n;
        self.n_left = n_left;
        self.n_right = n_right;
        self.init_vectors();
        self.calc_potential();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
    fn bias_setters_leave_the_potential_up_to_date() {
        // A bias setter must not leave `psi_f`/`psi_0` describing the
        // previous bias point: calling `calc_current()` straight after a
        // setter has to give the same answer as an explicit re-solve.
        let mut dev = Device::new(DeviceParams::default());

        dev.set_v_ds(0.3);
        let psi_f_after_setter = dev.psi_f.clone();
        let psi_0_after_setter = dev.psi_0;
        let current_after_setter = dev.calc_current();

        dev.calc_potential();
        assert_eq!(dev.psi_f, psi_f_after_setter);
        assert_eq!(dev.psi_0, psi_0_after_setter);
        assert_eq!(dev.calc_current(), current_after_setter);

        // Same for a setter that changes the grid size.
        dev.set_l_ch(20.0);
        assert_eq!(dev.psi_f.len(), dev.n);
        assert!(dev.psi_f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn increasing_v_ds_increases_current() {
        let mut dev = Device::new(DeviceParams::default());
        let i0 = dev.calc_current();

        dev.set_v_ds(0.3);
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
        let i0 = dev.calc_current();

        dev.set_v_g(0.3);
        let i1 = dev.calc_current();

        assert!(
            i1 > i0,
            "current should increase with gate bias: {i0} -> {i1}"
        );
    }

    #[test]
    #[should_panic(expected = "geo")]
    fn rejects_zero_geo() {
        Device::new(DeviceParams {
            geo: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "d_ch")]
    fn rejects_negative_d_ch() {
        Device::new(DeviceParams {
            d_ch: -1.0,
            ..Default::default()
        });
    }
}
