//! Physical constants (SI unless noted). Values are CODATA 2018 recommended
//! values. The original MATLAB code referenced an undefined `util.const`
//! package with these same names (`e`, `k_b`, `h`, `h_bar`, `m_e`) — this
//! module is the concrete replacement for it.

/// Elementary charge, C.
pub const E: f64 = 1.602_176_634e-19;
/// Boltzmann constant, J/K.
pub const K_B: f64 = 1.380_649e-23;
/// Planck constant, J*s.
pub const H: f64 = 6.626_070_15e-34;
/// Reduced Planck constant, J*s.
pub const H_BAR: f64 = 1.054_571_817e-34;
/// Electron rest mass, kg.
pub const M_E: f64 = 9.109_383_701_5e-31;
/// Vacuum permittivity, F/m. The legacy MATLAB code hardcoded `8.85e-12`
/// inline instead of using a constants module; we use the precise value.
pub const EPS_0: f64 = 8.854_187_812_8e-12;
