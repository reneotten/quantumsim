//! NEGForge: a Rust port of a 1D ballistic-MOSFET electrostatics + NEGF
//! transport model (originally MATLAB, see `legacy_matlab/`).
//!
//! See the crate modules for the physics of each piece, and the top-level
//! repository README for the overall architecture, build instructions, and
//! a list of deliberate deviations from the original code.

pub mod charge;
pub mod constants;
pub mod device;
pub mod error;
pub mod negf;
pub mod selfconsistent;
pub mod sweep;
mod tridiag;

pub use device::{Device, DeviceParams, GateGeometry};
pub use error::{NegForgeError, Result};
pub use negf::{GreenFunctionAlgorithm, GreenFunctionResult};
pub use selfconsistent::{SelfConsistentOptions, SelfConsistentResult};
pub use sweep::IvPoint;
