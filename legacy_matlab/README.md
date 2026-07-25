# Legacy MATLAB implementation

This directory preserves the original MATLAB coursework code (1D ballistic
MOSFET / NEGF transport simulator) that the Rust engine in `crates/` and the
notebook in `notebooks/` are based on. It is kept for reference and is not
part of the build.

Notes on the original code, carried over into the rewrite's design notes
(see the top-level README "Physics model & known limitations" section):

- `quantumsim.m` is the most complete version (electrostatics + ballistic
  current + NEGF local density of states / charge density).
- `simulation.m`, `simulation_v2.m`, `simulation_v3.m` are earlier
  procedural-script duplicates of the same model with minor sweep variants.
- `aufgabe1e.m` is a standalone, simpler Poisson-only exercise.
- All of them reference an undefined `util.const` package (elementary
  charge, Boltzmann constant, Planck constant, electron mass) that is not
  present in this repository, so none of this code runs as-is.
- The electrostatic solve and the NEGF charge calculation were never wired
  together into a feedback loop in the original code (`calc_potential`
  always used `rho = 0`). The Rust rewrite adds that self-consistent loop;
  see the top-level README for details and caveats.
- The rewrite does **not** treat this code as ground truth. Several pieces
  of its physics are wrong and are corrected rather than reproduced: the
  contact self-energy has the wrong sign of its imaginary part (giving a
  partly-negative density of states), `calc_n` omits spin degeneracy and
  uses an energy-independent Fermi weight with the wrong contact's band
  edge, and `calc_current` carries an unexplained `1e-3` factor. Each
  deviation is listed with its justification in the top-level README.
