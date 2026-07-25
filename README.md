# NEGForge

A 1D ballistic-MOSFET electrostatics + NEGF (Non-Equilibrium Green's
Function) transport simulator. The numerical engine is Rust; the frontend
is a Jupyter notebook, talking to the engine through a native PyO3
extension module.

This is a rewrite of a MATLAB nanoelectronics coursework project (see
[`legacy_matlab/`](legacy_matlab/)) that implemented a self-consistent-style
1D electrostatic solve, a ballistic (Landauer) current calculation, and an
NEGF local-density-of-states / charge-density calculation — but never
actually closed the loop between electrostatics and charge. This rewrite
does.

## Architecture

```
crates/negforge-core/   Pure-Rust physics engine (no Python dependency)
crates/negforge-py/     PyO3 bindings, compiled to the native module
                         negforge._negforge
python/negforge/        Pythonic wrapper package (numpy arrays, keyword
                         device construction, IVCurve helper)
notebooks/               Jupyter notebook frontend
legacy_matlab/           Original MATLAB code, kept for reference
docs/PHYSICS.md          Physics and implementation guide for legacy_matlab/
```

`negforge-core` has no PyO3/Python dependency at all — it's a normal Rust
library with its own unit and integration tests, usable from any Rust
program. `negforge-py` is a thin binding layer on top of it.

### `negforge-core` modules

- `constants` — physical constants (elementary charge, k_B, h, hbar,
  electron mass), replacing the undefined `util.const` package the
  original MATLAB code depended on but never included.
- `tridiag` — real and complex Thomas-algorithm tridiagonal solvers, used
  by both the electrostatics and NEGF modules.
- `device` — device geometry/parameters and the electrostatic
  (`calc_potential`) and ballistic-current (`calc_current`) calculations.
- `negf` — the NEGF retarded Green's-function calculation. Rather than the
  original's dense O(N^3) matrix inversion (`inv()` in MATLAB, fine for a
  one-off plot but too slow to call repeatedly in a self-consistent loop
  or a bias sweep), this uses a left-connected recursive Green's-function
  sweep (O(N)) for the diagonal (local density of states) and a direct
  tridiagonal solve (O(N)) for each of the two contact-column quantities
  needed for the charge density — both cross-checked against a dense
  reference solver in the test suite.
- `charge` — NEGF-derived electron density, with two bug fixes relative to
  the original `calc_n()` (see "Deviations from the original" below).
- `selfconsistent` — **new**: the Poisson&harr;NEGF self-consistency loop.
- `sweep` — bias sweeps (`sweep_v_g`, `sweep_v_ds`) and subthreshold-swing
  extraction, mirroring `plot_Vg_I` / `plot_Vds_I` / `plot_S`.

## Building and running

### Rust engine only

```bash
cargo build --workspace
cargo test --workspace
```

### Python package + notebook

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release          # builds negforge-py and installs it editable
pip install -e ".[notebook]"       # jupyter, matplotlib, ipywidgets
jupyter notebook notebooks/negforge_demo.ipynb
```

`maturin develop` compiles `crates/negforge-py` and installs the
resulting extension inside the `python/negforge/` package (as
`negforge._negforge`); `python/negforge/__init__.py` wraps it in a
friendlier, numpy-returning API (`negforge.Device`, `negforge.IVCurve`).

## Physics model

Source-channel-drain chain, 1D, with source/drain treated as long, heavily
screened contact regions:

- **Electrostatics**: not a full 2D/3D Poisson solve, but a 1D "natural
  length" (`lambda`) approximation common in compact models for
  thin-body/double-gate MOSFETs. `lambda = sqrt(k_si/k_ox * d_ch * d_ox /
  geo)`. The electrostatic operator is tridiagonal with a reflective
  (Neumann-like) boundary condition at both ends, solved via the Thomas
  algorithm.
- **Ballistic current**: the Landauer formula, `I = 2e/h * integral[f_s(E)
  - f_d(E)] dE`, with Fermi-Dirac occupation of the source/drain
  reservoirs.
- **NEGF**: a tight-binding discretization (on-site energy `2t + Psi_f(x)`,
  hopping `-t`) with open-boundary self-energies at the two contacts,
  giving the retarded Green's function, local density of states, and
  (with the self-consistent extension) the charge density.

Units follow the original code throughout: lengths in nm, energies and
potentials in eV, temperature in Kelvin.

## Deviations from the original MATLAB code

This is a faithful port plus closing the missing self-consistency loop,
not a from-scratch physics redesign. Everything below is a deliberate,
documented decision, not an accident:

1. **Self-consistent Poisson&harr;NEGF loop** (`selfconsistent.rs`). The
   original computed the electrostatic potential once with `rho = 0` and,
   separately, an NEGF charge density — but never fed one into the other.
   This is genuinely new: solve electrostatics, compute the NEGF electron
   density, convert it to a charge density, damp/mix it into `rho`,
   re-solve, iterate to convergence (or report a `NotConverged` error
   rather than silently returning garbage).

2. **Charge-density unit fix** (needed for #1 to be numerically meaningful
   at all). `calc_potential`'s RHS divides `rho + N_dot` by `EPS_0 *
   k_si`, with `EPS_0` in SI units (F/m); for that division to land in the
   same eV/nm^2 ballpark as the model's other terms, `rho` must be a
   genuine volume charge density in C/m^3. The original never exercised
   this since `rho` was always zero. The self-consistent loop converts the
   NEGF electron density (computed in nm^-1, the model's native length
   unit) to C/m^3 by treating the 1D chain as having an implicit unit (1
   m^2) cross-section and multiplying by `-e`. Skipping this and feeding a
   raw, differently-scaled density into `rho` was tried first and made the
   fixed-point iteration diverge by many orders of magnitude — see the
   module docs in `charge.rs` and `selfconsistent.rs` for the full
   derivation.

3. **NEGF broadening (`eta`) inside the self-consistent loop.** The
   original hardcoded `eta = 1e-8` for a one-off local-density-of-states
   plot. That's fine for a static plot but not for a feedback loop: a
   bound-state resonance whose energy happens to land within `eta` of an
   energy-grid point produces a `|G|^2` spike orders of magnitude larger
   than neighboring grid points, and feeding that grid-alignment-dependent
   spike back into the electrostatic solve makes the iteration diverge or
   oscillate. `SelfConsistentOptions::eta` defaults to `0.08` eV (tuned
   empirically against the default device geometry — see `negf.rs` module
   docs), large enough to resolve resonances smoothly across iterations.
   The original's small `eta = 1e-8` is preserved as `negf::DEFAULT_ETA`
   for the standalone, non-self-consistent `local_density_of_states` /
   LDOS-plot path, where the original's sharp-peak behavior is what you
   want to see.

4. **Two `calc_n()` bug fixes** (`charge.rs`), needed for the
   self-consistent charge density to respond to the actual physics rather
   than being a constant:
   - The original multiplies the whole energy sum by a single scalar
     `f(E_fs)` / `f(E_fd)` (evaluated once, outside the sum) instead of the
     energy-dependent Fermi occupation `f_s(E)` / `f_d(E)` used everywhere
     else in the model (e.g. `calc_current`). Fixed to use the proper
     per-energy weight.
   - The original reuses one `mask = E > Psi_f(1)` (the *source* band
     edge) for both the source and drain contact terms. Each contact's
     contribution is now masked by its own band edge, consistent with how
     `calc_green` already conditions each contact's self-energy on its own
     band edge.

5. **O(N) NEGF instead of O(N^3).** See "Architecture" above. Purely a
   performance change (validated against a dense reference solver in
   tests); the physics is unchanged.

Everything else — the electrostatic operator, the ballistic current
formula, the contact self-energy sign convention, the general unit
handling (nm/eV/K) — is a direct, unmodified port. In particular, the
model's overall dimensional consistency is inherited as-is from the
original teaching code (e.g. the electrostatic equation isn't a fully
rigorous SI-unit Poisson equation); this rewrite does not attempt to
re-derive the model's physics from first principles, only to make it run,
close its one clearly-missing feedback loop, and fix the bugs that stood
in the way of that loop actually doing something.

## Testing

- `crates/negforge-core/src/*.rs` — unit tests per module, including
  cross-checks of the O(N) tridiagonal/recursive-Green's-function solvers
  against dense (Gaussian-elimination / full-matrix-inversion) reference
  implementations on small systems.
- `crates/negforge-core/tests/self_consistent_realistic_device.rs` — an
  integration test at the model's default (non-toy) device scale,
  confirming the self-consistent loop converges and reproduces the
  expected ballistic-MOSFET trend (current increasing with gate bias).
- `notebooks/negforge_demo.ipynb` has been executed end-to-end
  (`jupyter nbconvert --execute`) to confirm the full frontend path works;
  outputs are cleared before committing since they go stale the moment the
  engine changes.

Run everything with `cargo test --workspace`.
