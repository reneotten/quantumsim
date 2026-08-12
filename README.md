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

### Validity of the model — read before trusting a number

Each approximation below is documented in more detail (with the reasoning
and, where relevant, a test demonstrating it) in the corresponding source
module's doc comments.

- **Natural-length electrostatics** (`device.rs`) assumes a thin body with a
  parabolic transverse potential profile. Reasonable for thin-body/
  double-gate/gate-all-around MOSFETs; not valid for bulk (thick-body)
  devices. It is also a classical (non-quantized) transverse charge model,
  in tension with the fully-quantized longitudinal NEGF treatment — this is
  a first-order, not a fully self-consistent 2D, model.
- **Ballistic Landauer current** (`device.rs`) assumes perfect transmission
  above the channel barrier and zero below it: no scattering, no
  sub-barrier tunneling. Good for short channels well below the carrier
  mean free path; overestimates on-current and underestimates subthreshold
  current otherwise.
- **Effective-mass, single-valley, parabolic band** (`m_eff`, used for both
  the ballistic hopping parameter and the NEGF tight-binding parameter):
  real silicon has multiple equivalent valleys and a non-parabolic band
  away from the edge; `m_eff` is a single fitting parameter standing in for
  both, so absolute currents should be read as illustrative trends rather
  than device-accurate predictions.
- **Tight-binding NEGF discretization** (`negf.rs`) reproduces the continuum
  parabolic dispersion only for grid spacings fine enough that the swept
  energy range stays well below the chain's `4*t_hop` bandwidth; worth
  checking via a grid-refinement test at new device scales.
- **Contact self-energy sign.** Causal, absorbing contacts require
  `Im(sigma) <= 0`, so the implementation uses
  `sigma = t_hop * exp(-i*k*a)` for the wavevector branch it computes. See
  `negf.rs` for the derivation. The charge-density calculation uses
  `sin(k)`, which is invariant under this branch/sign choice.
- **No explicit spin-degeneracy factor** in the self-consistent charge
  density (unlike `calc_current`'s explicit `2e/h`), carried over unchanged
  from the original `calc_n`, which was never exercised against a
  reference before this rewrite closed the feedback loop.

None of the above are believed to affect the qualitative trends the test
suite checks (current increasing with gate/drain bias, self-consistent
convergence to a stable, non-negative charge density) — but they mean
absolute numbers out of this model should be treated as illustrative of
device physics concepts, not as device-accurate predictions.

## Implementation differences from the legacy MATLAB code

The Rust implementation closes the electrostatics/charge feedback loop, fixes
the charge-density occupation and contact-mask calculations, and replaces the
dense NEGF solve with O(N) tridiagonal and recursive Green's-function methods.
It also uses a finite NEGF broadening during self-consistent iterations to
avoid grid-aligned resonances destabilizing the loop. The detailed rationale
and validation live in the module documentation and test suite.

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
