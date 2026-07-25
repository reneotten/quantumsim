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
  (`calc_potential`) and ballistic-current (`calc_current`) calculations,
  including the `GateGeometry`/`planar`/`fin_fet`/`nanoribbon` presets —
  see "Planar, FinFET and nanoribbon devices" below.
- `negf` — the NEGF retarded Green's-function calculation, with two
  interchangeable algorithms (`GreenFunctionAlgorithm::{Recursive,
  Dense}`) and a parallel energy loop — see "Performance" below.
- `charge` — NEGF-derived electron density, with two bug fixes relative to
  the original `calc_n()` (see "Deviations from the original" below).
- `selfconsistent` — **new**: the Poisson&harr;NEGF self-consistency loop.
- `sweep` — bias sweeps (`sweep_v_g`, `sweep_v_ds`) and subthreshold-swing
  extraction, mirroring `plot_Vg_I` / `plot_Vds_I` / `plot_S`. Each sweep
  point runs in parallel (see "Performance").

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

## Planar, FinFET and nanoribbon devices

`geo` in the natural-length formula above (`lambda = sqrt(k_si/k_ox *
d_ch * d_ox / geo)`) is the number of gates electrostatically controlling
the channel — the "generalized scale length" model from the multi-gate
MOSFET literature (Auth & Plummer 1997). It's the one parameter that
actually distinguishes device architectures in this 1D model: more gates
gives a smaller natural length, i.e. tighter electrostatic control and
better short-channel-effect immunity, for the same body thickness.
[`GateGeometry`] enumerates the standard cases (`SingleGate`=1,
`DoubleGate`=2, `TriGate`=3, `GateAllAround`=4), and `DeviceParams` has
matching presets with realistic body/oxide dimensions:

| Preset (Rust `DeviceParams::`, Python `negforge.Device.`) | Architecture | `geo` | Notes |
|---|---|---|---|
| `planar()` | Planar bulk/SOI MOSFET | 1 | Same as `default()` |
| `fin_fet()` | Double-gate FinFET | 2 | Thin fin (`d_ch` = fin width), sidewall gates only; `.with_gate_geometry(GateGeometry::TriGate)` if the fin top is also gated |
| `nanoribbon()` | Gate-all-around nanoribbon/nanowire | 4 | Narrow body (`d_ch` = ribbon width/diameter), gate wraps all sides |

This reproduces the textbook result
(`crates/negforge-core/examples/geometry_comparison.rs`, cross-checked in
`crates/negforge-core/tests/multigate_geometry.rs`): at a channel length
short enough for the planar device to show real short-channel-effect
degradation, FinFET and especially gate-all-around do much better —
subthreshold swing at `l_ch = 10 nm` (ideal thermal limit at 300 K is
59.6 mV/decade):

```
planar      lambda= 8.473 nm   S= 147.3 mV/decade
fin_fet     lambda= 4.151 nm   S=  86.4 mV/decade
nanoribbon  lambda= 1.895 nm   S=  64.3 mV/decade
```

All three architectures also run through the full self-consistent
Poisson&harr;NEGF loop and both NEGF algorithms without issue — see the
test file above.

**What this is not**: `d_ch` here is a single effective body-thickness
number, not an actually-resolved cross-section — there's no transverse-
mode/subband quantization, so it can't capture, say, the difference
between a wide-and-thin FinFET fin and a narrow-and-thick one with the
same `geo` and cross-sectional area, or volume-inversion effects specific
to very narrow gate-all-around wires. Modeling that requires resolving the
cross-section, which is what a real 3D (or mode-space) extension would
add — see below.

## Extending to a real 3D solver

Nothing here — every geometry preset above still uses the same 1D
natural-length electrostatics and 1D NEGF transport. A genuine 3D (or
"mode-space", the standard middle ground) solver is a substantially
different piece of software, not an incremental change to this one, and
wasn't built as part of this pass. Rough shape of what it would take, for
context if this becomes a real ask later:

1. **Mode-space NEGF** (the tractable approach real nanoscale-FET
   simulators use, e.g. nanoMOS/OMEN/NEMO — as opposed to full real-space
   3D NEGF, which is a research-grade undertaking on its own): at each
   slice along the transport direction, solve a 2D (FinFET) or
   cross-sectional Schrodinger equation for the transverse confinement
   eigenvalues (subbands) and eigenvectors, given the local potential.
2. Run this crate's existing 1D NEGF machinery once per subband, using
   each subband's confinement energy as an added effective potential
   floor, and sum the resulting charge/current across subbands.
3. Couple that to a 2D or 3D Poisson solve (cross-section x length, or a
   full 3D mesh) instead of the current 1D natural-length formula, and
   iterate to self-consistency the same way `selfconsistent.rs` already
   does for the 1D case.
4. None of the current O(N) tridiagonal tricks carry over directly — the
   Poisson operator is no longer tridiagonal in 2D/3D (a sparse
   iterative solver, e.g. conjugate gradient, would replace the Thomas
   algorithm), and the NEGF energy-loop parallelism generalizes but now
   has a subband dimension to parallelize over too.

This is weeks-to-months of numerical-methods and validation work, not a
follow-on patch, and getting it subtly wrong (e.g. a sign error in the
subband coupling) is easy to do and hard to notice without a trusted 2D/3D
reference to check against — unlike the 1D engine here, which could be
(and was) validated by cross-checking algorithms against each other and
against known physical trends. If this is wanted, it's worth scoping and
staffing as its own effort rather than folding into this codebase's
existing architecture.

## Deviations from the original MATLAB code

The user requested a faithful port plus closing the missing
self-consistency loop, not a from-scratch physics redesign. Everything
below is a deliberate, documented decision, not an accident:

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

5. **A choice of O(N) or O(N^3) NEGF, plus parallelism.** See
   "Performance" below. Purely an implementation/engineering addition
   (validated by cross-checking the two algorithms against each other in
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

## Performance

### Dense vs. recursive NEGF: the tradeoff

Every NEGF quantity (local density of states, the injected charge used by
the self-consistent loop) can be computed two ways, chosen via
`GreenFunctionAlgorithm` in Rust or an `algorithm: "recursive" | "dense"`
string argument in Python:

|                    | `Dense`                                             | `Recursive` (default)                                         |
|--------------------|------------------------------------------------------|-----------------------------------------------------------------|
| Method             | Build the full N×N complex matrix, invert it (Gauss-Jordan), read off the diagonal and two boundary columns — what the original MATLAB `inv()` call did. | Left-connected recursive Green's-function sweep (diagonal) + a direct tridiagonal solve (the two boundary columns). |
| Cost per energy point | O(N^3) time, O(N^2) memory                        | O(N) time, O(N) memory                                          |
| Why it exists      | Obviously correct: a direct definition-level matrix inversion, no tridiagonal-structure assumption, no recursion formula to get subtly wrong. Useful as a trusted reference (see tests) and as a fallback if the Hamiltonian ever gains longer-range hopping and stops being purely tridiagonal. | The one to use for anything performance-sensitive — the self-consistent loop and bias sweeps call this dozens to hundreds of times. |

Both are cross-checked against each other in
`negf::tests::dense_and_recursive_algorithms_agree_end_to_end` and
`selfconsistent::tests::dense_and_recursive_converge_to_the_same_potential`.
Measured with `cargo run --release --example benchmark -p negforge-core`
(single NEGF sweep, 100 energy points, on a 4-core machine):

```
     N      dense (s)  recursive (s)    speedup
    21         0.0018         0.0002        10x
    51         0.0144         0.0002        76x
   101         0.1202         0.0003       395x
   201         0.8849         0.0005      1889x
   351         6.2474         0.0009      6611x
```

Dense scales cubically and recursive is essentially flat, as expected. At
the model's default device size (N=561), a full self-consistent solve
using `Dense` did not finish a single iteration in several minutes — it is
not a realistic choice at that scale, only for small devices or
cross-validation.

### Parallelism

Every energy point in a Green's-function sweep is an independent
calculation (they only read the same fixed Hamiltonian), so the energy
loop runs in parallel via `rayon`, spreading points across all available
CPU cores automatically (respects `RAYON_NUM_THREADS` if you want to cap
it). This applies to both algorithms and is the default — there's no
opt-in needed. Measured on the same 4-core machine (Recursive, N=561, 900
energy points, one sweep):

```
1 thread:      0.0479 s
all cores (4):  0.0169 s
speedup:        2.8x-4.0x (varies by run/load)
```

Bias sweeps (`sweep_v_g`/`sweep_v_ds`) are parallel too, at the *point*
level rather than the energy level: `set_v_g`/`set_v_ds` reset a device's
`psi_g`/`psi_bi`/`rho` from scratch (see `Device::init_vectors`) and
re-solve `psi_f` for the new bias, so nothing carries over between bias
points — each one is evaluated on its
own cloned `Device`, in parallel. This is why `sweep_v_g`/`sweep_v_ds` take
`&Device` (a template whose bias is varied) rather than `&mut Device`: the
input device's own state is left untouched, which also fixed a surprising
side effect the earlier API had (a sweep silently leaving the device
parked at its last bias point).

The one thing that is **not** parallelizable is the self-consistent loop's
*iterations* — each iteration's charge depends on the previous iteration's
potential, a genuine sequential dependency. Parallelism instead comes from
within each iteration's NEGF sweep (above), which is where nearly all the
time goes.

### Using the switch

Rust:

```rust
use negforge_core::{GreenFunctionAlgorithm, SelfConsistentOptions};

let opts = SelfConsistentOptions {
    algorithm: GreenFunctionAlgorithm::Dense, // or ::Recursive (default)
    ..Default::default()
};
```

Python:

```python
dev.solve_self_consistent(algorithm="dense")   # or "recursive" (default)
dev.local_density_of_states(algorithm="dense")
dev.sweep_v_g(0.0, 0.4, 0.05, self_consistent=True, algorithm="dense")
```

## Testing

- `crates/negforge-core/src/*.rs` — unit tests per module, including
  cross-checks of the O(N) tridiagonal/recursive-Green's-function solvers
  against the dense (Gaussian-elimination / full-matrix-inversion)
  algorithm on small systems, and of the two `GreenFunctionAlgorithm`
  variants against each other end-to-end (`negf.rs`) and through the full
  self-consistent loop (`selfconsistent.rs`).
- `crates/negforge-core/tests/self_consistent_realistic_device.rs` — an
  integration test at the model's default (non-toy) device scale,
  confirming the self-consistent loop converges and reproduces the
  expected ballistic-MOSFET trend (current increasing with gate bias).
- `crates/negforge-core/tests/multigate_geometry.rs` — the
  planar/FinFET/nanoribbon physical sanity checks: natural-length
  ordering, all three architectures converging (decoupled and
  self-consistent), and subthreshold swing improving with more gates at a
  short channel length.
- `crates/negforge-core/examples/benchmark.rs` — the performance audit
  behind the numbers quoted above; run it with `cargo run --release
  --example benchmark -p negforge-core`.
- `crates/negforge-core/examples/geometry_comparison.rs` — the
  planar/FinFET/nanoribbon subthreshold-swing comparison quoted above; run
  it with `cargo run --release --example geometry_comparison -p
  negforge-core`.
- `notebooks/negforge_demo.ipynb` has been executed end-to-end
  (`jupyter nbconvert --execute`) to confirm the full frontend path works,
  including the dense-vs-recursive comparison cell; outputs are cleared
  before committing since they go stale the moment the engine changes.

Run everything with `cargo test --workspace`.
