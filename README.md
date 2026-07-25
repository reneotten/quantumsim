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
- `negf` — the NEGF retarded Green's-function calculation: contact
  self-energies, local density of states, Landauer-Caroli transmission and
  the current derived from it, with two interchangeable algorithms
  (`GreenFunctionAlgorithm::{Recursive, Dense}`), selectable outputs
  (`GreenOutputs`) and a parallel energy loop — see "Performance" below.
- `charge` — NEGF-derived electron density, with four corrections relative
  to the original `calc_n()` (see "Deviations from the original" below).
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
- **Ballistic current** (`calc_current`): the Landauer formula assuming
  perfect transmission above the barrier top, `I = 2e/h *
  integral[f_s(E) - f_d(E)] dE`, with Fermi-Dirac occupation of the
  source/drain reservoirs. Returns amperes per conducting subband.
- **NEGF**: a tight-binding discretization (on-site energy `2t + Psi_f(x)`,
  hopping `-t`) with retarded open-boundary self-energies `Sigma^r =
  t*exp(-i k a)` at the two contacts, giving the retarded Green's
  function, the local density of states, the charge density that closes
  the self-consistent loop, and the Landauer-Caroli transmission
  `T(E) = Gamma_s Gamma_d |G_{N-1,0}|^2`.
- **NEGF current** (`calc_current_negf`): the same Landauer integral with
  the *actual* `T(E)` instead of a step function, so it includes tunneling
  through the barrier and quantum reflection above it. It agrees with the
  ballistic formula to a few percent in the on-state and departs from it
  in regimes where the analytic model's assumptions break down — notably
  when a large `V_ds` pushes the conduction window past the discretized
  band's finite width `4t` (0.68 eV at the default `a = 0.5 nm`; shrink
  `a` to widen it).

Units follow the original code throughout: lengths in nm, energies and
potentials in eV, temperature in Kelvin. Charge densities (`rho`,
`N_dot`) are the one place that needs care: because the Laplacian is built
in nm^-2, they are SI C/m^3 scaled by `RHO_SI_TO_MODEL` (1e-18). The
self-consistent loop does that conversion, together with the division by
`cross_section_nm2` that turns the 1D electron density into a volume
density.

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

The original is a teaching code, not a reference implementation, and it is
not treated as ground truth here: where its physics is wrong, this port
fixes it and says so. Everything below is a deliberate, documented
decision, not an accident:

1. **Contact self-energy sign** (`negf.rs`) — the most consequential fix.
   The original attaches `Sigma = +t*exp(+i k a)` at each contact, which has
   `Im(Sigma) > 0`. A retarded self-energy must have `Im <= 0`; with the
   original sign the contact term *subtracts* from the `+i*eta`
   regularization instead of adding to it, and two things break:

   - The Green's function ends up retarded on sites reached only by `eta`
     and advanced on contact-coupled sites, so `Im(G_ii)` is not
     sign-definite. Measured on the default device, **32% of the
     "LDOS" entries came out negative** — a density of states cannot be
     negative.
   - Inside the self-consistent loop, where `eta` is comparable to
     `t sin(ka)`, the two nearly cancel and `|G|^2` at the contacts is
     inflated by **5-14x** (measured at the `eta` values the loop used),
     distorting the charge density fed back into Poisson.

   NEGForge uses `Sigma^r = t*exp(-i k a)`, so
   `Im[(E + i*eta) I - H - Sigma^r]` is positive definite and the LDOS
   (`-Im(G_ii) / (pi a)`) is positive by construction — asserted in both
   unit and integration tests. The self-energy is also attached at every
   energy rather than only inside the contact band; outside it, the
   closed form is real, which is the correct evanescent level shift
   (`Gamma = 0`) that the original dropped.

2. **Self-consistent Poisson&harr;NEGF loop** (`selfconsistent.rs`). The
   original computed the electrostatic potential once with `rho = 0` and,
   separately, an NEGF charge density — but never fed one into the other.
   This is genuinely new: solve electrostatics, compute the NEGF electron
   density, convert it to a charge density, damp/mix it into `rho`,
   re-solve, iterate to convergence (or report a `NotConverged` error
   rather than silently returning garbage).

3. **Charge-density units and the device cross-section** (needed for #2 to
   mean anything). Two separate conversions are involved, and both were
   implicit before:

   - The electrostatic solve's Laplacian is built in nm^-2 while
     `rho / (EPS_0 * k_si)` is in V/m^2, a factor of 1e18 apart. `rho` and
     `N_dot` are therefore documented as SI C/m^3 scaled by
     `RHO_SI_TO_MODEL`.
   - The transport model is 1D and yields a *linear* density (nm^-1);
     converting it to a volume density needs a channel cross-section.
     `DeviceParams::cross_section_nm2` makes that explicit, defaulting to
     `d_ch^2`.

   Leaving the cross-section implicit — as the original scaling did — is
   equivalent to assuming **1 nm^2**, which for the default geometry makes
   the charge term the dominant contribution to the potential and, with the
   sign fix in place, no longer converges at all. With the device's own
   `d_ch^2 = 25 nm^2`, the charge term settles at roughly a third of the
   gate/built-in term: a real but perturbative correction, which is what
   self-consistency in this regime should look like.

4. **Spin degeneracy in the charge density** (`charge.rs`). The original
   includes the factor 2 for spin in the current (`2e/h`) but omits it in
   `calc_n`, so the charge fed back into Poisson was a factor 2 light.
   Both now use `g_s = 2`.

5. **Two more `calc_n()` fixes** (`charge.rs`):
   - The original multiplies the whole energy sum by a single scalar
     `f(E_fs)` / `f(E_fd)` (evaluated once, outside the sum) instead of the
     energy-dependent Fermi occupation `f_s(E)` / `f_d(E)` used everywhere
     else in the model. Fixed to use the proper per-energy weight.
   - The original reuses one `mask = E > Psi_f(1)` (the *source* band edge)
     for both contact terms. Each contact's broadening now follows from its
     own self-energy and vanishes outside its own band automatically, so no
     mask is needed.

6. **NEGF broadening (`eta`) inside the self-consistent loop.** `eta` is a
   numerical-integration parameter: it only has to be large enough that a
   resonance is resolved across several energy-grid points, otherwise the
   charge estimate jumps around with grid alignment between iterations and
   the fixed point never settles. It therefore scales with the energy step:
   `SelfConsistentOptions::eta` defaults to `5 * d_e` (5 meV at the default
   `d_e = 1 meV`), the smallest multiple that converges robustly across
   device scales — `1 * d_e` does not converge. This is ~16x smaller than
   the value the loop needed before the self-energy sign was fixed, which
   is the point: that large `eta` was compensating for a bug. The
   original's `eta = 1e-8` is kept as `negf::DEFAULT_ETA` for the
   standalone LDOS and transmission paths, where sharp resonances are what
   you want to see.

7. **Landauer-Caroli current** (`negf::negf_current`, `calc_current_negf`
   in Python) — new. The original computed a Green's function but never a
   transmission, so the current was always the analytic `T(E) = 1`
   approximation even when the NEGF machinery was running. Since
   `T(E) = Gamma_s Gamma_d |G_{N-1,0}|^2` needs only the source column the
   sweep already computes, the NEGF current comes essentially for free.
   `calc_current` is kept as the analytic ballistic model rather than
   replaced: it is a legitimate compact model, and the two disagree
   precisely where each one's assumptions differ (see the discretization
   caveat in the `negf_current` docs).

8. **Unexplained `1e-3` in the current** (`device.rs`). The original
   multiplies the Landauer integral by `1e-3` with no stated justification;
   it made absolute currents wrong by three orders of magnitude. Dropped —
   `calc_current` now returns amperes per subband. Ratios and the
   subthreshold swing (a log-slope) are unaffected, so this changes no
   qualitative result.

9. **A choice of O(N) or O(N^3) NEGF, plus parallelism.** See
   "Performance" below. Purely an implementation/engineering addition
   (validated by cross-checking the two algorithms against each other in
   tests); the physics is unchanged.

Everything else — the electrostatic operator and its reflective boundary
condition, the `lambda` approximation, the region masks, the geometry, the
nm/eV/K unit conventions — is a direct, unmodified port.

### Known model limitations (inherited, not bugs)

- The electrostatics is the original's 1D "natural length" approximation,
  not a rigorous 2D/3D Poisson solve.
- The tight-binding chain reproduces a parabolic band only near the band
  bottom; its total width `4t` scales as `1/a^2`, so results that depend on
  states near the band top are discretization-limited (see
  `negf_current`'s docs).
- `calc_current`'s ballistic approximation ignores band structure entirely,
  which is why it and `calc_current_negf` diverge deep in subthreshold at
  large `V_ds`.
- The self-consistent loop uses simple linear mixing; it is not a Newton or
  Anderson scheme, so strongly-coupled regimes may need a smaller `mixing`.

## Performance

### Dense vs. recursive NEGF: the tradeoff

Every NEGF quantity (local density of states, the injected charge used by
the self-consistent loop) can be computed two ways, chosen via
`GreenFunctionAlgorithm` in Rust or an `algorithm: "recursive" | "dense"`
string argument in Python:

|                    | `Dense`                                             | `Recursive` (default)                                         |
|--------------------|------------------------------------------------------|-----------------------------------------------------------------|
| Method             | Build the full N×N complex matrix and eliminate on it — what the original MATLAB `inv()` call did. A full Gauss-Jordan inverse when the diagonal is wanted, otherwise the same elimination with just two right-hand sides. | Left-connected recursive Green's-function sweep (diagonal) + one shared tridiagonal factorization for both boundary columns. |
| Cost per energy point | O(N^3) time, O(N^2) memory                        | O(N) time, O(N) memory                                          |
| Why it exists      | Obviously correct: a direct definition-level matrix inversion, no tridiagonal-structure assumption, no recursion formula to get subtly wrong. Useful as a trusted reference (see tests) and as a fallback if the Hamiltonian ever gains longer-range hopping and stops being purely tridiagonal. | The one to use for anything performance-sensitive — the self-consistent loop and bias sweeps call this dozens to hundreds of times. |

Both are cross-checked against each other in
`negf::tests::dense_and_recursive_algorithms_agree_end_to_end` and
`selfconsistent::tests::dense_and_recursive_converge_to_the_same_potential`.
Measured with `cargo run --release --example benchmark -p negforge-core`
(single NEGF sweep, 100 energy points, on a 4-core machine):

```
     N      dense (s)  recursive (s)    speedup
    21         0.0020         0.0001        15x
    51         0.0158         0.0002        74x
   101         0.1170         0.0004       313x
   201         0.9209         0.0007      1232x
   351         5.9763         0.0010      5858x
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
1 thread:      0.0422 s
all cores (4):  0.0128 s
speedup:        3.3x (varies by run/load)
```

The sweep writes into flat, preallocated output buffers and each worker
reuses one scratch workspace, so an energy point allocates nothing. The
two boundary columns also share a single forward elimination instead of
running two independent tridiagonal solves — bit-for-bit identical output
(asserted in `tridiag::tests::boundary_columns_match_two_general_solves_exactly`),
about 25% off the column work.

### Computing only what the caller needs

`GreenOutputs` controls whether a sweep evaluates the Green's-function
diagonal at all. The self-consistent loop never reads it — the charge
density needs only the boundary columns — so it asks for
`GreenOutputs::BoundaryColumns` and skips both the diagonal recursion and
its output array (4 MB per iteration at the default device size). Measured
single-threaded at N=561, 900 energy points:

```
LDOS + columns (Full):                 0.0413 s
columns only (self-consistent loop):   0.0304 s
saving:                                1.36x
```

Against the pre-optimization implementation (0.0471 s for the same sweep),
a self-consistent iteration's NEGF work is about 1.55x faster. Under
`Dense` the saving is larger, since the alternative is a full N×N inverse
rather than two right-hand sides.

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

A self-consistent bias sweep therefore nests one `rayon` parallel loop
inside another. That is fine: rayon's work-stealing handles it, and
forcing the inner energy loop to run serially when nested measured
*slower* (2.85 s vs 2.58-2.64 s for a 21-point self-consistent sweep on 4
cores), so the nesting is left alone.

The one thing that is **not** parallelizable is the self-consistent loop's
*iterations* — each iteration's charge depends on the previous iteration's
potential, a genuine sequential dependency. Parallelism instead comes from
within each iteration's NEGF sweep (above), which is where nearly all the
time goes.

Bias sweeps also report convergence per point: a bias point whose
self-consistent solve fails comes back with a `NaN` current and
`converged = false` (a `RuntimeWarning` in Python) instead of discarding
the whole sweep.

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
- Physics invariants are asserted rather than assumed: the contact
  self-energy is retarded inside the band and decaying outside it, the LDOS
  is positive everywhere (unit *and* default-scale integration tests), the
  transmission stays within `0 <= T <= 1`, contact broadening vanishes
  outside each contact's band, and skipping the Green's-function diagonal
  leaves the boundary columns unchanged.
- `crates/negforge-core/tests/self_consistent_realistic_device.rs` — an
  integration test at the model's default (non-toy) device scale,
  confirming the self-consistent loop converges and reproduces the
  expected ballistic-MOSFET trend (current increasing with gate bias),
  that the LDOS is positive there too, and that the NEGF and analytic
  ballistic currents agree in the on-state.
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
