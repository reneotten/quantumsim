# NEGForge — Physics and Numerical Methods Tutorial

This document is a tutorial walkthrough of both the *physics* and the
*numerical methods* behind NEGForge: a 1D ballistic-MOSFET electrostatics +
NEGF (Non-Equilibrium Green's Function) transport simulator. It is written
to stand on its own — you should not need the reference textbooks open to
follow it — but it points to where each idea is developed in full so you
can go deeper, and it points at the exact source file and function that
implements each piece, so you can move between "why" and "how" freely.

**How to read this document.** Parts I–V follow the model's own data flow,
each one bottom-to-top: the physics problem being solved, the equations
that solve it, the discretization that turns those equations into linear
algebra, and the algorithm that solves that linear algebra efficiently.
Part VI collects validity/limitations in one place. Part VII explains the
testing strategy. If you only want the destination, jump to Part VIII's
glossary and the code-to-section index below; if you want the journey,
read start to finish.

| Code | What it is | Covered in |
|---|---|---|
| `legacy_matlab/aufgabe1e.m`, `simulation.m` | Original electrostatics-only MATLAB scripts | Part I |
| `crates/negforge-core/src/tridiag.rs` | Thomas-algorithm tridiagonal solvers | Part I.4 |
| `crates/negforge-core/src/device.rs` | Electrostatics (`calc_potential`) + ballistic current (`calc_current`) | Parts I, II |
| `crates/negforge-core/src/negf.rs` | Tight-binding Hamiltonian, contact self-energies, retarded Green's function | Part III |
| `crates/negforge-core/src/charge.rs` | NEGF-derived electron density | Part IV |
| `crates/negforge-core/src/selfconsistent.rs` | Poisson↔NEGF fixed-point loop | Part V |

Primary references used throughout:

- S. Datta, *Quantum Transport: Atom to Transistor*, Cambridge University
  Press, 2005 — Chapter 2 ("A Simple Solver") for the electrostatic model;
  Chapters 8–10 for the NEGF transport formalism.
- S. Datta, *Electronic Transport in Mesoscopic Systems*, Cambridge
  University Press, 1995 — background on the Landauer/NEGF picture of
  transport.
- M. Lundstrom & J. Guo, *Nanoscale Transistors: Device Physics, Modeling
  and Simulation*, Springer, 2006 — short-channel electrostatics, the
  "natural length" theory, and the top-of-the-barrier ballistic MOSFET
  model.
- Y. Taur & T. H. Ning, *Fundamentals of Modern VLSI Devices*, Cambridge
  University Press — the scale-length / natural-length model for
  double-gate and thin-body MOSFETs.
- A. Svizhenko, M. P. Anantram, T. R. Govindan, B. Biegel, R. Venugopal,
  "Two-dimensional quantum mechanical modeling of nanotransistors,"
  *J. Appl. Phys.* **91**, 2343 (2002) — the recursive Green's function
  (RGF) algorithm used for the O(N) NEGF sweep.
- R. Lake, G. Klimeck, R. C. Bowen, D. Jovanovic, "Single and multiband
  modeling of quantum electron transport through layered semiconductor
  devices," *J. Appl. Phys.* **81**, 7845 (1997) — another standard RGF
  reference, and the contact self-energy / broadening-function formalism.

---

# Part I — Electrostatics: from 3-D Poisson's equation to a 1-D solve

## I.1 What problem is being solved?

The device is a gate-controlled transistor (thin-body / double-gate /
gate-all-around MOSFET) split into three regions along the transport
direction `x`:

```
 |<------ source ------>|<------ channel ------>|<------ drain ------>|
 x = 0                 x = d_S            x = d_S + d_ch             x = N·a
        (grounded, V=0)      (under the gate)          (biased at V_ds)
```

The gate sits over the channel only (self-aligned gate assumption). We want
the electrostatic potential energy `φ(x)` along the device, because it is
what opens or closes the channel to current, and — the point of this whole
codebase — it becomes the on-site potential energy in the NEGF Hamiltonian
(Part III), and is in turn corrected by the charge density NEGF computes
(Part V). This chapter covers the electrostatics alone, exactly as
`legacy_matlab/aufgabe1e.m` and `simulation.m` implement it (a **linear**,
1-D reduced Poisson solve, `rho = 0`, no NEGF yet) — the ingredient the
rest of the document builds on.

## I.2 From 3-D electrostatics to a 1-D ODE: the natural length λ

The exact problem is 2-D (or 3-D) Poisson's equation inside the
semiconductor body, with boundary conditions set by capacitive coupling to
the gate through the oxide:

$$\nabla^2 \phi(x,y) = -\frac{\rho(x,y)}{\varepsilon_0 \varepsilon_{si}}$$

with a Robin-type boundary condition at the gate/oxide/silicon interface
(continuity of displacement field across the oxide, no free charge in the
oxide):

$$\varepsilon_{ox}\frac{\phi_g - \phi(x, y_{surface})}{t_{ox}} = \varepsilon_{si}\left.\frac{\partial \phi}{\partial y}\right|_{surface}$$

Solving the full 2-D problem numerically at every bias point is expensive.
Instead, following Taur & Ning and Datta's *Quantum Transport* §2.4, we
assume the potential is *parabolic in the transverse direction* `y`
(reasonable for a thin body between one or two gates) and satisfies the
interface boundary condition above. Substituting that ansatz into Poisson's
equation and integrating out `y` collapses the 2-D problem into a single
1-D ODE for the **centerline potential** `φ(x)`:

$$\frac{d^2\phi}{dx^2} - \frac{\phi(x)}{\lambda^2} = -\frac{\rho(x)}{\varepsilon_0\varepsilon_{si}} - \frac{\phi_g(x) + \phi_{bi}(x)}{\lambda^2}$$

where

$$\lambda = \sqrt{\frac{\varepsilon_{si}}{\varepsilon_{ox}}\cdot\frac{t_{ch}\, t_{ox}}{g}}$$

is the **natural length** (a.k.a. scale length), `t_ch` the body/channel
thickness, `t_ox` the oxide thickness, and `g` a geometry factor (1 for a
single gate, 2 for a symmetric double gate, etc. — the code exposes this as
`geo`).

**Physical reading of the ODE.** Far from the source/drain junctions (many
`λ` away), the `d²φ/dx²` term is negligible and `φ → φ_g + φ_bi`: the gate
fully sets the potential, exactly as in a long-channel MOSFET. Near a
junction, the fixed potential imposed by the heavily doped source/drain
"leaks" into the channel over a decay length `λ`. If the channel length `L`
is only a few `λ`, the source and drain potentials overlap in the middle of
the channel and the gate partially loses control — this *is* the
microscopic origin of short-channel effects: threshold voltage roll-off and
drain-induced barrier lowering (DIBL). `λ` is therefore the single number
that tells you, for a given `t_ch`/`t_ox`/gate geometry, how short a
channel can be made before electrostatic control degrades — which is why
the source/drain extension regions are sized as multiples of `λ`
(`d_S = d_D = 5·λ` in `aufgabe1e.m`; `l_ds = floor(lambda)*15` in
`quantumsim.m`/`device.rs`) — far enough that the boundary is effectively
in the "gate has full control" regime.

**Validity.** This is a thin-body approximation: it assumes the transverse
potential profile is well-described by a single parabola, which holds when
the body is thin compared to the channel length, and degrades for bulk
(thick-body) MOSFETs where the transverse charge distribution isn't
captured by one decay length. It is also a *classical*, non-quantized
charge model in the transverse direction, in tension with the fully
quantum-mechanical longitudinal treatment NEGF gives the same device in
Part III — the overall model is a first-order approximation, not a fully
self-consistent multi-dimensional quantum simulation.

## I.3 Discretization

Discretize `x` on a uniform grid with spacing `a` and replace the second
derivative with the standard 3-point finite-difference stencil:

$$\left.\frac{d^2\phi}{dx^2}\right|_i \approx \frac{\phi_{i+1} - 2\phi_i + \phi_{i-1}}{a^2}$$

which turns the ODE into a linear system `A φ = b` for the vector of
grid-point potentials, where `A` is tridiagonal (`-2/a²` on the diagonal,
`1/a²` on the off-diagonals, from the Laplacian, with `-1/λ²` added to
every diagonal entry — `device::Device::laplacian` builds exactly this),
and

$$b_i = \frac{1}{\varepsilon_0\varepsilon_{si}}\rho_i - \frac{\phi_{g,i} + \phi_{bi,i}}{\lambda^2}$$

(`device::Device::calc_potential` assembles this right-hand side and calls
the tridiagonal solver).

**Boundary conditions.** The far ends of the source and drain extensions
get **Neumann (zero-field) boundaries**, `dφ/dx = 0` — physically, "the
potential has flattened out by the time you're this far from the channel,
so there's no field left." This is implemented with the classic
ghost-point trick: reflecting the mirror point `φ_{-1} = φ_1` across the
boundary turns the stencil at the edge from `(φ_2 - 2φ_1 + φ_0)` into
`(2φ_2 - 2φ_1)`, i.e. doubling the corner off-diagonal entry of `A`
(`sup[0] = sub[n-2] = 2/a²` in `Device::laplacian`, matching `ddx2(1,2)=2` /
`ddx2(N,N-1)=2` in `aufgabe1e.m`).

## I.4 Tutorial: the Thomas algorithm for tridiagonal systems

`A φ = b` is an `N×N` system, but `A` is tridiagonal (only 3 nonzero
diagonals). Solving it with general-purpose Gaussian elimination costs
`O(N^3)`; because a tridiagonal matrix only has `O(N)` nonzero entries to
begin with, a specialized elimination — the **Thomas algorithm** — solves
it in `O(N)`. This is what `tridiag::solve_real` (and its complex
counterpart `solve_complex`, reused by the NEGF module in Part III)
implements, and it's worth understanding once since it recurs throughout
the codebase.

Write the system as, for each row `i`:

$$\text{sub}_{i-1}\, x_{i-1} + \text{diag}_i\, x_i + \text{sup}_i\, x_{i+1} = \text{rhs}_i$$

**Forward elimination.** Starting from row 0, eliminate `x_{i-1}` from row
`i` using row `i-1`'s (already-reduced) equation. Tracking only the
modified coefficients (since the matrix stays tridiagonal, elimination
never creates new nonzero entries — the key property that makes this
`O(N)` instead of `O(N^3)`):

$$c'_0 = \frac{\text{sup}_0}{\text{diag}_0}, \qquad d'_0 = \frac{\text{rhs}_0}{\text{diag}_0}$$

$$c'_i = \frac{\text{sup}_i}{\text{diag}_i - \text{sub}_{i-1} c'_{i-1}}, \qquad d'_i = \frac{\text{rhs}_i - \text{sub}_{i-1} d'_{i-1}}{\text{diag}_i - \text{sub}_{i-1} c'_{i-1}}$$

After this forward pass, row `i` reads `x_i + c'_i x_{i+1} = d'_i` — each
row now involves only `x_i` and `x_{i+1}`.

**Back substitution.** Starting from the last row (which has no `x_{i+1}`
term, so `x_{N-1} = d'_{N-1}` directly), walk backward:

$$x_{N-1} = d'_{N-1}, \qquad x_i = d'_i - c'_i\, x_{i+1}$$

Both passes are a single `O(N)` loop, so the whole solve is `O(N)` — this
is what makes it practical to call `calc_potential` once per
self-consistent iteration (Part V) without the electrostatics solve itself
becoming the bottleneck. `solve_complex` is the identical algorithm with
complex diagonal entries and right-hand side, used in Part III for the
NEGF Green's-function columns.

**Validated by:** `tridiag::tests::thomas_matches_dense_gaussian_elimination`
cross-checks this against a dense (`O(N^3)`) Gaussian-elimination reference
on a 12-site random system.

## I.5 Code walkthrough: the original MATLAB scripts

### I.5.1 `aufgabe1e.m` — dense-matrix version

| Line(s) | What it does |
|---|---|
| 1–5 | Physical constants: elementary charge `e`, relative permittivities of the oxide (`eps_ox`, SiO₂ ≈ 3.9) and silicon (`eps_si` ≈ 11.2), and vacuum permittivity `eps_0`. |
| 7–8 | Bias point: `V_ds` = drain-source voltage, `V_g` = gate voltage. |
| 10–13 | Device geometry in nm: channel thickness `d_ch`, oxide thickness `d_ox`, grid spacing `a` (here 0.01 nm), and the natural length `λ` from §I.2. |
| 14–15 | Source and drain extension lengths are each `5λ` (§I.2). |
| 17–20 | Convert physical lengths to grid-point counts (`N_S`, `N_ch`, `N_D`) and total grid size `N`. |
| 22–24 | Bandgap `E_g` (1.12 eV, silicon), reference Fermi level `E_f = 0`, and `phig = V_g`. |
| 26–27 | `phi_g`: nonzero only over the channel indices — the gate only overlaps the channel. |
| 28–30 | `phi_bi`: the channel's built-in potential is mid-gap (`E_f + E_g/2`), the drain region is offset by `-V_ds`. |
| 31 | `rho = 0`: no free/dopant charge in this script — a fully linear solve. |
| 33–36 | Build the tridiagonal `-2 - 1/λ²` / `+1` matrix from §I.3, with the Neumann ghost-point correction in the two corners. |
| 38 | Assemble the right-hand side `b` from §I.3. |
| 40 | Solve `A φ = b` directly (`\`, dense LU). |
| 41–42 | Plot the resulting potential profile and print its maximum (the barrier height). |

### I.5.2 `simulation.m` — sparse-matrix version with doping

A cleaned-up, more general rewrite of the same physics:

- **Fixed grid size, derived spacing.** Fixes `N = 500` grid points and
  derives spacing `a = l_g / (N-1)` from the total device length
  `l_g = 2·l_ds + l_ch` — the more numerically convenient direction to work
  in, since it gives direct control over matrix size / runtime.
- **Sparse assembly.** The same tridiagonal Laplacian is built with
  `spdiags` instead of `diag` — asymptotically faster and lighter on
  memory for large `N`.
- **`geo`, the geometry factor** lets you model a double-gate or wrap-gate
  device by changing a single number — a wrap-gate (cylindrical,
  gate-all-around) geometry couples to the channel more strongly than a
  planar single gate, which shortens `λ` and improves short-channel
  control for the same `t_ch`/`t_ox`.
- **`N_dot`, dopant charge density** is a placeholder for fixed
  ionized-dopant charge on the Poisson right-hand side — `0` in the
  default parameters (an undoped body). This is the field the
  self-consistent loop's NEGF-computed mobile charge is added alongside
  (Part V).
- Region assignment mirrors `aufgabe1e.m`: source grounded/zero, channel
  carries `-V_g` and mid-gap `φ_bi`, drain carries `-V_ds`.

## I.6 Worked numeric example

Plugging in `aufgabe1e.m`'s defaults (`eps_si=11.2`, `eps_ox=3.9`,
`d_ch=20 nm`, `d_ox=1 nm`):

$$\lambda = \sqrt{\frac{11.2}{3.9}\times 20 \times 1}\ \text{nm} \approx 7.58\ \text{nm}$$

so the source/drain extensions `d_S = d_D = 5λ ≈ 37.9 nm`, and with the
0.01 nm grid spacing the script builds roughly 9,600 grid points.

Physically: a 20 nm-thick, single-gated silicon body with a 1 nm oxide has
quite poor electrostatic control (`λ` is large relative to typical modern
channel lengths of 10–20 nm) — this is exactly why real advanced nodes use
thin bodies (small `t_ch`) and/or multiple gates (larger `g`) to shrink `λ`
and keep short-channel effects under control: halving `d_ch` to 10 nm
shrinks `λ` to ≈ 5.36 nm (a ~30% reduction), and moving to a double gate
(`g=2`) shrinks it by a further factor of `√2`.

`simulation.m`'s defaults (`k_Si=11.68`, `k_ox=3.9`, `d_ch=10 nm`,
`d_ox=1 nm`, `geo=1`) give `λ ≈ 5.47 nm`, with `l_g = 30 nm` spread over
`N=500` points (`a ≈ 0.0601 nm`), placing the channel roughly between grid
indices 166 and 334. `device::DeviceParams::default()` (`d_ch=5 nm,
d_ox=5 nm, k_si=11.2, k_ox=3.9`) gives `λ = sqrt(11.2/3.9*5*5) ≈ 8.47 nm`.

**What the potential profile means.** With `V_g = 0` and `V_ds > 0` you
should see: a flat, low region in the source, a barrier peak under the
gate (height ≈ mid-gap `E_f + E_g/2`, attenuated somewhat near the
junctions by the `1/λ²` coupling to the drain), and a step down to
`-V_ds` in the drain.

- **Increase `V_g`** (more negative `phi_g`/`Psi_g` in the channel, the
  n-type convention used here) → the channel barrier drops → the
  transistor turns "on."
- **Increase `V_ds`** with the channel short relative to `λ` → the barrier
  peak itself drops (not just the drain-side potential) → this is DIBL,
  visible directly as a change in `max(phi_f)` while sweeping `V_ds` at
  fixed `V_g`.
- **Shrink `d_ch`/`l_ch`** relative to `λ` → the flat "gate is in control"
  plateau at the top of the barrier shrinks or disappears, replaced by a
  rounded peak — the discretized signature of losing electrostatic
  control.

## I.7 A caveat in the original two scripts (not in `quantumsim.m` or the Rust port)

In `aufgabe1e.m` and `simulation.m`, the finite-difference Laplacian is
assembled as `(-2 - 1/λ², +1, +1)` **without dividing by `a²`**
(`aufgabe1e.m` even has this spelled out and commented out:
`%ddx2 = ddx2./a^2;`). The consistent discretization of §I.3 requires the
off-diagonal `+1` entries (and the whole matrix) scaled by `1/a²` relative
to the `-1/λ²` term. As written, those two scripts implicitly assume
`a = 1` (nm) in the balance between the second-derivative term and the
`1/λ²` term, even though `a` is set much finer (`0.01 nm` /
`≈0.06 nm`) — worth fixing, or deliberately confirming via a
grid-refinement/convergence test, before relying on their `φ_f` for
quantitative work.

**This does not affect `quantumsim.m` or the Rust rewrite.**
`quantumsim.m`'s `get_laplacian()` divides by `self.a.^2` explicitly
(`self.L_sparse = spdiags(...)/self.a.^2`), and `device::Device::laplacian`
scales by `1/a²` throughout (`inv_a2 = 1.0/a.powi(2)`) with the `1/λ²`
term left unscaled — the consistent discretization — which is what the
rest of this document (and the Rust engine) is built on.

---

# Part II — Ballistic current: the Landauer formula

## II.1 Physics

Once the electrostatic barrier `φ_f(x)` is known, the simplest possible
transport model treats the channel as **ballistic**: an electron injected
from a reservoir either sails over the top of the barrier unimpeded, or is
reflected — no scattering, no tunneling. This is the "top-of-the-barrier"
model (Lundstrom & Guo), a special case of the general Landauer–Büttiker
formula

$$I = \frac{2e}{h}\int T(E)\,\big[f_S(E) - f_D(E)\big]\,dE$$

with the transmission `T(E)` collapsed to a step function: `T(E) = 1` for
`E` above the barrier top `psi_0 = max(φ_f)`, `T(E) = 0` below it. `f_S`,
`f_D` are Fermi-Dirac occupations of the source/drain reservoirs at their
own electrochemical potentials:

$$f_{S,D}(E) = \frac{1}{\exp\!\left(\dfrac{E - E_{fS,fD}}{k_B T}\right) + 1}$$

with `E_fS = e_fs` (a fixed parameter) and `E_fD = -V_ds + 0.05` (the drain
Fermi level tracks the applied bias, offset by a fixed reference used
throughout the original coursework code). The prefactor `2e/h` is the
single-mode Landauer conductance quantum — the `2` is spin degeneracy, `e`
and `h` the elementary charge and Planck constant.

**Validity.** Good only when the channel is short compared to the carrier
mean free path (no scattering) — real devices lose current to
phonon/impurity/surface-roughness scattering, so this overestimates
on-current. Excluding sub-barrier tunneling also underestimates
subthreshold current. Both are standard, deliberate simplifications for a
first compact model, not oversights.

## II.2 Numerical implementation

`device::Device::calc_current` (mirrors `calc_current()`) approximates the
integral by a **rectangle rule** over a fixed energy step `d_e`, from
`psi_0` up to an upper bound `e_max`:

$$I \approx \frac{2e}{h}\, d_e \sum_{k=0}^{n} \big[f_S(E_k) - f_D(E_k)\big], \qquad E_k = \psi_0 + k\cdot d_e$$

`e_max` is chosen so the Fermi functions are numerically negligible past
it, rather than integrating to infinity: solving
`f_S(e_max) ≈ epsilon` for a small tolerance `epsilon` gives

$$e_{max} = E_{fS} - k_B T \ln(\epsilon) / e$$

(`device::Device::new`'s `e_max = params.e_fs - K_B*T*epsilon.ln()/E`,
with `epsilon = 10^{-14}` by default — comfortably past the point where
`f_S` has decayed to numerical noise).

## II.3 Units, and a deliberately-inherited quirk

Energies in this codebase are in eV throughout (`device.rs`'s module
docs), so the sum's argument `E_k - E_{fS,fD}` is in eV; the exponent needs
to be dimensionless, hence the explicit `* E` (elementary charge, used here
as the eV→Joule conversion factor: 1 eV = `e` Joules) inside the Fermi
function's exponent, dividing by `K_B * T` (Joules). The final expression
carries a second `* E` (converting the summed `d_e` — in eV — to Joules,
so that `2e/h * (Joules)` comes out in Amps) and an additional `* 1e-3`
whose exact provenance is not derivable from unit analysis alone; it is
inherited byte-for-byte from `calc_current()`'s
`I=2*e/h*sum(...)*dE*e*1e-3` and is preserved unchanged here, consistent
with this rewrite's policy of faithfully porting the original numerics
except where a bug was specifically identified and fixed (see the
top-level README's "Deviations" list).

---

# Part III — NEGF: quantum transport via the retarded Green's function

Ballistic transport (Part II) gets the qualitative trend right but throws
away all quantum interference and resonance structure, and can't produce a
charge density to feed back into the Poisson solve. NEGF is the more
complete (still 1D, still effective-mass) treatment that both of those
things.

## III.1 From the Schrödinger equation to a tight-binding chain

Start from the effective-mass, 1-D time-independent Schrödinger equation
with the electrostatic potential energy `U(x) = -e\,\phi_f(x)` from Part I
as the potential term:

$$-\frac{\hbar^2}{2m^*}\frac{d^2\psi}{dx^2} + U(x)\,\psi = E\,\psi$$

Discretize on the same grid as the electrostatics (spacing `a`), replacing
the second derivative with the same 3-point stencil as §I.3. Collecting
terms gives a matrix eigenvalue-like equation `H\psi = E\psi` with a
**tight-binding chain** Hamiltonian:

$$H_{ii} = 2t + U(x_i), \qquad H_{i,i\pm1} = -t, \qquad t = \frac{\hbar^2}{2 m^* a^2}$$

(`Device::new` computes `t` into the `t_hop` field;
`negf::green_function_sweep`'s `bulk_diag = 2*t + psi_f` and
`off_diag = t` build exactly this — note the off-diagonal of the *matrix
being inverted*, `(E+iη)I - H`, is `-H_{i,i±1} = +t`, positive, which is
what the code stores).

**Validity: how fine does `a` need to be?** An infinite tight-binding chain
with this `H` has dispersion `E(k) = 2t(1-\cos(ka))` — bounded above by a
bandwidth of `4t`, unlike the true continuum's unbounded parabola
`E = ħ²k²/2m*`. The two agree only for small `k·a` (Taylor-expand
`\cos(ka) ≈ 1 - (ka)^2/2` to see `E(k) ≈ t(ka)^2 = ħ²k²/2m*`, recovering
the continuum dispersion). So `a` needs to be fine enough that the energy
range actually swept (`psi_0` to `e_max`, roughly) stays well below `4t` —
worth checking (e.g. halve `a` and confirm results are unchanged) before
trusting NEGF results at a new device scale.

## III.2 Open boundaries: contact self-energies

A real device isn't an isolated chain — the source and drain are
semi-infinite reservoirs. Modeling that directly (an infinite matrix) isn't
possible, but its *effect* on the finite simulated region can be captured
exactly by an energy-dependent correction to the boundary sites' on-site
energy, called a **self-energy** `Σ(E)`. This is the standard NEGF
open-boundary technique (Datta, *Quantum Transport*, §8.2–8.3).

For a semi-infinite 1-D tight-binding chain (same `t`, on-site energy
`2t + U_{contact}`) attached at a boundary site, the self-energy has a
closed form in terms of the chain's own dispersion. Solving
`E = 2t(1-\cos(ka)) + U_{contact}` for `k` gives

$$\cos(k a) = \frac{2t + U_{contact} - E}{2t}$$

taking the branch `k·a \in [0,\pi]` for energies inside the contact's
propagating band, `U_{contact} \le E \le U_{contact} + 4t`. The self-energy
of the semi-infinite lead, evaluated at the boundary site, is then

$$\Sigma(E) = -t\,e^{i k(E) a}$$

**Causality requirement.** A contact that only lets particles escape into
the lead (absorbing, not amplifying) must have `Im(Σ) ≤ 0` — equivalently,
the **broadening function**

$$\Gamma(E) = -2\,\mathrm{Im}\big[\Sigma(E)\big] \ge 0$$

must be non-negative. This is not a sign convention you get to pick freely:
it is required for the resulting Green's function to actually be the
*retarded* one (poles in the lower half of the complex-`E` plane), and for
physical quantities built from it (density of states, injected charge) to
come out non-negative. Concretely: for a propagating mode
(`0 < k(E)a < \pi`), `\mathrm{Im}[-t e^{ika}] = -t\sin(ka) \le 0` — good,
this checks out for the formula above.

**Implementation note.** `negf::green_function_sweep` computes its own
wavevector via
`ka_s = acos(-(2*t - E + psi_0s)/(2*t))`, which is algebraically
`\pi - k(E)a` from the dispersion above (the reflected branch), not
`k(E)a` directly. Composed with `exp(-i*ka_s)` — as the code does —
the `\pi` phase from the reflection and the branch flip cancel exactly:

$$t\, e^{-i\, ka_s} = t\, e^{-i(\pi - k(E)a)} = t\, e^{-i\pi} e^{i k(E) a} = -t\, e^{i k(E) a} = \Sigma(E)$$

so `sigma_l = t * (-(Complex64::i() * ka_s)).exp()` reproduces the exact
textbook self-energy, without needing `ka_s` itself to be in the
"canonical" branch. (An earlier version of this code, and the original
`quantumsim.m` it was ported from, instead used `exp(+i*ka_s)` — which for
this same reflected branch gives `\mathrm{Im}(\Sigma) = +t\sin(k(E)a) > 0`,
backwards from the causality requirement above. That produced a
local-density-of-states proxy that could go visibly negative — impossible
for a genuine density of states — confirmed at the broadening used inside
the self-consistent loop before this was corrected. See
`negf::tests::contact_self_energy_has_non_positive_imaginary_part_for_propagating_modes`
and `negf::tests::g_diag_is_non_negative_at_self_consistent_loop_broadening`
for the regression tests that now guard this.)

Only propagating-band self-energies are applied (`if e >= psi_0s`/`psi_0d`
in the code); below the contact's own band edge the boundary site keeps its
bare on-site energy plus only the artificial `iη` broadening (§III.4) — a
simplification already present in the original model, not something this
rewrite added or removed.

## III.3 The retarded Green's function, LDOS, and broadening

With the Hamiltonian and self-energies assembled, the central object is the
**retarded Green's function**

$$G^r(E) = \big[(E+i\eta)I - H - \Sigma_S(E) - \Sigma_D(E)\big]^{-1}$$

where `η` is a small positive broadening (§III.4) and `Σ_S`, `Σ_D` are
nonzero only at the source/drain boundary sites. Two derived quantities
matter for this codebase:

- **Local density of states**, `D(x,E) = -\mathrm{Im}[G^r_{xx}(E)]/(\pi)`
  (per unit length, hence the extra `/a` the code applies —
  `GreenFunctionResult::g_diag` stores `-\mathrm{Im}(G_{ii})/a`, i.e. the
  LDOS without the `1/\pi` normalization, matching the original code's
  convention for what to plot).
- **Broadening / injection**, `\Gamma_{S,D}(E) = -2\,\mathrm{Im}[\Sigma_{S,D}(E)]`,
  which for our contact self-energy (§III.2) works out to
  `\Gamma_{S,D}(E) = 2t\sin(k_{S,D}(E)\,a)` for propagating contact modes —
  this is what drives the charge-density calculation in Part IV.

## III.4 The artificial broadening `η`

`η` in `(E+iη)I - H - \Sigma_S - \Sigma_D` is a mathematical device, not a
physical scattering rate: it nudges the poles of `G^r` off the real axis so
the matrix stays invertible, in the same spirit as the `+i0^+`
prescription in the Feynman propagator. `negf::DEFAULT_ETA = 1e-8` matches
the original `calc_green()`'s hardcoded `eta = 1i*1e-8`, appropriate for a
one-off local-density-of-states plot at a fixed potential, where you *want*
to see sharp, well-resolved resonances.

Inside the self-consistent loop (Part V) that small a value is a problem: a
narrow resonance whose energy happens to land within `η` of one of the
discrete energy-grid points produces a `|G|^2` spike orders of magnitude
larger than its neighbors, and feeding that grid-alignment-dependent spike
back into the electrostatic solve makes the fixed-point iteration diverge
or oscillate. `SelfConsistentOptions::eta` defaults instead to `0.08` eV —
several times the energy-grid spacing `d_e` — so each resonance is smeared
across multiple grid points and the charge estimate varies smoothly
between iterations. This need is independent of the self-energy sign
question in §III.2 — re-verified after that fix by sweeping `eta` down to
`0.005`/`0.01` on the default device, which still fails to converge within
50 iterations, confirming the requirement is genuinely about resolving
resonances, not about compensating for a sign error.

## III.5 Tutorial: the recursive Green's function (RGF) algorithm

Directly forming and inverting the `N×N` matrix `(E+iη)I - H - \Sigma_S -
\Sigma_D` (dense Gaussian elimination, as the original `inv()` MATLAB call
did) costs `O(N^3)` *per energy point*, which is the dominant cost of a
bias sweep or a self-consistent loop that calls this once per iteration.
But only two things are actually needed from the full inverse: its
diagonal (`g_diag`, the LDOS) and its first/last columns (the source/drain
-injected quantities `g_col_source`/`g_col_drain` that Part IV needs). Both
can be obtained in `O(N)` by exploiting the matrix's tridiagonal structure
— the standard **recursive Green's function** (RGF) technique used
throughout the mesoscopic-transport literature (Svizhenko et al. 2002;
Lake et al. 1997).

**The boundary columns** (`negf::green_function_sweep`'s `col_source`,
`col_drain`) are the easy half: the `i`-th column of `M^{-1}` is, by
definition, the solution `x` of `M x = e_i` for the `i`-th standard basis
vector. Since `M` is tridiagonal, this is exactly `tridiag::solve_complex`
from §I.4 — one `O(N)` Thomas-algorithm solve per column, no separate
derivation needed.

**The full diagonal** (`negf::full_diagonal`) needs a genuinely different
technique, since there's no single right-hand side that extracts "the
diagonal" from a linear solve. The implementation is two sweeps:

*Forward sweep* — build a **left-connected** Green's function `g_left[i]`,
defined as the (single-site) Green's function of the truncated chain
consisting of sites `{0,...,i}` only (i.e. as if site `i+1` didn't exist).
This is exactly the same recursion as the Thomas algorithm's forward
elimination (§I.4) applied to this matrix, viewed as a continued fraction —
each new site's effective on-site energy is renormalized by its coupling
to the (already-resolved) chain to its left:

$$g_{\text{left}}[0] = \frac{1}{\text{diag}_0}, \qquad
g_{\text{left}}[i] = \frac{1}{\text{diag}_i - t_{i-1}^2\, g_{\text{left}}[i-1]}$$

*Backward sweep* — combine `g_left` with a pass from the far end back to
site 0, producing the **full** diagonal (accounting for coupling in both
directions):

$$\text{full}[N-1] = g_{\text{left}}[N-1], \qquad
\text{full}[i] = g_{\text{left}}[i] + g_{\text{left}}[i]^2\, t_i^2\, \text{full}[i+1]$$

The seed is exact for the same reason `g_left[N-1]` is exact for the last
site alone: there's nothing to the right of site `N-1`, so "left-connected"
and "fully connected" coincide there. Each subsequent step folds in one
more site's worth of coupling to the right. Both passes are `O(N)`
(constant work per site), so the total remains `O(N)` for the whole
diagonal.

**Validated by:** `negf::tests::recursive_diagonal_matches_dense_inverse`
and `negf::tests::boundary_columns_match_dense_inverse` cross-check both
pieces above against a dense (`O(N^3)`) Gauss-Jordan matrix inversion on a
9-site random complex tridiagonal system, to within `1e-8` relative
tolerance — the appropriate way to validate a recursive numerical identity
like this one, and the reason this document states the recursion rather
than re-deriving the (slightly intricate) exact equivalence between the
above and a direct Cramer's-rule expansion from scratch.

## III.6 Code walkthrough: `negf::green_function_sweep`

For each energy `E` in the caller-supplied grid:

1. Compute the contact wavevectors `ka_s`, `ka_d` (§III.2) — complex in
   general, real only inside each contact's propagating band.
2. Build the bulk diagonal `(E+iη) - (2t + \psi_f(x_i))` for every site.
3. If `E` is above each contact's own band edge, overwrite that boundary
   site's diagonal entry with the self-energy-corrected version (§III.2).
4. Get the full diagonal via the RGF sweep (§III.5) → `g_diag`.
5. Solve for the source/drain-injected columns via two tridiagonal solves
   (§III.5) → `g_col_source`, `g_col_drain` (stored as `|G|^2`, the squared
   magnitudes Part IV's charge-injection formula needs).

The energy grid itself is chosen by the caller — the original hardcoded
`min(Psi_f) : dE : 0.7*E_max` (`negf_energy_grid` in `selfconsistent.rs`
reproduces this for the self-consistent loop; ad-hoc callers, e.g. for an
LDOS plot, can choose their own).

---

# Part IV — From Green's functions to charge density

## IV.1 Physics

The **electron density** at position `x` is built from the NEGF
correlation function `G^<(x,x,E)`, related to the retarded Green's function
and the contacts' occupation via the standard NEGF result

$$G^<(E) = G^r(E)\,\big[\Sigma_S^{in}(E) + \Sigma_D^{in}(E)\big]\,G^a(E), \qquad \Sigma_{S,D}^{in}(E) = \Gamma_{S,D}(E)\, f_{S,D}(E)$$

("in-scattering" from each contact, weighted by that contact's occupation
`f_{S,D}`, §II.1). For our boundary-only self-energies this reduces to
`G^<_{xx}(E) = \Gamma_S(E) f_S(E) |G^r_{x,1}(E)|^2 + \Gamma_D(E) f_D(E)
|G^r_{x,N}(E)|^2` — exactly the two `|G|^2` boundary columns Part III
computed — and the electron density is

$$n(x) = \int \frac{dE}{2\pi}\, G^<_{xx}(E) = \int \frac{dE}{2\pi}\Big[\Gamma_S(E) f_S(E)\, |G^r_{x,1}(E)|^2 + \Gamma_D(E) f_D(E)\, |G^r_{x,N}(E)|^2\Big]$$

**Why this is non-negative by construction.** Every factor —
`Γ_{S,D} ≥ 0` (§III.2's causality requirement), `f_{S,D} \in (0,1)`
(Fermi-Dirac), `|G|^2 \ge 0` — is non-negative, so `n(x) \ge 0` always
holds, *regardless* of any Green's-function sign convention question. This
is the one positivity guarantee the self-consistent loop (Part V) actually
leans on.

## IV.2 Code walkthrough: `charge::electron_density`

`charge::electron_density` implements the integral above as a rectangle
-rule sum over the same energy grid Part III swept, with two details worth
noting:

- `gamma_s = t_hop * sin(k)` is *half* of `Γ_S = 2t\sin(k)` from §III.3;
  combined with the function's `1/pi` prefactor (rather than the `1/2π`
  the formula above has), the two halvings cancel exactly:
  `t\sin(k)/\pi = (2t\sin(k))/(2\pi)`, reproducing the textbook formula's
  magnitude exactly, just factored differently.
- There is no explicit spin-degeneracy factor of two here, unlike
  `calc_current`'s explicit `2e/h` (§II.1). Whether that's correct depends
  on whether `f_S`/`f_D`/the density of states already implicitly assume a
  spin-summed convention — carried over unchanged from the original
  `calc_n()`, which (before this rewrite closed the feedback loop, Part V)
  was never checked against any reference calculation.

Two bug fixes relative to the original `calc_n()`, both necessary for the
self-consistent loop (Part V) to respond to the actual device physics
rather than returning an (effectively) constant:

1. **Per-energy Fermi weight.** The original evaluated `f(E_{fs})`/`f(E_{fd})`
   *once*, outside the energy sum, collapsing the density's entire energy
   dependence to a single number. Fixed to use the energy-dependent
   `f_S(E)`/`f_D(E)` inside the sum, matching `calc_current` (§II.2).
2. **Per-contact band-edge mask.** The original used one mask
   (`E > \psi_f(1)`, the *source* band edge) for both the source and drain
   contact terms. Fixed so each contact's term is gated by its own band
   edge — consistent with how `green_function_sweep` already conditions
   each contact's self-energy on its own band edge (§III.2).

---

# Part V — Closing the loop: self-consistent Poisson↔NEGF iteration

## V.1 Physics: why self-consistency matters

Parts I–IV, run once in sequence, are still not a complete device
simulation: `calc_potential` solved electrostatics assuming `\rho = 0`
(no mobile charge), and separately, `electron_density` computed what the
charge *would* be for that potential — but the two were never fed back
into each other. Physically, mobile charge *screens* the electrostatic
potential (that's what the `\rho` term in Poisson's equation is for), and
the potential in turn determines where charge accumulates — a genuine
feedback loop. `selfconsistent::solve_self_consistent` is the module that
closes it (no equivalent exists in the original MATLAB code):

1. Solve electrostatics (`calc_potential`) for the current `\rho`.
2. Run the NEGF sweep (Part III) at the resulting potential.
3. Compute the electron density (Part IV) from that sweep.
4. Convert it to a charge density and mix it into `\rho` (§V.2).
5. Repeat from step 1 until the potential profile stops changing (or a
   caller-supplied iteration budget runs out).

## V.2 Numerical method: Picard iteration with linear mixing

This is a **fixed-point iteration** (also called Picard iteration): define
a map `F: \rho \mapsto \rho'` (steps 1–4 above), and look for `\rho^*` with
`F(\rho^*) = \rho^*`. Applying `F` repeatedly (`\rho_{n+1} = F(\rho_n)`)
converges to a fixed point only if `F` is a contraction near it; blindly
substituting the full NEGF-computed density at every step
(`\rho_{n+1} = F(\rho_n)` exactly) tends to overshoot and oscillate or
diverge for this kind of strongly-coupled electrostatic/charge problem — a
well-known issue with naive Poisson–Schrödinger/NEGF self-consistency, not
specific to this codebase.

The standard fix, used here, is **linear (Picard) mixing**: instead of
jumping all the way to the newly computed target, take a damped step
toward it,

$$\rho_{n+1} = \rho_n + \alpha\,(\rho_{n+1}^{\text{target}} - \rho_n), \qquad \alpha = \texttt{mixing} \in (0, 1]$$

(`SelfConsistentOptions::mixing`, default `0.3`). Smaller `α` is more
stable but converges more slowly (more iterations to reach a given
tolerance); `α = 1` recovers plain, undamped fixed-point iteration.
Convergence is declared when the largest change in `\psi_f` between
iterations drops below `tolerance` (default `1e-6` eV); if
`max_iterations` (default 50) is exhausted first, `NotConverged` is
returned — reporting failure explicitly rather than silently returning a
garbage, unconverged potential.

**`mixing` and `eta` are numerical knobs, not physical parameters**: they
govern *whether and how fast* the iteration converges, not what it
converges to. A `NotConverged` result isn't evidence the underlying device
physics is invalid — it may just need more iterations, a smaller `mixing`,
or a larger `eta` (§III.4). Conversely, convergence is necessary but not
sufficient for physical correctness: a converged fixed point still
inherits every approximation from Parts I–IV (Part VI collects them).

## V.3 Charge-density units

`calc_potential`'s right-hand side (§I.3) divides `\rho + N_{dot}` by
`\varepsilon_0 \varepsilon_{si}` (`EPS_0` in SI units, F/m). For that
division to land in the same eV/nm² ballpark as the other right-hand-side
term (`(\psi_g + \psi_{bi})/\lambda^2`), `\rho` must be a genuine volume
charge density in C/m³ — the original code never exercised this (`\rho`
was always zero), so it was never validated. The self-consistent loop
converts Part IV's electron density accordingly:

1. `charge::electron_density` returns a **linear** density in nm⁻¹ (the
   model's native length unit — the original multiplied by `1e9` at this
   point to plot on a real-world axis, but feeding that ~10⁹×-too-large
   value straight into `\rho`, which shares scale with `\psi_g/\lambda^2`
   etc. still in nm, overwhelms the electrostatic solve and makes the
   iteration diverge by many orders of magnitude — this is why the
   conversion below is done explicitly, in the caller, rather than baked
   into `electron_density` the way the original baked in its `1e9`).
2. Convert nm⁻¹ → m⁻¹ (`* 1e9`) and treat the 1-D chain as having an
   implicit unit (1 m²) cross-section, so the linear density doubles as a
   volume density.
3. Multiply by `-e` (electrons are negatively charged):
   `\rho(x) = -e \cdot n_{electron}(x) \cdot 10^9` (`target_rho` in
   `solve_self_consistent`).

`N_dot` (fixed dopant charge) is left as-is (default zero, undoped body);
if used with a nonzero value it should be supplied in the same C/m³ units.

## V.4 Code walkthrough: `selfconsistent::solve_self_consistent`

```text
calc_potential()                          // rho = 0 initially
loop (up to max_iterations):
    energies      = negf_energy_grid(...)  // min(psi_f) .. 0.7*e_max, step d_e
    green         = green_function_sweep(energies, eta)   // Part III
    n_electron    = electron_density(green, ...)           // Part IV
    rho          += mixing * (target_rho(n_electron) - rho)   // §V.2, §V.3
    psi_before    = psi_f.clone()
    calc_potential()                       // re-solve with updated rho
    residual      = max(|psi_f - psi_before|)
    if residual < tolerance: return Ok(...)
return Err(NotConverged)
```

---

# Part VI — Validity and known limitations, all in one place

Each item below is documented in more detail (with the reasoning, and
where relevant, a regression test) in the corresponding source module's
doc comments; this is a consolidated summary.

- **Natural-length electrostatics** (Part I.2): thin-body approximation,
  not valid for bulk/thick-body devices; a classical (non-quantized)
  transverse charge model in tension with the quantum-mechanical
  longitudinal NEGF treatment.
- **Ballistic Landauer current** (Part II.1): `T=1` above the barrier,
  `T=0` below — no scattering, no sub-barrier tunneling. Good for channels
  short compared to the mean free path; overestimates on-current and
  underestimates subthreshold current otherwise.
- **Effective-mass, single-valley, parabolic band** (`m_eff`, Parts II–III):
  stands in for silicon's multiple equivalent valleys and non-parabolic
  dispersion away from the band edge. Treat absolute currents as
  illustrative trends, not device-accurate predictions. Spin degeneracy
  *is* included in `calc_current` via the explicit `2e/h`.
- **Tight-binding discretization range of validity** (Part III.1): valid
  only while the swept energy range stays well below the chain's `4t`
  bandwidth; check via grid refinement at new device scales.
- **Contact self-energy sign** (Part III.2): was an inherited bug (wrong
  causality sign, `Im(Σ) > 0` for propagating modes), now fixed to the
  standard `Σ = -t\,e^{ik(E)a}` (implemented as `t\,e^{-i\,ka_s}` for the
  branch this code computes). Verified by two regression tests; the
  self-consistent loop's convergence behavior and default `eta`/`mixing`
  were re-checked and did not need retuning after the fix.
- **No explicit spin-degeneracy factor** in the self-consistent charge
  density (Part IV.2), unlike `calc_current`'s explicit `2e/h`; carried
  over unchanged from the original `calc_n`, which was never checked
  against a reference before this rewrite closed the feedback loop.

None of the above are believed to affect the *qualitative* trends the test
suite checks (current increasing with gate/drain bias, self-consistent
convergence to a stable, non-negative charge density) — but they mean
absolute numbers from this model should be read as illustrating device
physics concepts, not as quantitatively validated predictions for a real
device.

---

# Part VII — Testing strategy: numerics vs. physics

The test suite (`cargo test --workspace`) validates two genuinely different
things, and it's worth knowing which is which when reading or extending it:

- **Numerical correctness** — "does the fast algorithm compute the same
  thing as a slow, obviously-correct reference?" `tridiag`'s Thomas
  algorithm is checked against dense Gaussian elimination
  (§I.4); `negf`'s recursive Green's function sweep and boundary-column
  solves are checked against a dense Gauss-Jordan matrix inversion
  (§III.5). These tests would catch an algebra mistake in the `O(N)`
  algorithms even if the underlying physics model were entirely different.
- **Physical correctness** — "does the model behave the way the physics
  says it should?" Current increasing with gate bias and with drain bias
  (§II, `device::tests`), the self-consistent loop converging to a stable,
  non-negative charge density (§V, `selfconsistent::tests`), and — since
  the sign fix in Part III.2 — the local density of states staying
  non-negative (`negf::tests::g_diag_is_non_negative_at_self_consistent_loop_broadening`)
  and the contact self-energy carrying the causally-required sign
  (`negf::tests::contact_self_energy_has_non_positive_imaginary_part_for_propagating_modes`).
  These tests wouldn't catch a subtle O(N) algorithm bug (a wrong-but
  -internally-consistent fast algorithm could still pass them), which is
  why both kinds of test exist side by side.

`crates/negforge-core/tests/self_consistent_realistic_device.rs` runs the
full pipeline (Parts I–V) at the model's actual default device scale rather
than a toy size, specifically to catch issues that only appear once the
NEGF energy grid and tight-binding bandwidth are at realistic proportions.
`notebooks/negforge_demo.ipynb` is executed end-to-end
(`jupyter nbconvert --execute`) before committing to confirm the full
Python/Jupyter frontend path works; its outputs are cleared before
committing since they go stale the moment the engine changes.

Run everything with `cargo test --workspace`.

---

# Part VIII — Glossary

| Symbol / term | Meaning | Units in this codebase |
|---|---|---|
| `φ`, `Psi_f` | Self-consistent electrostatic potential *energy* (not volts) | eV |
| `λ`, `lambda` | Natural (screening) length | nm |
| `a` | Grid spacing | nm |
| `t`, `t_hop` | Tight-binding hopping parameter, `ħ²/(2m*a²)` | eV |
| `η`, `eta` | Artificial imaginary broadening, `(E+iη)I - H` | eV |
| `Σ_S`, `Σ_D` | Contact retarded self-energies | eV |
| `Γ_S`, `Γ_D` | Contact broadening functions, `-2 Im(Σ)` | eV |
| `G^r` | Retarded Green's function | eV⁻¹ (energy-domain) |
| `f_S`, `f_D` | Source/drain Fermi-Dirac occupation | dimensionless |
| `k_B`, `h`, `ħ`, `e`, `m_e` | Physical constants (`constants.rs`) | SI |
| `ρ`, `rho` | Charge density fed into the Poisson solve | C/m³ |
| `n(x)` | Electron number density from NEGF | nm⁻¹ (`electron_density`'s return value) |
| `N_dot` | Fixed (dopant) charge term | same units as `rho` |
| `mixing`, `α` | Linear/Picard mixing factor for the self-consistent loop | dimensionless, `(0,1]` |

Lengths throughout are in nm, energies and potentials in eV, temperature in
Kelvin — matching the original MATLAB code's convention end to end.
