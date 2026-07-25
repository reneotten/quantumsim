# NEGForge — Physics and Implementation Guide

This document explains the physics behind NEGForge's electrostatics solver
(`aufgabe1e.m`, `simulation.m`) and how the MATLAB code implements it. It is
written to stand on its own — you should not need to have the reference
textbooks open to follow it — but it points to the books where each idea is
developed in full so you can go deeper.

Primary references used throughout:

- S. Datta, *Quantum Transport: Atom to Transistor*, Cambridge University
  Press, 2005 — Chapter 2 ("A Simple Solver") for the electrostatic model
  used here; Chapters 8–10 for the NEGF transport formalism this
  electrostatics feeds into.
- M. Lundstrom & J. Guo, *Nanoscale Transistors: Device Physics, Modeling
  and Simulation*, Springer, 2006 — short-channel electrostatics and the
  "natural length" theory.
- Y. Taur & T. H. Ning, *Fundamentals of Modern VLSI Devices*, Cambridge
  University Press — derivation of the scale length / natural length model
  for double-gate and thin-body MOSFETs.
- S. Datta, *Electronic Transport in Mesoscopic Systems*, Cambridge
  University Press, 1995 — background on the Landauer/NEGF picture of
  transport referenced in the "next steps" section below.

## 1. What problem is being solved?

Both scripts model the electrostatic potential along the channel of a
nanoscale, gate-controlled transistor (a thin-body / double-gate / gate-all
-around MOSFET). The device is split into three regions along the transport
direction `x`:

```
 |<------ source ------>|<------ channel ------>|<------ drain ------>|
 x = 0                 x = d_S            x = d_S + d_ch             x = N·a
        (grounded, V=0)      (under the gate)          (biased at V_ds)
```

The gate sits over the channel only (self-aligned gate assumption). What we
want is the potential `φ(x)` along the device, because that potential is
what ultimately opens or closes the channel to current — and, in the next
stage of a full simulation, becomes the on-site potential energy in a
Non-Equilibrium Green's Function (NEGF) Hamiltonian (hence the name
NEGForge — building the electrostatic backbone that NEGF transport
calculations are "forged" from).

This is a **linear, 1-D reduced Poisson solve**. It does not yet include
carrier charge feedback (self-consistency) or actual quantum transport —
both scripts solve a single linear system for `φ`. That is the correct
first stage of the pipeline described in Section 5.

## 2. From 3-D electrostatics to a 1-D ODE: the natural length λ

The exact problem is 2-D (or 3-D) Poisson's equation inside the
semiconductor body, with boundary conditions set by capacitive coupling to
the gate through the oxide:

$$\nabla^2 \phi(x,y) = -\frac{\rho(x,y)}{\varepsilon_0 \varepsilon_{si}}$$

with a Robin-type boundary condition at the gate/oxide/silicon interface
(continuity of displacement field across the oxide, no free charge in the
oxide):

$$\varepsilon_{ox}\frac{\phi_g - \phi(x, y_{surface})}{t_{ox}} = \varepsilon_{si}\left.\frac{\partial \phi}{\partial y}\right|_{surface}$$

Solving the full 2-D problem numerically at every bias point is expensive.
Instead, following Taur & Ning and the treatment in Datta's *Quantum
Transport* §2.4, we assume the potential is *parabolic in the transverse
direction* `y` (reasonable for a thin body between two gates) and satisfies
the interface boundary condition above. Substituting that ansatz back into
Poisson's equation and integrating out `y` collapses the 2-D problem into a
single 1-D ODE for the **centerline potential** `φ(x)`:

$$\frac{d^2\phi}{dx^2} - \frac{\phi(x)}{\lambda^2} = -\frac{\rho(x)}{\varepsilon_0\varepsilon_{si}} - \frac{\phi_g(x) + \phi_{bi}(x)}{\lambda^2}$$

where

$$\lambda = \sqrt{\frac{\varepsilon_{si}}{\varepsilon_{ox}}\cdot\frac{t_{ch}\, t_{ox}}{g}}$$

is the **natural length** (a.k.a. scale length), `t_ch` the body/channel
thickness, `t_ox` the oxide thickness, and `g` a geometry factor (1 for a
single gate, 2 for a symmetric double gate, etc. — `simulation.m` exposes
this as `geo`).

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
channel can be made before electrostatic control degrades. This is why both
scripts size the source/drain extension regions as multiples of `λ`
(`d_S = d_D = 5·λ` in `aufgabe1e.m`) — far enough that the boundary is
effectively in the "gate has full control" regime.

## 3. Discretization

Both scripts discretize `x` on a uniform grid with spacing `a` and replace
the second derivative with the standard 3-point finite-difference stencil:

$$\left.\frac{d^2\phi}{dx^2}\right|_i \approx \frac{\phi_{i+1} - 2\phi_i + \phi_{i-1}}{a^2}$$

which turns the ODE into a linear system `A φ = b` for the vector of
grid-point potentials, where `A` is tridiagonal (`-2` on the diagonal, `1`
on the off-diagonals, from the Laplacian) with `-1/λ²` added to every
diagonal entry, and

$$b_i = \frac{e}{\varepsilon_0\varepsilon_{si}}\rho_i - \frac{\phi_{g,i} + \phi_{bi,i}}{\lambda^2}$$

**Boundary conditions.** The far ends of the source and drain extensions
are given **Neumann (zero-field) boundaries**, `dφ/dx = 0` — physically,
"the potential has flattened out by the time you're this far from the
channel, so there's no field left." This is implemented with the classic
ghost-point trick: reflecting the mirror point `φ_{-1} = φ_1` across the
boundary turns the stencil at the edge from `(φ_2 - 2φ_1 + φ_0)` into
`(2φ_2 - 2φ_1)`, which is exactly the `2` placed in the corner of the
matrix (`ddx2(1,2)=2` / `ddx2(N,N-1)=2` in `aufgabe1e.m`, and the doubled
`super(2)`/`sub(end-1)` entries in `simulation.m`'s sparse construction).

## 4. Code walkthrough

### 4.1 `aufgabe1e.m` — dense-matrix version

| Line(s) | What it does |
|---|---|
| 1–5 | Physical constants: elementary charge `e`, relative permittivities of the oxide (`eps_ox`, SiO₂ ≈ 3.9) and silicon (`eps_si` ≈ 11.2), and vacuum permittivity `eps_0`. |
| 7–8 | Bias point: `V_ds` = drain-source voltage, `V_g` = gate voltage. |
| 10–13 | Device geometry in nm: channel thickness `d_ch`, oxide thickness `d_ox`, grid spacing `a` (here 0.01 nm — a very fine grid), and the natural length `λ` from Section 2. |
| 14–15 | Source and drain extension lengths are each set to `5λ` — long enough to reach the "gate has full control" asymptote described above, so the Neumann boundary at the far end is physically reasonable. |
| 17–20 | Convert physical lengths to grid-point counts (`N_S`, `N_ch`, `N_D`) and total grid size `N`. |
| 22–24 | Bandgap `E_g` (1.12 eV, silicon), reference Fermi level `E_f = 0`, and `phig = V_g` (gate workfunction/potential term). |
| 26–27 | `phi_g`: nonzero only over the channel indices — the gate only overlaps the channel. |
| 28–30 | `phi_bi`: the channel's built-in potential is mid-gap (`E_f + E_g/2`, i.e. intrinsic-like reference), while the drain region is offset by `-V_ds` (drain held `V_ds` below the grounded source). |
| 31 | `rho = 0`: no free/dopant charge in this script — a fully linear solve. |
| 33–36 | Build the tridiagonal `-2 - 1/λ²` / `+1` matrix from Section 3, with the Neumann ghost-point correction in the two corners. |
| 38 | Assemble the right-hand side `b` from Section 3. |
| 40 | Solve `A φ = b` directly (`\`, dense LU). |
| 41–42 | Plot the resulting potential profile and print its maximum (the barrier height). |

### 4.2 `simulation.m` — sparse-matrix version with doping

This is a cleaned-up, more general rewrite of the same physics:

- **Fixed grid size, derived spacing.** Instead of fixing `a` and deriving
  `N`, this version fixes `N = 500` grid points and derives the spacing
  `a = l_g / (N-1)` from the total device length `l_g = 2·l_ds + l_ch`
  (lines 36–39). This is the more numerically convenient direction to work
  in, since it gives you direct control over matrix size / runtime.
- **Sparse assembly (lines 22–33).** The same tridiagonal Laplacian is
  built with `spdiags` instead of `diag`, which is asymptotically much
  faster and lighter on memory for large `N` — the `tic`/`toc` calls exist
  specifically to demonstrate this (see line 33 vs. the sparse solve at
  line 65, "much quicker!").
- **`geo`, the geometry factor** (line 14, used in `λ`): lets you model a
  double-gate or wrap-gate device by changing a single number instead of
  rederiving the electrostatics — a wrap-gate (cylindrical, gate-all
  -around) geometry couples to the channel more strongly than a planar
  single gate, which shortens `λ` and improves short-channel control for
  the same `t_ch`/`t_ox`.
- **`N_dot`, dopant charge density** (line 17, used at line 65 as
  `(rho + N_dot)`): a placeholder for adding fixed ionized-dopant charge to
  the right-hand side of Poisson's equation — the natural next step toward
  a fully self-consistent charge model (see Section 5). It is `0` in the
  current default parameters, i.e. an undoped body.
- Region assignment (lines 44–57) mirrors `aufgabe1e.m`: source is
  grounded/zero, channel carries `-V_g` and mid-gap `φ_bi`, drain carries
  `-V_ds`.

## 5. Worked example: reading physical meaning off the numbers

Plugging in `aufgabe1e.m`'s defaults (`eps_si=11.2`, `eps_ox=3.9`,
`d_ch=20 nm`, `d_ox=1 nm`):

$$\lambda = \sqrt{\frac{11.2}{3.9}\times 20 \times 1}\ \text{nm} \approx 7.58\ \text{nm}$$

so the source/drain extensions `d_S = d_D = 5λ ≈ 37.9 nm`, and with the
0.01 nm grid spacing the script builds roughly 9,600 grid points
(`N_S ≈ N_D ≈ 3789`, `N_ch = 2000`).

Physically: a 20 nm-thick, single-gated silicon body with a 1 nm oxide has
quite poor electrostatic control (`λ` is large relative to typical modern
channel lengths of 10–20 nm) — this is exactly why real advanced nodes use
thin bodies (small `t_ch`) and/or multiple gates (larger `g` in the
denominator of `λ`) to shrink `λ` and keep short-channel effects under
control. You can see this directly in the code: halving `d_ch` to 10 nm
shrinks `λ` to ≈ 5.36 nm (a ~30% reduction), and moving to a double gate
(`g=2` in the `simulation.m` formula) shrinks it by a further factor of
`√2`.

`simulation.m`'s defaults (`k_Si=11.68`, `k_ox=3.9`, `d_ch=10 nm`,
`d_ox=1 nm`, `geo=1`) give `λ ≈ 5.47 nm`, with `l_g = 2·10 + 10 = 30 nm`
spread over `N=500` points (`a ≈ 0.0601 nm`), placing the channel roughly
between grid indices 166 and 334.

**What the output plot means.** `plot(phi_f)` (or `plot((0:N-1).*a, Psi_f)`
in `simulation.m`) shows the potential energy barrier along the channel.
With `V_g = 0` and `V_ds > 0` you should see: a flat, low region in the
source, a barrier peak under the gate (height ≈ mid-gap `E_f + E_g/2`,
attenuated somewhat near the junctions by the `1/λ²` coupling to the
drain), and a step down to `-V_ds` in the drain. Try it yourself:

- **Increase `V_g`** (more negative `phi_g`/`Psi_g` in the channel for an
  n-type device convention, as encoded here) → the channel barrier drops →
  models turning the transistor "on."
- **Increase `V_ds`** with the channel short relative to `λ` → the barrier
  peak itself drops (not just the drain-side potential) → this is DIBL,
  visible directly as a change in `max(phi_f)` as you sweep `V_ds` at fixed
  `V_g`.
- **Shrink `d_ch`/`l_ch`** relative to `λ` → the flat "gate is in control"
  plateau at the top of the barrier shrinks or disappears entirely,
  replaced by a rounded peak — the discretized signature of losing
  electrostatic control.

## 6. Implementation caveat worth knowing before extending this code

In both scripts, the finite-difference Laplacian is assembled as
`(-2 - 1/λ², +1, +1)` **without dividing by `a²`** — `aufgabe1e.m` even
has this spelled out and commented out (`%ddx2 = ddx2./a^2;`). The
consistent discretization of Section 3 requires the off-diagonal `+1`
entries (and the whole matrix) to be scaled by `1/a²` relative to the
`-1/λ²` term, i.e. the diagonal should read `-2/a² - 1/λ²` and the
off-diagonals `1/a²`. As written, the code implicitly assumes `a = 1`
(nm) in the balance between the second-derivative term and the `1/λ²`
term, even though `a` is set to a much finer spacing (`0.01 nm` in
`aufgabe1e.m`, `≈0.06 nm` in `simulation.m`). This changes the relative
weighting of "diffusion" vs. "screening" in the solve and is worth fixing
(or deliberately confirming via a grid-refinement/convergence test — halve
`a` and check that `φ_f` is unchanged) before relying on `φ_f` for
quantitative work such as feeding it into an NEGF Hamiltonian.

## 7. Where this fits in a full NEGF device simulation

`φ_f` (or `Psi_f`) computed here is exactly the ingredient a Non-Equilibrium
Green's Function transport solver needs as its potential energy term. This
section originally sketched a *roadmap* from the electrostatics-only MATLAB
scripts to a complete quantum transport simulation; that roadmap has since
been implemented, in Rust, in `crates/negforge-core` — following Datta,
*Quantum Transport*, Chapters 8–10:

1. **Build the device Hamiltonian.** Discretize the transport direction the
   same way as here, and put `U_i = -e\,\phi_{f,i}` on the diagonal of a
   tight-binding-like Hamiltonian `H` (nearest-neighbor hopping `t₀` off
   the diagonal, from the effective mass and the same grid spacing `a`).
   → `negf::green_function_sweep`'s `bulk_diag` (`crates/negforge-core/src/negf.rs`).
2. **Add contact self-energies.** The source and drain are modeled as
   semi-infinite reservoirs in local equilibrium at Fermi levels `E_f` and
   `E_f - eV_ds`. Their effect on the finite device is captured by
   self-energy matrices `Σ_S(E)`, `Σ_D(E)` (computed e.g. via the surface
   Green's function / Sancho-Rubio recursion), which get added to `H` only
   at the boundary sites.
   → the `sigma_l`/`sigma_r` terms in `negf::green_function_sweep` — but see
   §8 below, this is the one place the implementation is known to diverge
   from what this section describes.
3. **Compute the retarded Green's function:**
   $$G^r(E) = \left[EI - H - \Sigma_S(E) - \Sigma_D(E)\right]^{-1}$$
   → done via an O(N) recursive sweep + tridiagonal solves rather than a
   dense matrix inverse (`negf::full_diagonal`, `tridiag::solve_complex`),
   cross-checked against a dense reference solver in `negf.rs`'s tests.
4. **Get transmission and current** via the broadening matrices
   `Γ_{S,D} = i(Σ_{S,D} - Σ_{S,D}^†)`:
   $$T(E) = \mathrm{Tr}\left[\Gamma_S(E)\,G^r(E)\,\Gamma_D(E)\,G^a(E)\right]$$
   $$I = \frac{2e}{h}\int T(E)\,\big[f_S(E) - f_D(E)\big]\,dE$$
   (Landauer–Büttiker formula; `f_S`, `f_D` are Fermi functions at the
   source/drain electrochemical potentials.)
   → `device::Device::calc_current` implements this in its simplest limit:
   rather than computing `T(E)` from the Green's function, it assumes
   `T(E) = 1` for every state above the channel barrier top and `T(E) = 0`
   below it (no scattering, no sub-barrier tunneling) — see §8.
5. **Close the self-consistency loop.** The electron density implied by the
   NEGF solution (via the correlation function `G^<`) is fed back into
   `rho` in the Poisson solve implemented here, and steps 1–5 repeat until
   `φ_f` and the charge density converge together. This outer loop is what
   turns the present linear electrostatics solve into a fully
   self-consistent Poisson–NEGF device simulator — the stated purpose of
   this repository.
   → `selfconsistent::solve_self_consistent`, iterating
   `device::Device::calc_potential` against
   `charge::electron_density`.

`N_dot` in `simulation.m` is the natural entry point for step 5's feedback:
in the Rust port it's `DeviceParams::n_dot`, still a static placeholder for
fixed dopant charge, added alongside (not replaced by) the NEGF-computed
mobile charge density that `selfconsistent::solve_self_consistent` now
assigns to `rho` on each iteration of the outer loop.

## 8. Validity and known limitations of the implemented NEGF / self-consistent loop

The approximations below are documented in full, with derivations and (for
the last one) regression tests, in the doc comments of the corresponding
`crates/negforge-core/src` modules; this section is a summary for readers
who came here for the physics rather than the code.

- **Ballistic transport, no scattering.** `calc_current`'s `T(E) = 1` above
  the barrier / `T(E) = 0` below it (step 4 above) is the "top-of-the
  -barrier" thermionic-emission limit, not a general Landauer calculation
  from the NEGF transmission function. It's a reasonable approximation only
  when the channel is short compared to the carrier mean free path; it
  overestimates on-current and underestimates subthreshold current
  otherwise, since both scattering and sub-barrier tunneling are excluded
  by construction. (`device.rs`)
- **Effective-mass, single-valley, parabolic band.** `m_eff` (used for both
  the ballistic hopping parameter and the NEGF tight-binding hopping `t`)
  stands in for silicon's multiple equivalent conduction-band valleys and
  its non-parabolic dispersion away from the band edge. Treat absolute
  currents as illustrating trends, not as device-accurate predictions.
  (`device.rs`)
- **Tight-binding discretization range of validity.** The on-site/hopping
  parameters reproduce the continuum parabolic dispersion `E = ħ²k²/2m*`
  only for small `k·a`; the chain's actual dispersion is
  `E = 2t(1 - cos(ka))`, bounded above by a bandwidth of `4t`. The grid
  spacing `a` needs to be fine enough that the swept energy range (`psi_0`
  to `e_max`) stays well below that bandwidth — worth checking via a
  grid-refinement test before trusting results at a new device scale.
  (`negf.rs`)
- **Contact self-energy sign — a known, inherited issue, not introduced by
  the Rust rewrite.** Step 2 above describes the textbook self-energy
  `Σ_{S,D}(E)`; the actual formula ported from `quantumsim.m`'s
  `calc_green()` (`sigma = t·exp(i·k·a)`) has the opposite sign from what a
  causal, absorbing contact requires (a retarded self-energy needs
  `Im(Σ) ≤ 0`, so that the broadening `Γ = -2·Im(Σ)` is non-negative; this
  formula gives `Im(Σ) = t·sin(ka) > 0` for every propagating contact
  mode). This is confirmed directly by a unit test
  (`negf::tests::contact_self_energy_has_positive_imaginary_part_for_propagating_modes`),
  and its consequence — the `g_diag` local-density-of-states proxy going
  materially negative, not just noisy — is confirmed by another
  (`negf::tests::g_diag_can_go_negative_at_self_consistent_loop_broadening`)
  at the broadening actually used inside the self-consistent loop. The
  quantity that actually feeds the electrostatic solve,
  `charge::electron_density`, stays non-negative regardless — it's built
  entirely from squared Green's-function magnitudes `|G|²`, which can't go
  negative no matter the self-energy's sign — but its absolute magnitude
  isn't verified against a correctly-signed reference calculation, and this
  may well be why the self-consistent loop needs an empirically-tuned `eta`
  far larger than a physical dephasing rate to converge (a large enough
  artificial broadening keeps the *total* effective damping positive
  despite the contacts contributing negative damping). This is flagged
  rather than fixed because correcting it would change the self-consistent
  loop's numerical behavior — see `negf.rs` for the full derivation.
- **No explicit spin-degeneracy factor in the charge density.** Unlike
  `calc_current`'s explicit `2e/h` (which does include spin degeneracy),
  `charge::electron_density` has no analogous factor of two — carried over
  unchanged from the original `calc_n`, which (before this rewrite closed
  the feedback loop) was never actually exercised against any reference.
  (`charge.rs`)
- **Natural-length electrostatics** (§2 above) remains a thin-body
  approximation and is also a *classical*, non-quantized transverse charge
  model, in tension with the fully quantum-mechanical longitudinal NEGF
  treatment described in this section — the overall model is a first-order
  approximation, not a fully self-consistent 2D quantum simulation.

None of the above are believed to affect the *qualitative* trends the test
suite checks — current increasing with gate/drain bias, the self-consistent
loop converging to a stable, non-negative charge density — but they mean
absolute numbers from this model should be read as illustrating device
physics concepts, not as quantitatively validated predictions for a real
device.
