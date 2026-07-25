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
Green's Function transport solver needs as its potential energy term. The
roadmap from here to a complete quantum transport simulation — following
Datta, *Quantum Transport*, Chapters 8–10 — is:

1. **Build the device Hamiltonian.** Discretize the transport direction the
   same way as here, and put `U_i = -e\,\phi_{f,i}` on the diagonal of a
   tight-binding-like Hamiltonian `H` (nearest-neighbor hopping `t₀` off
   the diagonal, from the effective mass and the same grid spacing `a`).
2. **Add contact self-energies.** The source and drain are modeled as
   semi-infinite reservoirs in local equilibrium at Fermi levels `E_f` and
   `E_f - eV_ds`. Their effect on the finite device is captured by
   self-energy matrices `Σ_S(E)`, `Σ_D(E)` (computed e.g. via the surface
   Green's function / Sancho-Rubio recursion), which get added to `H` only
   at the boundary sites.
3. **Compute the retarded Green's function:**
   $$G^r(E) = \left[EI - H - \Sigma_S(E) - \Sigma_D(E)\right]^{-1}$$
4. **Get transmission and current** via the broadening matrices
   `Γ_{S,D} = i(Σ_{S,D} - Σ_{S,D}^†)`:
   $$T(E) = \mathrm{Tr}\left[\Gamma_S(E)\,G^r(E)\,\Gamma_D(E)\,G^a(E)\right]$$
   $$I = \frac{2e}{h}\int T(E)\,\big[f_S(E) - f_D(E)\big]\,dE$$
   (Landauer–Büttiker formula; `f_S`, `f_D` are Fermi functions at the
   source/drain electrochemical potentials.)
5. **Close the self-consistency loop.** The electron density implied by the
   NEGF solution (via the correlation function `G^<`) is fed back into
   `rho` in the Poisson solve implemented here, and steps 1–5 repeat until
   `φ_f` and the charge density converge together. This outer loop is what
   turns the present linear electrostatics solve into a fully
   self-consistent Poisson–NEGF device simulator — the stated purpose of
   this repository.

`N_dot` in `simulation.m` is the natural entry point for step 5's feedback:
today it's a static placeholder for fixed dopant charge, but the same slot
in the Poisson right-hand side is where the NEGF-computed mobile charge
density would eventually be added on each iteration of the outer loop.
