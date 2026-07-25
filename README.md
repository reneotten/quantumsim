# NEGForge (quantumsim)

Electrostatics solver for nanoscale MOSFETs, built as the foundation for a
self-consistent Poisson–NEGF (Non-Equilibrium Green's Function) quantum
transport simulator.

- `aufgabe1e.m` — dense-matrix reference implementation of the 1-D reduced
  Poisson solve (source/channel/drain regions, natural-length model).
- `simulation.m` — sparse-matrix rewrite with configurable gate geometry
  (`geo`) and a dopant-charge slot (`N_dot`).

For a full explanation of the underlying physics (the natural-length
electrostatic model, short-channel effects, boundary conditions) and a
line-by-line walkthrough of both scripts, see
**[docs/PHYSICS.md](docs/PHYSICS.md)**.
