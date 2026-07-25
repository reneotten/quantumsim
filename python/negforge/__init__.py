"""NEGForge: 1D ballistic-MOSFET electrostatics + NEGF transport simulator.

The numerical engine (self-consistent Poisson/NEGF solve, tridiagonal linear
algebra, recursive Green's-function evaluation) lives in Rust
(``negforge-core``, exposed here via PyO3 as ``negforge._negforge``).
This package is a thin, notebook-friendly Python wrapper around it: numpy
arrays instead of raw tuples, keyword-argument device construction, and a
small `IVCurve` helper for bias sweeps. See the top-level repository README
for the physics background, unit conventions, and known limitations.
"""

from __future__ import annotations

import numpy as np

from ._negforge import Device as _RustDevice

__all__ = ["Device", "IVCurve"]


class IVCurve:
    """Voltage/current arrays from a bias sweep ([`Device.sweep_v_g`] /
    [`Device.sweep_v_ds`])."""

    def __init__(self, voltage, current):
        self.voltage = np.asarray(voltage)
        self.current = np.asarray(current)

    def __iter__(self):
        return iter((self.voltage, self.current))

    def __repr__(self):
        return f"IVCurve({len(self.voltage)} points)"

    def subthreshold_swing(self, v_min: float = 0.0, v_max: float = 0.4) -> float:
        """Subthreshold swing from a `log10(I)` vs voltage linear fit over
        `[v_min, v_max]`, matching `plot_Vg_I`'s `polyfit` step in the
        original MATLAB code.

        The fit only means anything for strictly positive, finite currents;
        a zero/negative current (numerical noise deep in the off state, or a
        bias range where the ballistic window collapses) would make `log10`
        return `-inf`/`nan` and silently poison the fit, so it raises
        `ValueError` instead.
        """
        mask = (self.voltage >= v_min) & (self.voltage <= v_max)
        if mask.sum() < 2:
            raise ValueError("not enough points in [v_min, v_max] to fit a swing")
        current = self.current[mask]
        if not np.all(np.isfinite(current)) or np.any(current <= 0.0):
            bad = self.voltage[mask][~(np.isfinite(current) & (current > 0.0))]
            raise ValueError(
                "subthreshold swing needs strictly positive, finite currents for the "
                f"log10 fit; {len(bad)} point(s) in [{v_min}, {v_max}] fail this, "
                f"starting at V = {bad[0]:g}"
            )
        log_i = np.log10(current)
        slope, _ = np.polyfit(self.voltage[mask], log_i, 1)
        # `np.polyfit` on a perfectly flat curve returns a tiny non-zero
        # slope from the least-squares solve, so check the data too rather
        # than trusting `slope == 0` alone.
        if not np.isfinite(slope) or slope == 0.0 or np.ptp(log_i) == 0.0:
            raise ValueError(
                f"degenerate subthreshold-swing fit (slope {slope}): the current does "
                f"not vary with voltage over [{v_min}, {v_max}]"
            )
        return 1.0 / slope


class Device:
    """A 1D ballistic MOSFET device.

    Thin, Pythonic wrapper around the Rust engine: numpy arrays for
    profiles, and dataclass-like results instead of raw tuples. Keyword
    arguments match the Rust `DeviceParams` field names (see the top-level
    README for units and defaults), e.g.::

        dev = negforge.Device(v_ds=0.3, v_g=0.2, l_ch=40.0)
        dev.solve_self_consistent()
        current = dev.calc_current()
    """

    def __init__(self, **kwargs):
        # The Rust constructor already solves the electrostatic potential,
        # and so do the bias setters — a freshly built or re-biased device
        # is always ready for `calc_current()`/`local_density_of_states()`.
        self._inner = _RustDevice(**kwargs)

    def __repr__(self):
        return f"Device(n={self.n}, a={self.a} nm, screening_length={self.screening_length:.3f} nm)"

    # -- architecture presets ------------------------------------------------
    #
    # `geo` (number of gates) is what actually distinguishes a planar,
    # FinFET, or gate-all-around/nanoribbon device in this model's natural-
    # length electrostatics (lambda ~ 1/sqrt(geo) -- more gates means
    # tighter electrostatic control). These mirror the presets in
    # `negforge_core::DeviceParams` (`planar`/`fin_fet`/`nanoribbon`); see
    # the top-level README's "Planar, FinFET and nanoribbon devices"
    # section for the physics and for why this ordering (planar worst,
    # nanoribbon best short-channel behavior) is expected.

    @classmethod
    def planar(cls, **overrides) -> "Device":
        """A planar, single-gate (bulk or SOI) MOSFET. Same defaults as
        `Device()` -- planar/single-gate is this model's baseline."""
        return cls(geo=1.0, **overrides)

    @classmethod
    def fin_fet(cls, **overrides) -> "Device":
        """A double-gate FinFET: a thin fin (`d_ch` = fin width) gated on
        both sidewalls, with a thinner oxide than the planar preset. Pass
        `geo=3.0` to also gate the fin top (tri-gate)."""
        params = {"geo": 2.0, "d_ch": 8.0, "d_ox": 1.5}
        params.update(overrides)
        return cls(**params)

    @classmethod
    def nanoribbon(cls, **overrides) -> "Device":
        """A gate-all-around nanoribbon/nanowire FET: a narrow body
        (`d_ch` = ribbon width/diameter) fully wrapped by the gate."""
        params = {"geo": 4.0, "d_ch": 5.0, "d_ox": 1.0}
        params.update(overrides)
        return cls(**params)

    # -- electrostatics ----------------------------------------------------

    def calc_potential(self) -> "Device":
        """Decoupled electrostatic solve (`rho` unchanged). Matches the
        original `calc_potential()`.

        Rarely needed explicitly: construction and the `set_*` methods
        already leave the potential up to date. It is still the way to
        re-solve after mutating `rho` yourself.
        """
        self._inner.calc_potential()
        return self

    def solve_self_consistent(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        mixing: float = 0.3,
        eta: float = 0.08,
        algorithm: str = "recursive",
    ) -> tuple[int, float]:
        """Run the self-consistent Poisson<->NEGF loop (not present in the
        original MATLAB code — see the top-level README). Returns
        `(iterations, residual)` on success; raises `RuntimeError` if it
        does not converge within `max_iterations`, or `ValueError` for
        out-of-range arguments.

        `algorithm` is `"recursive"` (default, O(N) per NEGF energy point)
        or `"dense"` (O(N^3), matching the original MATLAB `inv()` call).
        Recursive is what you want for anything but cross-validating a
        suspicious result — each iteration here runs a full NEGF sweep, and
        dense makes that dramatically slower. See the README's "Performance"
        section for the tradeoff and benchmark numbers.
        """
        return self._inner.solve_self_consistent(max_iterations, tolerance, mixing, eta, algorithm)

    # -- bias control --------------------------------------------------------
    #
    # Each setter re-solves the decoupled electrostatic potential, so
    # `calc_current()`/`local_density_of_states()` can be called straight
    # afterwards without a stale potential from the previous bias point.
    # Note this also resets `rho` to zero: re-run `solve_self_consistent()`
    # if you want a self-consistent result at the new bias.

    def set_v_ds(self, v: float) -> "Device":
        """Set the drain-source bias and re-solve the potential."""
        self._inner.set_v_ds(v)
        return self

    def set_v_g(self, v: float) -> "Device":
        """Set the gate bias and re-solve the potential."""
        self._inner.set_v_g(v)
        return self

    def set_l_ch(self, l: float) -> "Device":
        """Set the channel length (re-deriving the grid) and re-solve the
        potential."""
        self._inner.set_l_ch(l)
        return self

    # -- observables -----------------------------------------------------------

    def calc_current(self) -> float:
        return self._inner.calc_current()

    @property
    def psi_f(self) -> np.ndarray:
        """Solved electrostatic potential energy profile, eV."""
        return np.asarray(self._inner.psi_f)

    @property
    def rho(self) -> np.ndarray:
        """Charge density term entering the electrostatic solve."""
        return np.asarray(self._inner.rho)

    @property
    def positions_nm(self) -> np.ndarray:
        return np.asarray(self._inner.positions_nm())

    @property
    def n(self) -> int:
        return self._inner.n

    @property
    def a(self) -> float:
        return self._inner.a

    @property
    def screening_length(self) -> float:
        """Natural (screening) length lambda, nm."""
        return self._inner.screening_length

    def local_density_of_states(self, algorithm: str = "recursive"):
        """NEGF sweep at the device's current potential.

        Returns `(energies, ldos)` where `ldos[k]` is the local density of
        states across all grid sites at `energies[k]`. Matches
        `calc_green()`'s `G_r_diag` in the original code. Energy points run
        in parallel across CPU cores. `algorithm` is `"recursive"` (default)
        or `"dense"` — see `solve_self_consistent` / the README for the
        tradeoff.
        """
        energies, ldos = self._inner.local_density_of_states(algorithm)
        return np.asarray(energies), np.asarray(ldos)

    def sweep_v_g(
        self,
        v_min: float,
        v_max: float,
        step: float,
        self_consistent: bool = False,
        algorithm: str = "recursive",
    ) -> IVCurve:
        """Gate-voltage sweep at the device's current drain bias. Matches
        `plot_Vg_I`. Points run in parallel across CPU cores and don't
        mutate this device.

        `algorithm` (`"recursive"`/`"dense"`) only changes the *result* when
        `self_consistent=True` — see `solve_self_consistent` — but it is
        always validated, so a misspelled name raises `ValueError` either
        way rather than being silently ignored. `v_min`/`v_max`/`step` must
        describe a forward range with a positive step.
        """
        voltage, current = self._inner.sweep_v_g(v_min, v_max, step, self_consistent, algorithm)
        return IVCurve(voltage, current)

    def sweep_v_ds(
        self,
        v_min: float,
        v_max: float,
        step: float,
        self_consistent: bool = False,
        algorithm: str = "recursive",
    ) -> IVCurve:
        """Drain-voltage sweep at the device's current gate bias. Matches
        `plot_Vds_I`. Points run in parallel across CPU cores and don't
        mutate this device.

        `algorithm` (`"recursive"`/`"dense"`) only changes the *result* when
        `self_consistent=True` — see `solve_self_consistent` — but it is
        always validated, so a misspelled name raises `ValueError` either
        way rather than being silently ignored. `v_min`/`v_max`/`step` must
        describe a forward range with a positive step.
        """
        voltage, current = self._inner.sweep_v_ds(v_min, v_max, step, self_consistent, algorithm)
        return IVCurve(voltage, current)
