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
        original MATLAB code."""
        mask = (self.voltage >= v_min) & (self.voltage <= v_max)
        if mask.sum() < 2:
            raise ValueError("not enough points in [v_min, v_max] to fit a swing")
        log_i = np.log10(self.current[mask])
        slope, _ = np.polyfit(self.voltage[mask], log_i, 1)
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
        self._inner = _RustDevice(**kwargs)
        self._inner.calc_potential()

    def __repr__(self):
        return f"Device(n={self.n}, a={self.a} nm, screening_length={self.screening_length:.3f} nm)"

    # -- electrostatics ----------------------------------------------------

    def calc_potential(self) -> "Device":
        """Decoupled electrostatic solve (`rho` unchanged). Matches the
        original `calc_potential()`."""
        self._inner.calc_potential()
        return self

    def solve_self_consistent(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        mixing: float = 0.3,
        eta: float = 0.08,
    ) -> tuple[int, float]:
        """Run the self-consistent Poisson<->NEGF loop (not present in the
        original MATLAB code — see the top-level README). Returns
        `(iterations, residual)` on success; raises `RuntimeError` if it
        does not converge within `max_iterations`.
        """
        return self._inner.solve_self_consistent(max_iterations, tolerance, mixing, eta)

    # -- bias control --------------------------------------------------------

    def set_v_ds(self, v: float) -> "Device":
        self._inner.set_v_ds(v)
        return self

    def set_v_g(self, v: float) -> "Device":
        self._inner.set_v_g(v)
        return self

    def set_l_ch(self, l: float) -> "Device":
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

    def local_density_of_states(self):
        """NEGF sweep at the device's current potential.

        Returns `(energies, ldos)` where `ldos[k]` is the local density of
        states across all grid sites at `energies[k]`. Matches
        `calc_green()`'s `G_r_diag` in the original code.
        """
        energies, ldos = self._inner.local_density_of_states()
        return np.asarray(energies), np.asarray(ldos)

    def sweep_v_g(self, v_min: float, v_max: float, step: float, self_consistent: bool = False) -> IVCurve:
        """Gate-voltage sweep at the device's current drain bias. Matches
        `plot_Vg_I`."""
        voltage, current = self._inner.sweep_v_g(v_min, v_max, step, self_consistent)
        return IVCurve(voltage, current)

    def sweep_v_ds(self, v_min: float, v_max: float, step: float, self_consistent: bool = False) -> IVCurve:
        """Drain-voltage sweep at the device's current gate bias. Matches
        `plot_Vds_I`."""
        voltage, current = self._inner.sweep_v_ds(v_min, v_max, step, self_consistent)
        return IVCurve(voltage, current)
