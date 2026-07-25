// PyO3's #[pymethods] macro expands `?` in ways that trip this lint on
// PyResult-returning methods; it's a known false positive, not a real
// no-op conversion in our code.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use negforge_core::{Device, DeviceParams, SelfConsistentOptions};

fn to_py_err(e: negforge_core::NegForgeError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Python-facing wrapper around [`negforge_core::Device`].
#[pyclass(name = "Device")]
struct PyDevice {
    inner: Device,
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (
        a=0.5, e_f=0.15, e_g=1.0, v_ds=0.0, v_g=0.0, d_ox=5.0, d_ch=5.0,
        k_si=11.2, k_ox=3.9, geo=1.0, l_ch=40.0, auto_size_contacts=true,
        l_ds=40.0, n_dot=0.0, epsilon=10e-15, e_fs=0.05, t=300.0, d_e=0.001
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        a: f64,
        e_f: f64,
        e_g: f64,
        v_ds: f64,
        v_g: f64,
        d_ox: f64,
        d_ch: f64,
        k_si: f64,
        k_ox: f64,
        geo: f64,
        l_ch: f64,
        auto_size_contacts: bool,
        l_ds: f64,
        n_dot: f64,
        epsilon: f64,
        e_fs: f64,
        t: f64,
        d_e: f64,
    ) -> Self {
        let params = DeviceParams {
            a,
            e_f,
            e_g,
            v_ds,
            v_g,
            d_ox,
            d_ch,
            k_si,
            k_ox,
            geo,
            l_ch,
            auto_size_contacts,
            l_ds,
            n_dot,
            epsilon,
            e_fs,
            t,
            d_e,
            m_eff: 0.9 * negforge_core::constants::M_E,
        };
        Self {
            inner: Device::new(params),
        }
    }

    fn calc_potential(&mut self) {
        self.inner.calc_potential();
    }

    fn calc_current(&self) -> f64 {
        self.inner.calc_current()
    }

    fn set_v_ds(&mut self, v: f64) {
        self.inner.set_v_ds(v);
    }

    fn set_v_g(&mut self, v: f64) {
        self.inner.set_v_g(v);
    }

    fn set_l_ch(&mut self, l: f64) {
        self.inner.set_l_ch(l);
    }

    #[pyo3(signature = (max_iterations=50, tolerance=1e-6, mixing=0.3, eta=0.08))]
    fn solve_self_consistent(
        &mut self,
        max_iterations: usize,
        tolerance: f64,
        mixing: f64,
        eta: f64,
    ) -> PyResult<(usize, f64)> {
        let opts = SelfConsistentOptions {
            max_iterations,
            tolerance,
            mixing,
            green_energy_fraction: 0.7,
            eta,
        };
        let result = negforge_core::selfconsistent::solve_self_consistent(&mut self.inner, &opts)
            .map_err(to_py_err)?;
        Ok((result.iterations, result.residual))
    }

    #[getter]
    fn psi_f(&self) -> Vec<f64> {
        self.inner.psi_f.clone()
    }

    #[getter]
    fn rho(&self) -> Vec<f64> {
        self.inner.rho.clone()
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    #[getter]
    fn a(&self) -> f64 {
        self.inner.params.a
    }

    /// Natural (screening) length, nm. Named `screening_length` rather than
    /// `lambda` on the Python side since `lambda` is a reserved keyword.
    #[getter]
    fn screening_length(&self) -> f64 {
        self.inner.lambda
    }

    fn positions_nm(&self) -> Vec<f64> {
        (0..self.inner.n)
            .map(|i| i as f64 * self.inner.params.a)
            .collect()
    }

    fn sweep_v_g(
        &mut self,
        v_min: f64,
        v_max: f64,
        step: f64,
        self_consistent: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let opts = SelfConsistentOptions::default();
        let sc = if self_consistent { Some(&opts) } else { None };
        let points = negforge_core::sweep::sweep_v_g(&mut self.inner, v_min, v_max, step, sc)
            .map_err(to_py_err)?;
        Ok((
            points.iter().map(|p| p.voltage).collect(),
            points.iter().map(|p| p.current).collect(),
        ))
    }

    fn sweep_v_ds(
        &mut self,
        v_min: f64,
        v_max: f64,
        step: f64,
        self_consistent: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let opts = SelfConsistentOptions::default();
        let sc = if self_consistent { Some(&opts) } else { None };
        let points = negforge_core::sweep::sweep_v_ds(&mut self.inner, v_min, v_max, step, sc)
            .map_err(to_py_err)?;
        Ok((
            points.iter().map(|p| p.voltage).collect(),
            points.iter().map(|p| p.current).collect(),
        ))
    }

    /// Run the NEGF sweep at the device's current potential and return
    /// `(energies, ldos)` where `ldos[k]` is the local density of states
    /// row (one value per grid site) at `energies[k]`.
    fn local_density_of_states(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
        let e_min = self
            .inner
            .psi_f
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let e_max = 0.7 * self.inner.e_max;
        let d_e = self.inner.params.d_e;
        let steps = ((e_max - e_min) / d_e).floor().max(0.0) as usize;
        let energies: Vec<f64> = (0..=steps).map(|k| e_min + k as f64 * d_e).collect();
        let result = negforge_core::negf::green_function_sweep(
            &self.inner,
            &energies,
            negforge_core::negf::DEFAULT_ETA,
        );
        (result.energies, result.g_diag)
    }
}

#[pymodule]
fn _negforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    Ok(())
}
