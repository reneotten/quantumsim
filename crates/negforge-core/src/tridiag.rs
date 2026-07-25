//! Tridiagonal linear system solvers (Thomas algorithm), real and complex.
//!
//! Both the electrostatic Poisson-like solve and the NEGF Green's-function
//! recursion operate on tridiagonal matrices, so this is the one place that
//! implements the O(N) forward-elimination/back-substitution algorithm.
//!
//! [`solve_real`] and [`solve_complex`] are the general routines. The NEGF
//! hot path needs two specialized ones — [`inverse_diagonal_complex`] and
//! [`boundary_columns_complex`] — which exploit the two properties of
//! `(E + i*eta) I - H` for a nearest-neighbor tight-binding chain: the
//! off-diagonal is a single real constant, and the right-hand sides are unit
//! vectors at the two ends. They write into caller-owned buffers so the
//! per-energy-point evaluation allocates nothing.

use num_complex::Complex64;

/// Solve `A x = rhs` for a real tridiagonal `A` given as three diagonals.
///
/// `sub[i]` is `A[i+1][i]` (length N-1), `diag[i]` is `A[i][i]` (length N),
/// `sup[i]` is `A[i][i+1]` (length N-1).
///
/// Panics if `n == 0` (there is no system to solve); the degenerate `n == 1`
/// case is handled directly as a scalar division.
pub fn solve_real(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    assert!(n > 0, "tridiagonal solve needs at least one equation");
    assert_eq!(sub.len(), n - 1);
    assert_eq!(sup.len(), n - 1);
    assert_eq!(rhs.len(), n);

    if n == 1 {
        return vec![rhs[0] / diag[0]];
    }

    let mut c_prime = vec![0.0; n - 1];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        // `i - 1 <= n - 2`, so `c_prime[i - 1]` is always in bounds here.
        let denom = diag[i] - sub[i - 1] * c_prime[i - 1];
        if i < n - 1 {
            c_prime[i] = sup[i] / denom;
        }
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

/// Complex counterpart of [`solve_real`], used by the NEGF module where the
/// matrix `(E + i*eta) I - H` has a real tridiagonal skeleton but complex
/// diagonal entries (contact self-energies) and the right-hand side may be
/// complex (unit vectors when extracting a single column of the Green's
/// function).
///
/// Same degenerate-size contract as [`solve_real`].
///
/// The NEGF hot path uses the specialized routines below instead; this
/// general version is kept as the reference they are validated against, and
/// for any future non-constant off-diagonal (longer-range hopping, a
/// position-dependent effective mass).
#[allow(dead_code)]
pub fn solve_complex(
    sub: &[f64],
    diag: &[Complex64],
    sup: &[f64],
    rhs: &[Complex64],
) -> Vec<Complex64> {
    let n = diag.len();
    assert!(n > 0, "tridiagonal solve needs at least one equation");
    assert_eq!(sub.len(), n - 1);
    assert_eq!(sup.len(), n - 1);
    assert_eq!(rhs.len(), n);

    if n == 1 {
        return vec![rhs[0] / diag[0]];
    }

    let mut c_prime = vec![Complex64::new(0.0, 0.0); n - 1];
    let mut d_prime = vec![Complex64::new(0.0, 0.0); n];

    c_prime[0] = Complex64::new(sup[0], 0.0) / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - c_prime[i - 1] * sub[i - 1];
        if i < n - 1 {
            c_prime[i] = Complex64::new(sup[i], 0.0) / denom;
        }
        d_prime[i] = (rhs[i] - d_prime[i - 1] * sub[i - 1]) / denom;
    }

    let mut x = vec![Complex64::new(0.0, 0.0); n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

/// Reusable scratch buffers for [`inverse_diagonal_complex`] and
/// [`boundary_columns_complex`]. Allocate one per worker thread and reuse it
/// across energy points.
pub struct ComplexWorkspace {
    /// Forward-elimination coefficients (`c'` in the Thomas algorithm), or
    /// the left-connected Green's function in the diagonal recursion.
    a: Vec<Complex64>,
    /// Pivot denominators of the forward elimination.
    b: Vec<Complex64>,
}

impl ComplexWorkspace {
    pub fn new(n: usize) -> Self {
        Self {
            a: vec![Complex64::new(0.0, 0.0); n],
            b: vec![Complex64::new(0.0, 0.0); n],
        }
    }

    fn resize(&mut self, n: usize) {
        if self.a.len() < n {
            self.a.resize(n, Complex64::new(0.0, 0.0));
            self.b.resize(n, Complex64::new(0.0, 0.0));
        }
    }
}

/// Diagonal of `A^-1` for a complex tridiagonal `A` whose off-diagonals are
/// both the same real constant `t`, written into `out`.
///
/// This is the standard left-connected recursive Green's-function sweep:
/// forward for the semi-infinite-to-the-left partial inverse, backward to
/// dress it with the right-hand part. O(N), no allocation.
pub fn inverse_diagonal_complex(
    t: f64,
    diag: &[Complex64],
    work: &mut ComplexWorkspace,
    out: &mut [Complex64],
) {
    let n = diag.len();
    assert!(n > 0, "tridiagonal solve needs at least one equation");
    assert_eq!(out.len(), n);
    work.resize(n);

    let one = Complex64::new(1.0, 0.0);
    if n == 1 {
        out[0] = one / diag[0];
        return;
    }

    let t2 = t * t;
    let g_left = &mut work.a;
    g_left[0] = one / diag[0];
    for i in 1..n {
        g_left[i] = one / (diag[i] - g_left[i - 1] * t2);
    }

    out[n - 1] = g_left[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = g_left[i] + g_left[i] * g_left[i] * t2 * out[i + 1];
    }
}

/// Columns `0` and `N-1` of `A^-1` for a complex tridiagonal `A` whose
/// off-diagonals are both the same real constant `t`.
///
/// Equivalent to two [`solve_complex`] calls with the unit vectors `e_0` and
/// `e_{N-1}` as right-hand sides, but shares the single forward elimination
/// between them (the matrix is the same) and skips the last column's forward
/// pass entirely — its `d'` vector is zero everywhere except the final entry.
/// O(N), no allocation, and bit-for-bit identical to the two separate solves
/// (see the tests).
pub fn boundary_columns_complex(
    t: f64,
    diag: &[Complex64],
    work: &mut ComplexWorkspace,
    col_first: &mut [Complex64],
    col_last: &mut [Complex64],
) {
    let n = diag.len();
    assert!(n > 0, "tridiagonal solve needs at least one equation");
    assert_eq!(col_first.len(), n);
    assert_eq!(col_last.len(), n);
    work.resize(n);

    let one = Complex64::new(1.0, 0.0);
    if n == 1 {
        col_first[0] = one / diag[0];
        col_last[0] = col_first[0];
        return;
    }

    // Shared forward elimination.
    let (c, denom) = (&mut work.a, &mut work.b);
    denom[0] = diag[0];
    c[0] = t / denom[0];
    for i in 1..n {
        denom[i] = diag[i] - c[i - 1] * t;
        if i < n - 1 {
            c[i] = t / denom[i];
        }
    }

    // First column: rhs = e_0, so d'[0] = 1/denom[0] and the rest of the
    // forward sweep just propagates it. Built in place, then back-substituted
    // in place (x[i] only needs d'[i] and x[i+1]).
    col_first[0] = one / denom[0];
    for i in 1..n {
        col_first[i] = -col_first[i - 1] * t / denom[i];
    }
    for i in (0..n - 1).rev() {
        col_first[i] -= c[i] * col_first[i + 1];
    }

    // Last column: rhs = e_{N-1}, so d' is zero until the final entry and the
    // back-substitution is all that remains.
    col_last[n - 1] = one / denom[n - 1];
    for i in (0..n - 1).rev() {
        col_last[i] = -c[i] * col_last[i + 1];
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Dense reference solver (Gaussian elimination with partial pivoting)
    /// used only in tests to validate the O(N) Thomas-algorithm solvers.
    fn dense_solve_real(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
        let n = diag.len();
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = diag[i];
            if i > 0 {
                a[i][i - 1] = sub[i - 1];
            }
            if i < n - 1 {
                a[i][i + 1] = sup[i];
            }
        }
        let mut b = rhs.to_vec();
        for col in 0..n {
            let mut pivot = col;
            for row in (col + 1)..n {
                if a[row][col].abs() > a[pivot][col].abs() {
                    pivot = row;
                }
            }
            a.swap(col, pivot);
            b.swap(col, pivot);
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for k in col..n {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = b[i];
            for j in (i + 1)..n {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        x
    }

    #[test]
    fn thomas_matches_dense_gaussian_elimination() {
        let n = 12;
        let sub: Vec<f64> = (0..n - 1).map(|i| 1.0 + 0.1 * i as f64).collect();
        let sup: Vec<f64> = (0..n - 1).map(|i| 0.7 + 0.05 * i as f64).collect();
        let diag: Vec<f64> = (0..n).map(|i| -3.0 - 0.2 * i as f64).collect();
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 1.0).collect();

        let fast = solve_real(&sub, &diag, &sup, &rhs);
        let reference = dense_solve_real(&sub, &diag, &sup, &rhs);

        for (a, b) in fast.iter().zip(reference.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn single_equation_systems_are_solved_as_scalars() {
        // A 1x1 system has no off-diagonals at all; the Thomas recursion has
        // nothing to eliminate, so it must degenerate to a plain division
        // rather than indexing into the empty sub/super-diagonals.
        let x = solve_real(&[], &[4.0], &[], &[2.0]);
        assert_eq!(x, vec![0.5]);

        let z = solve_complex(
            &[],
            &[Complex64::new(0.0, 2.0)],
            &[],
            &[Complex64::new(4.0, 0.0)],
        );
        assert_relative_eq!(z[0].re, 0.0, epsilon = 1e-12);
        assert_relative_eq!(z[0].im, -2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "at least one equation")]
    fn empty_system_panics_with_a_clear_message() {
        solve_real(&[], &[], &[], &[]);
    }

    /// Reference inputs for the specialized complex routines: a constant
    /// real off-diagonal and a complex diagonal, as in `(E + i*eta) I - H`.
    fn complex_system(n: usize) -> (f64, Vec<Complex64>) {
        let t = 0.37;
        let diag = (0..n)
            .map(|i| Complex64::new(-1.5 - 0.3 * i as f64, 0.05))
            .collect();
        (t, diag)
    }

    #[test]
    fn inverse_diagonal_matches_general_solves() {
        let n = 11;
        let (t, diag) = complex_system(n);
        let off = vec![t; n - 1];

        let mut work = ComplexWorkspace::new(n);
        let mut got = vec![Complex64::new(0.0, 0.0); n];
        inverse_diagonal_complex(t, &diag, &mut work, &mut got);

        for i in 0..n {
            let mut e_i = vec![Complex64::new(0.0, 0.0); n];
            e_i[i] = Complex64::new(1.0, 0.0);
            let column = solve_complex(&off, &diag, &off, &e_i);
            assert_relative_eq!(got[i].re, column[i].re, epsilon = 1e-12);
            assert_relative_eq!(got[i].im, column[i].im, epsilon = 1e-12);
        }
    }

    #[test]
    fn boundary_columns_match_two_general_solves_exactly() {
        let n = 11;
        let (t, diag) = complex_system(n);
        let off = vec![t; n - 1];

        let mut work = ComplexWorkspace::new(n);
        let mut first = vec![Complex64::new(0.0, 0.0); n];
        let mut last = vec![Complex64::new(0.0, 0.0); n];
        boundary_columns_complex(t, &diag, &mut work, &mut first, &mut last);

        let mut e_0 = vec![Complex64::new(0.0, 0.0); n];
        e_0[0] = Complex64::new(1.0, 0.0);
        let mut e_n = vec![Complex64::new(0.0, 0.0); n];
        e_n[n - 1] = Complex64::new(1.0, 0.0);

        // The shared factorization performs the same operations in the same
        // order as the two separate solves, so this is an exact match, not an
        // approximate one.
        assert_eq!(first, solve_complex(&off, &diag, &off, &e_0));
        assert_eq!(last, solve_complex(&off, &diag, &off, &e_n));
    }

    #[test]
    fn specialized_complex_routines_handle_a_single_site() {
        let diag = vec![Complex64::new(0.0, 2.0)];
        let mut work = ComplexWorkspace::new(1);
        let mut out = vec![Complex64::new(0.0, 0.0); 1];
        inverse_diagonal_complex(0.37, &diag, &mut work, &mut out);
        assert_relative_eq!(out[0].im, -0.5, epsilon = 1e-12);

        let mut first = vec![Complex64::new(0.0, 0.0); 1];
        let mut last = vec![Complex64::new(0.0, 0.0); 1];
        boundary_columns_complex(0.37, &diag, &mut work, &mut first, &mut last);
        assert_eq!(first, last);
        assert_relative_eq!(first[0].im, -0.5, epsilon = 1e-12);
    }

    #[test]
    fn workspace_grows_to_fit_a_larger_system() {
        // A workspace sized for a small device must still be usable after the
        // grid grows (e.g. `set_l_ch`), rather than panicking on a short slice.
        let mut work = ComplexWorkspace::new(2);
        let n = 9;
        let (t, diag) = complex_system(n);
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        inverse_diagonal_complex(t, &diag, &mut work, &mut out);
        assert!(out.iter().all(|g| g.re.is_finite() && g.im.is_finite()));
    }

    #[test]
    fn identity_system_returns_rhs() {
        let n = 5;
        let sub = vec![0.0; n - 1];
        let sup = vec![0.0; n - 1];
        let diag = vec![1.0; n];
        let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = solve_real(&sub, &diag, &sup, &rhs);
        assert_eq!(x, rhs);
    }
}
