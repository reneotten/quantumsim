//! Tridiagonal linear system solvers (Thomas algorithm), real and complex.
//!
//! Both the electrostatic Poisson-like solve and the NEGF Green's-function
//! recursion operate on tridiagonal matrices, so this is the one place that
//! implements the O(N) forward-elimination/back-substitution algorithm.

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
