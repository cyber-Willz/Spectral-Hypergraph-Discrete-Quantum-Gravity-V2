//! Seeley-DeWitt heat-kernel expansion: a cornerstone QFT-in-curved-
//! spacetime result, tested here against the crate's OWN heat-kernel
//! machinery (`heat_kernel::heat_trace`, already used elsewhere in this
//! crate for spectral-dimension flow) rather than a fresh standalone
//! calculation -- this is the "test it on the system" piece, connecting
//! curvature (already computed via `tensor_calculus.rs` in the GR work)
//! to the QFT heat-kernel trace.
//!
//! For the scalar Laplace-Beltrami operator on a closed Riemannian
//! 2-manifold, the short-time heat-trace expansion is
//!
//!   Tr(e^{-t Delta}) ~ Area/(4*pi*t) + chi/6 + O(t)
//!
//! where `chi` is the Euler characteristic (Gauss-Bonnet: `integral R dA
//! = 4*pi*chi` for the scalar-curvature convention `R = 2K`). The first
//! term is pure geometry (area); the SECOND term is where curvature
//! enters the heat kernel at all -- it's the `a_1` Seeley-DeWitt
//! coefficient, `(1/(4*pi)) * integral (R/6) dA`. This module tests that
//! the crate's heat-kernel trace, evaluated on the EXACT eigenvalues of
//! the round-sphere Laplacian (a case with known closed-form spectrum,
//! not from any approximate discretization), reproduces both terms.
//!
//! What this does NOT do: it doesn't compute the heat trace on an
//! arbitrary curved metric from `metrics.rs`/`tensor_calculus.rs`
//! directly (that would need the actual Laplacian spectrum of an
//! arbitrary metric, which this crate doesn't currently extract) -- the
//! sphere is used because its spectrum is known in closed form, making
//! this a genuine check of the heat-kernel machinery against a real
//! curved-space QFT result, not a claim that arbitrary `spectral_dqg`
//! metrics have been wired into it.

use crate::heat_kernel::heat_trace;

/// Eigenvalues (with multiplicity) of the Laplace-Beltrami operator on a
/// round 2-sphere of radius `r`: `lambda_l = l(l+1)/r^2`, degeneracy
/// `2l+1`, for `l = 0..=l_max`. Returned as a flat list with each
/// eigenvalue repeated `2l+1` times, ready for `heat_kernel::heat_trace`.
pub fn sphere_laplacian_eigenvalues(r: f64, l_max: u64) -> Vec<f64> {
    let mut eigs = Vec::new();
    for l in 0..=l_max {
        let lambda = (l * (l + 1)) as f64 / (r * r);
        for _ in 0..(2 * l + 1) {
            eigs.push(lambda);
        }
    }
    eigs
}

/// The residual `Tr(e^{-t Delta}) - Area/(4*pi*t)` after subtracting the
/// pure-geometry leading term, using this crate's own `heat_trace`. As
/// `t -> 0` (with `l_max` large enough that truncation is negligible at
/// that `t`), this should converge to the Seeley-DeWitt `a_1`
/// coefficient `chi/6` (`= 1/3` for the sphere, `chi=2`).
pub fn seeley_dewitt_a1_residual(r: f64, l_max: u64, t: f64) -> f64 {
    let eigs = sphere_laplacian_eigenvalues(r, l_max);
    let p_t = heat_trace(&eigs, t);
    let area = 4.0 * std::f64::consts::PI * r * r;
    let leading = area / (4.0 * std::f64::consts::PI * t);
    p_t - leading
}

#[cfg(test)]
mod tests {
    use super::*;

    /// As t shrinks (with l_max fixed and large enough that truncation
    /// error stays negligible across the whole sweep), the residual
    /// should converge to chi/6 = 1/3 for the sphere -- checked across a
    /// shrinking sequence of t, not a single data point, using the
    /// crate's own heat_kernel::heat_trace function.
    #[test]
    fn a1_coefficient_converges_to_euler_characteristic_over_six() {
        let r = 1.0;
        let l_max = 2000; // eigenvalues up to l(l+1) ~ 4e6, e^{-t*lambda}
                           // negligible for the smallest t tested (1e-3):
                           // exp(-1e-3 * 1.6e7) ~ 0, so truncation error
                           // is far below double precision here.
        let expected = 1.0 / 3.0; // chi/6, chi=2 for the sphere
        let mut last_err = f64::MAX;
        for &t in &[0.05, 0.02, 0.01, 0.005, 0.002, 0.001] {
            let residual = seeley_dewitt_a1_residual(r, l_max, t);
            let err = (residual - expected).abs();
            assert!(
                err < last_err + 1e-12,
                "residual should converge to chi/6 as t shrinks: t={t}, residual={residual}, err={err}, previous={last_err}"
            );
            last_err = err;
        }
        assert!(last_err < 1e-3, "should be tight at the smallest t, got err={last_err}");
    }

    /// The leading term alone (before curvature enters at all) should
    /// just track the sphere's area, independent of t -- a basic sanity
    /// check on the geometry side of the split.
    #[test]
    fn leading_term_is_pure_area_independent_of_radius_choice() {
        for &r in &[0.5, 1.0, 2.0, 3.0] {
            let l_max = 2000;
            let t = 0.001;
            let eigs = sphere_laplacian_eigenvalues(r, l_max);
            let p_t = heat_trace(&eigs, t);
            let area = 4.0 * std::f64::consts::PI * r * r;
            let leading = area / (4.0 * std::f64::consts::PI * t);
            let residual = p_t - leading;
            // residual should still be ~1/3 regardless of r (chi is
            // scale-invariant), confirming the area subtraction is doing
            // the right scaling with r.
            assert!(
                (residual - 1.0 / 3.0).abs() < 1e-3,
                "r={r}: residual={residual}, expected ~1/3"
            );
        }
    }
}
