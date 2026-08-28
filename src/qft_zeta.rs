//! Riemann zeta function, computed from first principles (no hardcoded
//! special values) via direct summation with an Euler-Maclaurin tail
//! correction for `s > 1`, and analytically continued to negative
//! integers via the functional equation. This exists to feed
//! `casimir.rs` an honestly-derived `zeta(-3)`, rather than the constant
//! `1/120` typed in as a magic number -- the same "derive it, then
//! cross-check against the known value" discipline as the rest of this
//! crate's QFT/QG modules.

use std::f64::consts::PI;

/// zeta(s) for real s > 1, via direct summation of the first `n_terms`
/// terms plus the standard Euler-Maclaurin asymptotic tail correction
/// `N^(1-s)/(s-1) + N^(-s)/2 + s*N^(-s-1)/12` for the remainder
/// `sum_{n=N+1}^infinity n^-s`.
pub fn zeta_gt1(s: f64, n_terms: u64) -> f64 {
    assert!(s > 1.0, "direct summation only converges for s > 1");
    let n = n_terms as f64;
    let mut total = 0.0;
    for k in 1..=n_terms {
        total += (k as f64).powf(-s);
    }
    let tail = n.powf(1.0 - s) / (s - 1.0) + 0.5 * n.powf(-s) + (s / 12.0) * n.powf(-s - 1.0);
    total + tail
}

/// Factorial as f64, for the small non-negative integers this module
/// needs (Gamma(n) = (n-1)! for positive integers).
fn factorial(n: u64) -> f64 {
    (1..=n).map(|k| k as f64).product::<f64>().max(1.0)
}

/// zeta(s) for s a negative integer, via the functional equation
/// `zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)`,
/// with `zeta(1-s)` (which has argument `> 1` since `s < 0`) computed by
/// `zeta_gt1` and `Gamma(1-s) = (-s)!` since `1-s` is then a positive
/// integer.
pub fn zeta_negative_integer(s: i64, n_terms_for_positive_side: u64) -> f64 {
    assert!(s < 0, "use zeta_gt1 for s > 1; s=0,1 are separate special cases not needed here");
    let s_f = s as f64;
    let one_minus_s = (1 - s) as u64; // positive integer, since s < 0
    let z_pos = zeta_gt1(one_minus_s as f64, n_terms_for_positive_side);
    let gamma_term = factorial(one_minus_s - 1); // Gamma(1-s) = (1-s-1)! = (-s)!
    2f64.powf(s_f) * PI.powf(s_f - 1.0) * (PI * s_f / 2.0).sin() * gamma_term * z_pos
}

#[cfg(test)]
mod tests {
    use super::*;

    /// zeta(4) = pi^4/90 (the s=4 case of the general zeta(2n) closed
    /// forms) -- validates the Euler-Maclaurin tail correction.
    #[test]
    fn zeta_4_matches_pi4_over_90() {
        let z = zeta_gt1(4.0, 2000);
        let exact = PI.powi(4) / 90.0;
        assert!((z - exact).abs() / exact < 1e-10, "zeta(4)={z}, exact={exact}");
    }

    /// zeta(2) = pi^2/6 (the Basel problem) -- a second, independent
    /// check of the same direct-summation + tail-correction machinery.
    /// Needs more terms than zeta(4) for the same precision: the
    /// Euler-Maclaurin tail correction's own error shrinks slower for
    /// smaller s.
    #[test]
    fn zeta_2_matches_basel_problem() {
        let z = zeta_gt1(2.0, 20_000);
        let exact = PI.powi(2) / 6.0;
        assert!((z - exact).abs() / exact < 1e-8, "zeta(2)={z}, exact={exact}");
    }

    /// zeta(-1) = -1/12, the famous Ramanujan-summation value -- via the
    /// functional equation, independent of the s=4/s=-3 pair the Casimir
    /// module actually uses, so this validates the functional-equation +
    /// Gamma-factorial machinery on a second, well-known case.
    #[test]
    fn zeta_negative_1_matches_minus_one_twelfth() {
        let z = zeta_negative_integer(-1, 20_000);
        assert!((z - (-1.0 / 12.0)).abs() < 1e-9, "zeta(-1)={z}, exact=-1/12");
    }

    /// zeta(-3) = 1/120 -- the value the Casimir-effect derivation
    /// actually needs (see casimir.rs). Computed here purely via the
    /// functional equation applied to the independently-validated
    /// zeta(4), not hardcoded.
    #[test]
    fn zeta_negative_3_matches_one_over_120() {
        let z = zeta_negative_integer(-3, 2000);
        assert!((z - 1.0 / 120.0).abs() < 1e-10, "zeta(-3)={z}, exact=1/120");
    }
}
