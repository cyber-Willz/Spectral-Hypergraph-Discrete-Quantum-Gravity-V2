//! The Casimir effect: a laboratory-confirmed QFT prediction (vacuum
//! zero-point energy between conducting plates), derived here via
//! zeta-function regularization rather than presented as a memorized
//! closed form.
//!
//! Derivation sketch (standard, e.g. Milonni, "The Quantum Vacuum"): for
//! the EM field between two perfectly-conducting plates of area A
//! separated by distance a, the mode sum for the zero-point energy,
//! after integrating out the continuous transverse wavevector, reduces
//! to a divergent sum `sum_{n=1}^infinity n^3` (from the discrete
//! `k_z = n*pi/a` modes and the density of transverse states). Assigning
//! that divergent sum its zeta-regularized value `zeta(-3) = 1/120`
//! (computed in `qft_zeta.rs`, not hardcoded here) gives the finite,
//! experimentally-confirmed result
//!
//!   E(a)/A = -(pi^2/6) * hbar * c * zeta(-3) / a^3 = -pi^2 hbar c / (720 a^3)
//!
//! This module takes `zeta(-3)` as an input (computed independently in
//! `qft_zeta.rs`) rather than hardcoding `1/120`, so the Casimir formula
//! and the closed form it's supposed to equal are only linked through
//! actually-computed machinery -- see the unit tests for that check, and
//! for a cross-validation against the real Mohideen & Roy 1998 AFM
//! measurement.
//!
//! What this module does NOT include: finite conductivity (real gold has
//! a finite skin depth, which matters once the separation is comparable
//! to it), surface roughness, and finite-temperature corrections -- all
//! of which the real experiment had to model to reach 1% agreement. This
//! is the T=0, perfect-conductor idealization only.

/// Casimir energy per unit area for two parallel, perfectly-conducting
/// plates separated by `a`: `E/A = -(pi^2/6) hbar c zeta(-3) / a^3`.
/// Pass in an independently-computed `zeta_neg3` (e.g. from
/// `qft_zeta::zeta_negative_integer(-3, ...)`) rather than a hardcoded
/// constant.
pub fn energy_per_area(hbar: f64, c: f64, a: f64, zeta_neg3: f64) -> f64 {
    -(std::f64::consts::PI.powi(2) / 6.0) * hbar * c * zeta_neg3 / a.powi(3)
}

/// Casimir force per unit area (pressure, negative = attractive):
/// `F/A = -dE/da / A = -(pi^2/2) hbar c zeta(-3) / a^4`.
pub fn force_per_area(hbar: f64, c: f64, a: f64, zeta_neg3: f64) -> f64 {
    -(std::f64::consts::PI.powi(2) / 2.0) * hbar * c * zeta_neg3 / a.powi(4)
}

/// Force between a sphere of radius `r` and a flat plate, separated by
/// `a`, via the proximity force approximation (PFA): `F = 2*pi*r *
/// (E/A)(a)` -- the standard geometry used by the actual AFM Casimir
/// experiments (Mohideen & Roy 1998 and successors), since a true
/// sphere-plate calculation is far harder than parallel plates and PFA
/// is accurate to a few percent when `a << r` (true here: a ~ 0.1-0.9
/// micron, r ~ 100 micron).
pub fn force_sphere_plate_pfa(hbar: f64, c: f64, r: f64, a: f64, zeta_neg3: f64) -> f64 {
    2.0 * std::f64::consts::PI * r * energy_per_area(hbar, c, a, zeta_neg3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qft_zeta::zeta_negative_integer;

    const HBAR: f64 = 1.054_571_817e-34;
    const C: f64 = 299_792_458.0;

    /// The zeta-regularized derivation should reproduce the standard
    /// closed form `E/A = -pi^2 hbar c/(720 a^3)` -- an algebra check
    /// that the two are actually the same formula, using an
    /// independently-computed zeta(-3) rather than assuming it.
    #[test]
    fn energy_per_area_matches_standard_closed_form() {
        let z = zeta_negative_integer(-3, 2000);
        let a = 1e-6; // 1 micron
        let e = energy_per_area(HBAR, C, a, z);
        let closed_form = -std::f64::consts::PI.powi(2) * HBAR * C / (720.0 * a.powi(3));
        let rel_err = (e - closed_form).abs() / closed_form.abs();
        assert!(rel_err < 1e-9, "e={e}, closed_form={closed_form}, rel_err={rel_err}");
    }

    /// Force should scale as 1/a^4 (parallel plates) -- a structural
    /// check independent of the exact zeta(-3) value used.
    #[test]
    fn force_per_area_scales_as_inverse_fourth_power() {
        let z = zeta_negative_integer(-3, 2000);
        let f1 = force_per_area(HBAR, C, 1e-6, z).abs();
        let f2 = force_per_area(HBAR, C, 2e-6, z).abs();
        let ratio = f1 / f2;
        assert!((ratio - 16.0).abs() < 1e-6, "doubling a should cut force/area by 16x, got ratio={ratio}");
    }

    /// Cross-validation against Mohideen & Roy 1998 (PRL 81, 4549): AFM
    /// measurement of the force between a gold-coated sphere (diameter
    /// 196 micron, so r=98 micron) and a flat plate, separations 0.1-0.9
    /// micron, forces of order 1-300 pN, RMS deviation from full theory
    /// (which includes finite conductivity/roughness/thermal corrections
    /// this module doesn't model) of 1.6 pN. This T=0/perfect-conductor
    /// idealization should land in the same pN-to-sub-pN range and the
    /// same 1/a^3-ish falloff (via PFA) across that separation range --
    /// it is expected to run somewhat HIGH of the real measurement at the
    /// smallest separations, since finite conductivity suppresses the
    /// real force there and isn't modeled here.
    #[test]
    fn sphere_plate_force_lands_in_mohideen_roy_1998_range() {
        let z = zeta_negative_integer(-3, 2000);
        let r = 98e-6; // 196 micron diameter sphere
        let f_100nm = force_sphere_plate_pfa(HBAR, C, r, 100e-9, z).abs();
        let f_900nm = force_sphere_plate_pfa(HBAR, C, r, 900e-9, z).abs();

        // Reported force range in the paper's Fig. 4 spans roughly a few
        // hundred pN at 100nm down to sub-pN near 900nm.
        assert!(
            f_100nm > 50e-12 && f_100nm < 500e-12,
            "F(100nm) should be order-100-pN, got {:e} N",
            f_100nm
        );
        assert!(
            f_900nm > 0.05e-12 && f_900nm < 5e-12,
            "F(900nm) should be sub-pN to a few pN, got {:e} N",
            f_900nm
        );
        // Falloff over the 9x separation range should be roughly a's
        // exponent cubed via PFA (900/100)^3 = 729x, order-of-magnitude
        // check on the scaling behavior itself.
        let falloff = f_100nm / f_900nm;
        assert!(
            falloff > 300.0 && falloff < 1500.0,
            "falloff over the 0.1-0.9 micron range should be ~ (9)^3=729x, got {falloff}"
        );
    }
}
