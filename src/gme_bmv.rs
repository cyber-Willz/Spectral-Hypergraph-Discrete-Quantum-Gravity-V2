//! Gravity-mediated entanglement (GME / QGEM): the most realistic
//! laboratory route to testing whether gravity is quantized, via the
//! Bose-Marletto-Vedral (BMV) protocol.
//!
//! Two massive particles are each placed in a spatial superposition of
//! two locations ("left"/"right" branches, separated by `dx1`, `dx2`),
//! held near each other (center separation `d`) for a time `t`. Each of
//! the four branch-pair configurations (LL, LR, RL, RR) sits at a
//! different separation and therefore accumulates a different
//! gravitational phase `phi = G m1 m2 t / (hbar r)`. If those four phases
//! are NOT all equal, the two particles' spin-position states become
//! entangled purely through the gravitational interaction -- and by the
//! LOCC theorem (local operations + classical communication cannot
//! generate entanglement), that forces the gravitational field itself to
//! have carried quantum degrees of freedom, since it's the only channel
//! connecting the two particles.
//!
//! What this module does:
//!   1. Computes the exact (not small-`dx`-expanded) Newtonian phase for
//!      each of the four branch configurations, and separately the
//!      standard small-`dx/d`-expansion `phi ~ G m1 m2 dx1 dx2 t / (hbar
//!      d^3)` used throughout the literature -- then checks the two
//!      agree in the appropriate limit, same "convergence, not a single
//!      coincidence" discipline as `geodesics.rs`'s perihelion test.
//!   2. Cross-validates against the actual published numbers: Bose et
//!      al. 2017 (Phys. Rev. Lett. 119, 240401) quote, for `m=1e-14 kg`,
//!      center separation `d=450 micron`, superposition width
//!      `dx=250 micron`, interaction time `tau~2.5 s`: `Delta phi_LR ~
//!      -0.2`, `Delta phi_RL ~ +0.7` (relative to the LL=RR baseline).
//!      This module's constant-branch-separation model doesn't capture
//!      the additional phase picked up during the Stern-Gerlach
//!      acceleration steps the real protocol includes, so the honest bar
//!      is "same sign, same order of magnitude", not exact reproduction
//!      -- see the test below for the actual numbers and that gap.
//!
//! What this module does NOT do: it does not compute the actual spin
//! entanglement witness `W` (a specific trigonometric combination of the
//! four phases plus the Stern-Gerlach recombination step), and it does
//! not model decoherence (Casimir-Polder, blackbody, residual gas
//! collisions) that determines whether the required coherence time is
//! experimentally achievable -- both are real next steps, left
//! undone rather than approximated with an invented formula.

/// Gravitational phase accumulated over time `t` at fixed separation `r`:
/// `phi = G m1 m2 t / (hbar r)`.
pub fn newtonian_phase(g: f64, m1: f64, m2: f64, r: f64, t: f64, hbar: f64) -> f64 {
    g * m1 * m2 * t / (hbar * r)
}

/// Separation between branch `s1` of particle 1 (at `s1 * dx1/2`, with
/// `s1 = -1` for "left", `+1` for "right") and branch `s2` of particle 2
/// (at `d + s2 * dx2/2`), for two particles whose superpositions are
/// collinear with the axis joining their interferometers' centers.
pub fn branch_separation(d: f64, dx1: f64, dx2: f64, s1: f64, s2: f64) -> f64 {
    (d + s2 * dx2 / 2.0) - (s1 * dx1 / 2.0)
}

/// The four branch-pair phases (LL, LR, RL, RR), each `phi =
/// G m1 m2 t / (hbar * separation)`, using the exact (non-small-`dx`)
/// separations.
pub struct FourBranchPhases {
    pub ll: f64,
    pub lr: f64,
    pub rl: f64,
    pub rr: f64,
}

pub fn four_branch_phases(
    g: f64,
    m1: f64,
    m2: f64,
    d: f64,
    dx1: f64,
    dx2: f64,
    t: f64,
    hbar: f64,
) -> FourBranchPhases {
    let phase = |s1: f64, s2: f64| {
        newtonian_phase(g, m1, m2, branch_separation(d, dx1, dx2, s1, s2), t, hbar)
    };
    FourBranchPhases {
        ll: phase(-1.0, -1.0),
        lr: phase(-1.0, 1.0),
        rl: phase(1.0, -1.0),
        rr: phase(1.0, 1.0),
    }
}

/// The standard small-`dx/d`-expansion relative phase used throughout the
/// BMV/QGEM literature: `phi ~ G m1 m2 dx1 dx2 t / (hbar d^3)`. Valid only
/// for `dx1, dx2 << d`; see `four_branch_phases` for the exact version.
pub fn leading_order_relative_phase(
    g: f64,
    m1: f64,
    m2: f64,
    dx1: f64,
    dx2: f64,
    d: f64,
    t: f64,
    hbar: f64,
) -> f64 {
    g * m1 * m2 * dx1 * dx2 * t / (hbar * d.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 6.674_30e-11;
    const HBAR: f64 = 1.054_571_817e-34;

    /// As dx/d -> 0, the genuinely *entangling* combination of the four
    /// branch phases -- `phi_LL + phi_RR - phi_LR - phi_RL`, the part
    /// that can't be absorbed into either particle's own local phase,
    /// since it alone depends on BOTH branch choices jointly -- should
    /// converge to `-2 * leading_order_relative_phase`, the standard
    /// small-dx literature formula. Checked across a shrinking sequence,
    /// not just one data point, so a coincidental match at one scale
    /// can't hide a wrong exponent. (Note: individual branch-phase
    /// differences like `phi_RL - phi_LL` are first-order in dx and
    /// dominate at small dx -- but that first-order piece is a local,
    /// non-entangling phase; it's this second-order cross term that
    /// carries the actual entanglement-generating physics, which is why
    /// the literature formula is second order in dx, not first.)
    #[test]
    fn cross_term_converges_to_leading_order_as_dx_shrinks() {
        let m1 = 1e-14;
        let m2 = 1e-14;
        let d = 450e-6;
        let t = 2.5;
        let mut last_rel_err = f64::MAX;
        for &dx in &[100e-6, 30e-6, 10e-6, 3e-6, 1e-6] {
            let branches = four_branch_phases(G, m1, m2, d, dx, dx, t, HBAR);
            let cross_term = branches.ll + branches.rr - branches.lr - branches.rl;
            let lo = leading_order_relative_phase(G, m1, m2, dx, dx, d, t, HBAR);
            let predicted = -2.0 * lo;
            let rel_err = (cross_term - predicted).abs() / predicted.abs();
            assert!(
                rel_err < last_rel_err + 1e-9,
                "convergence should improve as dx shrinks: dx={dx}, rel_err={rel_err}, previous={last_rel_err}"
            );
            last_rel_err = rel_err;
        }
        assert!(last_rel_err < 1e-3, "should converge tightly at the smallest dx, got {last_rel_err}");
    }

    /// Cross-validation against Bose et al. 2017 (Phys. Rev. Lett. 119,
    /// 240401), quoted values (their Eq./discussion around the SG
    /// free-fall step): m=1e-14 kg, d=450 micron, dx=250 micron,
    /// tau~2.5s give Delta phi_LR ~ -0.2, Delta phi_RL ~ +0.7 relative to
    /// the LL=RR baseline. This model (constant branch separation over
    /// the whole interaction time, no Stern-Gerlach acceleration-phase
    /// contribution) is honestly cruder than the real protocol, so the
    /// bar is matching SIGN and ORDER OF MAGNITUDE, not exact
    /// reproduction.
    #[test]
    fn bose_2017_grb_style_cross_check_matches_sign_and_order_of_magnitude() {
        let m = 1e-14;
        let d = 450e-6;
        let dx = 250e-6;
        let t = 2.5;
        let branches = four_branch_phases(G, m, m, d, dx, dx, t, HBAR);

        // LL and RR should coincide exactly (both equal to G m^2 t/(hbar d))
        // -- a structural symmetry of this collinear geometry, independent
        // of the specific numbers, and a basic self-consistency check.
        assert!(
            (branches.ll - branches.rr).abs() / branches.ll < 1e-12,
            "LL and RR should be exactly equal by construction: ll={}, rr={}",
            branches.ll,
            branches.rr
        );

        let delta_lr = branches.lr - branches.ll;
        let delta_rl = branches.rl - branches.ll;

        let published_lr = -0.2;
        let published_rl = 0.7;

        assert!(delta_lr < 0.0, "Delta phi_LR should be negative, got {delta_lr}");
        assert!(delta_rl > 0.0, "Delta phi_RL should be positive, got {delta_rl}");

        let rel_err_lr = (delta_lr - published_lr).abs() / published_lr.abs();
        let rel_err_rl = (delta_rl - published_rl).abs() / published_rl.abs();
        assert!(
            rel_err_lr < 1.0,
            "Delta phi_LR should be within an order of magnitude of published -0.2, got {delta_lr}"
        );
        assert!(
            rel_err_rl < 1.0,
            "Delta phi_RL should be within an order of magnitude of published 0.7, got {delta_rl}"
        );
    }
}
