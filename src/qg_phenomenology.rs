//! Quantum-gravity phenomenology: Lorentz-invariance-violation (LIV)
//! bounds from gamma-ray-burst photon time-of-flight.
//!
//! This is the piece of "honest evidentiary machinery" a real discrete/
//! quantum-gravity proof attempt actually needs, given the framing that
//! prompted this module: the Planck energy (`~1.22e19 GeV`) is
//! unreachable by any collider, so the field's actual empirical leverage
//! comes from indirect channels -- and the best-established of those is
//! searching for energy-dependent photon speed (a generic prediction of
//! many discretized-spacetime models) in gamma-ray bursts, where a ~GeV
//! photon and a ~keV photon that left the source together have traveled
//! billions of light-years, turning a tiny per-photon dispersion into a
//! measurable arrival-time difference.
//!
//! What this module does:
//!   1. Implements the standard leading-order LIV time-delay formula
//!      (Jacob & Piran 2008; Ellis et al.; used by the Fermi-LAT
//!      collaboration) with a real flat-LambdaCDM cosmological weighting
//!      integral, evaluated by Simpson's rule -- not a flat-space
//!      shortcut.
//!   2. Cross-validates it against the actual published result: Vasileiou
//!      et al. 2013 (Phys. Rev. D 87, 122001) derived `E_QG,1 > 7.6
//!      * E_Planck` from GRB 090510 (`z=0.903`, a `~31 GeV` photon
//!      arriving `~0.829 s` after the GBM trigger) using several
//!      dedicated statistical techniques on the full photon sample. This
//!      module's `liv_energy_scale_lower_bound` uses the much cruder
//!      "single highest-energy photon vs. trigger time" estimator, which
//!      is explicitly weaker than the paper's method -- so the honest bar
//!      to clear is "same order of magnitude, understood gap", not
//!      "reproduces 7.6 exactly". See the unit test below for the actual
//!      numbers and that comparison.
//!   3. Provides `naive_discreteness_energy_scale`, converting an assumed
//!      minimal-length discreteness scale into the QG energy scale
//!      `E_QG = hbar*c/ell` a graph/lattice model with that spacing would
//!      naively imply, so any candidate discreteness length can be
//!      checked against the GRB bound.
//!
//! What this module does NOT do: it does not compute a discreteness
//! length FROM `spectral_dqg`'s own hypergraph (the hypergraph is built
//! as abstract graph combinatorics with no committed physical length
//! scale per edge) -- doing that honestly would require calibrating the
//! model's lattice spacing against something, which this crate does not
//! currently do. Wiring that up is a real next step, deliberately left
//! undone rather than faked with an invented conversion factor.

/// Reduced Planck constant, J*s.
pub const HBAR: f64 = 1.054_571_817e-34;
/// Speed of light, m/s.
pub const C: f64 = 299_792_458.0;
/// Newtonian gravitational constant, m^3 kg^-1 s^-2.
pub const G: f64 = 6.674_30e-11;
/// 1 GeV in Joules.
pub const GEV_IN_JOULES: f64 = 1.602_176_634e-10;

/// Planck energy in GeV: `sqrt(hbar c^5 / G)`.
pub fn planck_energy_gev() -> f64 {
    (HBAR * C.powi(5) / G).sqrt() / GEV_IN_JOULES
}

/// Planck length in meters: `sqrt(hbar G / c^3)`.
pub fn planck_length_m() -> f64 {
    (HBAR * G / C.powi(3)).sqrt()
}

/// The flat-LambdaCDM cosmological weighting integral that enters the LIV
/// time-delay formula at leading order `n`:
/// `K_n(z) = integral_0^z (1+z')^n dz' / sqrt(Omega_m (1+z')^3 + Omega_lambda)`.
/// Evaluated by Simpson's rule with `n_steps` intervals (must be even).
pub fn cosmological_liv_kernel(z: f64, n: i32, omega_m: f64, omega_lambda: f64, n_steps: usize) -> f64 {
    assert!(n_steps % 2 == 0 && n_steps >= 2, "Simpson's rule needs an even step count");
    let integrand = |zp: f64| -> f64 {
        (1.0 + zp).powi(n) / (omega_m * (1.0 + zp).powi(3) + omega_lambda).sqrt()
    };
    let h = z / n_steps as f64;
    let mut sum = integrand(0.0) + integrand(z);
    for i in 1..n_steps {
        let zp = i as f64 * h;
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * integrand(zp);
    }
    sum * h / 3.0
}

/// Hubble constant in inverse seconds, given `H0` in km/s/Mpc.
pub fn hubble_per_second(h0_km_s_mpc: f64) -> f64 {
    let mpc_in_km = 3.085_677_581e19;
    h0_km_s_mpc / mpc_in_km
}

/// Leading-order-`n` LIV lower bound on the quantum-gravity energy scale
/// `E_QG,n` (in GeV), from observing a high-energy photon of energy
/// `e_high_gev` (with a low-energy reference photon of energy
/// `e_low_gev`, `<< e_high_gev`) arrive no more than `delta_t_max_s`
/// after emission, from a source at redshift `z`. Solves
/// `delta_t = (1+n)/(2 H0) * (E_high^n - E_low^n)/E_QG^n * K_n(z)`
/// for `E_QG`.
///
/// This is the crude single-photon-pair estimator, weaker than dedicated
/// multi-photon statistical techniques (see module docs) -- it answers
/// "what's the most conservative bound the raw arrival time alone
/// implies", not "the tightest bound the data support".
pub fn liv_energy_scale_lower_bound(
    e_high_gev: f64,
    e_low_gev: f64,
    z: f64,
    delta_t_max_s: f64,
    n_order: i32,
    h0_km_s_mpc: f64,
    omega_m: f64,
    omega_lambda: f64,
) -> f64 {
    let h0 = hubble_per_second(h0_km_s_mpc);
    let k_n = cosmological_liv_kernel(z, n_order, omega_m, omega_lambda, 2000);
    let n = n_order as f64;
    let energy_term = e_high_gev.powi(n_order) - e_low_gev.powi(n_order);
    let e_qg_n = (1.0 + n) / (2.0 * h0) * energy_term * k_n / delta_t_max_s;
    e_qg_n.powf(1.0 / n)
}

/// The QG energy scale `E_QG = hbar*c/ell` that a discreteness/lattice
/// length `ell` (meters) would naively imply under the simplest possible
/// "the discreteness length sets the dispersion scale" ansatz -- the
/// assumption underlying most LIV phenomenology, not a derivation from
/// any specific hypergraph/spin-foam/CDT model's actual dynamics.
pub fn naive_discreteness_energy_scale(ell_m: f64) -> f64 {
    (HBAR * C / ell_m) / GEV_IN_JOULES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planck_units_match_known_values() {
        // E_Planck ~= 1.22e19 GeV, ell_Planck ~= 1.62e-35 m (standard values).
        let e_p = planck_energy_gev();
        assert!((e_p - 1.22e19).abs() / 1.22e19 < 0.01, "E_Planck={e_p:e}");
        let l_p = planck_length_m();
        assert!((l_p - 1.616e-35).abs() / 1.616e-35 < 0.01, "ell_Planck={l_p:e}");
    }

    /// Small-z sanity check: near z=0 the integrand is ~1, so
    /// K_n(z) ~= z for small z, independent of n.
    #[test]
    fn cosmological_kernel_reduces_to_z_for_small_z() {
        let k = cosmological_liv_kernel(0.001, 1, 0.3, 0.7, 100);
        assert!((k - 0.001).abs() / 0.001 < 1e-3, "K_1(0.001)={k}");
    }

    #[test]
    fn cosmological_kernel_is_monotonic_increasing_in_z() {
        let mut prev = 0.0;
        for &z in &[0.1, 0.3, 0.6, 0.9, 1.5, 2.0] {
            let k = cosmological_liv_kernel(z, 1, 0.3, 0.7, 500);
            assert!(k > prev, "K_1 should increase with z: z={z}, k={k}, prev={prev}");
            prev = k;
        }
    }

    /// Cross-validation against the real published bound: Vasileiou et
    /// al. 2013 (Phys. Rev. D 87, 122001) report E_QG,1 > 7.6 * E_Planck
    /// from GRB 090510 (z=0.903) using several dedicated statistical
    /// techniques on the full LAT photon sample. This test uses the
    /// crude single-photon-pair estimator (the highest-energy ~31 GeV
    /// photon, ~0.829s after trigger, vs. a ~keV reference photon) and
    /// checks it lands in the same order of magnitude as -- but,
    /// correctly, BELOW -- the paper's tighter, more sophisticated bound.
    /// If this test required matching 7.6 exactly it would be checking
    /// the wrong thing; landing within an order of magnitude and on the
    /// weaker side is what a correctly-implemented crude estimator should
    /// do.
    #[test]
    fn grb_090510_naive_bound_is_same_order_of_magnitude_as_published_and_weaker() {
        let e_qg_1 = liv_energy_scale_lower_bound(
            31.0,      // e_high_gev: the ~31 GeV LAT photon
            1e-4,      // e_low_gev: ~100 keV reference, negligible next to 31 GeV
            0.903,     // z: GRB 090510 redshift
            0.829,     // delta_t_max_s: observed arrival delay after trigger
            1,         // linear LIV
            70.0,      // H0 = 70 km/s/Mpc
            0.3,       // Omega_m
            0.7,       // Omega_lambda
        );
        let e_planck = planck_energy_gev();
        let ratio_to_planck = e_qg_1 / e_planck;
        let published_ratio = 7.6;

        // Same order of magnitude as the published result...
        assert!(
            ratio_to_planck > 0.1 && ratio_to_planck < published_ratio,
            "naive E_QG,1/E_Planck={ratio_to_planck}, published=7.6 -- expected same order of \
             magnitude and weaker (smaller), since this is the crude single-photon estimator"
        );
    }

    /// Both a graph model with lattice spacing at the Planck length itself
    /// AND one 1000x coarser than that are already excluded by GRB
    /// 090510's bound -- illustrating just how little room the "naive
    /// discreteness = dispersion scale" ansatz has left, for any model
    /// that adopts it without additional suppression mechanisms (most
    /// serious QG approaches -- e.g. those with an emergent, not
    /// fundamental, Lorentz-violating discreteness -- build in exactly
    /// such a mechanism for this reason).
    #[test]
    fn planck_length_discreteness_is_already_excluded_by_grb_bound() {
        let e_qg_from_planck_length = naive_discreteness_energy_scale(planck_length_m());
        let e_qg_from_1000x_coarser = naive_discreteness_energy_scale(1000.0 * planck_length_m());
        let published_bound_gev = 7.6 * planck_energy_gev();

        assert!(
            e_qg_from_planck_length < published_bound_gev,
            "ell=ell_Planck implies E_QG={e_qg_from_planck_length:e}, below the published bound {published_bound_gev:e}"
        );
        assert!(
            e_qg_from_1000x_coarser < published_bound_gev,
            "ell=1000*ell_Planck implies E_QG={e_qg_from_1000x_coarser:e}, below the published bound {published_bound_gev:e}"
        );
    }

}
