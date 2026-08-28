//! Live demo of `qg_phenomenology.rs`: the actual empirical channel by
//! which discrete-spacetime models get tested, since the Planck energy
//! itself is unreachable. Prints real numbers, cross-validated against
//! the published Fermi-LAT GRB 090510 result.

use spectral_dqg::qg_phenomenology::{
    cosmological_liv_kernel, liv_energy_scale_lower_bound, naive_discreteness_energy_scale,
    planck_energy_gev, planck_length_m,
};

fn main() {
    println!("================================================================");
    println!(" QG phenomenology: Lorentz-invariance-violation bounds from GRBs");
    println!("================================================================\n");

    let e_planck = planck_energy_gev();
    let l_planck = planck_length_m();
    println!("Planck energy  = {e_planck:.4e} GeV  (unreachable: ~10^15x the LHC's ~14 TeV)");
    println!("Planck length  = {l_planck:.4e} m\n");

    println!("---- Step 1: cosmological LIV weighting kernel K_1(z) ----\n");
    println!("{:>6} {:>10}", "z", "K_1(z)");
    for &z in &[0.1, 0.3, 0.5, 0.903, 1.5, 2.0] {
        let k = cosmological_liv_kernel(z, 1, 0.3, 0.7, 2000);
        println!("{z:>6.3} {k:>10.4}");
    }

    println!("\n---- Step 2: GRB 090510 -- naive single-photon-pair bound vs. published ----\n");
    println!("Vasileiou et al. 2013 (Phys. Rev. D 87, 122001), using dedicated multi-photon");
    println!("statistical techniques on the full LAT sample, report E_QG,1 > 7.6 * E_Planck.");
    println!("This crude estimator instead uses only the single ~31 GeV photon's arrival");
    println!("time (0.829s post-trigger) vs. a ~keV reference -- it should land in the same");
    println!("order of magnitude but BELOW 7.6, since it throws away most of the statistical");
    println!("power the paper's methods use.\n");

    let e_qg_1 = liv_energy_scale_lower_bound(31.0, 1e-4, 0.903, 0.829, 1, 70.0, 0.3, 0.7);
    let ratio = e_qg_1 / e_planck;
    println!("naive E_QG,1            = {e_qg_1:.4e} GeV");
    println!("naive E_QG,1 / E_Planck = {ratio:.3}   (published: 7.6)");

    println!("\n---- Step 3: what discreteness scale would this bound already exclude? ----\n");
    println!("{:>16} {:>16} {:>10}", "ell / ell_Planck", "implied E_QG (GeV)", "excluded?");
    let published_bound_gev = 7.6 * e_planck;
    for &mult in &[1.0, 10.0, 1e3, 1e6, 1e9, 1e12] {
        let ell = mult * l_planck;
        let e_qg = naive_discreteness_energy_scale(ell);
        let excluded = e_qg < published_bound_gev;
        println!("{mult:>16.0e} {e_qg:>16.4e} {:>10}", if excluded { "yes" } else { "NO" });
    }
    println!(
        "\n(Since naive E_QG(ell) = hbar*c/ell scales as 1/ell, the exclusion threshold is\n\
         ell* = ell_Planck / 7.6 = {:.3e} m. Under this naive ansatz, GRB 090510 alone\n\
         already excludes EVERY discreteness length coarser than roughly the Planck length\n\
         itself -- there is essentially no room left between \"continuum\" and \"excluded\".\n\
         Any discrete-spacetime model that wants a lattice spacing at or above the Planck\n\
         length AND survives this bound needs an explicit suppression mechanism for LIV,\n\
         not just \"discreteness exists\" -- exactly what serious QG programs with an\n\
         emergent, not fundamental, discreteness build in.)",
        l_planck / 7.6
    );

    println!("\n================================================================");
    println!(" What this crate does NOT (yet) claim: a discreteness length FROM");
    println!(" spectral_dqg's own hypergraph. The hypergraph has no committed");
    println!(" physical edge length, so mapping graph combinatorics -> a length");
    println!(" in meters -> a testable E_QG would need an honest calibration this");
    println!(" crate doesn't currently have. This module is the machinery that");
    println!(" calibration would plug into, not a claim that it's been done.");
    println!("================================================================");
}
