//! Second GR demo, continuing `gr_demo.rs`: precession, redshift, and
//! (semiclassical) black-hole thermodynamics, all computed live from the
//! library functions in `geodesics.rs` / `semiclassical.rs` -- not
//! narrated claims. Run alongside `gr_demo.rs` for the full picture:
//! Ricci/Einstein tensors, light bending, energy/angular-momentum
//! conservation there; precession, redshift, Hawking temperature, and
//! horizon entropy here.

use spectral_dqg::geodesics::{
    schwarzschild_perihelion_precession, schwarzschild_radial_redshift_check,
};
use spectral_dqg::semiclassical::{
    bekenstein_hawking_entropy, horizon_area_schwarzschild, schwarzschild_hawking_temperature,
};

fn main() {
    println!("================================================================");
    println!(" GR demo 2: precession, redshift, black-hole thermodynamics");
    println!("================================================================\n");

    let r_s = 1.0;

    // ---- Step 1: Perihelion precession -------------------------------
    println!("---- Step 1: Perihelion precession -- converges to 3*pi*r_s/p as p/r_s -> infinity ----\n");
    println!("3*pi*r_s/p is only the LEADING weak-field term, so a single data point");
    println!("will legitimately disagree by several percent at moderate field strength.");
    println!("The real check is that the residual shrinks proportionally to r_s/p as the");
    println!("orbit widens -- the signature of a correctly-implemented higher-order term,");
    println!("not a bug.\n");
    println!(
        "{:>10} {:>12} {:>16} {:>16} {:>10} {:>14}",
        "r_min/r_s", "p/r_s", "measured", "predicted(LO)", "rel err", "err / (M/p)"
    );
    for &r_min_mult in &[20.0, 40.0, 80.0, 160.0, 320.0] {
        let r_min = r_min_mult * r_s;
        let r_max = 2.0 * r_min;
        let p = 2.0 * r_min * r_max / (r_min + r_max);
        let predicted = 3.0 * std::f64::consts::PI * r_s / p;
        let dlambda = 0.5 * (r_min_mult / 20.0).sqrt();
        match schwarzschild_perihelion_precession(r_s, r_min, r_max, dlambda, 1e-4) {
            Some(precession) => {
                let rel_err = (precession - predicted).abs() / predicted;
                let m_over_p = (r_s / 2.0) / p;
                println!(
                    "{r_min_mult:>10.0} {p:>12.2} {precession:>16.6} {predicted:>16.6} {:>9.2}% {:>14.3}",
                    rel_err * 100.0,
                    rel_err / m_over_p
                );
            }
            None => println!("{r_min_mult:>10.0}   did not converge within step budget"),
        }
    }

    // ---- Step 2: Gravitational redshift -------------------------------
    println!("\n---- Step 2: Gravitational redshift -- integrated photon geodesic vs closed form ----\n");
    let (drift, measured_ratio, r_final) =
        schwarzschild_radial_redshift_check(r_s, 5.0 * r_s, 50.0 * r_s, 0.01, 1e-4);
    let predicted_ratio =
        ((1.0 - r_s / (5.0 * r_s)) / (1.0 - r_s / r_final)).sqrt();
    println!("conserved photon energy E=f(r)u^t: fractional drift over the whole path = {drift:.3e} (expect ~0)");
    println!(
        "redshift ratio nu_obs/nu_emit at r={r_final:.3}: measured (from geodesic) = {measured_ratio:.6}, closed form sqrt(f(r_emit)/f(r_obs)) = {predicted_ratio:.6}"
    );

    // ---- Step 3: Hawking temperature -----------------------------------
    println!("\n---- Step 3: Hawking temperature (surface gravity read from the metric) ----\n");
    println!("Note: this is a semiclassical (QFT-on-curved-background) result, not pure");
    println!("GR -- this crate has no quantum sector, so what's checked is that the");
    println!("metric's horizon-adjacent derivative structure feeds the formula correctly.\n");
    println!("{:>8} {:>16} {:>16} {:>10}", "r_s", "T_H numeric", "T_H exact", "rel err");
    for &r_s_i in &[0.5, 1.0, 2.0, 5.0] {
        let t_h = schwarzschild_hawking_temperature(r_s_i, 1e-5 * r_s_i);
        let m = r_s_i / 2.0;
        let t_h_exact = 1.0 / (8.0 * std::f64::consts::PI * m);
        let rel_err = (t_h - t_h_exact).abs() / t_h_exact;
        println!("{r_s_i:>8.1} {t_h:>16.6} {t_h_exact:>16.6} {:>9.4}%", rel_err * 100.0);
    }

    // ---- Step 4: Bekenstein-Hawking entropy -----------------------------
    println!("\n---- Step 4: Bekenstein-Hawking entropy (horizon area from the metric) ----\n");
    println!("{:>8} {:>16} {:>16} {:>10}", "r_s", "S numeric", "S exact", "rel err");
    for &r_s_i in &[0.5, 1.0, 2.0, 5.0] {
        let area = horizon_area_schwarzschild(r_s_i, 400, 400);
        let s = bekenstein_hawking_entropy(area);
        let s_exact = std::f64::consts::PI * r_s_i * r_s_i;
        let rel_err = (s - s_exact).abs() / s_exact;
        println!("{r_s_i:>8.1} {s:>16.6} {s_exact:>16.6} {:>9.4}%", rel_err * 100.0);
    }

    println!("\n================================================================");
    println!(" Done. Full pass/fail assertions live in geodesics.rs and");
    println!(" semiclassical.rs unit tests (`cargo test --lib`).");
    println!("================================================================");
}
