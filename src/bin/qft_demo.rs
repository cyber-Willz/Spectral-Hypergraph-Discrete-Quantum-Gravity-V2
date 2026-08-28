//! Live demo of the new QFT modules: `qft_zeta.rs`, `casimir.rs`,
//! `seeley_dewitt.rs`. Prints real numbers, cross-validated against
//! known closed forms and a real experiment.

use spectral_dqg::casimir::{energy_per_area, force_sphere_plate_pfa};
use spectral_dqg::qft_zeta::{zeta_gt1, zeta_negative_integer};
use spectral_dqg::seeley_dewitt::seeley_dewitt_a1_residual;

const HBAR: f64 = 1.054_571_817e-34;
const C: f64 = 299_792_458.0;

fn main() {
    println!("================================================================");
    println!(" QFT tests: zeta regularization, Casimir effect, Seeley-DeWitt");
    println!("================================================================\n");

    println!("---- Step 1: Riemann zeta, computed (not hardcoded), positive and continued ----\n");
    let z4 = zeta_gt1(4.0, 2000);
    let z2 = zeta_gt1(2.0, 20_000);
    let zm1 = zeta_negative_integer(-1, 20_000);
    let zm3 = zeta_negative_integer(-3, 2000);
    println!("zeta(2)  = {z2:.10}   exact pi^2/6  = {:.10}", std::f64::consts::PI.powi(2) / 6.0);
    println!("zeta(4)  = {z4:.10}   exact pi^4/90 = {:.10}", std::f64::consts::PI.powi(4) / 90.0);
    println!("zeta(-1) = {zm1:.10}   exact -1/12   = {:.10}", -1.0 / 12.0);
    println!("zeta(-3) = {zm3:.10}   exact 1/120   = {:.10}", 1.0 / 120.0);

    println!("\n---- Step 2: Casimir effect, derived from zeta(-3), not the closed form ----\n");
    let a = 1e-6;
    let e = energy_per_area(HBAR, C, a, zm3);
    let closed_form = -std::f64::consts::PI.powi(2) * HBAR * C / (720.0 * a.powi(3));
    println!("E/A at a=1um: derived from zeta(-3) = {e:.6e} J/m^2");
    println!("              standard closed form  = {closed_form:.6e} J/m^2");

    println!("\n---- Step 3: cross-check against Mohideen & Roy 1998 (PRL 81, 4549) ----\n");
    println!("AFM measurement: gold-coated sphere (diameter 196um, r=98um) vs flat plate,");
    println!("separations 0.1-0.9um, forces of order 1-300pN, RMS deviation from full");
    println!("theory (finite conductivity + roughness + thermal) of 1.6pN. This is the");
    println!("T=0/perfect-conductor idealization only -- expected to run somewhat HIGH");
    println!("of the real measurement at the smallest separations.\n");
    println!("{:>10} {:>16}", "a (nm)", "F_ideal (pN)");
    let r = 98e-6;
    for &a_nm in &[100.0, 200.0, 300.0, 500.0, 900.0] {
        let a_m = a_nm * 1e-9;
        let f = force_sphere_plate_pfa(HBAR, C, r, a_m, zm3).abs();
        println!("{a_nm:>10.0} {:>16.3}", f * 1e12);
    }

    println!("\n---- Step 4: Seeley-DeWitt a_1 coefficient, via the crate's OWN heat_kernel ----\n");
    println!("Tr(e^-tD) ~ Area/(4*pi*t) + chi/6 + O(t) on the round sphere (chi=2).");
    println!("This reuses heat_kernel::heat_trace() -- the same function already used");
    println!("elsewhere in this crate for spectral-dimension flow -- on the sphere's");
    println!("exact Laplacian spectrum.\n");
    println!("{:>10} {:>14}", "t", "residual");
    for &t in &[0.05, 0.02, 0.01, 0.005, 0.002, 0.001] {
        let residual = seeley_dewitt_a1_residual(1.0, 2000, t);
        println!("{t:>10.4} {residual:>14.8}   (expect -> chi/6 = 0.33333333)");
    }

    println!("\n================================================================");
    println!(" What this doesn't do: finite-conductivity/roughness/thermal");
    println!(" Casimir corrections, and Seeley-DeWitt on an arbitrary");
    println!(" spectral_dqg metric (only the sphere's known closed-form");
    println!(" spectrum is used here). Both are real next steps.");
    println!("================================================================");
}
