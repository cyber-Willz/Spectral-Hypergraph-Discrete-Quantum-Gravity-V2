//! Live demo of `gme_bmv.rs`: the most realistic laboratory route to
//! proving gravity is quantized. Prints real numbers, cross-validated
//! against the published Bose et al. 2017 result.

use spectral_dqg::gme_bmv::{four_branch_phases, leading_order_relative_phase};

const G: f64 = 6.674_30e-11;
const HBAR: f64 = 1.054_571_817e-34;

fn main() {
    println!("================================================================");
    println!(" GME/BMV: gravity-mediated entanglement phase calculation");
    println!("================================================================\n");

    println!("---- Step 1: convergence of the entangling cross-term to the ----");
    println!("---- standard small-dx literature formula, as dx/d -> 0      ----\n");
    let m1 = 1e-14;
    let m2 = 1e-14;
    let d = 450e-6;
    let t = 2.5;
    println!(
        "{:>10} {:>16} {:>16} {:>12}",
        "dx (m)", "cross term", "-2*lo formula", "ratio"
    );
    for &dx in &[100e-6, 30e-6, 10e-6, 3e-6, 1e-6] {
        let branches = four_branch_phases(G, m1, m2, d, dx, dx, t, HBAR);
        let cross_term = branches.ll + branches.rr - branches.lr - branches.rl;
        let lo = leading_order_relative_phase(G, m1, m2, dx, dx, d, t, HBAR);
        let predicted = -2.0 * lo;
        println!(
            "{dx:>10.1e} {cross_term:>16.6e} {predicted:>16.6e} {:>12.6}",
            cross_term / predicted
        );
    }

    println!("\n---- Step 2: cross-check against Bose et al. 2017 (PRL 119, 240401) ----\n");
    println!("Published (their SG free-fall step): m=1e-14 kg, d=450 micron,");
    println!("dx=250 micron, tau~2.5s give Delta phi_LR ~ -0.2, Delta phi_RL ~ +0.7");
    println!("(relative to the LL=RR baseline). This model has no Stern-Gerlach");
    println!("acceleration-phase contribution, so matching SIGN and ORDER OF");
    println!("MAGNITUDE is the honest bar, not exact reproduction.\n");

    let dx = 250e-6;
    let branches = four_branch_phases(G, m1, m2, d, dx, dx, t, HBAR);
    let delta_lr = branches.lr - branches.ll;
    let delta_rl = branches.rl - branches.ll;
    println!("phi_LL = phi_RR = {:.6}  (exact equality by construction)", branches.ll);
    println!("Delta phi_LR = {delta_lr:.6}   (published: -0.2)");
    println!("Delta phi_RL = {delta_rl:.6}   (published: +0.7)");

    println!("\n================================================================");
    println!(" What this crate does NOT (yet) compute: the actual spin-");
    println!(" entanglement witness W (needs the full Stern-Gerlach");
    println!(" recombination-step formula), and decoherence timescales");
    println!(" (Casimir-Polder, blackbody, residual gas) that determine");
    println!(" whether the required coherence time is achievable. Both are");
    println!(" real next steps, left undone rather than approximated.");
    println!("================================================================");
}
