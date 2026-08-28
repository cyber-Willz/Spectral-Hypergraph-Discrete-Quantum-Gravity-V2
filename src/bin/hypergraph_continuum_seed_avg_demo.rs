//! Seed-averaged version of `hypergraph_continuum_demo.rs`: the single-seed
//! run there showed visible non-monotonicity between individual N values
//! (expected sampling noise, not hidden) and reported two rate exponents
//! with no sense of how much they'd wobble under a different seed draw.
//! This binary puts real numbers on that wobble via seed-averaging plus a
//! seed-resampling bootstrap on the fitted exponent, instead of reporting
//! a single point estimate as if it had no uncertainty.
//!
//! Scope, stated up front: both schemes here use dense eigendecomposition
//! (`nalgebra::SymmetricEigen`), which is O(N^3) -- measured directly
//! before writing this sweep: ~1.1s at N=800, ~8.9s at N=1600, ~70s at
//! N=3200 for a *single* seed. Multi-seed averaging at N=3200 would cost
//! minutes per data point, so this sweep is restricted to N <= 1600 (8
//! seeds each) where that's still tractable, with a single-seed N=3200
//! spot-check reported separately rather than silently dropped. Extending
//! the seed-averaged sweep past N=1600 would need this crate's own
//! matrix-free SLQ machinery (`spectral_trace.rs`) adapted to extract a
//! few low-lying eigenvalues rather than the full-spectrum heat-trace
//! quantity it currently targets -- a real follow-up, not implemented
//! here.

use spectral_dqg::hypergraph_continuum_limit::{
    bootstrap_rate, convergence_point, seed_errors_at_n, summarize, CONTINUUM_L2_L1_RATIO,
};

fn main() {
    println!("================================================================");
    println!(" Hypergraph discrete-to-continuum limit: seed-averaged rates");
    println!(" Target: S^2, exact continuum ratio lambda(l=2)/lambda(l=1) = {}", CONTINUUM_L2_L1_RATIO);
    println!("================================================================\n");

    let resolutions = [100usize, 200, 400, 800, 1600];
    let eps_c = 2.5_f64;
    let seeds: Vec<u64> = (0..8).map(|i| 1000 + i).collect();

    println!("eps_c = {eps_c}, {} seeds per N: {:?}\n", seeds.len(), seeds);
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "N", "meanErrA", "stdErrA", "meanErrB", "stdErrB"
    );

    let mut per_n_a: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut per_n_b: Vec<(usize, Vec<f64>)> = Vec::new();

    for &n in &resolutions {
        let se = seed_errors_at_n(n, eps_c, &seeds);
        let s = summarize(&se);
        println!(
            "{:>6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  (n_a={}, n_b={})",
            s.n, s.mean_err_a, s.std_err_a, s.mean_err_b, s.std_err_b, s.n_seeds_a, s.n_seeds_b
        );
        per_n_a.push((n, se.errs_a));
        per_n_b.push((n, se.errs_b));
    }

    let (pa_mean, pa_std, pa_p5, pa_p95) = bootstrap_rate(&per_n_a, 2000, 42);
    let (pb_mean, pb_std, pb_p5, pb_p95) = bootstrap_rate(&per_n_b, 2000, 43);

    println!("\nBootstrap (2000 resamples over the 8 seeds at each N):");
    println!(
        "  Scheme A (clique):  p = {pa_mean:.4} +/- {pa_std:.4}   (5th-95th pct: {pa_p5:.4} - {pa_p95:.4})"
    );
    println!(
        "  Scheme B (Zhou):    p = {pb_mean:.4} +/- {pb_std:.4}   (5th-95th pct: {pb_p5:.4} - {pb_p95:.4})"
    );

    let overlap = pa_p5.max(pb_p5) <= pa_p95.min(pb_p95);
    println!(
        "\n90% bootstrap intervals for the two schemes' rates {}.",
        if overlap { "OVERLAP -- not distinguishable at this seed count" }
        else { "DO NOT overlap -- schemes differ at this seed count" }
    );

    println!("\nSingle-seed N=3200 spot check (not part of the rate fit above,");
    println!("dense eigendecomposition cost there is ~70s/seed -- see module scope note):");
    let cp = convergence_point(3200, eps_c, 42);
    if let (Some(ra), Some(rb)) = (cp.ratio_a_clique, cp.ratio_b_zhou) {
        println!(
            "  ratio_A={ra:.6} (err={:.6}), ratio_B={rb:.6} (err={:.6})",
            (ra - CONTINUUM_L2_L1_RATIO).abs(),
            (rb - CONTINUUM_L2_L1_RATIO).abs()
        );
    }

    println!(
        "\nCaveat carried over honestly: this bootstrap resamples *within* the\n\
         8 already-drawn seeds per N, so it quantifies sensitivity to which\n\
         of those 8 seeds got averaged, not the sensitivity to genuinely new\n\
         sphere samples beyond them (see bootstrap_rate's doc comment)."
    );
}
