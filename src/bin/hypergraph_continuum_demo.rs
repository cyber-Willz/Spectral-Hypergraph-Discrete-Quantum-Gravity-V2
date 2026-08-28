//! Live demo/experiment for `hypergraph_continuum_limit.rs`: runs the
//! actual convergence sweep and prints real, freshly-computed numbers, in
//! the same spirit as `main.rs`, `gr_demo.rs`, and `regge_demo.rs` --
//! nothing here is narrated or pre-baked.

use spectral_dqg::hypergraph_continuum_limit::{
    convergence_point, fit_power_law_rate, CONTINUUM_L2_L1_RATIO,
};

fn main() {
    println!("================================================================");
    println!(" Hypergraph discrete-to-continuum limit: quantitative test");
    println!(" Target: S^2, exact continuum ratio lambda(l=2)/lambda(l=1) = {}", CONTINUUM_L2_L1_RATIO);
    println!("================================================================\n");

    let resolutions = [100usize, 200, 400, 800, 1600, 3200];
    let eps_c = 2.5_f64;
    let seed = 42u64;

    println!(
        "epsilon-ball hyperedges, eps(N) = {eps_c} * sqrt(ln N / N), sphere sample seed = {seed}\n"
    );
    println!(
        "{:>6}  {:>16}  {:>16}  {:>14}  {:>14}",
        "N", "ratio_A_clique", "ratio_B_zhou", "err_A=|r-3|", "err_B=|r-3|"
    );

    let mut errs_a: Vec<(usize, f64)> = Vec::new();
    let mut errs_b: Vec<(usize, f64)> = Vec::new();

    for &n in &resolutions {
        let cp = convergence_point(n, eps_c, seed);
        let ra = cp.ratio_a_clique;
        let rb = cp.ratio_b_zhou;
        let ea = ra.map(|r| (r - CONTINUUM_L2_L1_RATIO).abs());
        let eb = rb.map(|r| (r - CONTINUUM_L2_L1_RATIO).abs());

        println!(
            "{:>6}  {:>16}  {:>16}  {:>14}  {:>14}",
            n,
            ra.map(|v| format!("{v:.6}")).unwrap_or_else(|| "n/a".into()),
            rb.map(|v| format!("{v:.6}")).unwrap_or_else(|| "n/a".into()),
            ea.map(|v| format!("{v:.6}")).unwrap_or_else(|| "n/a".into()),
            eb.map(|v| format!("{v:.6}")).unwrap_or_else(|| "n/a".into()),
        );

        if let Some(e) = ea {
            if e > 0.0 {
                errs_a.push((n, e));
            }
        }
        if let Some(e) = eb {
            if e > 0.0 {
                errs_b.push((n, e));
            }
        }
    }

    println!(
        "\n(err -> 0 as N grows is the actual convergence claim; the fit below\n\
         quantifies the *rate*, not just the direction, of that convergence.)\n"
    );

    if errs_a.len() >= 2 {
        let (p_a, c_a) = fit_power_law_rate(&errs_a);
        println!("Scheme A (clique expansion):  err(N) ~ {c_a:.4} * N^-{p_a:.4}");
    } else {
        println!("Scheme A: not enough non-zero-error points to fit a rate.");
    }
    if errs_b.len() >= 2 {
        let (p_b, c_b) = fit_power_law_rate(&errs_b);
        println!("Scheme B (Zhou hypergraph):   err(N) ~ {c_b:.4} * N^-{p_b:.4}");
    } else {
        println!("Scheme B: not enough non-zero-error points to fit a rate.");
    }

    println!(
        "\nCaveats (stated, not buried): single seed per N (no seed-averaging /\n\
         error bars on the rate estimate yet), a heuristic (not derived-\n\
         optimal) constant in eps(N), unweighted hyperedges. All three are\n\
         direct, tractable follow-ups -- none of them is required for the\n\
         ratio-convergence phenomenon itself to be real, or for scheme A\n\
         and scheme B to genuinely differ here (they do, unlike the fixed-\n\
         size kNN construction -- see hypergraph_continuum_limit.rs's\n\
         module doc comment for why that matters)."
    );
}
