//! Distinction item: certified interval P(t) estimation past the dense-
//! diagonalization wall.
//!
//! `heat_trace_slq` (point estimate) is trusted only by analogy: it agreed
//! with dense diagonalization at N=2000 in `large_n_flow`'s Step A, and
//! that agreement is the entire basis for trusting it at N=10,000 in Step
//! B, where dense diagonalization no longer runs. `heat_trace_interval_slq`
//! replaces that analogy with a per-run mathematical certificate that
//! holds at any N: `certified_lower <= Tr(e^{-tL}) <= certified_upper`
//! with probability >= `confidence`, built from Gauss/Gauss-Radau
//! quadrature bounds (exact, not sampled) plus a Hoeffding/empirical-
//! Bernstein margin on the one genuinely stochastic piece (finite-probe
//! averaging).
//!
//! Same honesty structure as `large_n_flow`: Step A checks the certificate
//! actually contains the dense answer at a size we can still afford to
//! diagonalize; Step B runs it alone at a size dense diagonalization
//! cannot reach in reasonable time.

use spectral_dqg::continuum_limit::random_simple_regular_graph;
use spectral_dqg::heat_kernel::heat_trace as exact_heat_trace;
use spectral_dqg::laplacian::spectrum;
use spectral_dqg::sparse::SparseNormalizedLaplacian;
use spectral_dqg::spectral_trace::{heat_trace_interval_slq, heat_trace_slq};
use std::time::Instant;

fn main() {
    let d = 4usize;
    let t = 1.0;
    let confidence = 0.99;

    // --- Step A: does the certificate actually contain the exact answer,
    // at a size we can still afford to check? ---
    println!("=== Step A: certified interval vs exact dense P(t), N=2000, d={d}, t={t} ===");
    let n_check = 2000usize;
    let g_check = random_simple_regular_graph(n_check, d, 1, 20000);

    let t0 = Instant::now();
    let exact = spectrum(&g_check, true);
    let exact_time = t0.elapsed();
    let p_exact = exact_heat_trace(&exact.eigenvalues, t);

    let sparse_check = SparseNormalizedLaplacian::from_graph(&g_check);

    let t0 = Instant::now();
    let point = heat_trace_slq(&sparse_check, t, 60, 60, 11);
    let point_time = t0.elapsed();

    let t0 = Instant::now();
    let interval = heat_trace_interval_slq(&sparse_check, t, 60, 60, confidence, 11);
    let interval_time = t0.elapsed();

    println!("dense diagonalization : {exact_time:?}   exact P(t) = {p_exact:.4}");
    println!("point-estimate SLQ    : {point_time:?}   P(t) ~= {point:.4}  (no guarantee it brackets exact)");
    println!(
        "certified interval SLQ: {interval_time:?}   [{:.4}, {:.4}]  (width {:.4}, {}% confidence)",
        interval.certified_lower,
        interval.certified_upper,
        interval.width(),
        (confidence * 100.0) as u32
    );
    let contains = interval.certified_lower <= p_exact && p_exact <= interval.certified_upper;
    println!(
        "certificate contains exact dense answer: {}",
        if contains { "YES" } else { "NO -- investigate before trusting Step B" }
    );
    if !contains {
        eprintln!("Certificate failed to contain the dense reference at N={n_check}; aborting rather than proceeding to an unchecked large-N claim.");
        std::process::exit(1);
    }

    // --- Step B: the actual distinction-level result. N=10,000, no dense
    // diagonalization attempted at all -- matrix-free throughout, with a
    // certificate instead of a trust-by-analogy point estimate. ---
    println!("\n=== Step B: N=10,000, d={d}, t={t}, matrix-free certified estimate only ===");
    let n_big = 10_000usize;
    let t0 = Instant::now();
    let g_big = random_simple_regular_graph(n_big, d, 2, 20000);
    let build_time = t0.elapsed();
    let sparse_big = SparseNormalizedLaplacian::from_graph(&g_big);
    println!(
        "graph build: {:?}, N={}, nnz(L)={}",
        build_time,
        sparse_big.n,
        sparse_big.nnz()
    );

    let t0 = Instant::now();
    let interval_big = heat_trace_interval_slq(&sparse_big, t, 150, 80, confidence, 21);
    let big_time = t0.elapsed();

    println!("certified interval (150 probes, 80 Lanczos steps): {big_time:?}");
    println!(
        "P(t={t}) in [{:.4}, {:.4}]  (point estimate {:.4}, width {:.4}, {}% confidence)",
        interval_big.certified_lower,
        interval_big.certified_upper,
        interval_big.point_estimate(),
        interval_big.width(),
        (confidence * 100.0) as u32
    );
    println!(
        "\nFor comparison: dense diagonalization at N=2000 alone took {exact_time:?} (O(N^3)); \
         at N=10,000 it would be roughly {:.0}x that cost, and O(N^2) memory \
         (~{:.1} GB just for the dense matrix). This run touched only sparse matvecs \
         and produced a certificate, not just a number to trust by analogy.",
        (n_big as f64 / n_check as f64).powi(3),
        (n_big * n_big * 8) as f64 / 1e9
    );
}
