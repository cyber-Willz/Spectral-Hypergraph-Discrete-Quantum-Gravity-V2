//! Matrix-free estimation of the heat trace P(t) = Tr(e^{-tL}) for N large
//! enough that dense diagonalization (`laplacian::spectrum`, O(N^3)) is not
//! an option — this is what makes an N ≥ 10^4 `d_s(t)` sweep tractable.
//!
//! Method: Stochastic Lanczos Quadrature (Hutchinson trace estimator +
//! Gauss quadrature via the Lanczos tridiagonalization), following Ubaru,
//! Chen & Saad, "Fast Estimation of tr(f(A)) via Stochastic Lanczos
//! Quadrature", SIMAX 2017. For each of `n_probes` independent Rademacher
//! vectors v (entries ±1):
//!
//!   1. Run m-step Lanczos on L starting from v/||v||, producing the
//!      tridiagonal T_m (diagonal α, off-diagonal β) -- only matvecs, no L
//!      ever materialized. The three-term recurrence + reorthogonalization
//!      itself is delegated to `krylov_ds::Lanczos` (full
//!      reorthogonalization), an independently-tested general-purpose
//!      Krylov-subspace crate, rather than hand-rolled here: this module
//!      only implements `krylov_ds::LinearOperator` for
//!      `SparseNormalizedLaplacian` (in `sparse.rs`) and the SLQ-specific
//!      logic (probe sampling, Gauss quadrature, the t-sweep reuse below).
//!   2. Eigendecompose the small (m×m) T_m = Y Θ Y^T.
//!   3. v^T f(L) v ≈ ||v||^2 · Σ_i (Y[0,i])^2 f(θ_i)   (Gauss quadrature
//!      nodes θ_i, weights (Y[0,i])^2).
//!
//! Averaging Γ_k = v_k^T f(L) v_k over probes gives an unbiased estimator
//! of Tr(f(L)) since E[v v^T] = I for Rademacher v. We use full
//! reorthogonalization in the Lanczos loop (m is small -- a few dozen
//! steps -- so this costs O(m^2 N), negligible next to the O(m·nnz)
//! matvecs) because losing orthogonality silently manufactures spurious
//! duplicate Ritz values, which is exactly the kind of bug that would
//! produce a plausible-looking but wrong d_s(t) curve.
//!
//! `heat_trace_slq`/`heat_trace_flow_slq` above are point estimates,
//! trusted by cross-checking against dense diagonalization at small N
//! (see the tests below, and `large_n_flow`'s Step A). That trust doesn't
//! extend past N ~ 3200-10^4, because that is exactly where the dense
//! cross-check stops being affordable. `heat_trace_interval_slq` replaces
//! the trust-by-analogy with a per-run certificate,
//! `certified_lower <= Tr(e^{-tL}) <= certified_upper` at a stated
//! confidence, that holds at any N -- see its own doc comment for the two
//! ingredients (Gauss/Gauss-Radau quadrature bounds + a Hoeffding/
//! empirical-Bernstein margin on the finite-probe averaging error).

use crate::sparse::SparseNormalizedLaplacian;
use krylov_ds::{Lanczos, LinearOperator, Reorthogonalization};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// One m-step Lanczos run from a fixed start vector, delegating the actual
/// three-term recurrence + reorthogonalization to `krylov_ds::Lanczos`
/// (full reorthogonalization -- the same policy the hand-rolled version
/// this replaced used, for the same reason: losing orthogonality silently
/// manufactures spurious duplicate Ritz values). `krylov_ds` normalizes
/// `start` internally, so it need not be pre-normalized here.
///
/// Returns `(alpha, beta)` with `alpha.len() == m` (or fewer, on happy
/// breakdown) and `beta.len() == alpha.len() - 1`, the standard tridiagonal
/// convention `quadrature_nodes_weights` below expects: `krylov_ds` reports
/// one extra trailing `beta` entry when it completes the full requested
/// depth without breakdown (a residual-bound quantity, not part of the
/// tridiagonal projection itself, per its own docs), which is dropped here.
fn lanczos_tridiagonal(
    l: &SparseNormalizedLaplacian,
    start: &[f64],
    m: usize,
) -> (Vec<f64>, Vec<f64>) {
    // krylov_ds errors if max_dim > n rather than silently clamping --
    // clamp here so callers (e.g. an SLQ sweep with a fixed step budget
    // run against graphs of varying size) don't have to special-case small
    // graphs themselves.
    let max_dim = m.min(l.dim()).max(1);
    let result = Lanczos::new(max_dim, 1e-12, Reorthogonalization::Full)
        .run(l, start)
        .expect("Lanczos on a Rademacher probe vector should not hit a dimension/zero-vector error");
    let alpha = result.alpha;
    let beta_len = alpha.len().saturating_sub(1);
    let beta = result.beta[..beta_len.min(result.beta.len())].to_vec();
    (alpha, beta)
}

/// Quadrature nodes θ_i and weights (Y[0,i])^2 from a tridiagonal (α, β).
fn quadrature_nodes_weights(alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let m = alpha.len();
    let mut t = DMatrix::<f64>::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = alpha[i];
    }
    for i in 0..beta.len() {
        t[(i, i + 1)] = beta[i];
        t[(i + 1, i)] = beta[i];
    }
    let eig = SymmetricEigen::new(t);
    let nodes: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    let weights: Vec<f64> = (0..m).map(|i| eig.eigenvectors[(0, i)].powi(2)).collect();
    (nodes, weights)
}

/// Gauss-Radau quadrature nodes/weights with one node pinned exactly at
/// `mu`. Standard construction (Golub & Meurant, *Matrices, Moments and
/// Quadrature*): solve `(T_{m-1} - mu I) delta = beta_{m-2}^2 e_{m-2}` on
/// the leading `(m-1)x(m-1)` block, then replace the last diagonal entry
/// with `mu + delta_{m-2}` before eigendecomposing the full `m x m`
/// tridiagonal matrix. Used below with `mu = 0`, a valid lower bound on
/// L_sym's spectrum since L_sym is PSD.
fn quadrature_nodes_weights_radau(alpha: &[f64], beta: &[f64], mu: f64) -> (Vec<f64>, Vec<f64>) {
    let m = alpha.len();
    assert!(
        m >= 2 && beta.len() == m - 1,
        "Gauss-Radau needs at least a 2-step Lanczos run"
    );
    let k = m - 1;
    let mut tk = DMatrix::<f64>::zeros(k, k);
    for i in 0..k {
        tk[(i, i)] = alpha[i] - mu;
        if i + 1 < k {
            tk[(i, i + 1)] = beta[i];
            tk[(i + 1, i)] = beta[i];
        }
    }
    let mut rhs = nalgebra::DVector::<f64>::zeros(k);
    rhs[k - 1] = beta[k - 1] * beta[k - 1];
    let delta = tk
        .lu()
        .solve(&rhs)
        .expect("Gauss-Radau: (T_{m-1} - mu*I) singular for this probe/mu");

    let mut alpha_mod = alpha.to_vec();
    alpha_mod[m - 1] = mu + delta[k - 1];
    quadrature_nodes_weights(&alpha_mod, beta)
}

/// Stochastic-Lanczos-Quadrature estimate of Tr(e^{-tL}) at a single t,
/// reusing the same Lanczos runs (recomputed for each t here for clarity;
/// see `heat_trace_flow_slq` for the version that reuses one Lanczos run
/// across an entire t-sweep, which is what you actually want in practice).
pub fn heat_trace_slq(
    l: &SparseNormalizedLaplacian,
    t: f64,
    n_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> f64 {
    let mut rng = Pcg64::seed_from_u64(seed);
    let n = l.n;
    let mut total = 0.0;
    for _ in 0..n_probes {
        let v: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        let (alpha, beta) = lanczos_tridiagonal(l, &v, lanczos_steps);
        let (nodes, weights) = quadrature_nodes_weights(&alpha, &beta);
        let v_norm_sq = n as f64; // ||v||^2 = n exactly for Rademacher entries
        let gamma: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(&theta, &w)| w * (-t * theta).exp())
            .sum::<f64>()
            * v_norm_sq;
        total += gamma;
    }
    total / n_probes as f64
}

/// The efficient version: run the Lanczos recursion (the O(m·nnz) part)
/// exactly once per probe vector, then reuse its quadrature nodes/weights
/// across the *entire* t-sweep (the exp() evaluation is O(m) per t, so this
/// is essentially free). This is what `spectral_dimension_flow_slq` uses.
pub fn heat_trace_flow_slq(
    l: &SparseNormalizedLaplacian,
    ts: &[f64],
    n_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let n = l.n;
    let mut sums = vec![0.0_f64; ts.len()];
    for _ in 0..n_probes {
        let v: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        let (alpha, beta) = lanczos_tridiagonal(l, &v, lanczos_steps);
        let (nodes, weights) = quadrature_nodes_weights(&alpha, &beta);
        let v_norm_sq = n as f64;
        for (k, &t) in ts.iter().enumerate() {
            let gamma: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(&theta, &w)| w * (-t * theta).exp())
                .sum::<f64>()
                * v_norm_sq;
            sums[k] += gamma;
        }
    }
    sums.iter().map(|&s| s / n_probes as f64).collect()
}

/// d_s(t) via centered log-log finite difference, fed by SLQ-estimated
/// P(t) instead of an exact eigendecomposition. Mirrors
/// `heat_kernel::spectral_dimension_flow`'s estimator exactly so the two
/// are numerically comparable on graphs small enough to run both.
pub fn spectral_dimension_flow_slq(
    l: &SparseNormalizedLaplacian,
    t_min: f64,
    t_max: f64,
    n_samples: usize,
    n_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Vec<crate::heat_kernel::SpectralDimensionPoint> {
    assert!(n_samples >= 3);
    let log_min = t_min.ln();
    let log_max = t_max.ln();
    let ts: Vec<f64> = (0..n_samples)
        .map(|i| {
            let frac = i as f64 / (n_samples as f64 - 1.0);
            (log_min + frac * (log_max - log_min)).exp()
        })
        .collect();
    let ps = heat_trace_flow_slq(l, &ts, n_probes, lanczos_steps, seed);

    let mut out = Vec::with_capacity(n_samples - 2);
    for i in 1..n_samples - 1 {
        let d_ln_p = ps[i + 1].ln() - ps[i - 1].ln();
        let d_ln_t = ts[i + 1].ln() - ts[i - 1].ln();
        let d_s = -2.0 * d_ln_p / d_ln_t;
        out.push(crate::heat_kernel::SpectralDimensionPoint {
            t: ts[i],
            p_t: ps[i],
            d_s,
        });
    }
    out
}

/// Certified interval estimate of `Tr(e^{-tL})`: `certified_lower <=
/// Tr(e^{-tL}) <= certified_upper` with probability >= `confidence`.
///
/// This replaces the "trust `heat_trace_slq` by analogy, because it agreed
/// with dense diagonalization at small N" pattern -- which stops being
/// checkable exactly where dense diagonalization's O(N^3) wall is (see
/// `sparse.rs` module docs, and the `large_n_flow` bin's Step A/B split) --
/// with a per-run mathematical certificate that holds at any N, including
/// N far past where a dense cross-check could ever run.
///
/// Two ingredients:
///
/// 1. `f(x) = e^{-tx}` is completely monotone on `[0, ∞)`, so per-probe
///    Gauss quadrature (`quadrature_nodes_weights`) is a guaranteed
///    *lower* bound on `v^T f(L) v`, and Gauss-Radau quadrature pinned at
///    `mu = 0` (`quadrature_nodes_weights_radau`) -- valid since L_sym is
///    PSD -- is a guaranteed *upper* bound. (Golub & Meurant; Bai, Fahey &
///    Golub, "Some Large-Scale Matrix Computation Problems", 1996.) No
///    deterministic truncation error, for any N or m.
/// 2. The only remaining uncertainty is finite-probe averaging error in
///    the Hutchinson estimator. Bounded by the tighter of a Hoeffding
///    margin (worst-case range `[0, N]`, always valid) and an
///    empirical-Bernstein margin (observed probe-to-probe variance,
///    tighter when the probes actually agree with each other), each
///    computed at confidence `1 - delta/2` so their combination via `min`
///    is still valid at the overall requested `confidence` (Bonferroni).
///
/// Cost: `O(n_probes * lanczos_steps * nnz(L))`, independent of N^3 -- the
/// Lanczos work here is identical to `heat_trace_slq`'s; this only adds
/// one extra small (m x m) tridiagonal solve + eigendecomposition per
/// probe for the Radau side.
pub struct HeatTraceInterval {
    /// Mean of per-probe Gauss (lower) quadrature bounds.
    pub lower_mean: f64,
    /// Mean of per-probe Gauss-Radau (upper) quadrature bounds.
    pub upper_mean: f64,
    /// Margin added outward from `[lower_mean, upper_mean]` to account for
    /// finite-probe averaging error, at `confidence`.
    pub margin: f64,
    pub confidence: f64,
    /// `(lower_mean - margin).max(0.0)` -- the trace is never negative.
    pub certified_lower: f64,
    pub certified_upper: f64,
}

impl HeatTraceInterval {
    pub fn point_estimate(&self) -> f64 {
        0.5 * (self.lower_mean + self.upper_mean)
    }
    pub fn width(&self) -> f64 {
        self.certified_upper - self.certified_lower
    }
}

pub fn heat_trace_interval_slq(
    l: &SparseNormalizedLaplacian,
    t: f64,
    n_probes: usize,
    lanczos_steps: usize,
    confidence: f64,
    seed: u64,
) -> HeatTraceInterval {
    assert!(
        (0.0..1.0).contains(&confidence),
        "confidence must be in [0, 1)"
    );
    let mut rng = Pcg64::seed_from_u64(seed);
    let n = l.n;

    let mut per_probe_lower = Vec::with_capacity(n_probes);
    let mut per_probe_upper = Vec::with_capacity(n_probes);

    for _ in 0..n_probes {
        let v: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        let v_norm_sq = n as f64; // exact for Rademacher entries

        let (alpha, beta) = lanczos_tridiagonal(l, &v, lanczos_steps);
        if alpha.len() < 2 {
            // Early Lanczos breakdown (tiny n, or v landed in a small
            // invariant subspace): Gauss quadrature is already exact on
            // that subspace, so record lower == upper rather than an
            // artificially widened bracket.
            let (nodes, weights) = quadrature_nodes_weights(&alpha, &beta);
            let est: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(&th, &w)| w * (-t * th).exp())
                .sum::<f64>()
                * v_norm_sq;
            per_probe_lower.push(est);
            per_probe_upper.push(est);
            continue;
        }

        let (g_nodes, g_weights) = quadrature_nodes_weights(&alpha, &beta);
        let lower: f64 = g_nodes
            .iter()
            .zip(&g_weights)
            .map(|(&th, &w)| w * (-t * th).exp())
            .sum::<f64>()
            * v_norm_sq;

        let (r_nodes, r_weights) = quadrature_nodes_weights_radau(&alpha, &beta, 0.0);
        let upper: f64 = r_nodes
            .iter()
            .zip(&r_weights)
            .map(|(&th, &w)| w * (-t * th).exp())
            .sum::<f64>()
            * v_norm_sq;

        // Theory guarantees lower <= upper; guard only against float noise
        // on near-degenerate spectra rather than asserting and aborting.
        let (lo, hi) = if lower <= upper { (lower, upper) } else { (upper, lower) };
        per_probe_lower.push(lo);
        per_probe_upper.push(hi);
    }

    let np = n_probes as f64;
    let lower_mean = per_probe_lower.iter().sum::<f64>() / np;
    let upper_mean = per_probe_upper.iter().sum::<f64>() / np;

    let range = n as f64;
    let delta = 1.0 - confidence;
    let delta_half = delta / 2.0;

    // (a) Hoeffding, worst-case range [0, N]: each raw probe contribution
    // v_i^T f(L) v_i lies in [0, N] since 0 <= e^{-t*lambda} <= 1 for
    // lambda >= 0, t >= 0, and ||v_i||^2 = N. Always valid, but loose
    // whenever the true spread of probe values is far below N.
    let hoeffding_margin = range * ((2.0 / delta_half).ln() / (2.0 * np)).sqrt();

    // (b) Empirical Bernstein (Maurer & Pontil 2009) on the per-probe
    // bracket midpoints, using the *observed* variance across probes.
    // Since the midpoint is only a proxy for the (unobservable) exact
    // per-probe value, we pay for that with the largest per-probe bracket
    // half-width -- still valid, just possibly looser than (a) when
    // individual brackets are wide (e.g. lanczos_steps too small).
    let midpoints: Vec<f64> = per_probe_lower
        .iter()
        .zip(per_probe_upper.iter())
        .map(|(&lo, &hi)| 0.5 * (lo + hi))
        .collect();
    let mean_mid = midpoints.iter().sum::<f64>() / np;
    let max_half_width = per_probe_lower
        .iter()
        .zip(per_probe_upper.iter())
        .map(|(&lo, &hi)| 0.5 * (hi - lo))
        .fold(0.0_f64, f64::max);
    let bernstein_margin = if np > 1.0 {
        let sample_var = midpoints.iter().map(|&x| (x - mean_mid).powi(2)).sum::<f64>() / (np - 1.0);
        (2.0 * sample_var * (2.0 / delta_half).ln() / np).sqrt()
            + 7.0 * range * (2.0 / delta_half).ln() / (3.0 * (np - 1.0))
            + max_half_width
    } else {
        range
    };

    let margin = hoeffding_margin.min(bernstein_margin);

    HeatTraceInterval {
        lower_mean,
        upper_mean,
        margin,
        confidence,
        certified_lower: (lower_mean - margin).max(0.0),
        certified_upper: upper_mean + margin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::heat_trace as exact_heat_trace;
    use crate::hypergraph::Hypergraph;
    use crate::laplacian::spectrum;

    /// Cross-check: on a graph small enough to also diagonalize exactly,
    /// SLQ's estimate of P(t) must agree with the exact Σ e^{-tλ} to a few
    /// percent. This is the same "don't trust a new numerical method
    /// without checking it against a known-good one on a case you can
    /// afford to brute-force" discipline the rest of this crate uses.
    #[test]
    fn slq_matches_exact_heat_trace_on_small_graph() {
        // A moderately irregular graph: two triangles joined by a bridge,
        // plus a couple of extra chords, N = 12.
        let mut hg = Hypergraph::new(12);
        let triangle_edges = [
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
            (6, 7), (7, 8), (8, 6),
            (9, 10), (10, 11), (11, 9),
        ];
        for &(a, b) in &triangle_edges {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        // bridges connecting the four triangles into one component, plus
        // a couple of chords to break any residual symmetry
        for &(a, b) in &[(2, 3), (5, 6), (8, 9), (0, 7), (4, 10)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();

        let exact = spectrum(&g, true);
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);

        for &t in &[0.05, 0.3, 1.0, 3.0, 10.0] {
            let p_exact = exact_heat_trace(&exact.eigenvalues, t);
            // generous probe/step budget since N=12 is tiny -- the point
            // here is correctness of the method, not its N=10^4 economy
            let p_slq = heat_trace_slq(&sparse_l, t, 200, 12, 7);
            let rel_err = (p_exact - p_slq).abs() / p_exact;
            assert!(
                rel_err < 0.03,
                "t={t}: exact P(t)={p_exact}, SLQ P(t)={p_slq}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn lanczos_steps_ge_n_recovers_exact_trace_deterministically() {
        // If lanczos_steps >= N, the Krylov space spans all of R^N (barring
        // degeneracy), so a *single* Rademacher probe already gives an
        // essentially exact quadrature for a generic vector -- this
        // isolates *quadrature exactness* from *Hutchinson sampling
        // variance*: with a full-rank Krylov space each individual probe's
        // v^T f(L) v is exact, but averaging finitely many Rademacher
        // probes still carries genuine Monte Carlo variance (Var[v^T A v]
        // = 2 sum_{i != j} A_ij^2 for Rademacher v), which is why the
        // tolerance below isn't machine epsilon even though the
        // quadrature step itself is exact here.
        let mut hg = Hypergraph::new(9);
        for &(a, b) in &[(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3),(5,6),(6,7),(7,8),(8,6)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let exact = spectrum(&g, true);
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);

        let t = 1.0;
        let p_exact = exact_heat_trace(&exact.eigenvalues, t);
        let p_slq = heat_trace_slq(&sparse_l, t, 4000, 9, 3);
        let rel_err = (p_exact - p_slq).abs() / p_exact;
        assert!(rel_err < 0.02, "rel_err={rel_err}");
    }

    /// Cross-check for `heat_trace_interval_slq`: on a graph small enough
    /// to diagonalize exactly, the certified interval must actually
    /// contain the exact P(t) -- not just be close to it, contain it,
    /// with the stated confidence's worth of margin.
    #[test]
    fn interval_slq_certificate_contains_exact_heat_trace() {
        let mut hg = Hypergraph::new(14);
        let edges = [
            (0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3),
            (5, 6), (6, 7), (7, 8), (8, 6), (8, 9), (9, 10), (10, 11),
            (11, 9), (11, 12), (12, 13), (13, 0),
        ];
        for &(a, b) in &edges {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let exact = spectrum(&g, true);
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);

        for &t in &[0.1, 1.0, 5.0] {
            let p_exact = exact_heat_trace(&exact.eigenvalues, t);
            let result = heat_trace_interval_slq(&sparse_l, t, 60, 13, 0.99, 11);
            assert!(
                result.certified_lower <= p_exact + 1e-8 && p_exact <= result.certified_upper + 1e-8,
                "t={t}: certified interval [{}, {}] must contain exact P(t)={p_exact}",
                result.certified_lower,
                result.certified_upper
            );
        }
    }

    /// The Gauss/Gauss-Radau brackets must hold *per probe*, not just on
    /// average -- checked directly against the exact quadratic form
    /// v^T e^{-tL} v computed via dense diagonalization.
    #[test]
    fn gauss_lower_and_radau_upper_bounds_hold_per_probe() {
        use crate::laplacian::normalized_laplacian;

        let mut hg = Hypergraph::new(10);
        for &(a, b) in &[(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(5,6),(6,7),(7,8),(8,9),(9,5)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let dense_l = normalized_laplacian(&g);
        let dense_eig = nalgebra::SymmetricEigen::new(dense_l);
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);
        let t = 0.7;

        let mut rng = Pcg64::seed_from_u64(3);
        for _ in 0..15 {
            let v: Vec<f64> = (0..10).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }).collect();
            let vv = nalgebra::DVector::from_vec(v.clone());
            let coords = dense_eig.eigenvectors.transpose() * &vv;
            let exact_form: f64 = dense_eig
                .eigenvalues
                .iter()
                .zip(coords.iter())
                .map(|(&lam, &c)| c * c * (-t * lam).exp())
                .sum();

            let (alpha, beta) = lanczos_tridiagonal(&sparse_l, &v, 8);
            let (gn, gw) = quadrature_nodes_weights(&alpha, &beta);
            let lower: f64 = gn.iter().zip(&gw).map(|(&th, &w)| w * (-t * th).exp()).sum::<f64>() * 10.0;
            let (rn, rw) = quadrature_nodes_weights_radau(&alpha, &beta, 0.0);
            let upper: f64 = rn.iter().zip(&rw).map(|(&th, &w)| w * (-t * th).exp()).sum::<f64>() * 10.0;

            assert!(lower <= exact_form + 1e-8, "Gauss must lower-bound: {lower} > {exact_form}");
            assert!(upper >= exact_form - 1e-8, "Radau must upper-bound: {upper} < {exact_form}");
        }
    }

    /// Calibration check: run the certificate many times independently at
    /// a stated confidence and confirm it's violated no more often than
    /// that allows, with slack for the randomness of a finite number of
    /// trials. This is what distinguishes "computes something called a
    /// confidence interval" from "the confidence interval is calibrated" --
    /// it would catch a margin formula that's silently backwards or
    /// missing a factor, which the single-run bracket-containment test
    /// above would not reliably catch.
    #[test]
    fn interval_slq_certificate_is_not_violated_more_often_than_stated() {
        let mut hg = Hypergraph::new(11);
        for &(a, b) in &[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,0),(0,5),(2,8)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let exact = spectrum(&g, true);
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);
        let t = 1.0;
        let p_exact = exact_heat_trace(&exact.eigenvalues, t);

        let confidence = 0.90; // deliberately loose so violations are observable
        let trials = 300;
        let mut violations = 0;
        for seed in 0..trials {
            let r = heat_trace_interval_slq(&sparse_l, t, 10, 9, confidence, seed as u64);
            if p_exact < r.certified_lower || p_exact > r.certified_upper {
                violations += 1;
            }
        }
        let observed_rate = violations as f64 / trials as f64;
        let nominal_rate = 1.0 - confidence;
        // Generous slack: this is a stochastic check on a stochastic
        // guarantee, meant to catch a badly wrong margin, not to certify
        // calibration to high precision.
        assert!(
            observed_rate <= nominal_rate * 3.0 + 0.05,
            "certificate violated {violations}/{trials} ({observed_rate:.3}) times, \
             far more than the nominal {nominal_rate:.3} failure rate"
        );
    }

    /// More probes should shrink the certified interval (the Hoeffding/
    /// Bernstein margin shrinks with n_probes), all else equal.
    #[test]
    fn more_probes_shrink_the_interval_slq_certificate() {
        let mut hg = Hypergraph::new(10);
        for &(a, b) in &[(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(5,6),(6,7),(7,8),(8,9),(9,5)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);
        let t = 0.8;

        let loose = heat_trace_interval_slq(&sparse_l, t, 5, 8, 0.99, 5);
        let tight = heat_trace_interval_slq(&sparse_l, t, 150, 8, 0.99, 5);
        assert!(
            tight.width() < loose.width(),
            "more probes should shrink the certificate: {} vs {}",
            tight.width(),
            loose.width()
        );
    }
}
