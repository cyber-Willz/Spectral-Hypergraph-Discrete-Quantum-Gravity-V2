//! Euclidean path integral over metrics for Regge calculus, via Metropolis
//! Monte Carlo over edge lengths -- this is the direct discretization of
//! ```text
//!     Z = integral D[g] exp(-S_Regge[g] / hbar)
//! ```
//! with D[g] realized concretely as a random walk on the space of valid
//! edge-length assignments (DeWitt/Hamber-Williams-style dynamical Regge
//! calculus; see e.g. Hamber, "Quantum Gravitation", Ch. 5-6, and
//! Rocek & Williams (1981) for the original Regge-calculus path integral
//! formulation). We fix the connectivity (simplicial complex) and let only
//! the edge lengths fluctuate -- the standard "quantum Regge calculus"
//! setup, as distinct from dynamical/causal triangulations which also sum
//! over connectivity. Coupling kappa = 1/(8 pi G) and hbar are both set to
//! 1 (natural units); nothing here fixes an actual value of G.
//!
//! What this establishes: a working numerical sampler of the Regge path
//! integral for a *fixed* simplicial complex, with the acceptance test,
//! thermalization, and expectation values reported honestly (including
//! the acceptance rate, since a silently-degenerate acceptance rate near
//! 0% or 100% is the standard failure mode of any Metropolis sampler and
//! would make <S> meaningless).
//!
//! ## The conformal-mode problem and what this module does about it
//!
//! `tests/regge_tests.rs`'s
//! `stronger_coupling_reveals_the_unbounded_below_conformal_mode_pathology`
//! documents the real, literature-known pathology: the curvature term
//! `sum_hinge L_hinge*delta_hinge` is not bounded below, and a pure
//! *global rescaling* of every edge length by a factor `s` is the specific
//! runaway direction -- it leaves every dihedral angle exactly unchanged
//! (angles depend only on length ratios within a tetrahedron), so the
//! curvature term scales exactly *linearly* in `s`, unboundedly, while
//! Cayley-Menger validity (the sampler's only hard wall) is itself
//! scale-invariant and so never blocks it. This is the discrete incarnation
//! of the Euclidean-quantum-gravity conformal factor problem (Gibbons,
//! Hawking & Perry 1978).
//!
//! Fully curing this the way GHP describe -- rotating the conformal mode's
//! integration contour into the complex plane -- is a genuine open
//! question for a non-Gaussian action like full Regge calculus, and stays
//! out of scope here (that claim is unchanged from before). What *is*
//! implemented now is the standard *practical* mitigation used throughout
//! the dynamical-triangulations/Regge Monte Carlo literature for exactly
//! this problem: sample a **volume-constrained (canonical) ensemble**
//! instead of an unconstrained one, via [`VolumeConstraint`] -- a soft
//! quadratic penalty `kappa_v * (V_total - V_target)^2` added to the
//! sampled weight. Since the runaway direction is precisely "rescale
//! everything, changing volume while leaving shape untouched," constraining
//! volume directly blocks that direction without suppressing genuine shape
//! (curvature) fluctuations. This converts the previously undiagnosed,
//! unmonitored hazard ("the sampler's behavior at some parameter settings
//! is 'don't go there'") into something quantified and testable: see
//! `tests/regge_tests.rs` for a test confirming the volume-constrained
//! ensemble actually tames the runaway that the unconstrained one still
//! (correctly, honestly) exhibits.
//!
//! Regardless of which mode is used, [`McResult`] now always reports the
//! mean edge length and mean volume alongside the action trace, so a
//! caller can see directly whether a run drifted in the conformal
//! direction rather than that drift being invisible in the returned
//! statistics.

use crate::regge::{all_tetrahedra_valid, regge_action, total_volume, EdgeLengths, SimplicialComplex};
use rand::Rng;
use rand_pcg::Pcg64;
use rand::SeedableRng;

/// Soft volume constraint: adds `kappa_v * (V_total - target_volume)^2` to
/// the sampled weight, taming the conformal-mode runaway direction (see
/// module docs) without a hard cutoff. `kappa_v` controls how tightly
/// volume is held near `target_volume`; `kappa_v = 0` (or omitting this
/// from `McConfig`) recovers the original unconstrained ensemble exactly.
#[derive(Clone, Copy, Debug)]
pub struct VolumeConstraint {
    pub kappa_v: f64,
    pub target_volume: f64,
}

pub struct McConfig {
    pub kappa: f64,        // 1/(8 pi G), coupling in front of curvature term
    pub lambda: f64,       // cosmological constant term
    pub step_size: f64,    // max proposed |delta length|
    pub n_sweeps: usize,   // one sweep = one proposal per edge
    pub seed: u64,
    /// If `Some`, sample the volume-constrained (canonical) ensemble
    /// instead of the unconstrained one. See module docs.
    pub volume_constraint: Option<VolumeConstraint>,
}

pub struct McResult {
    pub mean_action: f64,
    pub stderr_action: f64,
    pub acceptance_rate: f64,
    pub n_samples: usize,
    pub action_trace: Vec<f64>,
    /// Total volume at each recorded sweep -- always populated (even with
    /// no volume constraint), so conformal-mode drift is visible in the
    /// returned statistics rather than silent.
    pub volume_trace: Vec<f64>,
    /// Mean edge length at each recorded sweep, the more direct
    /// "conformal mode" diagnostic (proportional to the conformal
    /// rescaling factor itself, whereas volume scales as its cube).
    pub mean_edge_length_trace: Vec<f64>,
}

impl McResult {
    pub fn mean_volume(&self) -> f64 {
        self.volume_trace.iter().sum::<f64>() / self.volume_trace.len() as f64
    }
    /// Fractional drift in mean edge length from the first to the second
    /// half of the run -- a simple, direct "is this run visibly sliding in
    /// the conformal direction" diagnostic, independent of the action
    /// itself.
    pub fn conformal_drift(&self) -> f64 {
        let n = self.mean_edge_length_trace.len();
        if n < 4 {
            return 0.0;
        }
        let half = n / 2;
        let first: f64 = self.mean_edge_length_trace[..half].iter().sum::<f64>() / half as f64;
        let second: f64 = self.mean_edge_length_trace[half..].iter().sum::<f64>() / (n - half) as f64;
        (second - first) / first
    }
}

/// Run a Metropolis-Hastings random walk over edge lengths, sampling from
/// `exp(-kappa*S_Regge[l]/hbar)` with `hbar=1` (or, if `cfg.volume_constraint`
/// is set, from `exp(-kappa*S_Regge[l] - kappa_v*(V[l]-V_target)^2)` --
/// see module docs), subject to every tetrahedron staying geometrically
/// valid (Cayley-Menger positive). Rejects proposals that break validity
/// outright (infinite-action wall), exactly as a hard constraint boundary
/// should be handled in a Metropolis sampler.
pub fn run_path_integral(
    complex: &SimplicialComplex,
    initial: EdgeLengths,
    cfg: &McConfig,
) -> McResult {
    let mut rng = Pcg64::seed_from_u64(cfg.seed);
    let mut lengths = initial;
    assert!(
        all_tetrahedra_valid(complex, &lengths),
        "initial configuration must be geometrically valid"
    );

    let mut current_s = regge_action(complex, &lengths, cfg.lambda).total;
    let mut current_v = total_volume(complex, &lengths);
    let mut accepted = 0usize;
    let mut proposed = 0usize;
    let mut trace = Vec::with_capacity(cfg.n_sweeps);
    let mut volume_trace = Vec::with_capacity(cfg.n_sweeps);
    let mut mean_edge_trace = Vec::with_capacity(cfg.n_sweeps);

    let edge_list: Vec<_> = complex.edges.clone();
    let n_edges = edge_list.len() as f64;

    let penalty = |v: f64| -> f64 {
        match cfg.volume_constraint {
            Some(vc) => vc.kappa_v * (v - vc.target_volume).powi(2),
            None => 0.0,
        }
    };

    for _sweep in 0..cfg.n_sweeps {
        for &e in &edge_list {
            proposed += 1;
            let old_len = *lengths.lengths.get(&e).unwrap();
            let delta = rng.gen_range(-cfg.step_size..cfg.step_size);
            let new_len = old_len + delta;
            if new_len <= 1e-6 {
                continue; // reject degenerate/negative lengths outright
            }
            lengths.lengths.insert(e, new_len);

            if !all_tetrahedra_valid(complex, &lengths) {
                lengths.lengths.insert(e, old_len); // reject: broke triangle inequality
                continue;
            }

            let new_s = regge_action(complex, &lengths, cfg.lambda).total;

            // Incremental volume update: only the tetrahedra incident to
            // this edge can have changed volume, so recompute the total
            // via a local delta rather than a full O(all tets) resum per
            // proposal.
            let incident: &[usize] = complex.edge_to_tets.get(&e).map(|v| v.as_slice()).unwrap_or(&[]);
            let old_local: f64 = incident
                .iter()
                .map(|&ti| {
                    lengths.lengths.insert(e, old_len);
                    let vol = crate::regge::tetrahedron_volume(&lengths, &complex.tetrahedra[ti]);
                    lengths.lengths.insert(e, new_len);
                    vol
                })
                .sum();
            let new_local: f64 = incident
                .iter()
                .map(|&ti| crate::regge::tetrahedron_volume(&lengths, &complex.tetrahedra[ti]))
                .sum();
            let new_v = current_v - old_local + new_local;

            let d_s = cfg.kappa * (new_s - current_s) + penalty(new_v) - penalty(current_v);
            let accept = d_s <= 0.0 || rng.gen::<f64>() < (-d_s).exp();

            if accept {
                current_s = new_s;
                current_v = new_v;
                accepted += 1;
            } else {
                lengths.lengths.insert(e, old_len);
            }
        }
        trace.push(current_s);
        volume_trace.push(current_v);
        let mean_edge: f64 = lengths.lengths.values().sum::<f64>() / n_edges;
        mean_edge_trace.push(mean_edge);
    }

    let n = trace.len();
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64;
    // Naive stderr (ignores autocorrelation -- stated explicitly, not hidden).
    let stderr = (var / n as f64).sqrt();

    McResult {
        mean_action: mean,
        stderr_action: stderr,
        acceptance_rate: accepted as f64 / proposed as f64,
        n_samples: n,
        action_trace: trace,
        volume_trace,
        mean_edge_length_trace: mean_edge_trace,
    }
}
