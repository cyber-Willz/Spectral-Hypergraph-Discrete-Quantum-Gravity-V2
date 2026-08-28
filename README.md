# spectral_dqg

A Rust implementation of the computable core of the write-up's pipeline:
**hypergraph → discrete Laplacian → Ihara-Selberg zeta function → (honest)
continuum-limit diagnostics.**

The system can probably be used to Benchmark & Design Quantum Hardware

## Test structure

Everything in this crate falls into one of two roles, and they should not
be presented as equally weighted:

**Chapter 3 — Validation of the computational engine.** The continuum GR
tensor-calculus engine and its exact-solution/geodesic cross-checks
(`tensor_calculus.rs`, `metrics.rs`, `geodesics.rs`), the discrete-GR
machinery (`regge.rs`, `regge_pi.rs`, `regge_eom.rs`, `cheeger.rs`), the
Ihara-zeta / non-backtracking-operator infrastructure (`nonbacktracking.rs`,
`ihara_zeta.rs`, `bulk_spacing.rs`), the semiclassical and QFT modules
(`semiclassical.rs`, `qg_phenomenology.rs`, `gme_bmv.rs`, `qft_zeta.rs`,
`casimir.rs`, `seeley_dewitt.rs`), and the GIS bridge modules all belong
here. Every one of them is a correctness check against an *already-known*
closed-form result, published bound, or exact identity (Schwarzschild
Kretschmann scalar, GRB 090510 LIV bound, Mohideen & Roy 1998, the
Schläfli identity, Friedman's theorem, ...). They establish that the
engine computes real, correct mathematics and physics — which is
necessary before trusting it on anything new — but none of them is itself
a new result. 

**Chapter 4 — the actual contribution: a quantitative discrete-to-continuum
convergence test.** `hypergraph_continuum_limit.rs` is the one module in
this crate that produces something not already sitting in the literature
in this form: a **convergence-rate measurement** for hypergraph
discretizations of a curved manifold (S², chosen for its closed-form
Laplace-Beltrami spectrum) as sampling resolution N grows, and a
**head-to-head comparison of two discretization schemes** (clique
expansion vs. the Zhou-Huang-Schölkopf normalized hypergraph Laplacian)
built from the *same* underlying hyperedge set. See that module's own doc
comment for the full derivation, including a real dead end (fixed-size kNN
hyperedges make the two schemes mathematically degenerate — proven, not
just observed) and how switching to epsilon-ball hyperedges fixes it. Run
`cargo run --release --bin hypergraph_continuum_demo` for the single-seed
live numbers, or `hypergraph_continuum_seed_avg_demo` for the seed-averaged,
bootstrapped version — the more trustworthy of the two, since the single-
seed run's scheme comparison did not survive bootstrapping (see below).
This is deliberately the narrowest thing in the write-up's original "Step
3" that can be answered honestly and quantitatively — not a claim to have
solved the general H_N → M convergence problem, which remains open exactly
as `continuum_limit.rs`'s own doc comment (Chapter 3 material) says.

Depends on `krylov_ds` as a local path dependency, vendored directly in
this repo at `./krylov_ds` — nothing extra to check out.

Run it straight from this directory:

```
cargo run --release
cargo test --release
```

## What's actually implemented, and how it's verified

**Step 1 — Discrete kinematics.** Hypergraphs (`hypergraph.rs`) reduce to a
weighted graph via clique expansion (Zhou–Huang–Schölkopf convention). The
normalized Laplacian `L = I - D^{-1/2} A D^{-1/2}` (`laplacian.rs`) gives a
real, nonnegative spectrum. `heat_kernel.rs` computes the heat trace
`P(t) = Σ e^{-tλ_j}` and estimates the running spectral dimension
`d_s(t) = -2 d ln P / d ln t`. Verified by unit tests: `P(0) = N`, `P(t)` is
monotone decreasing, `λ_0 = 0`. For `N` too large for dense diagonalization,
`spectral_trace.rs` estimates `P(t)` matrix-free via Stochastic Lanczos
Quadrature, with the Lanczos tridiagonalization itself delegated to
[`krylov_ds`](./krylov_ds) (a separate, independently-tested Krylov-
subspace crate, added as a local path dependency) rather than hand-rolled
here — `sparse.rs` only implements `krylov_ds::LinearOperator` for the
existing matrix-free Laplacian; see `RUN_LOG.txt` for the integration notes
and the one real convention mismatch (a trailing `beta` entry) it surfaced.

**Step 2 — Ihara-Selberg zeta function.** `nonbacktracking.rs` builds the
Hashimoto non-backtracking edge matrix `B`. `ihara_zeta.rs` computes
`Z_H(u)^{-1}` two *independent* ways — the spectral form `det(I-uB)` and the
Bass n×n determinant formula — and cross-checks them against each other
(they agree to ~1e-15). A third, purely combinatorial check compares
`Tr(B^k)` against a brute-force DFS count of closed non-backtracking walks.
**This cross-check caught a real bug** during development: the brute-force
counter was missing the wrap-around non-backtracking condition (the closing
step back into the first step must also not immediately reverse), which
silently overcounted at certain walk lengths. That's the point of building
independent verification paths rather than trusting one derivation.

**Step 3 — Continuum limit.** The write-up's Step 3 asks for something that
isn't a runnable computation: proving a *specific* hypergraph sequence
converges to a *specific* smooth hyperbolic manifold with a matching
classical Selberg zeta function is an open research problem (essentially the
discrete-to-continuum problem from causal dynamical triangulations /
causal-set theory). `continuum_limit.rs` implements the strongest thing that
*is* honestly computable and genuinely analogous: convergence of random
regular graphs to the **Kesten–McKay law** as N → ∞ (verified numerically —
RMS deviation shrinks with N), plus a Ramanujan/Alon–Boppana spectral-gap
diagnostic.

## Chapter 4 material: `hypergraph_continuum_limit.rs`

See the module doc comment for the full derivation. Summary: build a kNN or
epsilon-ball hyperedge set from points sampled on S², compute its spectrum
two ways (clique-expansion normalized Laplacian; Zhou hypergraph Laplacian
from the incidence matrix directly), and track the ratio of the two lowest
nonzero eigenvalue bands against the exact continuum value 3.0 (= 6/2, from
S²'s l(l+1) spectrum) as N grows — a ratio, not raw eigenvalues, specifically
because it's scale-invariant and so immune to the schemes' differing
normalization constants. Two real findings from actually building this,
not asserted up front:

1. **Fixed-size kNN hyperedges make the two schemes mathematically
   degenerate.** When every hyperedge has the same size m, Zhou's Laplacian
   equals exactly `(m-1)/m` times the clique-expansion Laplacian — a pure
   global rescaling, verified by direct derivation and by a unit test
   (`zhou_laplacian_is_exactly_half_normalized_graph_laplacian_on_plain_graph`).
   A ratio-based comparison cannot see any difference between the schemes
   in that construction — not a subtle numerical near-miss, an exact
   algebraic identity. Fixed by switching to epsilon-ball hyperedges
   (`eps_ball_hypergraph`), where hyperedge size fluctuates with local
   sampling density and the two schemes' spectra genuinely diverge
   (verified directly, `eps_ball_schemes_a_and_b_genuinely_diverge`).
2. **The heuristic connectivity radius needed retuning against measured
   data, not assumed.** The first constant tried (`c=1.5` in
   `heuristic_eps`) left the sampled graph disconnected at every N tested
   (2–60 near-zero Laplacian eigenvalues instead of the expected 1),
   silently invalidating the whole spectral comparison for those runs.
   Swept `c` from 1.5 to 4.0 against the actual zero-eigenvalue count and
   found `c=2.5` gives exactly one zero eigenvalue (connected) uniformly
   from N=100 to N=3200 — the constant now in both the demo binary and the
   tests, chosen because it was measured to work, not picked a priori.

Live run at N=100..3200 (single seed, `eps_c=2.5`) fits an empirical
convergence rate of roughly N^-0.41 (clique expansion) and N^-0.52 (Zhou),
i.e. the same order but not identical, with visible non-monotonicity
between individual N values — expected single-seed sampling noise, not
smoothed over. Seed-averaging to get real error bars on those two exponents
is the direct next step, explicitly not done here yet (see the demo
binary's own printed caveats).

**Update, same session's follow-up work:** that single-seed comparison
was checked against 8-seed averaging + a seed-resampling bootstrap
(`seed_errors_at_n`, `bootstrap_rate`,
`cargo run --release --bin hypergraph_continuum_seed_avg_demo`, restricted
to N ≤ 1600 by the O(N³) cost of dense diagonalization — see that binary's
own doc comment) and the single-seed comparison did not hold up: the two
schemes' bootstrapped 90% rate intervals (p_A = 0.807 ± 0.137, p_B = 0.834
± 0.132) **overlap substantially**, so "scheme B converges faster than
scheme A" is not something this data actually supports — it was an
artifact of comparing two point estimates with no uncertainty attached.
The honest summary is narrower: the ratio does converge, probably faster
than N^-1/2, and the two schemes' rates are not currently distinguishable
at this seed count. The seed-averaged mean-error curve is also not cleanly
monotonic (a real rise from N=400 to N=800 before falling again at
N=1600, on both schemes) — reported directly in `RUN_LOG.txt` rather than
treated as resolved by averaging alone.

That diagnostic first exposed a real problem: the initial configuration-model
generator (`random_regular_graph`) permits self-loops/multi-edges and drops
them rather than resampling, so it isn't actually a *simple* d-regular graph
— and the demo showed it, with the second-largest adjacency eigenvalue
sitting at ~3.98–3.998 against a theoretical bound of 3.464 (a real,
near-total violation, not noise). `random_simple_regular_graph` fixes this
properly via rejection sampling (retry until no self-loops/multi-edges,
matching the recipe NetworkX's own `random_regular_graph` uses), and running
the same experiment with it closes the gap directly: the second eigenvalue
drops to ~3.36–3.45, inside the bound, matching Friedman's theorem that
generic random simple d-regular graphs are near-Ramanujan whp. Both versions
are still in the crate so you can see the before/after for yourself by
running `cargo run --release`.

**Step 4 — Regge calculus (discrete GR action, path integral, and the
honestly-scoped residue of diffeomorphism invariance).** `regge.rs` builds
a genuine 3D simplicial complex from 4-vertex hyperedges (treated as literal
tetrahedra, not clique-expanded), computes dihedral/deficit angles and the
Regge action `S = Σ_hinges L_hinge · δ_hinge` (the exact discretization of
∫R√g) via Regge's edge-length formulation -- no ambient embedding of the
whole complex is needed, only per-tetrahedron Cayley-Menger validity.
`regge_pi.rs` runs a Metropolis path integral `Z = ∫D[length] exp(-κS)` over
edge lengths at fixed connectivity. Verified: a regular tetrahedron's volume
matches the closed form `l³/(6√2)` to 2.22e-16; a flat cube gives deficit
angle and total action ~1e-15 (zero to float precision) under **two
completely different tetrahedralizations** (fanning around either of the
cube's two main diagonals) -- the honest, narrowly-scoped discrete fact that
*is* exactly true (flat-region physics doesn't depend on the triangulation
describing it), as opposed to full continuum diffeomorphism invariance,
which Regge calculus is well known to only approximately recover in the
continuum limit.

**Step 5 — Regge equations of motion (Schläfli identity).** `regge_eom.rs`
numerically verifies the fact that makes the Regge action (Step 4) an
actual discretization of a *field equation*, not just an action-shaped
formula: the Schläfli differential identity, `dS/dL_e = δ_e` at every
interior hinge (Regge 1961). Computed two independent ways — a central
finite difference of the *full* action vs. the bare deficit angle — and
cross-checked to ~1e-4. **This caught a real scope issue during
development**: the identity was first tested on the same flat cube used in
`regge_tests.rs` and failed badly, because that complex has a boundary and
`regge_action` (by design, see Step 4 above) sums only interior hinges —
the cancellation the identity relies on needs every edge of every touching
tetrahedron to itself be counted in the action, which structurally isn't
true near a boundary with no discretized Gibbons-Hawking-York term. Fixed
by testing on a genuinely closed complex instead — the boundary of a
4-simplex (5 tetrahedra, 10 edges, the minimal closed triangulated
3-manifold, topologically S³) — where the identity holds cleanly on both a
regular and an irregular (no symmetry to hide a bug) curved configuration.
See `RUN_LOG.txt` for the full diagnostic trail.

**Cheeger's inequality.** `cheeger.rs` computes the exact (brute-force,
small-N) Cheeger constant / edge conductance and cross-checks it against
the normalized-Laplacian spectral gap via `λ₁/2 ≤ h(G) ≤ √(2λ₁)` (Chung,
*Spectral Graph Theory*, Thm 2.2) — verified against closed-form values on
`C_10`/`K_6` and on irregular and disconnected graphs. This is the purely
combinatorial counterpart to `continuum_limit.rs`'s Ramanujan/Alon–Boppana
diagnostic for "how expander-like is this graph".

The path-integral coupling test caught a genuine physics subtlety, not a
bug: stronger coupling κ (~1/G) does *not* pull the ensemble toward
flatness -- ⟨S⟩ runs from -4.88 (κ=0.2) to -20.69 (κ=5.0), because the
curvature term is unbounded below and naive `exp(-κS)` sampling is pulled
toward large negative-deficit configurations, capped only by the hard
tetrahedron-validity wall. This is the discrete incarnation of the
conformal-factor problem in Euclidean quantum gravity (Gibbons, Hawking &
Perry 1978) -- reproduced correctly and reported, not silently "fixed" by
asserting the naive-but-wrong expectation. See `RUN_LOG.txt` for the full
diagnostic trail and `src/bin/regge_demo.rs` for a runnable end-to-end demo.

**Step 6 — Continuum tensor calculus, exact solutions, and geodesics.**
Everything above (and the GNSS module below) was either purely discrete or a
single hand-derived special case. `tensor_calculus.rs` is a generic
numerical engine: given *any* metric `g_{ab}(x)` as a callback, it computes
Christoffel symbols, the Riemann tensor, Ricci tensor/scalar, Einstein
tensor, and the Kretschmann scalar by numerically differentiating the metric
(central differences; the Riemann tensor requires differentiating the
Christoffel symbols a second time). `metrics.rs` supplies Schwarzschild and
FRW as metric callbacks together with their independently-known closed-form
results, and cross-checks the numerical engine against them: Schwarzschild
is verified Ricci-flat (`R_{ab}=0`, the vacuum condition that originally
defines it) and its Kretschmann scalar matches the textbook `48M^2/r^6` to
better than 1%; FRW's Ricci scalar matches the textbook
`6[a''/a+(a'/a)^2+k/a^2]` for a matter-dominated `a(t)=t^(2/3)` universe, and
is independently checked to be spatially homogeneous (a structural
consequence of FRW symmetry, not something the closed-form check alone would
catch if broken). `geodesics.rs` builds an RK4 geodesic integrator directly
on top of the same numerical Christoffel symbols and verifies it three ways:
norm conservation (`g_{ab}u^au^b` constant along any geodesic), the
*emergent* conservation of the Schwarzschild Killing charges (energy and
angular momentum, not hard-coded into the integrator), and a genuine
end-to-end physical prediction -- integrating a real null geodesic past a
mass and recovering the weak-field light-bending formula `Delta\phi ~ 2r_s/b`
to within a few percent.

That light-bending test caught a real bug during development, of the kind
this crate's README keeps a running record of: the first version compared
the total swept angle to `pi`, which is only the deflection baseline in the
`r0 -> infinity` limit. At the finite starting radius `r0` any integration
actually uses, the correct flat-space (zero-mass) baseline is
`pi - 2*asin(b/r0)`, not `pi` -- at `b/r0=0.25` that correction is *larger*
than the GR effect being measured, which is exactly what produced a wrong-
sign, wrong-magnitude result on the first attempt. Tracing the raw
trajectory in Cartesian coordinates (not just staring at the polar-angle
output) is what surfaced it; fixed by subtracting the exact finite-`r0`
flat-space baseline instead of `pi`, and cross-checked stable across three
different choices of `r0`.

## What's deliberately *not* claimed

- No claim that clique expansion is the "right" hypergraph Laplacian —
  tensor/simplicial hypergraph spectral theory is an open area.
- No literal Selberg trace formula on a target manifold; no metric
  reconstruction `g_{μν}(x)` for the general hypergraph (only per-tetrahedron
  local embeddings, used solely to read off dihedral angles). Proving that
  any specific simplicial complex here approximates a specific smooth
  manifold remains a research problem, not a library function.
- No fix for the conformal-factor / unbounded-below-action pathology in the
  Regge path integral (§Step 4) -- it is correctly reproduced and reported,
  not solved. A well-defined quantum Regge calculus path integral needs a
  proper measure on edge-length space and/or a rotation of the conformal
  mode; neither is implemented.
- No boundary term (discrete Gibbons-Hawking-York) anywhere in this crate --
  `regge_eom.rs`'s Schläfli-identity check is therefore only verified (and
  only claimed to hold) on closed, boundary-free complexes; on a complex
  with boundary, `dS/dL_e = δ_e` does not hold with the current
  boundary-term-free action, and this is reported rather than glossed over.
- No solver: `regge_eom.rs` verifies the identity the discrete vacuum field
  equations rest on and classifies hinges as vacuum/non-vacuum for a given
  configuration; it does not implement an extremization routine to search
  for solutions of `δ_e = 0` everywhere.
- `cheeger.rs`'s Cheeger constant is exact brute force, tractable only for
  small graphs (a few tens of vertices) -- no scalable approximation
  (spectral partitioning, sweep cuts) is implemented.
- No claim of continuum diffeomorphism invariance -- only the narrower,
  exactly-true discrete fact that flat-region physics is independent of
  which triangulation describes it (verified numerically to float
  precision under two different triangulations of the same flat cube).
- Spectral-dimension flow is shown honestly with its known finite-graph
  artifacts (UV/IR flattening) rather than papered over.
- `tensor_calculus.rs` is not a computer-algebra system: curvature is
  obtained by numerically differentiating a metric callback (twice, for the
  Riemann tensor), not by symbolic differentiation, so accuracy depends on
  the finite-difference step `h` and is verified empirically (~1e-2 to 1e-4
  relative error against closed-form results), not claimed to machine
  precision the way the closed-form Regge calculus in this crate is.
- `metrics.rs` implements exactly two exact solutions (Schwarzschild, FRW),
  chosen for having the least-ambiguous textbook closed-form curvature
  invariants to check against -- not a general solution catalog. No Kerr, no
  charged/rotating solutions, no matched interior/exterior solutions.
- No stress-energy tensor and no Einstein-equation solver: `tensor_calculus`
  computes curvature *from* a given metric; it does not solve
  `G_{ab}=8\pi T_{ab}` for an unknown metric given matter content. FRW's
  scale factor `a(t)` is caller-supplied, not derived from a Friedmann
  matter sector.
- `geodesics.rs` uses fixed-step RK4 with no adaptive step control or
  horizon/singularity detection beyond the specific "radius crosses back
  above its start" logic the light-bending test uses -- not a general
  geodesic solver, and not safe to point at trajectories that pass close to
  the photon sphere or horizon without hand-tuning the step size.
- No bridge (yet) between this continuum engine and the discrete Regge
  machinery above -- e.g. nothing compares a Regge deficit angle to a
  continuum curvature component on a matching geometry. Both now exist in
  this crate; connecting them is future work, not implemented here.
- `hypergraph_continuum_limit.rs` tests convergence of a single scalar
  ratio (the l=2/l=1 eigenvalue-band ratio), which is necessary but not
  sufficient evidence for convergence of the full discrete operator to the
  continuum Laplace-Beltrami operator -- not a claim of operator-norm or
  spectral-measure convergence in general.
- Its convergence-rate exponents come from a bounded seed budget (8 seeds,
  N ≤ 1600, capped by O(N^3) dense-diagonalization cost) and a heuristic
  (not derived-optimal) connectivity constant, with unweighted hyperedges.
  The bootstrap shows the two schemes' rates are not currently
  distinguishable (overlapping 90% intervals) -- this is reported as the
  actual finding, not papered over by picking whichever single-seed run
  looked more like a clean comparison. Extending past N=1600 needs a
  matrix-free low-eigenvalue extraction method (adapting
  `spectral_trace.rs`'s SLQ machinery, built for heat-trace sums, not
  individual low-lying eigenvalues) -- identified, not implemented.

## GIS bridge modules (`gis_*.rs`)

Three small modules connect specific, *verifiable* pieces of geodetic/GIS
mathematics to machinery already in this crate, rather than importing the
GIS domain wholesale (most of it — datums, map projections, remote sensing,
kriging, raster hydrology — has no functional relationship to discrete
quantum gravity and is deliberately left out):

- `gis_ellipsoid.rs` — WGS84 Gaussian curvature `K = 1/(M·N)`, plus a
  literal discrete Gauss–Bonnet check (icosahedron, Σδ = 4π) as the
  verified 2D analogue of the 3D Regge deficit-angle identity in
  `regge.rs`. Both are "curvature read off a metric"; only that narrow
  claim is made.
- `gis_spherical_spectrum.rs` — the true Laplace–Beltrami spectrum of a
  sphere (`λ_l = l(l+1)/R²`, the basis EGM2008-style gravity models expand
  in) fed through this crate's *existing* `heat_trace` /
  `spectral_dimension_flow`, recovering the textbook `d_s ≈ 2` plateau for
  a genuine 2-manifold as a sanity reference point.
- `gis_gnss_relativity.rs` — GPS satellite special- and general-relativistic
  clock corrections from first principles, cross-checked against the
  well-known net ~38 μs/day figure. The one piece of the GIS material with
  a direct physical (not just mathematical) link to GR — but it is a
  standalone continuum calculation, not wired into the Regge action or
  path integral.

No datum transforms, map projections, coordinate conversions, remote
sensing, spatial statistics, or raster/hydrology math are implemented here:
they are real GIS mathematics but do not connect to what this crate
actually does.

## Layout

```
src/hypergraph.rs        hypergraph + clique expansion
src/laplacian.rs          normalized/unnormalized Laplacian, spectrum
src/heat_kernel.rs         heat trace, spectral-dimension flow
src/nonbacktracking.rs    Hashimoto matrix, Tr(B^k), brute-force cross-check
src/ihara_zeta.rs         Ihara + Bass zeta computation & cross-check
src/continuum_limit.rs    Kesten-McKay convergence, Ramanujan diagnostic (Chapter 3)
src/hypergraph_continuum_limit.rs  sphere discretization, clique-expansion vs. Zhou
                          hypergraph Laplacian, scale-free spectral-ratio convergence
                          test and rate fit (Chapter 4 -- the thesis's actual contribution)
src/bin/hypergraph_continuum_demo.rs  live run of the Chapter 4 convergence sweep
src/bin/hypergraph_continuum_seed_avg_demo.rs  seed-averaged + bootstrapped version:
                          puts real error bars on the two rate exponents and checks
                          whether the two schemes' rates are actually distinguishable
src/regge.rs              simplicial complex, deficit angles, Regge action
src/regge_pi.rs           Metropolis path integral over edge lengths
src/regge_eom.rs          Schläfli identity / discrete Regge equations of motion
src/cheeger.rs            Cheeger constant vs. spectral gap (Cheeger's inequality)
src/tensor_calculus.rs    generic numerical Christoffel/Riemann/Ricci/Einstein/Kretschmann engine
src/metrics.rs            Schwarzschild and FRW metrics + their closed-form curvature cross-checks
src/geodesics.rs          RK4 geodesic integrator: conserved-quantity checks, light-bending,
                          perihelion precession, radial-redshift cross-check
src/semiclassical.rs      Hawking temperature (surface gravity) + Bekenstein-Hawking entropy
                          (horizon area), both read from the metric, not the closed forms
src/qg_phenomenology.rs   Lorentz-invariance-violation phenomenology: real flat-LambdaCDM
                          GRB time-delay formula, cross-validated against the published
                          Fermi-LAT GRB 090510 bound (Vasileiou et al. 2013)
src/gme_bmv.rs            Gravity-mediated entanglement (BMV/QGEM): four-branch Newtonian
                          phase calculation, cross-validated against the published Bose
                          et al. 2017 (PRL 119, 240401) phase values
src/qft_zeta.rs           Riemann zeta function via direct summation + Euler-Maclaurin
                          tail correction, analytically continued to negative integers
                          via the functional equation -- feeds casimir.rs a derived
                          zeta(-3), not a hardcoded 1/120
src/casimir.rs            Casimir effect derived via zeta-function regularization,
                          cross-validated against Mohideen & Roy 1998 (PRL 81, 4549)
src/seeley_dewitt.rs      Seeley-DeWitt heat-kernel a_1 coefficient on the sphere, reusing
                          the crate's own heat_kernel.rs machinery -- connects curvature
                          (Gauss-Bonnet) to the QFT heat-kernel trace
src/bin/gr_demo.rs        end-to-end live demo of Steps 6 (tensor calculus/metrics/geodesics)
src/bin/gr_demo2.rs       live demo: precession convergence sweep, redshift, Hawking
                          temperature, Bekenstein-Hawking entropy
src/bin/qg_phenomenology_demo.rs  live demo: GRB 090510 LIV bound + what discreteness
                          scales it already excludes under the naive dispersion ansatz
src/bin/gme_bmv_demo.rs   live demo: entangling-phase convergence + Bose et al. 2017
                          cross-check
src/bin/qft_demo.rs       live demo: zeta regularization, Casimir force vs. Mohideen &
                          Roy 1998, Seeley-DeWitt a_1 convergence
src/main.rs               end-to-end demo / report (Steps 0-3)
src/bin/regge_demo.rs     end-to-end demo / report (Step 4, Regge calculus)
tests/cross_checks.rs     the Steps 0-3 verification suite
tests/regge_tests.rs      the Step 4 (Regge calculus) verification suite
```
