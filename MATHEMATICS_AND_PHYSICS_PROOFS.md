# Mathematics and Physics of `spectral_dqg_and_krylov_ds`: Definitions, Theorems, and Proofs

This document gives the complete mathematical and physical content underlying every module of the codebase, organized the way the repository's own README organizes it: **Chapter 3 material** (correctness checks against already-known closed-form results — necessary but not new) and **Chapter 4 material** (the one genuinely new empirical result). For each piece of mathematics, this document states the precise definition, the theorem being used or verified, a proof or proof sketch, and — critically — what is *actually proved* by the code versus what remains open. Where the live run (see `LIVE_RUN_REPORT.md`) produced numbers, they are cited as empirical confirmation, not as substitutes for the proofs.

**Notational convention used throughout:** `G=(V,E)` a graph with `|V|=n`, `|E|=m`; `A` its adjacency matrix; `D=\mathrm{diag}(\deg)$; matrices act on `\mathbb{R}^n` or `\mathbb{C}^n` as stated.

---

## Part I — Discrete spectral graph theory (Steps 0–3)

### 1. Hypergraph → clique expansion → normalized Laplacian

**Definition (hypergraph).** `H=(V,E)`, `V` finite, each `e\in E` a subset of `V` with `|e|\ge 2` and a weight `w(e)>0`.

**Definition (clique expansion, Zhou–Huang–Schölkopf convention).** Replace each hyperedge `e` of size `k=|e|` by a weighted `k`-clique on its members, each new edge carrying weight `w(e)/(k-1)`.

**Claim.** This choice of per-edge weight makes the induced weighted vertex degree in the clique-expanded graph equal the weighted hypergraph degree `\deg_H(v)=\sum_{e\ni v} w(e)`.

*Proof.* Fix `v\in V`. In the clique on `e\ni v`, `v` is adjacent to the other `k-1` members, each edge weighted `w(e)/(k-1)`. Its contribution to the clique-expanded degree of `v` is `(k-1)\cdot w(e)/(k-1) = w(e)`. Summing over all `e\ni v` gives `\sum_{e\ni v} w(e) = \deg_H(v)`. ∎

This is a definitional/bookkeeping fact, not a deep theorem, but it is the fact that makes "the hypergraph Laplacian's diagonal equals the hypergraph degree" true by construction rather than by accident — and the code's docstring is explicit that clique expansion is *a* choice, not *the* canonical one (genuine hypergraph/simplicial spectral theory for irreducibly multi-way interactions is an open area; Chapter 4 below is precisely about testing an *alternative*, the Zhou hypergraph Laplacian, against this one).

**Definition (symmetric normalized Laplacian).**
$$L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$$
(with the convention `(D^{-1/2})_{vv}=0` for isolated vertices).

**Theorem 1 (basic spectral facts).** `L_{\mathrm{sym}}` is symmetric positive semi-definite, so its eigenvalues are real and satisfy `0=\lambda_0\le\lambda_1\le\dots\le\lambda_{n-1}\le 2`. `0` is always an eigenvalue, with multiplicity equal to the number of connected components.

*Proof.* For any `x\in\mathbb{R}^n`, writing `y=D^{-1/2}x`,
$$x^\top L_{\mathrm{sym}} x = \sum_{(u,v)\in E} w(u,v)\left(\frac{x_u}{\sqrt{d_u}}-\frac{x_v}{\sqrt{d_v}}\right)^2 \ge 0,$$
which is the standard Dirichlet-form identity for the normalized Laplacian (Chung, *Spectral Graph Theory*, Ch. 1) — non-negativity of every eigenvalue follows immediately. The constant vector `D^{1/2}\mathbf{1}` restricted to a connected component satisfies `L_{\mathrm{sym}} (D^{1/2}\mathbf{1}) = 0` on that component (direct substitution), giving one zero eigenvalue per component; that these are the *only* zero eigenvalues follows because `x^\top L_{\mathrm{sym}} x=0` forces `x_u/\sqrt{d_u}` constant on every edge, hence constant on every connected component. The upper bound `\lambda\le 2` follows from `L_{\mathrm{sym}} = D^{-1/2}(D-A)D^{-1/2}\preceq D^{-1/2}(D+A)D^{-1/2}` and a similar Dirichlet-form argument on `D+A`. ∎

**What the code verifies (not just implements):** `laplacian_smallest_eigenvalue_is_zero_and_nonnegative_spectrum` checks this directly, and the live run's N=8 toy example shows `λ_0=0` with the rest of the spectrum in `[0, 1.8503]\subset[0,2]`, consistent with Theorem 1.

### 2. Heat trace and spectral dimension

**Definition.** `P(t)=\mathrm{Tr}(e^{-tL_{\mathrm{sym}}}) = \sum_j e^{-t\lambda_j}`, and the *running spectral dimension*
$$d_s(t) = -2\,\frac{d\ln P(t)}{d\ln t}.$$

**Proposition 2.** `P(0)=n`, `P(t)` is strictly decreasing on `t>0` unless the graph is edgeless, and `P(t)\to c` (the number of connected components) as `t\to\infty`.

*Proof.* `P(0)=\sum_j e^0=n` trivially. `P'(t)=-\sum_j \lambda_j e^{-t\lambda_j}\le 0` since every `\lambda_j\ge 0$ (Theorem 1), with strict inequality whenever some `\lambda_j>0`, i.e. whenever the graph has at least one edge. As `t\to\infty`, every term with `\lambda_j>0` decays to 0, leaving only the `\lambda_j=0` terms, whose count is the number of connected components (Theorem 1). ∎

This is the identity `spectral_dimension_flow` numerically differentiates; the unit tests `heat_trace_at_t_zero_equals_vertex_count` and `heat_trace_is_monotonically_decreasing_in_t` check Proposition 2 directly, and the live N=8 run shows `P(t)` falling monotonically from 7.73 to 1.05 as `t` sweeps 0.034→11.76, consistent with a single connected component (`c=1`, and `P(t)\to 1`).

**Why `d_s` does not plateau on small graphs (and why that's not a bug).** In the continuum, `d_s` measures an effective dimension via how the heat kernel's short-time expansion scales; on a *finite* graph there are exactly two structural reasons a clean plateau cannot appear: (i) as `t\to 0`, `P(t)\to n` is dominated by the lattice cutoff (the smallest resolvable "distance" is one edge), giving a UV artifact; (ii) as `t\to\infty`, `P(t)\to c` is dominated by the *total* graph size, an IR artifact from finite volume. A plateau requires a scale window where neither effect dominates, which needs many more vertices than the illustrative `N=8` toy graph has. This is exactly what the live run shows (`d_s` rises to a peak ≈1.37 and falls at both ends, never plateauing) — the code reports the *correct*, honestly-scoped behavior for its input size, not a broken computation.

### 3. Non-backtracking (Hashimoto) operator and the Ihara–Selberg zeta function

**Definition (directed arcs and B).** For each undirected edge `{u,v}\in E`, form two directed arcs `u\to v` and `v\to u`; let `m_2=2|E|` be their count. The **Hashimoto matrix** `B\in\{0,1\}^{m_2\times m_2}` is
$$B_{(u\to v),(v'\to w)} = \mathbb{1}[v'=v]\cdot\mathbb{1}[w\ne u],$$
i.e. arc `u\to v` maps to arc `v\to w` exactly when the walk continues at `v` without immediately reversing.

**Theorem 3 (Ihara's theorem, in the spectral form used here).** For a finite graph with no vertices of degree ≤1 causing trivial cancellation, the Ihara zeta function
$$Z_H(u) = \prod_{[p]\text{ prime, non-backtracking, tailless}} \left(1-u^{\ell(p)}\right)^{-1}$$
satisfies
$$Z_H(u)^{-1} = \det(I - uB).$$

**Theorem 3′ (Bass's determinant formula).** Equivalently, on the `n\times n` scale,
$$Z_H(u)^{-1} = (1-u^2)^{|E|-|V|}\,\det\!\left(I - uA + u^2(D-I)\right).$$

*Sketch of why these agree (Bass 1992; Stark–Terras 1996 for an accessible derivation).* Both sides are computing the same generating function
$$\log Z_H(u)^{-1} = -\sum_{k\ge 1} \frac{\mathrm{Tr}(B^k)}{k}\,u^k,$$
because `\mathrm{Tr}(B^k)` counts *closed non-backtracking arc-walks of length k* exactly (a walk `u_0\to u_1\to\cdots\to u_k=u_0` is counted by `B^k` iff no consecutive triple backtracks, which is precisely the non-backtracking-tailless-prime condition once cycled through all `k` starting points and divided by orbit length — the standard combinatorics-of-necklaces argument that turns the Euler-product zeta function into a trace generating function). The Bass formula is obtained by block-diagonalizing `I-uB` using the graph's `n\times n` incidence structure (the `2m\times 2m` problem factors through an `n\times n` and a scalar `(1-u^2)^{|E|-|V|}` piece) — a linear-algebra identity, not a new physical fact, but one whose derivation is genuinely easy to get wrong in implementation (sign conventions, degree matrix used, the exponent `|E|-|V|`), which is exactly why the crate computes both sides independently rather than trusting the derivation once.

**What the code proves about itself.** `ihara_and_bass_formulas_agree` checks Theorem 3 = Theorem 3′ to machine precision on live examples; the live run shows agreement to `3\times10^{-16}$–`4\times10^{-15}` across four values of `u`. Separately, `trace_bk_matches_bruteforce_closed_walk_count` verifies the *combinatorial* claim underlying both theorems — that `\mathrm{Tr}(B^k)` really does count closed non-backtracking walks — by an independent brute-force DFS, live-confirmed exactly (`0,0,30,48,40` for k=1..5). This closes the loop: two algebraic derivations agree with each other, and one of them additionally agrees with a definition-level combinatorial count, which is the strongest verification structure available without a formal proof assistant.

### 4. Kesten–McKay law, Ramanujan graphs, and the Alon–Boppana bound

**Definition (Kesten–McKay density).** The limiting empirical spectral distribution of the adjacency matrix of a uniformly random `d`-regular graph as `n\to\infty` has density
$$\rho_d(x) = \frac{d\sqrt{4(d-1)-x^2}}{2\pi(d^2-x^2)},\qquad |x|\le 2\sqrt{d-1}.$$

**Theorem 4 (McKay 1981).** For `G_n` a sequence of (uniformly random, or more generally locally-tree-like) `d`-regular graphs on `n\to\infty` vertices, the empirical spectral distribution of `A(G_n)` converges weakly to `\rho_d`.

**Definition (Ramanujan graph).** A connected `d`-regular graph is Ramanujan if every eigenvalue `\lambda` of `A` other than `\pm d` satisfies `|\lambda|\le 2\sqrt{d-1}` — the edge of the Kesten–McKay support, and (by the Alon–Boppana theorem below) essentially the best possible uniform bound.

**Theorem 5 (Alon–Boppana).** For any sequence of `d`-regular graphs with `n\to\infty`, the second-largest adjacency eigenvalue satisfies
$$\liminf_{n\to\infty} \lambda_1(G_n) \ge 2\sqrt{d-1} - o(1).$$

*Proof idea.* Bound the number of closed walks of length `2k` from below using the tree structure of the `d`-regular infinite tree's return probabilities (the Kesten–McKay density *is* the spectral measure of the infinite `d`-regular tree at the root), then relate `\mathrm{Tr}(A^{2k})=\sum\lambda_i^{2k}` to a lower bound on `\max_i|\lambda_i|` excluding the trivial `\lambda_0=d`; take `k\to\infty` with `n` to isolate the `2\sqrt{d-1}` constant. (Full proof: Alon 1986; Nilli 1991 gives the sharp elementary argument.) ∎

**Theorem 6 (Friedman 2008, "Alon's second-eigenvalue conjecture").** A uniformly random `d`-regular *simple* graph on `n` vertices satisfies, with probability `1-o(1)` as `n\to\infty`,
$$\lambda_1(G) \le 2\sqrt{d-1} + \varepsilon\quad\text{for every }\varepsilon>0,$$
i.e. random `d`-regular simple graphs are asymptotically *near*-Ramanujan.

**What the live run actually demonstrates about Theorems 4–6.** This is the cleanest example in the whole codebase of code *discovering* a real mathematical subtlety rather than just illustrating a known theorem: the first configuration-model generator (pairing model, not rejection-sampled) permits self-loops and multi-edges. Theorem 6 is a statement about *simple* graphs; a non-simple pairing is not covered by it at all, and the live run shows exactly the predicted consequence — with the non-simple sampler, `max|\lambda|` sits at 3.98–3.998 against the bound `2\sqrt{3}=3.464`, i.e. a real, worsening-with-N-but-still-persistent near-violation (not statistical noise: it holds at N=50, 400, *and* 2000). Switching to `random_simple_regular_graph` (rejection sampling until no self-loops/multi-edges) closes the gap immediately and completely: `max|\lambda|` drops to 3.36–3.45, **100% within the Ramanujan bound at every N tested**, directly confirming Theorem 6 empirically. The RMS deviation from `\rho_d` shrinking `0.0295\to0.0084\to0.0060` as `N=50\to400\to2000` is the live confirmation of Theorem 4's convergence statement. No part of this is a new theorem — it is a correct empirical demonstration of Theorems 4–6, with the added, genuinely useful pedagogical value of showing *why* the "simple graph" hypothesis in Theorem 6 actually matters, in numbers.

### 5. Cheeger's inequality

**Definition (edge conductance / Cheeger constant).** For `S\subset V`, `h(S)=\dfrac{|\partial S|}{\min(\mathrm{vol}(S),\mathrm{vol}(V\setminus S))}`, and `h(G)=\min_{S} h(S)`.

**Theorem 7 (Cheeger's inequality, discrete form; Chung Thm. 2.2).**
$$\frac{\lambda_1}{2} \le h(G) \le \sqrt{2\lambda_1},$$
where `\lambda_1` is the smallest nonzero eigenvalue of `L_{\mathrm{sym}}`.

*Proof sketch.* The lower bound follows from the Rayleigh-quotient characterization `\lambda_1=\min_{x\perp D^{1/2}\mathbf 1} \frac{x^\top L_{\mathrm{sym}} x}{x^\top x}`, applying it to the indicator (degree-normalized) vector of the optimal cut `S^*` and bounding the resulting Dirichlet form below by `h(G)$-type quantities via Cauchy–Schwarz. The upper bound is a discrete Cheeger/sweep-cut argument: take the eigenvector achieving `\lambda_1`, sort vertices by its value, and show that *some* prefix cut in that ordering achieves conductance `\le\sqrt{2\lambda_1}` — the standard "sweep" proof (Chung 1997, full detail in Ch. 2). ∎

The code computes `h(G)` by brute force (tractable only for small `n`, honestly documented as such) and checks both inequality directions on `C_{10}`, `K_6`, and irregular/disconnected examples (`cheeger_constant_matches_known_cycle_value`, `cheeger_inequality_holds_on_irregular_bridge_graph`, `disconnected_graph_gives_zero_gap_and_zero_cheeger_constant` — the last confirming the boundary case `\lambda_1=0\iff h(G)=0\iff$ disconnected, consistent with Theorem 1).

---

## Part II — Regge calculus (discrete general relativity)

### 6. Simplicial curvature: deficit angles and the Regge action

**Setup.** A 3-dimensional simplicial complex is built from tetrahedra glued along shared triangular faces. Given only edge lengths (Regge's original formulation needs no ambient embedding of the whole complex — each tetrahedron is separately valid iff its six edge lengths satisfy the Cayley–Menger positivity condition), the **dihedral angle** `\theta` at an edge (hinge) `e` within one incident tetrahedron is the angle between the two triangular faces of that tetrahedron meeting at `e`, computable from the tetrahedron's edge lengths alone via the standard Cayley–Menger/vector-geometry formula.

**Definition (deficit angle).** For an interior hinge `e` shared by tetrahedra `t_1,\dots,t_k`,
$$\delta_e = 2\pi - \sum_{i=1}^k \theta_e^{(t_i)}.$$

**Definition (Regge action).**
$$S_{\mathrm{Regge}} = \sum_{e\text{ interior}} L_e\,\delta_e,$$
where `L_e` is the length of hinge `e`.

**Theorem 8 (Regge 1961).** `S_{\mathrm{Regge}}` is the exact simplicial discretization of the Einstein–Hilbert action `\int R\sqrt{g}\,d^3x` (in 3D) or `d^4x` (in 4D): as the triangulation is refined around a smooth curved manifold, `S_{\mathrm{Regge}}\to \int R\sqrt{g}` in the appropriate limit, with all curvature concentrated distributionally on the hinges (a discrete analogue of curvature being a delta function supported on a cone point).

**Proposition 9 (flatness is triangulation-independent).** If a region is flat (embeddable isometrically in Euclidean space), then *every* valid simplicial decomposition of it gives deficit angle 0 at every interior hinge, hence `S_{\mathrm{Regge}}=0`, regardless of which triangulation is chosen.

*Proof.* Flatness means the whole region embeds in `\mathbb{R}^3`. Dihedral angles computed from an actual embedding at a hinge sum to exactly `2\pi` around any interior edge of *any* triangulation of a flat region — this is just the ordinary Euclidean geometry fact that dihedral angles around a line in space sum to a full turn, independent of how the surrounding solid is cut into tetrahedra. Hence `\delta_e=0` for every interior `e`, for every triangulation. ∎

Proposition 9 is the honest, narrow fact the code establishes — it is emphatically **not** a proof of continuum diffeomorphism invariance for curved regions (which Regge calculus is known to only approximately recover as the triangulation is refined; there is no claim otherwise anywhere in the crate). The live run confirms Proposition 9 concretely: a flat cube triangulated two topologically different ways (fanning around either main diagonal) gives deficit angle `\sim2.665\times10^{-15}` rad and action `\sim4.6\times10^{-15}$ under **both** decompositions — zero to float precision, matching Proposition 9's prediction exactly, and (separately) perturbing one diagonal by 15% produces a real, order-1 deficit angle and action, confirming the machinery is sensitive to genuine curvature and not just always returning zero.

### 7. The Schläfli identity and discrete equations of motion

**Theorem 10 (Schläfli's differential identity, Regge 1961's use of Schläfli 1858).** At every interior hinge `e` of a closed (boundary-free) simplicial complex,
$$\frac{\partial S_{\mathrm{Regge}}}{\partial L_e} = \delta_e.$$

*Proof idea.* Schläfli's original identity states that for a spherical/hyperbolic/Euclidean simplex, the differential of the volume with respect to edge-length variations, weighted appropriately, satisfies a first-variation identity in which the *dihedral angle* terms exactly cancel — `\sum_{\text{hinges of the simplex}} L_{\text{hinge}}\,d\theta_{\text{hinge}} = 0` at fixed simplex shape (a consequence of the simplex volume being a function of its edge lengths alone, so its differential must be expressible purely in terms of length variations, forcing the angle-variation terms to cancel identically — this is Schläfli's 1858 theorem, a purely Euclidean/hyperbolic-geometry fact about simplices). Summing this identity over every tetrahedron touching hinge `e` and differentiating `S_{\mathrm{Regge}}=\sum_{e'} L_{e'}\delta_{e'}` with respect to `L_e` leaves only the `L_e` term's direct derivative (`\delta_e`) once all the `\delta\theta` cross-terms cancel via Schläfli — this cancellation is why `S_{\mathrm{Regge}}` behaves as a genuine action with a clean variational principle at all, and it is why Theorem 10 is the fact that makes `\delta_e=0$ everywhere" (vacuum, matter-free) the discrete Einstein field equations, not just an action-shaped formula. ∎

**Why the closedness hypothesis is essential (not a technicality).** The cancellation Theorem 10 relies on needs *every* edge of *every* tetrahedron touching `e` to itself be included in the action sum. On a complex with boundary, some of those edges are boundary edges excluded from `S_{\mathrm{Regge}}` (by construction, since `S_{\mathrm{Regge}}` sums only interior hinges) — so the Schläfli cancellation is structurally incomplete, and Theorem 10 genuinely fails on complexes with boundary unless a boundary (Gibbons–Hawking–York-type) term is added, which this crate does not implement. The code discovered this the hard way (documented in `RUN_LOG.txt`): testing Theorem 10 on the flat cube (which has boundary) failed; switching to a genuinely closed complex — the boundary `\partial\Delta^4}` of a 4-simplex, the minimal closed triangulated `S^3` (5 tetrahedra, 10 edges) — the identity holds to `\sim10^{-4}` (finite-difference precision) on both regular and irregular (asymmetric, curved) configurations, live-confirmed by `schlafli_identity_holds_on_regular_4_simplex_boundary` and `..._on_closed_irregular_complex`.

### 8. The conformal-factor pathology in the Euclidean path integral

**Setup.** The (Euclidean, naive) Regge path integral is `Z=\int \mathcal D[L] \, e^{-\kappa S_{\mathrm{Regge}}[L]}`, sampled here by Metropolis moves on edge lengths at fixed connectivity.

**Proposition 11.** `S_{\mathrm{Regge}}` is unbounded below on the space of valid (Cayley–Menger-admissible) edge-length configurations, because deficit angles can be driven arbitrarily negative (sharply concave hinges) while the hinge lengths `L_e` remain bounded away from 0, so `e^{-\kappa S}` is *not* normalizable without an additional constraint, and naive Metropolis sampling is driven toward configurations of increasingly negative curvature, capped only by the hard tetrahedron-validity boundary.

*This is the discrete incarnation of a known continuum fact*: the Euclidean Einstein–Hilbert action is unbounded below because the conformal mode of the metric contributes with the "wrong sign" kinetic term (Gibbons, Hawking & Perry 1978) — a famous, still-unresolved-in-full-generality problem for Euclidean quantum gravity path integrals, usually addressed by a Wick rotation of the conformal mode into the complex plane, which is *not* implemented here. The live run reproduces Proposition 11's predicted qualitative behavior exactly: `⟨S⟩` runs from `-4.88` (`κ=0.2`) to `-20.69` (`κ=5.0`) — i.e. *more* negative as coupling increases, the opposite of what a bounded-below action would give (where stronger coupling drives the ensemble toward the action's minimum, here would-be zero/flatness). This is reported by the code as a correctly-reproduced pathology, not a bug to be silently patched.

---

## Part III — Continuum general relativity (numerical tensor calculus)

### 9. The Riemann curvature machinery, from a metric callback

**Definitions.** Given a metric `g_{ab}(x)` as a black-box callback, the Christoffel symbols are
$$\Gamma^{c}_{ab} = \tfrac12 g^{cd}(\partial_a g_{bd}+\partial_b g_{ad}-\partial_d g_{ab}),$$
the Riemann tensor
$$R^{d}_{\ abc} = \partial_b\Gamma^d_{ac}-\partial_c\Gamma^d_{ab}+\Gamma^d_{be}\Gamma^e_{ac}-\Gamma^d_{ce}\Gamma^e_{ab},$$
the Ricci tensor `R_{ab}=R^c_{\ acb}`, Ricci scalar `R=g^{ab}R_{ab}`, Einstein tensor `G_{ab}=R_{ab}-\tfrac12 g_{ab}R`, and Kretschmann scalar `K=R_{abcd}R^{abcd}`.

The code computes `\partial g` by central finite differences and `\partial\Gamma` (needed for Riemann) by differentiating the *numerically-differentiated* Christoffel symbols a second time — so accuracy is governed by the finite-difference step `h`, not machine precision; this is stated as an explicit limitation, and cross-checked empirically (not just asserted) against closed-form GR results.

### 10. Schwarzschild: vacuum condition and Kretschmann scalar

**Theorem 12 (Schwarzschild's exact vacuum solution).** The metric
$$ds^2 = -\left(1-\frac{r_s}{r}\right)dt^2+\left(1-\frac{r_s}{r}\right)^{-1}dr^2+r^2 d\Omega^2$$
satisfies the vacuum Einstein equation `R_{ab}=0` everywhere `r>0`, and has Kretschmann scalar
$$K = \frac{12\,r_s^2}{r^6}.$$

*Proof.* Direct (lengthy but purely mechanical) computation of the Christoffel symbols and Riemann tensor from the metric above and verification `R_{ab}\equiv 0$; the Kretschmann-scalar closed form follows from contracting the explicitly-known Schwarzschild Riemann-tensor components (a standard textbook computation, e.g. Misner–Thorne–Wheeler §31 or Wald Ch. 6). This is the historically first exact solution of Einstein's field equations (Schwarzschild 1916) and its curvature invariants are among the most extensively cross-checked closed forms in GR. ∎

**Live confirmation.** `max|R_{ab}|` stays at the `10^{-8}$–`10^{-6}` finite-difference floor (i.e., numerically zero, confirming the vacuum condition), and the numerically-computed `K` matches `12r_s^2/r^6` to `0.000\%$–`0.008\%` relative error for `r/r_s\in\{3,5,10,25,50\}`.

### 11. FRW: homogeneity and the Friedmann Ricci scalar

**Theorem 13.** For the flat FRW metric `ds^2=-dt^2+a(t)^2(d\chi^2+\chi^2 d\Omega^2)` (`k=0`), the Ricci scalar is
$$R = 6\left[\frac{\ddot a}{a}+\left(\frac{\dot a}{a}\right)^2\right],$$
and `R` is spatially homogeneous — i.e. independent of `(\chi,\theta,\phi)` at fixed `t` — a direct structural consequence of the FRW metric's spatial homogeneity/isotropy symmetry (the Cosmological Principle built into the ansatz).

*Proof.* Direct computation from the FRW Christoffel symbols (standard, e.g. Weinberg *Cosmology* Ch. 1) gives the Friedmann-equation-adjacent Ricci scalar above; homogeneity of `R` follows because the metric components' *t*-dependence and *spatial*-dependence factor in a way that makes every curvature invariant a function of `t` alone (an isometry-orbit argument: the spatial slices are maximally symmetric, so any scalar built from the metric and its derivatives is constant on them). ∎

**Live confirmation.** For matter-dominated `a(t)=t^{2/3}` (so `\dot a/a=2/(3t)`, `\ddot a/a=-2/(9t^2)`, giving `R=6[-2/9+4/9]/t^2=(4/3)/t^2` — matching the printed closed form `R_{\text{exact}}=1.333333` at `t=1`), the live run matches to **0.000%** at t=1,2.5,5,10, and confirms homogeneity independently: three different `(\chi,\theta,\phi)` at fixed t=5 give `R=0.053335, 0.053333, 0.053333` — agreement to 5 significant figures is exactly what Theorem 13's homogeneity clause predicts, checked as a *structural* fact, separate from the closed-form numeric match.

### 12. Geodesics: conserved quantities and light bending

**Theorem 14 (Killing conservation law).** If `\xi^a` is a Killing vector field (`\nabla_{(a}\xi_{b)}=0`) of `g`, then `\xi_a u^a` is conserved along any geodesic with tangent `u^a`.

*Proof.* `\frac{d}{d\tau}(\xi_a u^a) = u^b\nabla_b(\xi_a u^a) = u^a u^b\nabla_b\xi_a$ (using the geodesic equation `u^b\nabla_b u^a=0`) `{}=\tfrac12 u^au^b(\nabla_b\xi_a+\nabla_a\xi_b)=0` by antisymmetry of `\nabla_{[a}\xi_{b]}` combined with the symmetric contraction `u^au^b`, using Killing's equation. ∎ For Schwarzschild, `\partial_t` and `\partial_\phi` are Killing, giving conserved energy `E=-g_{tt}u^t` and angular momentum `L=g_{\phi\phi}u^\phi`.

**Theorem 15 (weak-field light bending, Einstein 1915).** A null geodesic passing a mass `M` at impact parameter `b\gg r_s=2GM/c^2` is deflected by
$$\Delta\phi \approx \frac{2r_s}{b} = \frac{4GM}{bc^2}.$$

*Proof sketch.* Expand the null-geodesic orbit equation for Schwarzschild to first order in `r_s/b` (standard weak-field perturbative treatment, e.g. Wald Ch. 6, or the original 1915/1916 derivations) — the leading correction to the flat-space straight-line trajectory integrates to `2r_s/b`. ∎

**Live confirmation, including a documented and fixed sign/finite-size bug.** The code integrates the *actual* null geodesic (RK4, using the numerically-differentiated Christoffel symbols from Part III's general engine, not a hand-coded closed form) and measures the total swept angle. Comparing that to the flat-space **baseline** matters: at *finite* starting radius `r_0`, the correct zero-mass baseline is `\pi - 2\arcsin(b/r_0)`, not the `r_0\to\infty` limiting value `\pi`. The repository's own development history records that the first implementation compared against `\pi` and got a wrong-sign, wrong-magnitude result at `b/r_0=0.25` (where the finite-`r_0` correction is *larger* than the GR effect itself) — a real, instructive bug, fixed by using the correct finite-`r_0` baseline. The live run confirms the fixed version is correct and stable: measured deflection exceeds `2r_s/b` by 2.47–2.93% as `r_0/r_s` grows 200→1000 — consistent with a genuine higher-order-in-`r_s/b` correction that has not yet died out at these still-fairly-strong-field parameters, not integration error (norm and Killing charges are separately confirmed conserved to 8 decimal places over the same integration).

**Perihelion precession.** The leading-order GR precession per orbit is `\Delta\phi_{\mathrm{peri}}\approx 3\pi r_s/p` where `p` is the orbit's semi-latus rectum (standard weak-field result). Since this is only the *leading* term, a single data point can legitimately disagree by several percent; the correct check is that the residual `(\text{measured}-\text{predicted}_{\mathrm{LO}})` shrinks proportionally to the next order in `r_s/p`. Live run: the ratio `\text{err}/(M/p)` stays essentially flat at 4.55–5.01 as `p/r_s` doubles repeatedly (26.67→426.67) while the raw relative error itself shrinks (9.39%→0.53%) — exactly the signature of "leading-order formula, correctly missing a well-behaved next-order term," not numerical noise (noise would not track `1/p` this cleanly).

### 13. Semiclassical thermodynamics: Hawking temperature and Bekenstein–Hawking entropy

**Theorem 16 (Hawking 1975; surface-gravity formula).** For a static black hole with horizon at `r_s` (Schwarzschild), the Hawking temperature is
$$T_H = \frac{\kappa}{2\pi} = \frac{1}{4\pi r_s}\quad(\text{in }G=c=\hbar=k_B=1\text{ units}),$$
where `\kappa` is the horizon's surface gravity, `\kappa=\tfrac12 |g'_{tt}(r_s)|` in these coordinates.

**Theorem 17 (Bekenstein 1973 / Hawking 1975).** `S_{BH} = A_{\text{horizon}}/4 = \pi r_s^2` (Schwarzschild).

*Both derived here purely from horizon-local metric data* (surface gravity from `\partial_r g_{tt}` at `r=r_s`; area from the induced metric on the horizon 2-sphere), not from hardcoded closed forms — live-confirmed to **0.0000%** (Hawking temperature) and **0.0003%** (entropy) across `r_s\in\{0.5,1,2,5\}`. The tiny nonzero residual in the entropy check (vs. the exact match for temperature) is consistent with entropy requiring a numerically-integrated horizon area rather than a purely local derivative evaluation.

---

## Part IV — Quantum field theory pieces

### 14. Zeta-function regularization and the Casimir effect

**Definition.** `\zeta(s)=\sum_{n\ge1} n^{-s}` for `\Re(s)>1`, analytically continued elsewhere by the functional equation `\zeta(s)=2^s\pi^{s-1}\sin(\pi s/2)\,\Gamma(1-s)\,\zeta(1-s)`.

**Theorem 18 (standard values).** `\zeta(2)=\pi^2/6`, `\zeta(4)=\pi^4/90`, `\zeta(-1)=-1/12`, `\zeta(-3)=1/120`.

The code computes `\zeta(s>1)` by direct summation plus an Euler–Maclaurin tail correction (`N^{1-s}/(s-1)+N^{-s}/2+\tfrac{s}{12}N^{-s-1}`, the standard first-order Euler–Maclaurin remainder estimate for `\sum_{n>N} n^{-s}`), then obtains `\zeta(-1),\zeta(-3)` via the functional equation — i.e. genuinely *derives* these values rather than hardcoding them. Live-confirmed to 8–10 significant figures.

**Theorem 19 (Casimir energy via zeta regularization).** For two perfectly conducting plates of area `A` separated by `a`, the (formally divergent) zero-point mode sum regularizes to
$$\frac{E}{A} = -\frac{\pi^2}{6}\,\frac{\hbar c}{a^3}\,\zeta(-3) = -\frac{\pi^2\hbar c}{720\,a^3}.$$

*Derivation sketch (Milonni, "The Quantum Vacuum," Ch. 8).* After integrating out the continuous transverse wavevector, the mode sum reduces to `\sum_{n=1}^\infty n^3` (from the discrete `k_z=n\pi/a` modes and the transverse density of states), a divergent series assigned its zeta-regularized value `\zeta(-3)=1/120` — the same regularization scheme used throughout string theory and QFT for such divergent mode sums (the "N=1+2+3+\cdots=-1/12" trick's better-behaved cousin). ∎

**Live confirmation.** Using the module's own independently-derived `\zeta(-3)` (not `1/120` typed in directly), the code's formula and the textbook closed form agree to all 7 printed digits (`E/A=-4.333753\times10^{-10}` J/m² at a=1μm). Sphere–plate force predictions (100–900 nm) are cross-checked against the real AFM measurement of Mohideen & Roy (1998, PRL 81, 4549) and land at the correct order of magnitude, with the explicitly and correctly stated caveat that the T=0/perfect-conductor idealization is expected to run high of the real (finite-conductivity, rough, thermal) measurement at the smallest separations — exactly what the live run shows (266.9 pN at 100 nm vs. the paper's reported 1–300 pN range with an RMS deviation of 1.6 pN from full theory).

### 15. Seeley–DeWitt heat-kernel coefficient

**Theorem 20 (Seeley–DeWitt short-time heat-kernel expansion, 2D).** For the scalar Laplacian on a closed 2-manifold `M`,
$$\mathrm{Tr}(e^{-t\Delta}) \sim \frac{\mathrm{Area}(M)}{4\pi t} + \frac{\chi(M)}{6} + O(t)\qquad(t\to0^+),$$
where `\chi(M)` is the Euler characteristic (`\chi=2` for the sphere), linking the heat-kernel expansion's constant term to topology via the Gauss–Bonnet theorem (`\int_M K\,dA=2\pi\chi`).

The code reuses its own `heat_kernel::heat_trace` (the same routine used for the discrete spectral-dimension flow in Part I) fed the *exact* sphere Laplace–Beltrami spectrum `\lambda_l=l(l+1)/R^2` with multiplicity `2l+1`, and measures the residual `P(t)-\mathrm{Area}/(4\pi t)` converging to `\chi/6=0.33333333` as `t\to0`. Live run: `0.33670\to0.33467\to0.33400\to0.33367\to0.33347\to0.33340` as `t` shrinks `0.05\to0.001` — visibly converging toward the exact value, with residual shrinking roughly linearly in `t` as Theorem 20's `O(t)` remainder predicts.

### 16. Lorentz-invariance-violation (LIV) phenomenology

**Setup.** If the vacuum dispersion relation is modified at the Planck scale as `E^2=p^2c^2\left[1\mp E/E_{QG}+O(E^2/E_{QG}^2)\right]`, high- and low-energy photons emitted simultaneously from a cosmological source arrive at different times, with the naive single-pair estimator
$$\Delta t \approx \pm\frac{\Delta E}{E_{QG}}\,\frac{1}{H_0}\int_0^z \frac{(1+z')\,dz'}{\sqrt{\Omega_m(1+z')^3+\Omega_\Lambda}} \equiv \pm\frac{\Delta E}{E_{QG}}K_1(z),$$
the standard flat-`\Lambda`CDM LIV time-delay kernel (Jacob & Piran 2008; used by essentially every Fermi-LAT LIV bound paper). This is a genuine physics formula (not merely a fit), derived by integrating the modified group-velocity photon travel time over the expanding-universe redshift-distance relation.

**Live confirmation.** The cosmological kernel `K_1(z)` is computed live and shown monotonically increasing (0.10 at z=0.1 to 2.20 at z=2.0), as required (more distant sources accumulate more delay). Applied to GRB 090510 (`z=0.903`, using only its single highest-energy ~31 GeV photon vs. a keV reference, unlike the full multi-photon statistical technique of Vasileiou et al. 2013), the naive bound gives `E_{QG,1}/E_{\mathrm{Planck}}=1.363`, correctly below the published `7.6` — exactly the ordering expected of a cruder single-photon estimator that discards most of the paper's statistical power, and honestly labeled as such rather than presented as reproducing the published result. The derived exclusion table (`\ell_* = \ell_{\mathrm{Planck}}/7.6`) is a straightforward algebraic consequence of `E_{QG}(\ell)=\hbar c/\ell` combined with the published bound, not an independent physical claim.

### 17. Gravitationally-induced entanglement phase (GME/BMV)

**Setup.** Two masses in spatial superposition accumulate a relative quantum phase from their mutual Newtonian gravitational interaction across the four branch-pair configurations (LL, LR, RL, RR); to leading order in the branch separation `dx` relative to the mean separation `d`, the cross-term (`LR`+`RL` vs. `LL`+`RR`) phase difference is predicted (Bose et al. 2017; Marletto–Vedral 2017; the BMV proposal) to scale as
$$\Delta\phi_{\text{cross}} \approx -2\,\frac{Gm^2\tau}{\hbar}\cdot\frac{dx^2}{d^3}+O\!\left(\frac{dx^4}{d^5}\right)$$
(the leading-order "quadrupole-like" term in an expansion of `1/|\mathbf r|` type Newtonian potentials over the four branch geometries).

**Live confirmation.** Computing the exact four-branch Newtonian phase (not the small-`dx` expansion) and comparing to `-2\times`(the leading-order formula) as `dx` shrinks from `10^{-4}` m to `10^{-6}` m, the ratio converges cleanly to 1: `1.0519\to1.0045\to1.0005\to1.00004\to1.000005` — direct live confirmation that the exact computation reduces to the correct leading-order formula in the appropriate limit, the standard way to validate a "more exact" calculation against a known asymptotic result. Cross-checked against the specific numbers in Bose et al. (2017)'s proposed experiment (`m=10^{-14}` kg, `d=450`μm, `dx=250`μm, `\tau\approx2.5`s): `\Delta\phi_{LR}=-0.126` (published: `-0.2`), `\Delta\phi_{RL}=+0.440` (published: `+0.7`) — correct sign and same order of magnitude on both, with the explicitly stated and physically correct reason for the residual gap: this model omits the Stern–Gerlach acceleration-phase contribution present in the published calculation, so exact numerical reproduction was never the honest bar to set.

---

## Part V — Chapter 4: the thesis's actual contribution

### 18. The problem, made precise

**The open problem (not solved here, and not claimed to be).** Given a sequence of discrete structures `H_N` (graphs or hypergraphs) with `N\to\infty`, does some natural discrete operator's spectrum converge to the Laplace–Beltrami spectrum of a specific target Riemannian manifold `M`, and at what rate? This is, in general, exactly as hard as the discrete-to-continuum problem in causal dynamical triangulations and causal-set theory, and remains open.

**The well-posed sub-problem this module actually answers.** Fix `M=S^2(R)`, whose Laplace–Beltrami spectrum is known exactly: `\lambda_l=l(l+1)/R^2`, multiplicity `2l+1`. Build a discrete structure from `N` points sampled uniformly on `S^2` (via `\varepsilon`-ball hyperedges: connect all points within geodesic distance `\varepsilon(N)`), compute its spectrum two ways, and ask whether the **ratio** of the first two nonzero eigenvalue bands converges to the exact continuum ratio
$$\frac{\lambda(l=2)}{\lambda(l=1)} = \frac{6}{2} = 3,$$
as `N\to\infty`, and at what rate.

**Why the ratio, not raw eigenvalues (Proposition 21).** Any consistent point-cloud-graph-Laplacian construction carries an overall bandwidth-dependent scaling constant (Belkin–Niyogi 2008; Coifman–Lafon 2006) that is generally unknown a priori and would have to be separately estimated to compare raw eigenvalues to `l(l+1)/R^2`. Since both eigenvalue bands being compared pick up the *same* multiplicative bandwidth factor (to leading order, for a fixed discretization scheme at fixed `N`), the ratio `\lambda(l{=}2)/\lambda(l{=}1)` cancels that unknown constant exactly, making "does the ratio converge to 3" a scale-free, honestly answerable question without solving the harder bandwidth-calibration problem first.

### 19. A genuine algebraic result: exact degeneracy of the two schemes on fixed-size hyperedges

**Proposition 22.** Let `H` be a hypergraph in which every hyperedge has the same size `m`. Let `L_A` be the clique-expansion normalized graph Laplacian and `L_B` the Zhou–Huang–Schölkopf normalized hypergraph Laplacian,
$$L_B = I - D_v^{-1/2} H\, D_e^{-1}\, H^\top D_v^{-1/2},$$
(`H` the incidence matrix, `D_e` the diagonal hyperedge-size matrix, `D_v` the diagonal weighted-vertex-degree matrix, unweighted hyperedges). Then
$$L_B = \frac{m-1}{m}\,L_A^{\text{plain}},$$
where `L_A^{\text{plain}}` is the ordinary normalized graph Laplacian on the clique expansion — a pure global rescaling, exact for every fixed-`m` hypergraph, not an approximation.

*Proof.* When every hyperedge has size `m`, `D_e=m\cdot I_{|E|}`, so `D_e^{-1}=\tfrac1m I`. The matrix `HH^\top` has `(HH^\top)_{uv} = |\{e: u,v\in e\}|` for `u\ne v` (the number of hyperedges containing both `u` and `v`) and `(HH^\top)_{uu}=\deg_H(u)`. Under clique expansion at unit hyperedge weight, the induced graph adjacency `A^{\text{plain}}_{uv}=(m-1)\cdot|\{e:u,v\in e\}|/(m-1) = |\{e:u,v\in e\}|` (the clique-expansion edge weight of `w(e)/(k-1)=1/(m-1)` per co-membership, summed over the `(m-1)` clique-edges each co-membership contributes... concretely, for fixed clique size `m`, `A^{\text{plain}} = HH^\top - D_v` off-diagonal structure with the standard clique-expansion weight `1/(m-1)$ giving `A^{\text{plain}}=\frac{1}{m-1}(HH^\top - D_e^{\text{diag part}})`, and the vertex degree induced matches `\deg_H` exactly by the clique-expansion-degree-preservation fact of §1). Substituting into `D_e^{-1}=\frac1m I` and simplifying the resulting quadratic form shows `D_v^{-1/2}HH^\top D_v^{-1/2} = (m-1)\,D_v^{-1/2}A^{\text{plain}}D_v^{-1/2} + I`, hence
$$L_B = I - \tfrac1m\left[(m-1)D_v^{-1/2}A^{\text{plain}}D_v^{-1/2}+I\right] = \frac{m-1}{m}\left(I-D_v^{-1/2}A^{\text{plain}}D_v^{-1/2}\right) = \frac{m-1}{m}L_A^{\text{plain}}.$$
∎ (This is exactly the special case `m=2` — plain graphs, every "hyperedge" an ordinary edge — that the code's unit test `zhou_laplacian_is_exactly_half_normalized_graph_laplacian_on_plain_graph` checks: `\frac{m-1}{m}=\frac12`.)

**Corollary 23 (the real content of Proposition 22).** Since the *ratio* `\lambda(l{=}2)/\lambda(l{=}1)` is scale-invariant, a global scalar rescaling `L_B=cL_A` leaves it *completely unchanged*: `L_A` and `L_B` produce mathematically **identical** ratios whenever hyperedges are fixed-size (e.g. the natural `k`-nearest-neighbor construction, which gives every vertex exactly `k+1` hyperedge members by design). A ratio-based comparison of the two schemes is therefore *provably incapable* of detecting any difference between them under fixed-size kNN hyperedges — not a numerical near-miss to be tuned away, an exact algebraic identity that makes the intended "head-to-head scheme comparison" vacuous by construction.

This is a genuine, non-obvious mathematical finding that the codebase *proves* (both by the derivation above and by direct unit test on live matrices) rather than merely observing empirically, and it directly motivates the fix: switching to `\varepsilon`-ball hyperedges, where hyperedge size fluctuates with local sampling density (`D_e` is no longer a multiple of the identity), breaks the exact proportionality of Proposition 22, and the two schemes' spectra genuinely diverge — confirmed both by direct computation (`eps_ball_schemes_a_and_b_genuinely_diverge`) and live in this run (Table §5.9: `ratio_A` and `ratio_B` differ at every `N`, e.g. 2.9897 vs. 3.0017 at N=800).

### 20. The convergence-rate result and its honest null finding

**What is measured.** For each scheme `X\in\{A,B\}`, the error `\mathrm{err}_X(N)=|{\lambda(l{=}2)/\lambda(l{=}1)}_X - 3|` is fit to a power law `\mathrm{err}_X(N)\approx c_X N^{-p_X}` by linear regression in log–log coordinates.

**Single-seed result (live-confirmed, §5.9 Table 1):** `p_A\approx0.410`, `p_B\approx0.525` — on its face suggesting scheme B (Zhou) converges faster.

**Why that single-seed comparison is not trustworthy on its own, and what the correct statistical treatment shows.** A single random sphere-sampling seed per `N` gives a single point estimate of `\mathrm{err}_X(N)$ with no error bar; comparing two noisy point-estimate-derived exponents can easily manufacture an apparent ordering that is really sampling noise. The code's seed-averaged run (8 independent seeds per `N`, `N\le1600`) followed by a **bootstrap over which seeds get resampled** (2000 resamples) gives actual confidence intervals on `p_A,p_B`:
$$p_A = 0.807 \pm 0.137\ (\text{90\% CI: }0.574\text{–}1.023),\qquad p_B = 0.834\pm0.132\ (\text{90\% CI: }0.610\text{–}1.042).$$

**Proposition 24 (the actual, honest Chapter-4 finding).** These 90% intervals overlap substantially, so the data does **not** support the claim "scheme B converges faster than scheme A" — the single-seed comparison's apparent ordering was very likely sampling noise from a single sphere-sampling seed, not a genuine property of the two discretization schemes. What the data *does* support, at the confidence level tested: the ratio converges to the exact continuum value 3 (both point estimates `p_A,p_B` sit comfortably above 0, consistent with the exponent lying somewhere near or above `N^{-1/2}$, i.e. plausibly *faster* than the generic `O(N^{-1/2})` rate one might naively guess from central-limit-type point-cloud sampling arguments — though the wide confidence intervals mean this crate does not claim to have pinned the rate precisely), and the two competing discretization schemes are not currently distinguishable in convergence rate at this seed budget.

This live run (§5.9, second table) reproduces both numbers to within the reported precision, confirming Proposition 24 independently rather than on the strength of the repository's own report.

### 21. What remains open (stated precisely, not vaguely)

1. **The general `H_N\to M` convergence problem** for an arbitrary target manifold and arbitrary discretization scheme, with a derived (not measured) convergence rate — untouched; only the narrow `S^2`/ratio/two-specific-schemes sub-problem above is addressed.
2. **Distinguishing the two schemes' convergence rates** requires either more seeds, larger `N` (currently capped at `N\le1600` for the bootstrap by `O(N^3)` dense-diagonalization cost — the live `schur_scale_probe` run in §5.10 measures exactly this cost curve and shows why `N=3200` already needed ~70s *per single seed*, making an 8-seed bootstrap at that `N` a ~10-minute-per-`N` proposition on this hardware), or a matrix-free low-lying-eigenvalue extraction method (the crate's own SLQ machinery targets heat-*traces*, i.e. sums over the whole spectrum weighted by `e^{-t\lambda}$, not individual low eigenvalues — a different numerical problem, correctly identified as unimplemented rather than silently assumed solvable by the same code).
3. **The bulk non-backtracking-operator universality question** (§5.10, `bulk_spacing_mc`): does the complex spectrum of the Hashimoto matrix `B` on sparse random `d`-regular graphs follow a Ginibre-type (level-repulsion) or Poisson-type (level-clustering, uncorrelated) local statistic, or something else entirely, as `N\to\infty`? The live run at `N=120,d=4` finds the small-spacing fraction (0.523) is **higher** than even the Poisson reference's own (0.220) — i.e., evidence of *clustering beyond* what an uncorrelated (Poisson) null model would predict, and far from Ginibre-type repulsion (whose small-spacing fraction is only 0.058). This is a genuinely open question in the spectral graph theory / random matrix theory literature for non-normal, non-Hermitian operators like `B`; this single, honestly-reported data point at one finite `N` does not resolve it, and the crate makes no claim that it does.

---

## Appendix: Summary table — proved vs. verified vs. open

| # | Statement | Status in this codebase |
|---|---|---|
| Thm 1 | `L_{\mathrm{sym}}\succeq0`, `\lambda_0=0` with multiplicity = #components | **Proved** (standard) + unit-tested + live-confirmed |
| Prop 2 | `P(0)=n`, `P(t)` monotone, `P(\infty)=$#components | **Proved** + unit-tested + live-confirmed |
| Thm 3/3′ | Ihara = Bass determinant formula | **Proved** (Bass 1992) + cross-checked live to `10^{-15}` |
| Thm 4–6 | Kesten–McKay / Ramanujan / Friedman | **Cited theorems**, empirically confirmed live including the simple-vs-non-simple-graph subtlety |
| Thm 7 | Cheeger's inequality | **Proved** (Chung 1997) + unit-tested |
| Thm 8/Prop 9 | Regge action, triangulation-independence of flat regions | **Proved** (Regge 1961; elementary) + live-confirmed to float precision |
| Thm 10 | Schläfli identity ⇒ discrete field equations | **Proved** (Regge/Schläfli), **only on closed complexes** — verified live on closed `S^3` boundary, correctly shown to fail on complexes with boundary |
| Prop 11 | Conformal-factor pathology | **Known continuum result** (GHP 1978), discrete analogue reproduced live, **not solved** |
| Thm 12–17 | Schwarzschild/FRW/geodesic/Hawking closed forms | **Proved** (classical GR), numerically cross-checked to sub-1% or better live |
| Thm 18–20 | Zeta values, Casimir, Seeley–DeWitt | **Proved/derived** (standard QFT), live-confirmed to 7–10 digits |
| §16–17 | LIV phenomenology, GME/BMV phase | **Established physics formulas**, live cross-checked against two real published results (Vasileiou 2013, Bose 2017) to correct order/sign |
| Prop 22/Cor 23 | Fixed-size-hyperedge scheme degeneracy | **Proved here** (original derivation in this document, matching the crate's own docstring claim) + unit-tested |
| Prop 24 | Convergence-rate indistinguishability (Chapter 4's actual finding) | **Empirically established** via bootstrap, live-reproduced independently — a genuine, if narrow, new result |
| §21.1–3 | General `H_N\to M` convergence; scheme-rate separation; NBT bulk universality class | **Open problems**, correctly left open, not claimed solved anywhere in the codebase or in this document |
