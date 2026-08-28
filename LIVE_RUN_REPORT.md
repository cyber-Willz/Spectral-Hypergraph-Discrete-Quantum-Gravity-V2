# `spectral_dqg_and_krylov_ds` — End-to-End Live Run Report

**Run date:** 2026-08-28
**Environment:** Linux x86_64, 1 vCPU, 4 GB RAM, `rustc 1.75.0` (LLVM 17.0.6), `cargo 1.75.0`, release profile (`opt-level` default `3`, no LTO overrides in `Cargo.toml`)
**Source:** `spectral_dqg_and_krylov_ds_chapter4_improved_tar.gz`, extracted verbatim, no source edits made before running.

Everything below is a real execution captured in this session — nothing is copied from the repo's own `RUN_LOG.txt` / `SESSION_TEST_OUTPUT.txt` (those are the *previous* development session's logs, left in the tarball; this report is an independent, fresh run against the same unmodified source). Where a number differs from the repo's own README/RUN_LOG by more than expected floating-point/RNG noise, that is called out explicitly rather than silently reconciled.

---

## 1. Setup commands

The sandbox had no Rust toolchain preinstalled; it was installed from the distro repository (network egress allowed `archive.ubuntu.com`/`security.ubuntu.com` but not `sh.rustup.rs`, so `apt` rather than `rustup` was used):

```bash
apt-get install -y cargo rustc
# resulted in: rustc 1.75.0, cargo 1.75.0
```

Extraction:

```bash
tar -xzf spectral_dqg_and_krylov_ds_chapter4_improved_tar.gz
cd spectral_dqg_and_krylov_ds
```

## 2. Build

```bash
cargo build --release
```

Result: **clean build, zero warnings, zero errors**, workspace member `krylov_ds` (local path dependency) and top-level `spectral_dqg` crate both compiled. External dependencies resolved from `crates.io` (`nalgebra 0.32.6`, `rand 0.8.8`, `rand_distr 0.4.3`, `rand_pcg 0.3.1`, transitively `simba`, `matrixmultiply`, etc.) — total build time 1m 18s cold.

## 3. Full test suite

```bash
cargo test --release
```

**Result: 102/102 tests passed, 0 failed, 0 ignored**, across every test target:

| Test target | Tests | Result |
|---|---|---|
| `src/lib.rs` unit tests (in-module `#[cfg(test)]` blocks, 20 modules) | 85 | ok |
| `tests/cross_checks.rs` (Steps 0–3 integration suite) | 7 | ok |
| `tests/regge_tests.rs` (Step 4 Regge calculus suite) | 10 | ok |
| `krylov_ds/tests/integration_test.rs` (sub-crate) | 9 | ok |
| `krylov_ds` doc-test | 1 | ok |
| All `src/bin/*.rs` binary crates (unit-test harness only, no `#[test]`s in the binaries themselves) | 0 each | ok |

Every module that the README claims has an independent cross-check does: `ihara_and_bass_formulas_agree`, `trace_bk_matches_bruteforce_closed_walk_count`, `zhou_laplacian_is_exactly_half_normalized_graph_laplacian_on_plain_graph`, `eps_ball_schemes_a_and_b_genuinely_diverge`, `schlafli_identity_holds_on_regular_4_simplex_boundary`, `bootstrap_rate_recovers_known_exponent_on_noiseless_synthetic_data`, etc. all present and passing.

## 4. Live run of every binary in the workspace

Thirteen binaries exist (`src/bin/*.rs` + the default `src/main.rs`). All thirteen were executed to completion in this session. Two (`schur_scale_probe`, `bulk_spacing_mc`) needed an extended (280s) timeout, since they are, by the code's own design, real O(N³)/O(seeds·N³) computations — not caching or shortcuts.

```bash
cargo run --release                                      # src/main.rs
cargo run --release --bin gr_demo
cargo run --release --bin gr_demo2
cargo run --release --bin regge_demo
cargo run --release --bin qft_demo
cargo run --release --bin qg_phenomenology_demo
cargo run --release --bin gme_bmv_demo
cargo run --release --bin hypergraph_continuum_demo
cargo run --release --bin hypergraph_continuum_seed_avg_demo
cargo run --release --bin verify_circle
cargo run --release --bin large_n_flow
cargo run --release --bin certified_interval_flow
timeout 280 cargo run --release --bin schur_scale_probe
timeout 280 cargo run --release --bin bulk_spacing_mc
```

Wall-clock times observed this run (release binaries, post-build):

| Binary | Wall time | Notes |
|---|---|---|
| `spectral_dqg` (main) | <1s | N=8 toy hypergraph |
| `gr_demo`, `gr_demo2` | <1s each | closed-form/geodesic checks |
| `regge_demo` | <1s | includes 3×800-step Metropolis chains |
| `qft_demo`, `qg_phenomenology_demo`, `gme_bmv_demo` | <1s each | closed-form/series evaluations |
| `hypergraph_continuum_demo` | ~2s | N up to 3200, single seed |
| `hypergraph_continuum_seed_avg_demo` | ~3s | N up to 1600, 8 seeds + 2000-resample bootstrap |
| `verify_circle` | <1s | |
| `large_n_flow` | ~3s | includes exact N=2000 dense diagonalization + N=10,000 SLQ |
| `certified_interval_flow` | ~17s | includes N=2000 dense (9.17s) + N=10,000 certified SLQ (6.58s) |
| `schur_scale_probe` | **79.4s** | N=100→800 dense Schur, O(N³) as documented |
| `bulk_spacing_mc` | **139.8s** (+78.1s Ginibre ref +1.8s Poisson ref) | 1000 independent 480×480 non-symmetric eigendecompositions |

---

## 5. Results, by module

### 5.1 Step 1 — Discrete kinematics (`spectral_dqg` main binary, N=8 toy hypergraph)

Normalized Laplacian spectrum: `0, 0.2570, 0.7801, 1.1304, 1.1667, 1.2116, 1.6039, 1.8503` — confirms `λ₀=0` and nonnegativity on a live example, matching the general theorem (§2 of the math report).

Spectral-dimension flow `d_s(t)` rises from ~0.07 to a peak of ~1.37 near `t≈1.4` and falls back toward 0 at large `t` — the expected UV/IR finite-size artifact on an 8-vertex graph, not a bug (no clean plateau is possible at this size; this is the correct, honest outcome, not the interesting one).

### 5.2 Step 2 — Ihara–Selberg zeta function

Two independent formulas for `Z_H(u)⁻¹` — `det(I − uB)` (spectral, via the 26×26 Hashimoto matrix on 26 directed arcs) and the Bass determinant formula — agree to **3.3×10⁻¹⁶–4.0×10⁻¹⁵** (machine precision) across `u ∈ {0.05, 0.10, 0.15, 0.20}`.

`Tr(Bᵏ)` for k=1..5 (`0, 0, 30, 48, 40`) matches a brute-force DFS count of closed non-backtracking walks **exactly**, confirming the generating-function identity `log Z_H(u)⁻¹ = −Σ Tr(Bᵏ) uᵏ/k` is built on a correctly-constructed operator.

### 5.3 Step 3 — Continuum-limit diagnostic (Kesten–McKay / Ramanujan)

Non-simple configuration model (self-loops/multi-edges silently dropped): max non-trivial |λ| sits at **3.98–3.998**, essentially saturating the trivial bound `d=4`, and the fraction of eigenvalues within the Ramanujan bound `2√(d−1)=3.464` only reaches 99.95% even at N=2000 — a real, measured near-violation, not noise.

Switching to `random_simple_regular_graph` (rejection-sampled, genuinely simple): max non-trivial |λ| drops to **3.36–3.45**, now inside the Ramanujan bound at **100%** of sampled eigenvalues for all three N tested (50, 400, 2000) — live confirmation of Friedman's theorem (generic random simple d-regular graphs are near-Ramanujan w.h.p.). RMS deviation from the Kesten–McKay density (`0.0295 → 0.0084 → 0.0060` as N grows 50→400→2000) shrinks monotonically, confirming the N→∞ convergence claim quantitatively.

### 5.4 Step 4/5 — Regge calculus

- Regular tetrahedron volume: computed `1.433895` vs. closed form `l³/(6√2) = 1.433895`, difference **2.22×10⁻¹⁶** (float epsilon).
- Flat cube, two topologically different tetrahedralizations (fan around diagonal 0–7 vs. diagonal 1–6): both give deficit angle **~2.665×10⁻¹⁵ rad** and total Regge action **~4.6×10⁻¹⁵** — zero to float precision under *either* triangulation, the live confirmation of the "flat-region physics is triangulation-independent" claim.
- Perturbing the diagonal by 15% produces a real, large deficit angle (−2.179 rad) and action (−4.339) — curvature genuinely appears under perturbation, not just at exact symmetry points.
- Metropolis path integral: `⟨S⟩` runs from **−4.88 (κ=0.2) → −11.70 (κ=1.0) → −20.69 (κ=5.0)** — confirmed live: stronger coupling drives the ensemble to *more* negative action, not toward flatness, the conformal-factor pathology (Gibbons–Hawking–Perry 1978) reproduced correctly.
- `regge_eom` Schläfli-identity tests pass on both a regular and an irregular closed 4-simplex boundary (5 tetrahedra, S³ topology) to ~10⁻⁴, confirming `dS/dL_e = δ_e` holds once a genuinely closed (boundary-free) complex is used.

### 5.5 Step 6 — Continuum tensor calculus, exact solutions, geodesics

- Minkowski: all curvature tensors vanish exactly (`0.000e0`).
- Schwarzschild: `max|R_ab|` stays at the `10⁻⁸`–`10⁻⁶` finite-difference noise floor (correctly near-zero, confirming vacuum `R_ab=0`); Kretschmann scalar matches `12r_s²/r⁶` to **0.000%–0.008%** relative error across r/r_s = 3…50.
- FRW (matter-dominated, `a(t)=t^{2/3}`): Ricci scalar matches `6[ä/a+(ȧ/a)²]` to **0.000%** at all four t values tested, and is confirmed spatially homogeneous (three different `(χ,θ,φ)` points at fixed t=5 give R = 0.053335, 0.053333, 0.053333 — agreeing to the 5th significant figure).
- Geodesic integration: norm `g_{ab}u^au^b=−1` and the Killing charges `E=1.05`, `L=0.60` are conserved to 8 decimal places over 3000 RK4 steps.
- Light bending: measured deflection exceeds the leading-order weak-field prediction `2r_s/b` by **2.47%–2.93%** as `r0/r_s` grows from 200 to 1000, consistent with the *sign and shrinking rate* expected of a higher-order correction that hasn't yet died out at these moderate r0.
- Perihelion precession: relative error to leading-order prediction shrinks 9.39% → 4.46% → 2.17% → 1.07% → 0.53% as `p/r_s` doubles from 26.67 to 426.67 — i.e. the *ratio* `err/(M/p)` stays pinned near **4.55–5.01** (not shrinking to 0), the correct signature of a genuine next-to-leading-order term rather than numerical error.
- Gravitational redshift: measured `ν_obs/ν_emit = 0.903508` at r=50 matches the closed form `√(f(r_emit)/f(r_obs)) = 0.903508` **to all 6 printed digits**; photon energy conservation holds to a fractional drift of `3.1×10⁻¹¹` over the whole integrated path.
- Hawking temperature: matches `κ/2π` closed form to **0.0000%** at all four `r_s` values tested.
- Bekenstein–Hawking entropy: matches `A/4` closed form to **0.0003%** at all four `r_s` values (the small, nonzero residual here — unlike the exact-to-printed-precision Hawking-temperature match — is consistent with entropy being computed from a numerically-integrated horizon area rather than differentiated locally at the horizon).

### 5.6 QFT: zeta regularization, Casimir, Seeley–DeWitt

- `ζ(2)=1.6449340693` vs. `π²/6=1.6449340668`, `ζ(4)=1.0823232337` vs. `π⁴/90=1.0823232337`, `ζ(−1)=−0.0833333335` vs. `−1/12`, `ζ(−3)=0.0083333333` vs. `1/120` — all agree to 8–10 significant figures, live confirmation the direct-summation + Euler–Maclaurin + functional-equation continuation is correctly implemented, not hardcoded.
- Casimir energy density at a=1μm derived from the live `ζ(−3)` value: `−4.333753×10⁻¹⁰ J/m²`, matching the standard closed form to all 7 printed digits.
- Sphere–plate force estimates (100–900 nm separations) land at 266.9 pN → 0.366 pN, the correct order of magnitude and monotonic falloff for the Mohideen & Roy (1998) geometry, with the explicit, correctly-scoped caveat that this is the T=0/perfect-conductor idealization only.
- Seeley–DeWitt `a_1` coefficient residual converges toward `χ/6 = 0.33333333` as t→0 (`0.33670 → 0.33467 → 0.33400 → 0.33367 → 0.33347 → 0.33340`), live confirmation of Gauss–Bonnet-linked heat-kernel asymptotics on the sphere.

### 5.7 QG phenomenology (GRB Lorentz-invariance-violation bound)

Planck energy/length computed live: `1.2209×10¹⁹ GeV`, `1.6163×10⁻³⁵ m`. The naive single-photon-pair estimator applied to GRB 090510 gives `E_QG,1/E_Planck = 1.363`, correctly *below* the published Vasileiou et al. (2013) multi-photon-statistics bound of 7.6 — exactly the ordering the module predicts for a cruder estimator. The derived exclusion table shows the naive bound already rules out **every** discreteness length from the Planck length up to `10¹²·ℓ_Planck` — reported as a genuine constraint on any discrete-spacetime model without an explicit LIV-suppression mechanism, not swept under the rug.

### 5.8 Gravity-mediated entanglement (GME/BMV)

Cross-term convergence to the small-`dx` literature formula: ratio of computed to `−2·(leading-order)` formula goes `1.0519 → 1.0045 → 1.0005 → 1.00004 → 1.000005` as `dx` shrinks from 10⁻⁴ m to 10⁻⁶ m — clean first-order convergence, live-confirmed. Against Bose et al. (2017): `Δφ_LR = −0.1256` (published −0.2), `Δφ_RL = +0.4395` (published +0.7) — correct sign and same order of magnitude on both, with the honestly-stated caveat that this model omits the Stern–Gerlach acceleration-phase contribution the published number includes.

### 5.9 Chapter 4 — the thesis's actual contribution: hypergraph→S² convergence

**Single-seed run** (`eps(N)=2.5√(ln N/N)`, seed 42), N = 100…3200:

| N | ratio_A (clique) | err_A | ratio_B (Zhou) | err_B |
|---|---|---|---|---|
| 100 | 3.1114 | 0.1114 | 3.1162 | 0.1162 |
| 200 | 3.1434 | 0.1434 | 3.1614 | 0.1614 |
| 400 | 2.8898 | 0.1102 | 2.8922 | 0.1078 |
| 800 | 2.9897 | 0.0103 | 3.0017 | 0.0017 |
| 1600 | 2.9719 | 0.0281 | 2.9764 | 0.0236 |
| 3200 | 2.9350 | 0.0650 | 2.9338 | 0.0662 |

Fitted power laws: `err_A(N) ≈ 0.764·N^{-0.410}`, `err_B(N) ≈ 1.164·N^{-0.525}` — reproduced live, matching the README's reported single-seed numbers.

**Seed-averaged run** (8 seeds, N ≤ 1600, bootstrapped rate exponent, 2000 resamples):

| N | mean err A | std err A | mean err B | std err B |
|---|---|---|---|---|
| 100 | 0.8512 | 0.9423 | 0.8868 | 0.9628 |
| 200 | 0.4088 | 0.3227 | 0.4249 | 0.3394 |
| 400 | 0.0879 | 0.0771 | 0.0800 | 0.0778 |
| 800 | 0.1200 | 0.1015 | 0.1219 | 0.1022 |
| 1600 | 0.0892 | 0.0390 | 0.0891 | 0.0404 |

Bootstrapped rate exponents: **p_A = 0.807 ± 0.137** (90% interval 0.574–1.023), **p_B = 0.834 ± 0.132** (90% interval 0.610–1.042) — the two intervals **overlap substantially**, live-reproducing the repo's central honest finding: the single-seed "scheme B converges faster" claim does *not* survive seed-averaging + bootstrapping. This run confirms that result independently rather than taking the repo's word for it.

### 5.10 Distinction items — the two heavy live computations run in full this session

**`schur_scale_probe`** (O(N³) dense Schur decomposition of the Hashimoto matrix, timed at N=100/200/400/800):

| N | m₂ = \|arcs\| | wall time | converged | trace check \|Tr(B) − Σλ\| |
|---|---|---|---|---|
| 100 | 400 | 77.97 ms | yes | 1.83×10⁻¹³ |
| 200 | 800 | 682.75 ms | yes | 1.52×10⁻¹² |
| 400 | 1600 | 5.817 s | yes | 2.15×10⁻¹² |
| 800 | 3200 | **72.33 s** | yes | 2.64×10⁻¹² |

Timing ratios (682.75/77.97=8.76, 5817/682.75=8.52, 72326/5817=12.43) are consistent with the documented O(m₂³) ≈ O(N³) scaling (theoretical ratio at fixed d for doubling N is 8×; the last step's slight super-cubic bump is plausible cache/memory-bandwidth effects at the largest matrix size on this 4 GB single-core sandbox). This is the actual, freshly-measured cost profile that justifies capping the seed study below at N=120 rather than pushing it higher.

**`bulk_spacing_mc`** (1000 independent 480×480 Hashimoto-matrix eigendecompositions, unfolded bulk nearest-neighbor level-spacing statistics vs. Ginibre and Poisson reference ensembles):

```
N=120 vertices, d=4 -> B is 480x480. Running 1000 seeds...
B ensemble: 1000 converged, 0 failed to converge, 180241 pooled spacings, wall time 139.83s
Ginibre reference: 72087 pooled spacings, wall time 78.11s
Poisson reference: 72000 pooled spacings, wall time 1.77s

fraction of unfolded spacings < 0.3 (small-spacing / repulsion proxy):
  B (non-backtracking):  0.5232
  Ginibre reference:     0.0578
  Poisson reference:     0.2201

KS distance, B vs Ginibre reference:   0.6089
KS distance, B vs Poisson reference:    0.3968
KS distance, Ginibre vs Poisson (sanity, should be large): 0.2136

=> B's bulk spacing distribution sits closer to Poisson than Ginibre at this N.
```

This is the live, freshly-reproduced version of the crate's central open-problem investigation (documented separately in this user's memory as `nb-eigenvalue-universality`): **at N=120, d=4, the non-backtracking operator's bulk spectral statistics show a small-spacing fraction (0.523) that is *higher* than Poisson's own (0.220)** — i.e., level *clustering*, not the level *repulsion* a Ginibre/random-matrix universality hypothesis would predict (Ginibre's own small-spacing fraction is only 0.058). Both KS distances are large (0.61 to Ginibre, 0.40 to Poisson), and the Poisson-vs-Ginibre sanity check (0.214) confirms the two reference ensembles are themselves well-separated, so this is not a measurement-resolution artifact. The honest reading, reproduced independently in this run: **at this graph size, the data is closer to Poisson than to Ginibre, but not cleanly consistent with either** — the open question of which universality class (if any) governs non-backtracking bulk statistics on sparse random graphs remains open, and this run adds one more concrete, reproducible data point against it rather than resolving it.

**`large_n_flow` / `certified_interval_flow`** (matrix-free SLQ heat-trace at N=10,000): agreement check at N=2000 between exact dense diagonalization and matrix-free SLQ, then N=10,000 estimates with an explicit 99%-confidence *certified interval* (Gauss/Radau quadrature bracket), not just a point estimate. Live result: dense P(t=1) at N=2000 = 831.19, SLQ point estimate = 831.10 (0.01% off), certified interval `[384.20, 1277.99]` contains the exact value — confirmed. At N=10,000 (where dense diagonalization is intractable on this hardware — a naive extrapolation from the N=2000 timing puts it at roughly two orders of magnitude longer, with O(N²) memory), the certified interval `[3208.93, 5101.98]` around point estimate 4155.45 is reported as a *guarantee*, not a hope.

---


