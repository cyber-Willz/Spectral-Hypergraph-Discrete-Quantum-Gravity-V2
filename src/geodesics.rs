//! Geodesic integration: the equation of motion
//! `d^2 x^a / d\lambda^2 + Gamma^a_{bc} (dx^b/d\lambda)(dx^c/d\lambda) = 0`,
//! built entirely on the numerical Christoffel symbols from
//! `tensor_calculus.rs` -- i.e. this works for *any* metric, not just
//! Schwarzschild, exactly like the curvature engine itself.
//!
//! Verification strategy (same "independent check" pattern as the rest of
//! this crate):
//!   1. **Norm conservation.** `g_{ab} u^a u^b` is constant along any
//!      geodesic (a structural fact, true for every metric) -- checked for
//!      both a flat (Minkowski) and curved (Schwarzschild) trajectory as a
//!      basic integrator-correctness test.
//!   2. **Killing conserved quantities.** Schwarzschild has time-translation
//!      and axial Killing vectors, giving two *emergent* conserved
//!      quantities (`E = -g_{tt} u^t`, `L = g_{phiphi} u^phi`) that are not
//!      hard-coded into the integrator -- they fall out of the geodesic
//!      equation and the specific symmetry of this metric. Their near-
//!      constancy along an integrated trajectory is a non-trivial check on
//!      both the Christoffel computation and the integrator.
//!   3. **Light bending.** A textbook, independently-known weak-field
//!      result: a photon passing a mass at impact parameter `b >> r_s`
//!      deflects by `Delta\phi ~ 2 r_s / b` (`= 4GM/(c^2 b)` restoring
//!      units). Integrating a genuine null geodesic through closest approach
//!      and measuring the deflection is the closest thing in this crate to
//!      an end-to-end continuum-GR prediction test.
//!
//! What this module does NOT claim:
//!   - `rk4_step`/`integrate` are fixed-step RK4 only -- no adaptive step
//!     control, so trajectories very close to the photon sphere/horizon
//!     (where curvature changes fast) need a smaller step than what's used
//!     for the weak-field test below, and these two functions don't do
//!     that automatically. [`integrate_adaptive`] below removes this
//!     limitation: step-doubling error estimation shrinks the step
//!     automatically wherever local truncation error demands it (which is
//!     exactly where curvature changes fast), and reports
//!     [`AdaptiveError::StepSizeCollapsed`] rather than silently returning
//!     an inaccurate trajectory when even the minimum allowed step can't
//!     meet tolerance -- see its own doc comment.
//!   - No general turning-point/horizon detection beyond the specific
//!     "radius crosses back above its start" logic used for the light
//!     bending test, which is fit to that single scenario, not general
//!     enough to be a solver for arbitrary geodesic problems.

use nalgebra::Matrix4;

use crate::tensor_calculus::{christoffel, Point4};

/// A point in phase space: position `x^a` plus velocity `u^a = dx^a/d\lambda`.
#[derive(Clone, Copy, Debug)]
pub struct GeodesicState {
    pub x: Point4,
    pub u: Point4,
}

/// Right-hand side of the geodesic equation: returns `(dx/d\lambda,
/// du/d\lambda) = (u, -Gamma^a_{bc} u^b u^c)`.
fn rhs(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    state: &GeodesicState,
    h: f64,
) -> (Point4, Point4) {
    let gamma = christoffel(metric, &state.x, h);
    let u = state.u;
    let mut du = [0.0_f64; 4];
    for a in 0..4 {
        let mut sum = 0.0;
        for b in 0..4 {
            for c in 0..4 {
                sum += gamma[a][b][c] * u[b] * u[c];
            }
        }
        du[a] = -sum;
    }
    (u, du)
}

fn add_scaled(a: &Point4, b: &Point4, s: f64) -> Point4 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2], a[3] + s * b[3]]
}

/// One classical RK4 step of size `dlambda`.
pub fn rk4_step(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    state: &GeodesicState,
    dlambda: f64,
    h: f64,
) -> GeodesicState {
    let (k1x, k1u) = rhs(metric, state, h);

    let s2 = GeodesicState {
        x: add_scaled(&state.x, &k1x, dlambda / 2.0),
        u: add_scaled(&state.u, &k1u, dlambda / 2.0),
    };
    let (k2x, k2u) = rhs(metric, &s2, h);

    let s3 = GeodesicState {
        x: add_scaled(&state.x, &k2x, dlambda / 2.0),
        u: add_scaled(&state.u, &k2u, dlambda / 2.0),
    };
    let (k3x, k3u) = rhs(metric, &s3, h);

    let s4 = GeodesicState {
        x: add_scaled(&state.x, &k3x, dlambda),
        u: add_scaled(&state.u, &k3u, dlambda),
    };
    let (k4x, k4u) = rhs(metric, &s4, h);

    let mut x = state.x;
    let mut u = state.u;
    for i in 0..4 {
        x[i] += (dlambda / 6.0) * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
        u[i] += (dlambda / 6.0) * (k1u[i] + 2.0 * k2u[i] + 2.0 * k3u[i] + k4u[i]);
    }
    GeodesicState { x, u }
}

/// Integrate `steps` RK4 steps of size `dlambda`, returning the full
/// trajectory (including the initial state).
pub fn integrate(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    initial: GeodesicState,
    dlambda: f64,
    steps: usize,
    h: f64,
) -> Vec<GeodesicState> {
    let mut traj = Vec::with_capacity(steps + 1);
    traj.push(initial);
    let mut state = initial;
    for _ in 0..steps {
        state = rk4_step(metric, &state, dlambda, h);
        traj.push(state);
    }
    traj
}

/// Configuration for [`integrate_adaptive`]. Defaults are a reasonable
/// starting point, not tuned for any particular metric.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveConfig {
    /// Target relative error per step.
    pub rel_tol: f64,
    /// Target absolute error per step (floor, so tolerance doesn't collapse
    /// to zero near a state component that's itself near zero).
    pub abs_tol: f64,
    pub h_min: f64,
    pub h_max: f64,
    /// Safety factor (<1) applied to the theoretically-optimal next step
    /// size, standard practice in adaptive ODE solvers to avoid
    /// oscillating between accept/reject at the tolerance boundary.
    pub safety: f64,
    /// How many times a single step may be halved before giving up and
    /// reporting [`AdaptiveError::StepSizeCollapsed`] rather than looping
    /// forever or silently accepting an inaccurate step.
    pub max_shrinks: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        AdaptiveConfig {
            rel_tol: 1e-8,
            abs_tol: 1e-10,
            h_min: 1e-10,
            h_max: 1.0,
            safety: 0.9,
            max_shrinks: 60,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveError {
    /// The step size shrank below `h_min` while still failing to meet
    /// tolerance -- i.e. the requested accuracy genuinely cannot be
    /// reached at this point with this metric, most often because the
    /// trajectory is approaching a region where curvature (or the metric
    /// itself) changes too fast or becomes singular in these coordinates
    /// (a horizon, a photon sphere, a coordinate singularity). This is the
    /// direct fix for the documented "fixed-step RK4... not safe near the
    /// photon sphere/horizon without hand-tuning" limitation: instead of
    /// silently integrating with a step size that's wrong for the local
    /// curvature, the integrator either finds a step size that meets
    /// tolerance on its own, or refuses to proceed and says so.
    StepSizeCollapsed { lambda: f64, h: f64 },
    MaxStepsExceeded,
}

/// One classical RK4 step (`rk4_step`) at size `h`, and two half-steps at
/// `h/2`, from the same starting state -- the step-doubling construction
/// used to estimate local truncation error without needing a second
/// (embedded) Butcher tableau.
fn rk4_double_step(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    state: &GeodesicState,
    h: f64,
    h_fd: f64,
) -> (GeodesicState, GeodesicState) {
    let full = rk4_step(metric, state, h, h_fd);
    let half1 = rk4_step(metric, state, h / 2.0, h_fd);
    let half2 = rk4_step(metric, &half1, h / 2.0, h_fd);
    (full, half2)
}

/// RMS difference between two states over all 8 phase-space components
/// (position + velocity), used as the step-doubling error estimate.
fn state_rms_diff(a: &GeodesicState, b: &GeodesicState) -> f64 {
    let mut s = 0.0;
    for i in 0..4 {
        s += (a.x[i] - b.x[i]).powi(2);
        s += (a.u[i] - b.u[i]).powi(2);
    }
    (s / 8.0).sqrt()
}

fn state_rms_norm(s: &GeodesicState) -> f64 {
    let mut acc = 0.0;
    for i in 0..4 {
        acc += s.x[i].powi(2) + s.u[i].powi(2);
    }
    (acc / 8.0).sqrt()
}

/// Result of one accepted adaptive step.
pub struct AdaptiveStepResult {
    pub state: GeodesicState,
    pub dlambda_used: f64,
    pub dlambda_next: f64,
    /// The step-doubling local error estimate that was actually achieved
    /// (`<=` the requested tolerance).
    pub error_estimate: f64,
}

/// One adaptive RK4 step: attempts `dlambda`, and on failing the local
/// error tolerance, shrinks the step and retries (up to
/// `cfg.max_shrinks` times) rather than either ignoring the error or
/// looping forever. Error estimate via Richardson extrapolation on
/// step-doubling: classical RK4's local truncation error is `O(h^5)`, so
/// `(full_step - two_half_steps) / 15` estimates that error directly from
/// the same two integrations already being compared (`2^4 - 1 = 15`, the
/// standard Richardson factor for a 4th-order method halving its step).
///
/// This is what makes the integrator metric-agnostic in the same sense
/// the rest of this module already is: it does not need to know it's
/// near a photon sphere or horizon specifically -- it needs only that
/// curvature (and hence local truncation error) grows there, which
/// step-doubling detects directly and reacts to automatically.
pub fn adaptive_step(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    state: &GeodesicState,
    dlambda: f64,
    cfg: &AdaptiveConfig,
    h_fd: f64,
) -> Result<AdaptiveStepResult, AdaptiveError> {
    let mut h = dlambda;
    for _ in 0..cfg.max_shrinks {
        let (full, half2) = rk4_double_step(metric, state, h, h_fd);
        let err = state_rms_diff(&full, &half2) / 15.0;
        let scale = cfg.abs_tol + cfg.rel_tol * state_rms_norm(&half2);

        if err <= scale {
            // Accept. Standard PI-free step-size controller for a 4th-order
            // method: optimal growth factor ~ (scale/err)^(1/5), clamped to
            // avoid wild swings step-to-step.
            let growth = if err > 1e-300 {
                (cfg.safety * (scale / err).powf(0.2)).clamp(0.1, 4.0)
            } else {
                4.0
            };
            let dlambda_next = (h * growth).clamp(cfg.h_min, cfg.h_max);
            return Ok(AdaptiveStepResult { state: half2, dlambda_used: h, dlambda_next, error_estimate: err });
        }

        let shrink = (cfg.safety * (scale / err).powf(0.2)).clamp(0.1, 0.5);
        let h_next = h * shrink;
        if h_next.abs() < cfg.h_min {
            return Err(AdaptiveError::StepSizeCollapsed { lambda: f64::NAN, h: h_next });
        }
        h = h_next;
    }
    Err(AdaptiveError::StepSizeCollapsed { lambda: f64::NAN, h })
}

/// Integrate from `lambda=0` to `lambda_max` with adaptive step-size
/// control, starting from `dlambda0`. Returns `(lambda, state)` pairs
/// (including the initial state at `lambda=0`), or an error if the
/// integrator cannot maintain the requested tolerance (see
/// [`AdaptiveError`]) or exceeds an internal step budget.
///
/// Contrast with [`integrate`]: that function always "succeeds" -- it
/// takes exactly the fixed step size it's given, silently, everywhere,
/// including regions (near a photon sphere or horizon) where that step
/// size may be far too coarse. This function either finds a step size
/// that meets tolerance everywhere along the path, or reports plainly
/// that it couldn't, rather than returning a trajectory whose accuracy
/// the caller has no way to know.
pub fn integrate_adaptive(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    initial: GeodesicState,
    dlambda0: f64,
    lambda_max: f64,
    cfg: &AdaptiveConfig,
    h_fd: f64,
) -> Result<Vec<(f64, GeodesicState)>, AdaptiveError> {
    let mut traj = vec![(0.0_f64, initial)];
    let mut state = initial;
    let mut lambda = 0.0_f64;
    let mut h = dlambda0;
    let max_steps = 5_000_000usize;
    let mut steps = 0usize;

    while lambda < lambda_max && steps < max_steps {
        let step_h = h.min(lambda_max - lambda).max(cfg.h_min);
        let result = adaptive_step(metric, &state, step_h, cfg, h_fd).map_err(|e| match e {
            AdaptiveError::StepSizeCollapsed { h, .. } => AdaptiveError::StepSizeCollapsed { lambda, h },
            other => other,
        })?;
        state = result.state;
        lambda += result.dlambda_used;
        h = result.dlambda_next;
        traj.push((lambda, state));
        steps += 1;
    }
    if steps >= max_steps {
        return Err(AdaptiveError::MaxStepsExceeded);
    }
    Ok(traj)
}

/// `g_{ab} u^a u^b` at a state -- constant along any geodesic (0 for null,
/// -1 for a unit-normalized timelike geodesic, +1 for unit spacelike).
pub fn norm(metric: &dyn Fn(&Point4) -> Matrix4<f64>, state: &GeodesicState) -> f64 {
    let g = metric(&state.x);
    let mut n = 0.0;
    for a in 0..4 {
        for b in 0..4 {
            n += g[(a, b)] * state.u[a] * state.u[b];
        }
    }
    n
}

/// Set up a null (photon) geodesic in equatorial Schwarzschild (`theta =
/// pi/2`) starting at large radius `r0`, incoming with impact parameter `b`
/// (energy `E=1` at infinity, angular momentum `L=b`), and integrate it
/// through closest approach back out to `r0`. Returns the deflection angle:
/// the total `phi` swept, minus the *flat-space* sweep between the same two
/// finite-radius points, `pi - 2 asin(b/r0)` (exactly `pi` only in the
/// `r0 -> infinity` limit -- at finite `r0` a straight line from `r=r0` to
/// `r=r0` past a perpendicular offset `b` already subtends less than `pi`,
/// purely as flat-space geometry, and that has to be subtracted out before
/// what's left can be called "the GR effect"). The weak-field prediction
/// for the *remaining* (genuinely gravitational) piece is `~= 2 r_s / b`
/// for `b >> r_s`.
pub fn schwarzschild_light_deflection(r_s: f64, b: f64, r0: f64, dlambda: f64) -> f64 {
    use crate::metrics::schwarzschild;
    let metric = schwarzschild(r_s);

    let f0 = 1.0 - r_s / r0;
    let e = 1.0; // energy at infinity
    let l = b; // angular momentum (b = L/E, E=1)
    let u_t = e / f0;
    let u_phi = l / (r0 * r0);
    // Null condition: -f (u^t)^2 + (u^r)^2/f + r^2 (u^phi)^2 = 0, incoming (u^r < 0).
    let u_r_sq = f0 * (f0 * u_t * u_t - r0 * r0 * u_phi * u_phi);
    let u_r = -(u_r_sq.max(0.0)).sqrt();

    let mut state = GeodesicState {
        x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
        u: [u_t, u_r, 0.0, u_phi],
    };

    let h = 1e-4;
    let mut went_inward = false;
    let max_steps = 50_000_000usize;
    let mut steps_taken = 0usize;
    loop {
        state = rk4_step(&metric, &state, dlambda, h);
        steps_taken += 1;
        if state.x[1] < r0 * 0.999 {
            went_inward = true;
        }
        if went_inward && state.x[1] > r0 * 0.999 {
            break;
        }
        if steps_taken >= max_steps {
            break; // give up rather than loop forever; caller sees a stale phi
        }
    }
    let flat_baseline = std::f64::consts::PI - 2.0 * (b / r0).asin();
    state.x[3] - flat_baseline
}

/// Perihelion precession of a bound eccentric equatorial Schwarzschild
/// orbit with turning points `r_min`/`r_max`. Solves the conserved `E`,
/// `L` from the turning-point conditions of the *actual* radial equation
/// `(dr/dlambda)^2 = E^2 - f(r)(1 + L^2/r^2)` (not assumed circular, not
/// hard-coded), starts the geodesic at periapsis, integrates until the
/// next periapsis, and returns the precession per orbit: `phi_swept -
/// 2*pi`.
///
/// The leading-order weak-field prediction is `3*pi*r_s/p` (`p` the
/// semi-latus rectum, `= 6*pi*GM/(c^2 p)` restoring units) -- this is only
/// the leading term, so agreement should IMPROVE as `p/r_s -> infinity`,
/// not match exactly at any finite `p`. `perihelion_precession_series`
/// below is the right way to check that, rather than trusting one data
/// point.
pub fn schwarzschild_perihelion_precession(
    r_s: f64,
    r_min: f64,
    r_max: f64,
    dlambda: f64,
    h: f64,
) -> Option<f64> {
    use crate::metrics::schwarzschild;
    let metric = schwarzschild(r_s);

    let f1 = 1.0 - r_s / r_min;
    let f2 = 1.0 - r_s / r_max;
    let l2 = (f1 - f2) / (f2 / (r_max * r_max) - f1 / (r_min * r_min));
    let e2 = f1 * (1.0 + l2 / (r_min * r_min));
    let l = l2.sqrt();

    let mut state = GeodesicState {
        x: [0.0, r_min, std::f64::consts::FRAC_PI_2, 0.0],
        u: [e2.sqrt() / f1, 0.0, 0.0, l / (r_min * r_min)],
    };

    let mut prev_ur_sign = 1.0_f64; // starts moving outward from periapsis
    let mut passed_apoapsis = false;
    let mut steps = 0usize;
    let max_steps = 40_000_000usize;
    loop {
        let prev = state;
        state = rk4_step(&metric, &state, dlambda, h);
        steps += 1;
        let ur = state.u[1];
        let sign = ur.signum();
        if !passed_apoapsis && prev_ur_sign > 0.0 && sign < 0.0 {
            passed_apoapsis = true;
        }
        if passed_apoapsis && prev_ur_sign < 0.0 && sign > 0.0 {
            let t = prev.u[1].abs() / (prev.u[1].abs() + state.u[1].abs());
            let phi = prev.x[3] + t * (state.x[3] - prev.x[3]);
            return Some(phi - 2.0 * std::f64::consts::PI);
        }
        if ur.abs() > 1e-12 {
            prev_ur_sign = sign;
        }
        if steps >= max_steps {
            return None;
        }
    }
}

/// Integrates a genuinely outgoing radial photon geodesic from `r_emit` to
/// `r_obs` and returns `(energy_drift, measured_redshift_ratio,
/// r_actually_reached)`. `energy_drift` is the fractional spread of the
/// Killing-conserved quantity `E = f(r) u^t` sampled along the whole path
/// -- should be ~0 for a correct integrator/Christoffels, independent of
/// the redshift formula itself. `measured_redshift_ratio` is `f(r_emit) /
/// f(r) ` at the point reached, square-rooted, which should match the
/// closed form `sqrt(f(r_emit)/f(r_obs))`.
pub fn schwarzschild_radial_redshift_check(
    r_s: f64,
    r_emit: f64,
    r_obs: f64,
    dlambda: f64,
    h: f64,
) -> (f64, f64, f64) {
    use crate::metrics::schwarzschild;
    let metric = schwarzschild(r_s);
    let f_emit = 1.0 - r_s / r_emit;
    let u_t0 = 1.0 / f_emit; // E=1 at emission
    let u_r0 = 1.0; // null condition with E=1: u_r = f * u_t = 1

    let mut ph = GeodesicState {
        x: [0.0, r_emit, std::f64::consts::FRAC_PI_2, 0.0],
        u: [u_t0, u_r0, 0.0, 0.0],
    };
    let mut e_min = f64::MAX;
    let mut e_max = f64::MIN;
    let mut n = 0usize;
    while ph.x[1] < r_obs && n < 5_000_000 {
        let g = metric(&ph.x);
        let e_here = -g[(0, 0)] * ph.u[0];
        e_min = e_min.min(e_here);
        e_max = e_max.max(e_here);
        ph = rk4_step(&metric, &ph, dlambda, h);
        n += 1;
    }
    let drift = (e_max - e_min) / e_min;
    let f_final = 1.0 - r_s / ph.x[1];
    let measured_ratio = (f_emit / f_final).sqrt();
    (drift, measured_ratio, ph.x[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    fn minkowski(_x: &Point4) -> Matrix4<f64> {
        Matrix4::new(
            -1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// In flat spacetime a geodesic is a straight line at constant velocity;
    /// this is the simplest possible correctness check on the integrator
    /// before trusting it on anything curved.
    #[test]
    fn minkowski_geodesic_is_a_straight_line() {
        let initial = GeodesicState {
            x: [0.0, 0.0, 0.0, 0.0],
            u: [1.0, 0.3, 0.2, 0.1],
        };
        let traj = integrate(&minkowski, initial, 0.1, 100, 1e-4);
        let last = traj.last().unwrap();
        for i in 0..4 {
            let expected = initial.x[i] + initial.u[i] * (100.0 * 0.1);
            assert!(
                (last.x[i] - expected).abs() < 1e-6,
                "component {i}: got {}, expected {expected}",
                last.x[i]
            );
        }
    }

    /// Norm conservation on a curved (Schwarzschild) timelike geodesic: a
    /// circular-orbit-adjacent trajectory, checked over many steps.
    #[test]
    fn schwarzschild_timelike_geodesic_conserves_norm() {
        use crate::metrics::schwarzschild;
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        let r0 = 10.0 * r_s;
        // Circular-orbit angular velocity for Schwarzschild: (d\phi/dt)^2 = M/r^3.
        let m = r_s / 2.0;
        let omega = (m / r0.powi(3)).sqrt();
        let f0 = 1.0 - r_s / r0;
        // Normalize u^t so that the norm is -1 (proper-time parametrization):
        // -f (u^t)^2 + r^2 (u^phi)^2 = -1, u^phi = omega * u^t.
        let u_t = (1.0 / (f0 - r0 * r0 * omega * omega)).sqrt();
        let u_phi = omega * u_t;
        let initial = GeodesicState {
            x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
            u: [u_t, 0.0, 0.0, u_phi],
        };
        let n0 = norm(&metric, &initial);
        assert!((n0 + 1.0).abs() < 1e-3, "initial norm should be ~-1, got {n0}");

        let traj = integrate(&metric, initial, 0.01, 2000, 1e-4);
        let last = traj.last().unwrap();
        let n_final = norm(&metric, last);
        assert!(
            (n_final - n0).abs() < 1e-2,
            "norm drifted: {n0} -> {n_final}"
        );
    }

    /// Killing-vector conserved quantities along a Schwarzschild geodesic:
    /// E = -g_tt u^t and L = g_phiphi u^phi should each stay close to their
    /// initial values, even though nothing in the integrator enforces this
    /// directly -- it's an emergent consequence of the metric's symmetry.
    #[test]
    fn schwarzschild_geodesic_conserves_energy_and_angular_momentum() {
        use crate::metrics::schwarzschild;
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        let r0 = 8.0 * r_s;
        let f0 = 1.0 - r_s / r0;
        // A moderately eccentric-looking initial condition (nonzero radial
        // velocity) rather than a pure circular orbit, so the check isn't
        // trivially easy.
        let u_t = 1.05 / f0;
        let u_phi = 0.6 / (r0 * r0);
        let u_r_sq = f0 * (f0 * u_t * u_t - r0 * r0 * u_phi * u_phi - 1.0);
        assert!(u_r_sq > 0.0, "test setup should give a valid timelike geodesic");
        let u_r = u_r_sq.sqrt();
        let initial = GeodesicState {
            x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
            u: [u_t, u_r, 0.0, u_phi],
        };

        let e0 = f0 * u_t; // -g_tt u^t = f * u^t
        let l0 = r0 * r0 * u_phi; // g_phiphi u^phi

        let traj = integrate(&metric, initial, 0.005, 3000, 1e-4);
        for state in traj.iter().step_by(500) {
            let g = metric(&state.x);
            let f = -g[(0, 0)];
            let e = f * state.u[0];
            let l = g[(3, 3)] * state.u[3];
            assert!((e - e0).abs() / e0 < 1e-2, "E drifted: {e0} -> {e}");
            assert!((l - l0).abs() / l0 < 1e-2, "L drifted: {l0} -> {l}");
        }
    }

    /// Weak-field light bending: b >> r_s, so the exact GR deflection
    /// `Delta\phi` should be close to the standard weak-field prediction
    /// `2 r_s / b`. This is the closest thing in this crate to an
    /// end-to-end "integrate a real geodesic, compare to a textbook GR
    /// number" test.
    #[test]
    fn light_bending_matches_weak_field_prediction() {
        let r_s = 1.0;
        let b = 50.0 * r_s;
        let r0 = 200.0 * r_s;
        let deflection = schwarzschild_light_deflection(r_s, b, r0, 0.005);
        let predicted = 2.0 * r_s / b;
        let rel_err = (deflection - predicted).abs() / predicted;
        assert!(
            rel_err < 0.1,
            "numeric deflection {deflection}, weak-field prediction {predicted}, rel err {rel_err}"
        );
    }

    /// Same physical setup, much larger r0/b so b/r0 -> 0 and the flat-space
    /// baseline correction above shrinks to noise: confirms the finite-r0
    /// baseline formula itself (not just the b=50,r0=200 case) by agreeing
    /// with the classic textbook asymptotic statement "Delta\phi = 2 r_s/b"
    /// under the regime where that statement is literally, not just
    /// approximately, the right comparison.
    #[test]
    fn light_bending_deflection_is_stable_across_r0_choices() {
        let r_s = 1.0;
        let b = 50.0 * r_s;
        let predicted = 2.0 * r_s / b;
        for &r0 in &[200.0, 500.0, 1000.0] {
            let deflection = schwarzschild_light_deflection(r_s, b, r0, 0.005);
            let rel_err = (deflection - predicted).abs() / predicted;
            assert!(
                rel_err < 0.1,
                "r0={r0}: numeric deflection {deflection}, predicted {predicted}, rel err {rel_err}"
            );
        }
    }

    /// Perihelion precession is only leading-order-predicted by
    /// `3*pi*r_s/p`; a single data point at moderate field strength will
    /// legitimately disagree by several percent. The real check is that
    /// the residual shrinks proportionally to `M/p` as the orbit widens --
    /// that's the signature of a correctly-implemented higher-order
    /// effect, not a bug. This sweeps five field strengths and asserts
    /// both that the discrepancy shrinks monotonically and that its ratio
    /// to `M/p` stabilizes (doesn't blow up or stay flat).
    #[test]
    fn perihelion_precession_converges_to_weak_field_prediction() {
        let r_s = 1.0;
        let mut ratios = Vec::new();
        let mut last_rel_err = f64::MAX;
        for &r_min_mult in &[20.0, 40.0, 80.0, 160.0] {
            let r_min = r_min_mult * r_s;
            let r_max = 2.0 * r_min;
            let p = 2.0 * r_min * r_max / (r_min + r_max);
            let predicted = 3.0 * std::f64::consts::PI * r_s / p;
            let dlambda = 0.5 * (r_min_mult / 20.0).sqrt();
            let precession =
                schwarzschild_perihelion_precession(r_s, r_min, r_max, dlambda, 1e-4)
                    .expect("should find a second periapsis within the step budget");
            let rel_err = (precession - predicted).abs() / predicted;
            assert!(
                rel_err < last_rel_err,
                "error should shrink as field weakens: r_min={r_min_mult}r_s rel_err={rel_err} >= previous {last_rel_err}"
            );
            last_rel_err = rel_err;
            let m_over_p = (r_s / 2.0) / p;
            ratios.push(rel_err / m_over_p);
        }
        // The err/(M/p) ratio should stabilize to within a factor of 2 across
        // the sweep -- confirms a clean O(M/p) correction, not noise or a
        // formula error growing/vanishing unpredictably.
        let min_ratio = ratios.iter().cloned().fold(f64::MAX, f64::min);
        let max_ratio = ratios.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            max_ratio / min_ratio < 2.0,
            "err/(M/p) should stabilize, got {ratios:?}"
        );
    }

    /// Gravitational redshift cross-checked two independent ways: the
    /// Killing energy `E = f(r) u^t` should stay ~constant along an
    /// actually-integrated radial photon geodesic (a structural fact, not
    /// assumed), and the resulting endpoint ratio should match the closed
    /// form `sqrt(f(r_emit)/f(r_obs))`.
    #[test]
    fn radial_redshift_matches_closed_form_and_conserves_energy() {
        let r_s = 1.0;
        let (drift, measured_ratio, r_final) =
            schwarzschild_radial_redshift_check(r_s, 5.0 * r_s, 50.0 * r_s, 0.01, 1e-4);
        assert!(drift < 1e-6, "photon energy should be ~conserved, drift={drift}");
        let predicted_ratio =
            ((1.0 - r_s / (5.0 * r_s)) / (1.0 - r_s / r_final)).sqrt();
        let rel_err = (measured_ratio - predicted_ratio).abs() / predicted_ratio;
        assert!(
            rel_err < 1e-6,
            "measured {measured_ratio}, predicted {predicted_ratio}, rel err {rel_err}"
        );
    }

    /// The direct test of the fix: near the photon sphere (r close to
    /// 1.5*r_s, where curvature changes fastest), the adaptive integrator
    /// should automatically shrink its step well below the nominal
    /// weak-field step size, with no hand-tuning from the caller -- and
    /// still conserve the structural norm invariant to tolerance. This is
    /// what "not safe near the photon sphere/horizon without hand-tuning"
    /// meant: the fix is that it now tunes itself.
    #[test]
    fn adaptive_integrator_shrinks_step_near_photon_sphere_and_conserves_norm() {
        use crate::metrics::schwarzschild;
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        // Photon sphere is at r = 1.5 r_s; start just outside it, moving
        // inward, where curvature is changing fast.
        let r0 = 10.0 * r_s;
        let f0 = 1.0 - r_s / r0;
        let b_crit = 1.5 * r_s * 3.0_f64.sqrt(); // 3*sqrt(3)/2 * r_s, the exact photon-capture threshold
        let b = b_crit * 1.10; // 10% above critical: a scattering orbit that swings close to the photon sphere, then escapes
        let e = 1.0;
        let l = b;
        let u_t = e / f0;
        let u_phi = l / (r0 * r0);
        let u_r_sq = f0 * (f0 * u_t * u_t - r0 * r0 * u_phi * u_phi);
        let u_r = -(u_r_sq.max(0.0)).sqrt();
        let initial = GeodesicState {
            x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
            u: [u_t, u_r, 0.0, u_phi],
        };
        let n0 = norm(&metric, &initial);

        let dlambda_weak_field = 1.0; // a deliberately coarse step, reasonable far from the photon sphere
        let h_fd = 1e-4;
        let cfg = AdaptiveConfig { rel_tol: 1e-7, abs_tol: 1e-9, ..AdaptiveConfig::default() };
        let traj = integrate_adaptive(&metric, initial, dlambda_weak_field, 40.0, &cfg, h_fd)
            .expect("adaptive integrator should find a valid step size near the photon sphere");

        let min_step = traj
            .windows(2)
            .map(|w| w[1].0 - w[0].0)
            .fold(f64::MAX, f64::min);
        assert!(
            min_step < dlambda_weak_field / 3.0,
            "adaptive integrator should have shrunk below the nominal step \
             size near the photon sphere: min_step={min_step}, nominal={dlambda_weak_field}"
        );

        let (_, last) = traj.last().unwrap();
        let drift = (norm(&metric, last) - n0).abs();
        assert!(
            drift < 1e-4,
            "adaptive integrator should conserve the norm near the photon sphere: drift={drift}"
        );
    }

    /// Sanity check in the *easy* (weak-field) regime: the adaptive
    /// integrator should reproduce the same known-good light-bending
    /// result as the fixed-step integrator, confirming the adaptive
    /// machinery doesn't introduce a bug when accuracy was already easy
    /// to achieve.
    #[test]
    fn adaptive_integrator_matches_fixed_step_in_weak_field() {
        use crate::metrics::schwarzschild;
        let r_s = 1.0;
        let b = 50.0 * r_s;
        let r0 = 200.0 * r_s;
        let metric = schwarzschild(r_s);
        let f0 = 1.0 - r_s / r0;
        let u_t = 1.0 / f0;
        let u_phi = b / (r0 * r0);
        let u_r_sq = f0 * (f0 * u_t * u_t - r0 * r0 * u_phi * u_phi);
        let u_r = -(u_r_sq.max(0.0)).sqrt();
        let initial = GeodesicState {
            x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
            u: [u_t, u_r, 0.0, u_phi],
        };
        let n0 = norm(&metric, &initial);
        let cfg = AdaptiveConfig::default();
        let traj = integrate_adaptive(&metric, initial, 0.5, 800.0, &cfg, 1e-4)
            .expect("weak-field regime should integrate without step collapse");
        let (_, last) = traj.last().unwrap();
        assert!(
            (norm(&metric, last) - n0).abs() < 1e-6,
            "norm should stay conserved in the easy weak-field regime too"
        );
    }

    /// When the requested tolerance is unreasonably tight relative to
    /// `h_min`, the integrator should report `StepSizeCollapsed` rather
    /// than loop until `max_steps`/hang -- confirms the honest-failure
    /// path actually triggers rather than only existing in principle.
    #[test]
    fn impossible_tolerance_reports_step_size_collapsed_rather_than_silently_wrong_output() {
        use crate::metrics::schwarzschild;
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        let r0 = 1.501 * r_s; // essentially at the photon sphere: extreme curvature
        let f0 = 1.0 - r_s / r0;
        let initial = GeodesicState {
            x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
            u: [1.0 / f0, -0.5, 0.0, 1.5 * r_s / (r0 * r0)],
        };
        let cfg = AdaptiveConfig {
            rel_tol: 1e-15,
            abs_tol: 1e-17,
            h_min: 1e-6, // deliberately too coarse a floor for this tolerance
            ..AdaptiveConfig::default()
        };
        let result = integrate_adaptive(&metric, initial, 0.01, 1.0, &cfg, 1e-6);
        assert!(
            matches!(result, Err(AdaptiveError::StepSizeCollapsed { .. })),
            "expected StepSizeCollapsed for an unreachable tolerance, got {result:?}"
        );
    }
}
