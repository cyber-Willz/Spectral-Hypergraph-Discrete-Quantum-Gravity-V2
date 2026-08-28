//! Semiclassical black-hole thermodynamics: surface gravity, Hawking
//! temperature, horizon area, and Bekenstein-Hawking entropy.
//!
//! These are NOT pure-GR results -- Hawking temperature and the area law
//! come from quantum field theory on a fixed curved background plus
//! horizon thermodynamics, not from the Einstein equations alone. This
//! crate has no quantum sector, so what's actually checked here is
//! narrower than "the crate does Hawking radiation": it's that the
//! metric's horizon-adjacent derivative structure (`f'(r_s)`) and induced
//! area element, both read straight from the metric the crate already
//! builds, reproduce the values those semiclassical formulas require. In
//! units `G = c = hbar = k_B = 1`.

use crate::metrics::schwarzschild;
use crate::tensor_calculus::Point4;
use nalgebra::Matrix4;
use std::f64::consts::PI;

/// Surface gravity `kappa = f'(r_s) / 2` at the Schwarzschild horizon,
/// obtained by numerically differentiating `f(r) = -g_tt` straight from
/// the metric callback (not from the closed form `f(r) = 1 - r_s/r`)
/// using a central difference that straddles the horizon. `g_tt` itself
/// is finite and well-behaved on both sides of `r_s` (only `g_rr = 1/f`
/// is singular there), so this is safe as long as `h` stays away from
/// `g_rr`.
pub fn surface_gravity(metric: &dyn Fn(&Point4) -> Matrix4<f64>, r_s: f64, h: f64) -> f64 {
    let x_plus: Point4 = [0.0, r_s + h, PI / 2.0, 0.0];
    let x_minus: Point4 = [0.0, r_s - h, PI / 2.0, 0.0];
    let f_plus = -metric(&x_plus)[(0, 0)];
    let f_minus = -metric(&x_minus)[(0, 0)];
    let f_prime = (f_plus - f_minus) / (2.0 * h);
    f_prime / 2.0
}

/// Hawking temperature `T_H = kappa / (2*pi)`.
pub fn hawking_temperature(kappa: f64) -> f64 {
    kappa / (2.0 * PI)
}

/// Convenience: surface gravity + Hawking temperature for Schwarzschild
/// with horizon radius `r_s`, both computed from the metric rather than
/// the closed form.
pub fn schwarzschild_hawking_temperature(r_s: f64, h: f64) -> f64 {
    let metric = schwarzschild(r_s);
    hawking_temperature(surface_gravity(&metric, r_s, h))
}

/// Horizon area, computed by numerically integrating the induced-metric
/// area element `sqrt(g_theta_theta * g_phi_phi) d_theta d_phi` over the
/// sphere at `r = r_s`, reading `g_theta_theta`/`g_phi_phi` straight from
/// the metric (not from the closed form `4*pi*r_s^2`). Midpoint
/// quadrature with `n_theta` x `n_phi` cells.
pub fn horizon_area_schwarzschild(r_s: f64, n_theta: usize, n_phi: usize) -> f64 {
    let metric = schwarzschild(r_s);
    let dtheta = PI / n_theta as f64;
    let dphi = 2.0 * PI / n_phi as f64;
    let mut area = 0.0;
    for i in 0..n_theta {
        let theta = (i as f64 + 0.5) * dtheta;
        let x: Point4 = [0.0, r_s, theta, 0.0];
        let g = metric(&x);
        let g_theta = g[(2, 2)];
        let g_phi = g[(3, 3)];
        let elem = (g_theta * g_phi).sqrt();
        area += elem * dtheta * dphi * n_phi as f64;
    }
    area
}

/// Bekenstein-Hawking entropy `S = A / 4`.
pub fn bekenstein_hawking_entropy(area: f64) -> f64 {
    area / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Surface gravity and Hawking temperature, checked against the exact
    /// closed forms `kappa = 1/(2*r_s)`, `T_H = 1/(8*pi*M)` for several
    /// horizon sizes -- not just one.
    #[test]
    fn hawking_temperature_matches_closed_form() {
        for &r_s in &[0.5, 1.0, 2.0, 5.0] {
            let metric = schwarzschild(r_s);
            let kappa = surface_gravity(&metric, r_s, 1e-5 * r_s);
            let kappa_exact = 1.0 / (2.0 * r_s);
            assert!(
                (kappa - kappa_exact).abs() / kappa_exact < 1e-4,
                "r_s={r_s}: kappa={kappa}, exact={kappa_exact}"
            );

            let t_h = hawking_temperature(kappa);
            let m = r_s / 2.0;
            let t_h_exact = 1.0 / (8.0 * PI * m);
            assert!(
                (t_h - t_h_exact).abs() / t_h_exact < 1e-4,
                "r_s={r_s}: T_H={t_h}, exact={t_h_exact}"
            );
        }
    }

    /// Horizon area and Bekenstein-Hawking entropy, checked against the
    /// exact closed forms `A = 4*pi*r_s^2`, `S = pi*r_s^2 = 4*pi*M^2`.
    #[test]
    fn bekenstein_hawking_entropy_matches_closed_form() {
        for &r_s in &[0.5, 1.0, 2.0, 5.0] {
            let area = horizon_area_schwarzschild(r_s, 400, 400);
            let area_exact = 4.0 * PI * r_s * r_s;
            assert!(
                (area - area_exact).abs() / area_exact < 1e-3,
                "r_s={r_s}: area={area}, exact={area_exact}"
            );

            let s = bekenstein_hawking_entropy(area);
            let s_exact = PI * r_s * r_s;
            assert!(
                (s - s_exact).abs() / s_exact < 1e-3,
                "r_s={r_s}: S={s}, exact={s_exact}"
            );
        }
    }
}
