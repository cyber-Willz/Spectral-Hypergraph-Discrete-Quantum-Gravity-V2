//! Forward-mode automatic differentiation: a second-order "jet" (truncated
//! Taylor expansion) in the 4 spacetime coordinates, tracking a function's
//! value, its 4 first partial derivatives, and its 4x4 (symmetric) second
//! partial derivatives simultaneously through ordinary arithmetic.
//!
//! Why this exists: `tensor_calculus.rs`'s original engine differentiates
//! the metric *numerically* (central differences), which is deliberately
//! generic -- it works for any `Fn(&Point4) -> Matrix4<f64>` metric
//! callback, including ones with no closed-form derivative (e.g. an
//! arbitrary caller-supplied FRW scale factor `a(t)`) -- but is not
//! machine-precision: relative error `~1e-4`-`1e-6` for a well-chosen step
//! `h`, as documented there.
//!
//! `Jet2` replaces finite differences with *exact* differentiation for any
//! metric that can be written generically over the [`Scalar`] trait (i.e.
//! using only `+ - * /`, `sqrt`, `sin`, `cos`, `powi`, `powf` on its inputs
//! -- which covers every metric currently in `metrics.rs`, and any metric
//! expressible in closed form). The key fact making this sufficient for
//! the Riemann tensor specifically: the Riemann tensor is built from the
//! metric's value and its first and second derivatives only (see the
//! derivation in `tensor_calculus::curvature_at_exact`) -- so tracking a
//! function to *second* order, and no further, already gives everything
//! the curvature engine needs, machine-precision and closed-form, with no
//! step-size parameter to choose at all.
//!
//! Every arithmetic/transcendental operation below is implemented via the
//! ordinary multivariate chain rule
//! ```text
//!   h = a(f)  =>  h_i = a'(f) f_i,   h_{ij} = a''(f) f_i f_j + a'(f) f_{ij}
//! ```
//! applied through the private `unary` helper, which takes only `(a(f),
//! a'(f), a''(f))` evaluated at `f`'s current value -- this keeps every
//! operation's derivative formula a one-line, checkable fact about a
//! single-variable function, rather than a bespoke derivation each time.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Anything a metric formula can be generic over: real numbers (`f64`,
/// evaluated as usual) or [`Jet2`] (evaluated with exact derivatives
/// riding along for free). Implement this for a new numeric type to make
/// every metric in `metrics.rs` automatically differentiable through it.
pub trait Scalar:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
{
    fn from_f64(v: f64) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, p: f64) -> Self;
}

impl Scalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    fn powf(self, p: f64) -> Self {
        f64::powf(self, p)
    }
}

/// A second-order jet: `val = f(x)`, `grad[i] = df/dx^i`, `hess[i][j] =
/// d^2f/dx^i dx^j` (symmetric), at a fixed point `x`, propagated through
/// arithmetic via the exact multivariate chain rule -- not sampled, not
/// estimated.
#[derive(Debug, Clone, Copy)]
pub struct Jet2 {
    pub val: f64,
    pub grad: [f64; 4],
    pub hess: [[f64; 4]; 4],
}

impl Jet2 {
    /// A constant: zero derivatives.
    pub fn constant(v: f64) -> Self {
        Jet2 { val: v, grad: [0.0; 4], hess: [[0.0; 4]; 4] }
    }

    /// The jet for "coordinate `i`" itself, at value `v`: `grad = e_i`,
    /// `hess = 0` (a coordinate is an exactly linear function of itself and
    /// the others, so its own second derivatives vanish identically -- not
    /// an approximation).
    pub fn variable(i: usize, v: f64) -> Self {
        let mut grad = [0.0; 4];
        grad[i] = 1.0;
        Jet2 { val: v, grad, hess: [[0.0; 4]; 4] }
    }

    /// Seed all 4 coordinate jets for a point `x` in one call.
    pub fn variables(x: &[f64; 4]) -> [Jet2; 4] {
        std::array::from_fn(|i| Jet2::variable(i, x[i]))
    }

    /// Apply a scalar function `a` to this jet via the exact multivariate
    /// chain rule, given `a`'s value/first/second derivative *at this
    /// jet's current value* (`a_val = a(self.val)`, `da = a'(self.val)`,
    /// `d2a = a''(self.val)`). Every unary op below (`sqrt`, `sin`, `cos`,
    /// `powi`, `powf`, `recip`) is one call to this with a one-line
    /// derivative fact plugged in, rather than a separately-derived
    /// formula per operation.
    fn unary(self, a_val: f64, da: f64, d2a: f64) -> Jet2 {
        let mut grad = [0.0; 4];
        let mut hess = [[0.0; 4]; 4];
        for i in 0..4 {
            grad[i] = da * self.grad[i];
            for j in 0..4 {
                hess[i][j] = d2a * self.grad[i] * self.grad[j] + da * self.hess[i][j];
            }
        }
        Jet2 { val: a_val, grad, hess }
    }

    pub fn recip(self) -> Jet2 {
        let v = self.val;
        self.unary(1.0 / v, -1.0 / (v * v), 2.0 / (v * v * v))
    }

    pub fn sqrt(self) -> Jet2 {
        let v = self.val;
        let s = v.sqrt();
        self.unary(s, 0.5 / s, -0.25 / (v * s))
    }

    pub fn sin(self) -> Jet2 {
        let (s, c) = (self.val.sin(), self.val.cos());
        self.unary(s, c, -s)
    }

    pub fn cos(self) -> Jet2 {
        let (s, c) = (self.val.sin(), self.val.cos());
        self.unary(c, -s, -c)
    }

    pub fn powi(self, n: i32) -> Jet2 {
        let v = self.val;
        self.unary(v.powi(n), (n as f64) * v.powi(n - 1), (n as f64) * ((n - 1) as f64) * v.powi(n - 2))
    }

    pub fn powf(self, p: f64) -> Jet2 {
        let v = self.val;
        self.unary(v.powf(p), p * v.powf(p - 1.0), p * (p - 1.0) * v.powf(p - 2.0))
    }
}

impl Add for Jet2 {
    type Output = Jet2;
    fn add(self, rhs: Jet2) -> Jet2 {
        let mut grad = [0.0; 4];
        let mut hess = [[0.0; 4]; 4];
        for i in 0..4 {
            grad[i] = self.grad[i] + rhs.grad[i];
            for j in 0..4 {
                hess[i][j] = self.hess[i][j] + rhs.hess[i][j];
            }
        }
        Jet2 { val: self.val + rhs.val, grad, hess }
    }
}

impl Sub for Jet2 {
    type Output = Jet2;
    fn sub(self, rhs: Jet2) -> Jet2 {
        self + (-rhs)
    }
}

impl Neg for Jet2 {
    type Output = Jet2;
    fn neg(self) -> Jet2 {
        let mut grad = [0.0; 4];
        let mut hess = [[0.0; 4]; 4];
        for i in 0..4 {
            grad[i] = -self.grad[i];
            for j in 0..4 {
                hess[i][j] = -self.hess[i][j];
            }
        }
        Jet2 { val: -self.val, grad, hess }
    }
}

impl Mul for Jet2 {
    type Output = Jet2;
    fn mul(self, rhs: Jet2) -> Jet2 {
        // Product rule, second order: (fg)_i = f_i g + f g_i;
        // (fg)_{ij} = f_{ij} g + f_i g_j + f_j g_i + f g_{ij}.
        let mut grad = [0.0; 4];
        let mut hess = [[0.0; 4]; 4];
        for i in 0..4 {
            grad[i] = self.grad[i] * rhs.val + self.val * rhs.grad[i];
            for j in 0..4 {
                hess[i][j] = self.hess[i][j] * rhs.val
                    + self.grad[i] * rhs.grad[j]
                    + self.grad[j] * rhs.grad[i]
                    + self.val * rhs.hess[i][j];
            }
        }
        Jet2 { val: self.val * rhs.val, grad, hess }
    }
}

impl Div for Jet2 {
    type Output = Jet2;
    fn div(self, rhs: Jet2) -> Jet2 {
        self * rhs.recip()
    }
}

impl Scalar for Jet2 {
    fn from_f64(v: f64) -> Self {
        Jet2::constant(v)
    }
    fn sqrt(self) -> Self {
        Jet2::sqrt(self)
    }
    fn sin(self) -> Self {
        Jet2::sin(self)
    }
    fn cos(self) -> Self {
        Jet2::cos(self)
    }
    fn powi(self, n: i32) -> Self {
        Jet2::powi(self, n)
    }
    fn powf(self, p: f64) -> Self {
        Jet2::powf(self, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check every Jet2 operation against finite differences on a
    /// hand-picked scalar test function of 4 variables, at a generic
    /// (non-symmetric, non-special) point -- the same "independent check"
    /// discipline as the rest of this crate, applied to the autodiff
    /// engine itself before trusting it to replace finite differences
    /// anywhere else.
    fn f(x: &[f64; 4]) -> f64 {
        // A function exercising every op: +, -, *, /, sqrt, sin, cos, powi.
        let a = x[0].sin() * x[1].cos();
        let b = (x[2] * x[2] + 1.0).sqrt();
        let c = x[3] / (x[0] * x[0] + 2.0);
        a - b * c + x[1].powi(3)
    }

    fn f_jet(x: &[Jet2; 4]) -> Jet2 {
        let a = x[0].sin() * x[1].cos();
        let b = (x[2] * x[2] + Jet2::constant(1.0)).sqrt();
        let c = x[3] / (x[0] * x[0] + Jet2::constant(2.0));
        a - b * c + x[1].powi(3)
    }

    #[test]
    fn jet2_first_and_second_derivatives_match_finite_differences() {
        let x = [0.3, -0.7, 1.1, 0.5];
        let jets = Jet2::variables(&x);
        let j = f_jet(&jets);

        let h = 1e-4;
        assert!((j.val - f(&x)).abs() < 1e-12, "value mismatch");

        for i in 0..4 {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd_grad = (f(&xp) - f(&xm)) / (2.0 * h);
            assert!(
                (j.grad[i] - fd_grad).abs() < 1e-5,
                "grad[{i}]: exact={}, finite-diff={fd_grad}",
                j.grad[i]
            );
        }

        for i in 0..4 {
            for k in 0..4 {
                let fd_hess = if i == k {
                    let mut xp = x;
                    let mut xm = x;
                    xp[i] += h;
                    xm[i] -= h;
                    (f(&xp) - 2.0 * f(&x) + f(&xm)) / (h * h)
                } else {
                    let mut xpp = x;
                    let mut xpm = x;
                    let mut xmp = x;
                    let mut xmm = x;
                    xpp[i] += h;
                    xpp[k] += h;
                    xpm[i] += h;
                    xpm[k] -= h;
                    xmp[i] -= h;
                    xmp[k] += h;
                    xmm[i] -= h;
                    xmm[k] -= h;
                    (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h)
                };
                assert!(
                    (j.hess[i][k] - fd_hess).abs() < 5e-3,
                    "hess[{i}][{k}]: exact={}, finite-diff={fd_hess}",
                    j.hess[i][k]
                );
            }
        }
    }

    /// Symmetry of mixed partials (`d^2f/dxidxj = d^2f/dxjdxi`) should hold
    /// exactly (to float roundoff), not approximately -- a structural
    /// property of the chain-rule formulas above, not something that needs
    /// a separate finite-difference cross-check to establish.
    #[test]
    fn jet2_hessian_is_exactly_symmetric() {
        let x = [0.9, -0.2, 0.4, 1.3];
        let jets = Jet2::variables(&x);
        let j = f_jet(&jets);
        for i in 0..4 {
            for k in 0..4 {
                assert_eq!(j.hess[i][k], j.hess[k][i], "hess[{i}][{k}] != hess[{k}][{i}]");
            }
        }
    }
}
