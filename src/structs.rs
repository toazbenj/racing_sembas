use core::fmt;

use nalgebra::{Const, OMatrix, SVector};

use crate::{adherer_core::SamplingError, utils::vector_to_string};

pub trait Classifier<const N: usize> {
    fn classify(&mut self, p: SVector<f64, N>) -> Result<bool, SamplingError<N>>;
}

// pub type PointNode<const N: usize> = (SVector<f64, N>, bool);

pub struct PointNode<const N: usize> {
    pub p: SVector<f64, N>,
    pub class: bool,
}

#[derive(Clone, Copy)]
pub struct Halfspace<const N: usize> {
    pub b: SVector<f64, N>,
    pub n: SVector<f64, N>,
}

pub struct Span<const N: usize> {
    u: SVector<f64, N>,
    v: SVector<f64, N>,
}

pub struct Domain<const N: usize> {
    low: SVector<f64, N>,
    high: SVector<f64, N>,
}

impl<const N: usize> Span<N> {
    // Provides a span, composed of two vectors which will be orthonormalized.
    pub fn new(u: SVector<f64, N>, v: SVector<f64, N>) -> Self {
        let u = u.normalize();
        let v = v.normalize();
        let v = (v - u * u.dot(&v)).normalize();
        Span { u, v }
    }

    pub fn u(&self) -> SVector<f64, N> {
        self.u
    }
    pub fn v(&self) -> SVector<f64, N> {
        self.v
    }

    // Provides a rotater function rot(angle: f64) which returns a rotation matrix
    // that rotates by angle radians along the @&self span.
    pub fn get_rotater(&self) -> impl Fn(f64) -> OMatrix<f64, Const<N>, Const<N>> {
        let identity = OMatrix::<f64, Const<N>, Const<N>>::identity();

        let a = self.u * self.v.transpose() - self.v * self.u.transpose();
        let b = self.v * self.v.transpose() + self.u * self.u.transpose();

        move |angle: f64| identity + a * angle.sin() + b * (angle.cos() - 1.0)
    }
}

impl<const N: usize> Domain<N> {
    /// Returns a domain bounded by the two points.
    pub fn new(p1: SVector<f64, N>, p2: SVector<f64, N>) -> Self {
        let low = p1.zip_map(&p2, |a, b| a.min(b));
        let high = p1.zip_map(&p2, |a, b| a.max(b));

        Domain { low, high }
    }

    pub fn normalized() -> Self {
        let low = SVector::<f64, N>::zeros();
        let high = SVector::<f64, N>::repeat(1.0);
        Domain { low, high }
    }

    pub fn low(&self) -> SVector<f64, N> {
        self.low
    }

    pub fn high(&self) -> SVector<f64, N> {
        self.high
    }

    pub fn contains(&self, p: SVector<f64, N>) -> bool {
        let below_low = SVector::<bool, N>::from_fn(|i, _| p[i] < self.low[i]);
        if below_low.iter().any(|&x| x) {
            return false;
        }

        let above_high = SVector::<bool, N>::from_fn(|i, _| p[i] > self.high[i]);
        if above_high.iter().any(|&x| x) {
            return false;
        }

        true
    }

    pub fn dimensions(&self) -> SVector<f64, N> {
        self.high - self.low
    }

    pub fn translate_point_domains(
        p: SVector<f64, N>,
        from: Domain<N>,
        to: Domain<N>,
    ) -> SVector<f64, N> {
        ((p - from.low).component_div(&from.dimensions())).component_mul(&to.dimensions()) + to.low
    }
}

impl<const N: usize> fmt::Display for PointNode<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PointNode(p: {}, class: {})",
            vector_to_string(&self.p),
            self.class
        )
    }
}

impl<const N: usize> fmt::Display for Span<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Span({}, {})",
            vector_to_string(&self.u),
            vector_to_string(&self.v)
        )
    }
}

#[cfg(test)]
mod span_tests {
    use super::*;

    const ATOL: f64 = 1e-10;

    fn approx(a: f64, b: f64, atol: f64) -> bool {
        (a - b).abs() < atol
    }

    #[test]
    fn is_orthogonal() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        assert!(approx(span.u.angle(&span.v).to_degrees(), 90.0, ATOL));
    }

    #[test]
    fn is_normal() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        assert!(approx(span.u.norm(), 1.0, ATOL));
        assert!(approx(span.v.norm(), 1.0, ATOL));
    }

    #[test]
    fn rotater_90() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        let x0 = v;
        let angle = 90.0f64.to_radians();

        let x1 = span.get_rotater()(angle) * x0;

        assert!(approx(x0.angle(&x1), angle, ATOL));
    }

    #[test]
    fn rotater_25() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        let x0 = v;
        let angle = 25.0f64.to_radians();

        let x1 = span.get_rotater()(angle) * x0;

        assert!(approx(x0.angle(&x1), angle, ATOL));
    }
}

// #[cfg(test)]
// mod domain_tests {
//     use super::*;

//     const ATOL: f64 = 1e-10;

//     #[test]
//     fn () {

//     }

// }
