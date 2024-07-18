use nalgebra::SVector;

use super::{OutOfMode, WithinMode};

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryPair<const N: usize> {
    t: WithinMode<N>,
    x: OutOfMode<N>,
}

/// A halfspace is the smallest discrete unit of a hyper-geometry's surface. It
/// describes the location (the boundary point, b) and the direction of the surface
/// (the ortho[n]ormal surface vector, n).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Halfspace<const N: usize> {
    pub b: WithinMode<N>,
    pub n: SVector<f64, N>,
}

impl<const N: usize> BoundaryPair<N> {
    pub fn new(t: WithinMode<N>, x: OutOfMode<N>) -> Self {
        BoundaryPair { t, x }
    }

    pub fn t(&self) -> &WithinMode<N> {
        &self.t
    }

    pub fn x(&self) -> &OutOfMode<N> {
        &self.x
    }
}
