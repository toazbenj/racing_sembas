use nalgebra::SVector;

use super::{OutOfMode, Sample, WithinMode};

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
        Self { t, x }
    }

    pub fn from_samples(s1: Sample<N>, s2: Sample<N>) -> Option<BoundaryPair<N>> {
        if let (Sample::WithinMode(t), Sample::OutOfMode(x))
        | (Sample::OutOfMode(x), Sample::WithinMode(t)) = (s1, s2)
        {
            // unreadable mess omg keel over and die past me
            Some(BoundaryPair { t, x })
        } else {
            None
        }
    }

    pub fn t(&self) -> &WithinMode<N> {
        &self.t
    }

    pub fn x(&self) -> &OutOfMode<N> {
        &self.x
    }
}

pub mod backprop {
    use petgraph::graph::NodeIndex;

    pub trait Backpropegation<const N: usize> {
        fn backprop(&mut self, id: NodeIndex, margin: f64);
    }
}
