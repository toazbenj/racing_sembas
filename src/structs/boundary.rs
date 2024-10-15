use nalgebra::SVector;

use super::{OutOfMode, Sample, WithinMode};

pub type Boundary<const N: usize> = [Halfspace<N>];

/// A pair of points, t and x, where t falls within the target performance mode and x
/// falls outside of the performance mode. When a boundary pair exists, a boundary
/// must exist between t and x.
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
    /// Creates a BoundaryPair from known target and non-target samples
    pub fn new(t: WithinMode<N>, x: OutOfMode<N>) -> Self {
        Self { t, x }
    }

    /// Creates a BoundaryPair from two samples. If both are of the same class, None
    /// is returned.
    pub fn from_samples(s1: Sample<N>, s2: Sample<N>) -> Option<Self> {
        if let (Sample::WithinMode(t), Sample::OutOfMode(x))
        | (Sample::OutOfMode(x), Sample::WithinMode(t)) = (s1, s2)
        {
            Some(BoundaryPair { t, x })
        } else {
            None
        }
    }

    /// The target performance sample that falls within the target performance mode.
    pub fn t(&self) -> &WithinMode<N> {
        &self.t
    }

    /// The non-target performance sample that falls outside of the target
    /// performance mode.
    pub fn x(&self) -> &OutOfMode<N> {
        &self.x
    }
}

pub mod backprop {
    use petgraph::graph::NodeIndex;

    /// Backpropagation is the updating of previous surface direction information
    /// from newly added surface information. This can improve the surface vector
    /// approximations, and in turn improve sampling efficiency.
    pub trait Backpropagation<const N: usize> {
        fn backprop(&mut self, id: NodeIndex, margin: f64);
    }
}
