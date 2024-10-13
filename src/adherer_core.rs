use core::fmt;

use nalgebra::SVector;

use crate::structs::{Classifier, Halfspace, Sample, SamplingError};

/// A valid state of an adherer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdhererState<const N: usize> {
    Searching,
    FoundBoundary(Halfspace<N>),
}

/// An Adherer acquires a boundary halfspace relative to a known adjacent halfspace.
pub trait Adherer<const N: usize> {
    /// Takes a step in the adherence process.
    /// ## Arguments
    /// * classifier: The system under test with a performance envelope whose surface
    ///   is being explored.
    /// ## Returns
    /// * sample: Either a PointNode or a Sampling Error
    fn sample_next(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<&Sample<N>, SamplingError<N>>;

    /// Returns the current state of the adherer, either Searching or
    /// FoundBoundary(hs) where hs is the resulting halfspace.
    fn get_state(&self) -> AdhererState<N>;
}

/// Builds an Adherer and returns it. Provides a means of decoupling Explorers from
/// Adherers, such that any Explorer can use any Adherer.
pub trait AdhererFactory<const N: usize> {
    /// Constructs an Adherer that will find a boundary halfspace neighboring the
    /// given @hs halfspace in the given direction @v.
    /// ## Arguments
    /// * hs: A known boundary halfspace from which to explore the neighboring
    ///   boundary.
    /// * v: A displacement vector that is orthogonal to @hs. Warning: If this vector
    ///   is too large it can miss the envelope, resulting in
    ///   SamplingError:BoundaryLost.
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> Box<dyn Adherer<N>>;
}

impl<const N: usize> fmt::Debug for SamplingError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingError::BoundaryLost => write!(f, "Boundary lost during adherence."),
            SamplingError::OutOfBounds => {
                write!(f, "Boundary was sampled out of domain bounds.")
            }
            SamplingError::MaxSamplesExceeded => write!(f, "Exceeded max samples."),
            SamplingError::InvalidClassifierResponse(msg) => write!(f, "{msg}"),
        }
    }
}
