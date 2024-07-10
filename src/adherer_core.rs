use core::fmt;

use nalgebra::SVector;

use crate::structs::{Classifier, Halfspace, PointNode};

pub enum SamplingError<const N: usize> {
    BoundaryLost(SVector<f64, N>, String),
    OutOfBounds(SVector<f64, N>, String),
    MaxSamplesExceeded(SVector<f64, N>, String),
    InvalidClassifierResponse(String),
}

#[derive(Clone, Copy)]
pub enum AdhererState<const N: usize> {
    Searching,
    FoundBoundary(Halfspace<N>),
}

pub trait Adherer<const N: usize> {
    fn sample_next(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<PointNode<N>, SamplingError<N>>;
    fn get_state(&self) -> AdhererState<N>;
}

pub trait AdhererFactory<const N: usize> {
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> Box<dyn Adherer<N>>;
}

impl<const N: usize> fmt::Display for SamplingError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingError::BoundaryLost(_p, _msg) => write!(f, "BLE"),
            SamplingError::OutOfBounds(_p, _msg) => write!(f, "OOB"),
            SamplingError::MaxSamplesExceeded(_, _) => write!(f, "Exceeded max samples"),
            SamplingError::InvalidClassifierResponse(msg) => write!(f, "{msg}"),
        }
    }
}
