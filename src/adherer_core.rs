// The core components of an adherer

use core::fmt;

use nalgebra::SVector;

use crate::structs::{Halfspace, PointNode};

pub enum SamplingError<const N: usize> {
    BoundaryLostError(SVector<f64, N>, String),
    OutOfBoundsError(SVector<f64, N>, String),
    MaxSamplesExceeded(SVector<f64, N>, String),
}

#[derive(Clone, Copy)]
pub enum AdhererState<const N: usize> {
    Searching,
    FoundBoundary(Halfspace<N>),
}

pub trait Adherer<const N: usize> {
    fn sample_next(&mut self) -> Result<PointNode<N>, SamplingError<N>>;
    fn get_state(&self) -> AdhererState<N>;
}

pub trait AdhererFactory<const N: usize> {
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> Box<dyn Adherer<N>>;
}

impl<const N: usize> fmt::Display for SamplingError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingError::BoundaryLostError(_p, _msg) => write!(f, "BLE"),
            SamplingError::OutOfBoundsError(_p, _msg) => write!(f, "OOB"),
            SamplingError::MaxSamplesExceeded(_, _) => write!(f, "Exceeded max samples"),
            // MyError::CustomError(msg) => write!(f, "Custom Error: {}", msg),
            // MyError::IOError(err) => write!(f, "IO Error: {}", err),
            // MyError::NotFound => write!(f, "Not found error"),
        }
    }
}
