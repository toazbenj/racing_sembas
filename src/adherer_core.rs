// The core components of an adherer

use core::fmt;

use nalgebra::SVector;

use crate::structs::{Halfspace, PointNode};

pub enum AdhererError<const N: usize> {
    BoundaryLostError(SVector<f64, N>, String),
    OutOfBoundsError(SVector<f64, N>, String),
}

#[derive(Clone, Copy)]
pub enum AdhererState<const N: usize> {
    Searching,
    FoundBoundary(Halfspace<N>),
}

pub trait Adherer<const N: usize> {
    fn sample_next(&mut self) -> Result<PointNode<N>, AdhererError<N>>;
    fn get_state(&self) -> AdhererState<N>;
}

pub trait AdhererFactory<const N: usize> {
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> Box<dyn Adherer<N>>;
}

impl<const N: usize> fmt::Display for AdhererError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdhererError::BoundaryLostError(_p, _msg) => write!(f, "BLE"),
            AdhererError::OutOfBoundsError(_p, _msg) => write!(f, "OOB"),
            // MyError::CustomError(msg) => write!(f, "Custom Error: {}", msg),
            // MyError::IOError(err) => write!(f, "IO Error: {}", err),
            // MyError::NotFound => write!(f, "Not found error"),
        }
    }
}
