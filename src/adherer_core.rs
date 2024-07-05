// The core components of an adherer

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
    fn adhere_from(&self, hs: &Halfspace<N>, v: &SVector<f64, N>) -> Box<dyn Adherer<N>>;
}
