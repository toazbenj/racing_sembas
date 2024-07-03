// The core components of an adherer

use nalgebra::{OMatrix, SVector};

use crate::structs::Halfspace;

pub enum AdhererErrors {
    BoundaryLostError(String),
    OutOfBoundsError(String),
}

type PointNode<const N: usize> = (SVector<f64, N>, bool);

pub trait Adherer<const N: usize> {
    fn sample_next(&mut self) -> Result<PointNode<N>, AdhererErrors>;
    fn has_next(&self) -> bool;
    fn get_result(&self) -> Option<Halfspace<N>>;
}

pub trait AdhererFactory<const N: usize> {
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>);
}
