use crate::{
    adherer_core::SamplingError,
    structs::{Halfspace, PointNode},
};

pub trait Explorer<const N: usize> {
    fn step(&mut self) -> Result<Option<PointNode<N>>, SamplingError<N>>;

    fn boundary(&self) -> &Vec<Halfspace<N>>;
    fn boundary_count(&self) -> usize;
}
