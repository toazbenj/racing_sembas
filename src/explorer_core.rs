use crate::{
    adherer_core::AdhererError,
    structs::{Halfspace, PointNode},
};

pub trait Explorer<const N: usize> {
    fn step(&mut self) -> Result<Option<PointNode<N>>, AdhererError<N>>;

    fn get_boundary(&self) -> &Vec<Halfspace<N>>;
    fn get_boundary_count(&self) -> usize;
}
