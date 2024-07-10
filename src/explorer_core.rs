use crate::{
    adherer_core::SamplingError,
    structs::{Classifier, Halfspace, PointNode},
};

pub trait Explorer<const N: usize> {
    fn step(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<Option<PointNode<N>>, SamplingError<N>>;

    fn boundary(&self) -> &Vec<Halfspace<N>>;
    fn boundary_count(&self) -> usize;
}
