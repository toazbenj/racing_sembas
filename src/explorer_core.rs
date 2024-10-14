use crate::{
    prelude::{report::ExplorationStatus, AdhererFactory},
    structs::{Classifier, Halfspace, Result, Sample},
};

/// The system responsible for the full boundary exploration process. Leverages
/// Adherers to find neighboring boundary points.
pub trait Explorer<const N: usize, F>
where
    F: AdhererFactory<N>,
{
    /// Take a step in the boundary exploration process.
    /// ## Arguments
    /// * classifier: The system under test whose target performance boundaries are
    ///   being explored.
    /// ## Returns
    /// * sample: A PointNode or SamplingError.
    fn step<C: Classifier<N>>(&mut self, classifier: &mut C) -> Result<Option<Sample<N>>>;

    /// Gets the current state of the explored boundary.
    /// ## Returns
    /// * boundary: A Vec<Halfspace<N>>, which contains all of the acquired boundary
    ///   halfspaces found so far.
    fn boundary(&self) -> &Vec<Halfspace<N>>;

    /// Gets the total number of boundary halfspaces found so far.
    /// ## Returns
    /// * boundary_count: The number of halfspaces that have been found.
    fn boundary_count(&self) -> usize;

    fn describe(&self) -> ExplorationStatus<N, F>;
}
