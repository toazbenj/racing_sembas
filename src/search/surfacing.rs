use nalgebra::SVector;

use crate::{
    adherer_core::SamplingError,
    structs::{BoundaryPair, Classifier, Halfspace},
};

/// Finds the surface of an envelope, i.e. the initial halfspace for beginning
/// surface exploration by iteratively splitting the space in half until a desireable
/// distance from the boundary has been reached.
/// # Arguments
/// * `d` The desired maximum distance from the boundary.
/// * `t0` A target sample
/// * `x0` A non-target sample
/// * `max_samples` The maximum number of samples before the failing the process.
pub fn binary_surface_search<const N: usize>(
    d: f64,
    b_pair: &BoundaryPair<N>,
    max_samples: u32,
    classifier: &mut Box<dyn Classifier<N>>,
) -> Result<Halfspace<N>, SamplingError<N>> {
    let mut p_t = b_pair.t().into_inner();
    let mut p_x = b_pair.x().into_inner();
    let mut s = (p_x - p_t) / 2.0;
    let mut i = 0;

    while s.norm() > d && i < max_samples {
        if classifier.classify(p_t + s)? {
            p_t += s;
        } else {
            p_x -= s;
        }

        s = (p_x - p_t) / 2.0;
        i += 1
    }

    if i >= max_samples && s.norm() > d {
        return Err(SamplingError::MaxSamplesExceeded);
    }

    let n = s.normalize();

    Ok(Halfspace { b: p_t, n })
}
