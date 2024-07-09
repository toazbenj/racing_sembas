use nalgebra::SVector;

use crate::{adherer_core::SamplingError, structs::Halfspace};

/// Finds the surface of an envelope, i.e. the initial halfspace for beginning
/// surface exploration by iteratively splitting the space in half until a desireable
/// distance from the boundary has been reached.
/// # Arguments
/// * `d` The desired maximum distance from the boundary.
/// * `t0` A target sample
/// * `x0` A non-target sample
/// * `max_samples` The maximum number of samples before the failing the process.
fn binary_surface_search<const N: usize>(
    d: f64,
    t0: SVector<f64, N>,
    x0: SVector<f64, N>,
    max_samples: u32,
    classifier: fn(SVector<f64, N>) -> Result<bool, SamplingError<N>>,
) -> Result<Halfspace<N>, SamplingError<N>> {
    let mut p_t = t0;
    let mut p_x = x0;
    let mut s = (p_x - p_t) / 2.0;
    let mut i = 0;

    while s.norm() > d && i < max_samples {
        if classifier(p_t + s)? {
            p_t = p_t + s;
        } else {
            p_x = p_x - s;
        }

        s = (p_x - p_t) / 2.0;
        i += 1
    }

    if i >= max_samples && s.norm() > d {
        return Err(SamplingError::MaxSamplesExceeded(
            p_t,
            format!("Exceeded {max_samples} samples.").to_string(),
        ));
    }

    let n = s.normalize();

    return Ok(Halfspace { b: p_t, n });
}
