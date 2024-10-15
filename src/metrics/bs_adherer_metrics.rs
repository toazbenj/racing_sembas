use crate::prelude::ParameterError;

/// Provides suggested jump distance and angle for constant adherers.
/// ## WARNING
/// Although it provides a reasonable recommendation under some circumstances, it
/// fails when dealing with max_err ~= min(axes). It is best when max_err is
/// significantly smaller than min(axes)!
/// ## Arguments
/// * axes : A subset of lengths for each axis of the envelope. Not providing all
///   axes is allowed, although risks overshooting the envelope at lower resolutions.
/// * max_err : The desired error from the boundary.
/// * resolution : A scalar value, 0 < r <= 1, which measures how densely pact the
///   samples are. A value of 1 is the most possible, 0 is the least possible, but
///   you cannot use a value of 0.
/// ## Return (Ok(tuple))
/// * d : Jump distance between samples.
/// * initial_angle : The first angle to rotate the displacement by.
/// * n : The number of samples to take for each boundary point to reach desired
///   error.
/// ## Error (Err)
/// Returns error when max_err > min(axes), since no amount of rotation will result
/// in the desired max_err without requiring the jump distance to exceed the length
/// of the envelope. If this is to occur, revert to a lower max_err.
pub fn get_bs_params_by_envelope_size(
    axes: &[f64],
    max_err: f64,
    resolution: f64,
) -> Result<(f64, f64, u32), ParameterError> {
    assert!(resolution > 0.0 && resolution <= 1.0);
    let d_max = axes
        .iter()
        .copied()
        .min_by(|a, b| a.total_cmp(b))
        .expect("Must provide a non-empty list of axis lengths!");

    if max_err > d_max {
        return Err(ParameterError::Invalid(
            format!("Unable to produce recommendations due to max_err > smallest axis (min(axes)), resulting in impossible error target. Min axis: {d_max}, Max error: {max_err}").to_string()));
    }

    let d = max_err + (d_max - max_err) * (1.0 - resolution);
    let angle_0 = 110.0f64.to_radians();
    let angle_n = (max_err / d).asin();
    let n = (angle_0 / angle_n).log2().ceil() as u32 + 1;

    Ok((d, angle_0, n))
}
