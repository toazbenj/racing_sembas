/// Provides suggested jump distance and angle for constant adherers.
/// ## Arguments
/// * axes : A subset of lengths for each axis of the envelope. Not providing all
///   axes is allowed, although risks overshooting the envelope at lower resolutions.
/// * max_err : The desired error from the boundary.
/// * resolution : A scalar value, 0 < r <= 1, which measures how densely pact the
///   samples are. A value of 1 is the most possible, 0 is the least possible, but
///   you cannot use a value of 0.
/// ## Return
/// * d : Jump distance between samples.
/// * angle : The angle to rotate the displacement by.
pub fn get_const_params_by_envelope_size(
    axes: &[f64],
    max_err: f64,
    resolution: f64,
) -> (f64, f64) {
    assert!(resolution > 0.0 && resolution <= 1.0);
    let d = axes
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .expect("Must provide a non-empty list of axis lengths!")
        * resolution;
    (d, (max_err / d).asin())
}
