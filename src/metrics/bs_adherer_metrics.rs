/// Provides suggested jump distance and angle for constant adherers.
/// ## Arguments
/// * axes : A subset of lengths for each axis of the envelope. Not providing all
///   axes is allowed, although risks overshooting the envelope at lower resolutions.
/// * max_err : The desired error from the boundary.
/// * resolution : A scalar value, 0 < r <= 1, which measures how densely pact the
///   samples are. A value of 1 is the most possible, 0 is the least possible, but
///   you cannot use a value of 0.
/// ## Return (tuple)
/// * d : Jump distance between samples.
/// * initial_angle : The first angle to rotate the displacement by.
/// * n : The number of samples to take for each boundary point to reach desired
///   error.
pub fn get_bs_params_by_envelope_size(
    axes: &[f64],
    max_err: f64,
    resolution: f64,
) -> (f64, f64, u32) {
    assert!(resolution > 0.0 && resolution <= 1.0);
    let d = axes
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .expect("Must provide a non-empty list of axis lengths!")
        * (1.0 - resolution);

    let angle_0 = 110.0f64.to_radians();
    let angle_n = (max_err / d).asin();
    let n = (angle_0 / angle_n).log2().ceil() as u32 + 1;
    (d, angle_0, n)
}

#[cfg(test)]
mod test {
    use super::get_bs_params_by_envelope_size;

    #[test]
    fn test() {
        let (d, a, n) = get_bs_params_by_envelope_size(&[0.5, 0.5, 0.5], 0.03, 0.5);
        println!("{d}, {}, {n}", a.to_degrees());

        let an = a / (2.0f64.powi(n as i32));
        let err = an * d;
        println!("{}, {}", an.to_degrees(), err);
    }
}
