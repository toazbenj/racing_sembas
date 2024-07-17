use nalgebra::SVector;

use crate::{
    adherer_core::SamplingError,
    structs::{BoundaryPair, Classifier, Halfspace, WithinMode},
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
    let mut p_t = **b_pair.t();
    let mut p_x = **b_pair.x();
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

    Ok(Halfspace {
        b: WithinMode(p_t),
        n,
    })
}

#[cfg(test)]
mod test_surfacer {
    use nalgebra::SVector;

    use crate::{
        adherer_core::SamplingError,
        structs::{BoundaryPair, Classifier, Domain, OutOfMode, Sample, WithinMode},
    };

    use super::binary_surface_search;

    const RADIUS: f64 = 0.25;
    const DIAMETER: f64 = 2.0 * RADIUS;

    struct Sphere<const N: usize> {
        pub radius: f64,
        pub center: SVector<f64, N>,
        pub domain: Domain<N>,
    }

    impl<const N: usize> Classifier<N> for Sphere<N> {
        fn classify(&mut self, p: SVector<f64, N>) -> Result<bool, SamplingError<N>> {
            if !self.domain.contains(&p) {
                return Err(SamplingError::OutOfBounds);
            }

            Ok((p - self.center).norm() <= self.radius)
        }
    }

    fn setup_sphere<const N: usize>() -> Box<dyn Classifier<N>> {
        let radius = 0.25;
        let center = SVector::from_fn(|_, _| 0.5);
        let domain = Domain::normalized();

        Box::new(Sphere {
            radius,
            center,
            domain,
        })
    }

    #[test]
    fn finds_sphere_surface() {
        const D: usize = 10;
        let mut sphere = setup_sphere::<D>();
        let d = 0.05;

        // a little perturbed from the center
        let mut t = SVector::from_fn(|_, _| 0.5);
        t[D - 1] += 0.15;
        let t = WithinMode(t);
        let mut x = SVector::from_fn(|_, _| 0.15);
        x[0] += 0.3;
        let x = OutOfMode(x);

        let hs = binary_surface_search(d, &BoundaryPair::new(t, x), 100, &mut sphere)
            .expect("Failed to find boundary?");

        assert!(
            sphere.classify(*hs.b).unwrap(),
            "Halfspace outside of geometry?"
        );
        assert!(
            !sphere.classify(*hs.b + hs.n * d).unwrap(),
            "Halfspace not on boundary?"
        );
    }

    #[test]
    fn finds_sphere_surface_within_maximum_samples() {
        const D: usize = 10;
        let mut sphere = setup_sphere::<D>();
        let d = 0.05;

        // a little perturbed from the center
        let mut t = SVector::from_fn(|_, _| 0.5);
        t[D - 1] += 0.15;
        let t = WithinMode(t);
        let mut x = SVector::from_fn(|_, _| 0.15);
        x[0] += 0.3;
        let x = OutOfMode(x);
        let max_samples = (DIAMETER / d).log2().ceil() as u32;

        let hs = binary_surface_search(d, &BoundaryPair::new(t, x), max_samples, &mut sphere)
            .expect("Failed to find boundary within the maximum number of samples ({max_samples})");

        assert!(
            sphere.classify(*hs.b).unwrap(),
            "Halfspace outside of geometry?"
        );
        assert!(
            !sphere.classify(*hs.b + hs.n * d).unwrap(),
            "Halfspace not on boundary?"
        );
    }

    #[test]
    fn fails_finding_sphere_surface_below_expected_sample_count() {
        const D: usize = 10;
        let mut sphere = setup_sphere::<D>();
        let d = 0.05;

        // a little perturbed from the center
        let t = SVector::from_fn(|_, _| 0.5);
        let t = WithinMode(t);

        let mut x = SVector::from_fn(|_, _| 0.5);
        x[0] = 0.0;
        let x = OutOfMode(x);
        let minimum_sample_count = (RADIUS / d).log2().ceil() as u32;

        let err: SamplingError<10> = binary_surface_search(
            d,
            &BoundaryPair::new(t, x),
            minimum_sample_count - 1,
            &mut sphere,
        )
        .expect_err(
            format!(
                "Found boundary despite not having enough samples ({})?",
                minimum_sample_count - 1
            )
            .as_str(),
        );
        assert_eq!(
            err,
            SamplingError::MaxSamplesExceeded,
            "Unexpected error type? Got: {err:?}"
        )
    }
}
