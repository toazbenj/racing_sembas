use nalgebra::SVector;
use surfacing::binary_surface_search;

use crate::{
    extensions::Queue,
    structs::{BoundaryPair, Classifier, Domain, OutOfMode, SamplingError, WithinMode},
};

#[cfg(feature = "global_search")]
pub mod global_search;

#[cfg(feature = "surfacing")]
pub mod surfacing;

pub enum SearchMode {
    Full,
    Nearest,
}

/// Searches the space between two points for a specific performance mode.
/// # Arguments
/// * target_clas : The performance mode (in-mode or out-of-mode) to search for.
///   Function termins which target class is found.
/// * max_samples : A limit on how many samples are taken before terminating. Returns
///   None if max samples is reached before finding the @target_cls.
/// * p1, p2 : The two points to search between.
/// * classifier : The FUT.
/// # Returns
/// * Some(p) : The point that is classified as target_cls, i.e.
///   classifier.classify(p) == target_cls
/// * None : No target_cls points were found within max_samples number of iterations.
pub fn binary_search_between<const N: usize>(
    mode: SearchMode,
    target_cls: bool,
    max_samples: u32,
    p1: SVector<f64, N>,
    p2: SVector<f64, N>,
    classifier: &mut Box<dyn Classifier<N>>,
) -> Option<SVector<f64, N>> {
    let mut pairs = vec![(p1, p2)];

    for _ in 0..max_samples {
        let (p1, p2) = pairs
            .dequeue()
            .expect("Unexpectedly ran out of pairs to explore during search?");
        let s = p2 - p1;
        let mid = p1 + s / 2.0;
        let cls = classifier.classify(&mid).expect(
            "Classifier threw error when sampling. Make sure @p1 and @p2 are valid samples?",
        );
        if cls == target_cls {
            return Some(mid);
        }

        match mode {
            SearchMode::Full => {
                pairs.enqueue((p1, mid));
                pairs.enqueue((mid, p2));
            }
            SearchMode::Nearest => pairs.enqueue((p1, mid)),
        }
    }

    None
}

/// Finds a boundary point on the opposite side of an envelope given a starting
/// boundary point and directional vector of the chord between these boundary points.
/// **Note: If there is no Out-of-Mode samples between @b and the edge of the domain,
/// the point lying on the edge of the domain will be returned.**
/// # Arguments
/// * d : The maximum allowable distance from the boundary for the opposing boundary
///   point.
/// * b : A point that lies on the boundary of the envelope whose diameter is being
///   measured.
/// * v : A unit vector describing the direction of the chord. Note: v must point
///   TOWARDS the envelope, not away. i.e. its dot product with the OSV should be
///   negative, n.dot(v) < 0,
/// * domain : The domain to constrain exploration to be within.
/// * classifier : The classifier for the FUT.
/// # Returns (Ok)
/// * Ok(b2) : The point within the envelope opposite that of @b.
/// # Error (Err)
/// * Err(SamplingError) : Failed to find any target performance samples in the
///   direction @v. Often caused by an invalid v, n.dot(v) > 0, or insufficient
///   @max_samples.
pub fn find_opposing_boundary<const N: usize>(
    d: f64,
    b: SVector<f64, N>,
    v: SVector<f64, N>,
    domain: Domain<N>,
    classifier: &mut Box<dyn Classifier<N>>,
    num_checks: u32,
    num_iter: u32,
) -> Result<WithinMode<N>, SamplingError<N>> {
    let dist = domain.distance_to_edge(&b, &v)? * 0.999;
    let p = b + v * dist;

    let cls = classifier.classify(&p).expect(
        "A point that was supposed to be on the edge of the domain (yet inside) fell outside of the classifier's domain. Incorrect @domain?");

    let (mut t, mut x) = if cls {
        (p, None)
    } else {
        // Find next target hit
        let t = binary_search_between(SearchMode::Nearest, true, num_checks, b, p, classifier)
            .expect("@num_checks is too small, was unable to re-acquire envelope.");
        (t, Some(p))
    };

    // While there are gaps, explore towards envelope from @b
    while let Some(gap) =
        binary_search_between(SearchMode::Full, false, num_checks, t, b, classifier)
    {
        t = binary_search_between(
            SearchMode::Nearest,
            true,
            num_checks,
            b,
            gap,
            classifier,
        ).expect("Fatal error. Although able to find envelope originally, envelope was lost while eleminating intermediate envelopes?");
        x = Some(gap);
    }

    let b2 = match x {
        Some(x) => binary_surface_search(
            d,
            &BoundaryPair::new(WithinMode(t), OutOfMode(x)),
            num_iter,
            classifier,
        ).expect("Unexpected sampling error during final binary surface search of opposing boundary point.").b,
        None => WithinMode(p),
    };

    Ok(b2)
}

#[cfg(test)]
mod binary_search_between {
    use super::binary_search_between;
    use crate::structs::{Classifier, Domain};
    use nalgebra::SVector;

    struct EmptyClassifier<const N: usize> {}
    impl<const N: usize> Classifier<N> for EmptyClassifier<N> {
        fn classify(
            &mut self,
            _: &SVector<f64, N>,
        ) -> Result<bool, crate::prelude::SamplingError<N>> {
            Ok(false)
        }
    }

    struct Sphere<const N: usize> {
        c: SVector<f64, N>,
        r: f64,
        domain: Domain<N>,
    }

    impl<const N: usize> Sphere<N> {
        fn new(c: SVector<f64, N>, r: f64) -> Box<Sphere<N>> {
            Box::new(Sphere {
                c,
                r,
                domain: Domain::normalized(),
            })
        }
    }

    impl<const N: usize> Classifier<N> for Sphere<N> {
        fn classify(
            &mut self,
            p: &SVector<f64, N>,
        ) -> Result<bool, crate::prelude::SamplingError<N>> {
            if !self.domain.contains(p) {
                Err(crate::structs::SamplingError::OutOfBounds)
            } else {
                Ok((p - self.c).magnitude() < self.r)
            }
        }
    }

    fn create_sphere<const N: usize>() -> Box<dyn Classifier<N>> {
        let c: SVector<f64, N> = SVector::from_fn(|_, _| 0.5);
        let r = 0.25;

        Sphere::new(c, r)
    }

    #[test]
    fn finds_sphere() {
        let mut classifier = create_sphere::<10>();
        let p1: SVector<f64, 10> = SVector::zeros();
        let p2 = SVector::from_fn(|_, _| 1.0);

        let r = binary_search_between(super::SearchMode::Full, true, 10, p1, p2, &mut classifier)
            .expect("Failed to find sphere when it should have?");

        assert!(
            classifier
                .classify(&r)
                .expect("Unexpected out of bounds sample from BSB result?"),
            "Returned non-target (incorrect) sample?"
        )
    }

    #[test]
    fn returns_none_when_no_envelope_exists() {
        let p1: SVector<f64, 10> = SVector::zeros();
        let p2 = SVector::from_fn(|_, _| 1.0);
        let mut classifier: Box<dyn Classifier<10>> = Box::new(EmptyClassifier {});

        let r = binary_search_between(super::SearchMode::Full, true, 10, p1, p2, &mut classifier);

        assert_eq!(r, None, "Somehow found an envelope when none existed?")
    }

    #[test]
    fn returns_none_with_insufficient_max_samples() {
        let p1: SVector<f64, 10> = SVector::zeros();
        let p2 = SVector::from_fn(|_, _| 1.0);

        let c = p2 / 8.0;
        let mut classifier: Box<dyn Classifier<10>> = Sphere::new(c, 0.1);
        let num_steps_to_find = 4;

        let r = binary_search_between(
            super::SearchMode::Full,
            true,
            num_steps_to_find - 1,
            p1,
            p2,
            &mut classifier,
        );

        assert_eq!(r, None, "Found the envelope when it shouldn't have.");
    }

    #[test]
    fn finds_sphere_with_exact_max_samples() {
        let p1: SVector<f64, 10> = SVector::zeros();
        let p2 = SVector::from_fn(|_, _| 1.0);

        let c = p2 / 8.0;
        let mut classifier: Box<dyn Classifier<10>> = Sphere::new(c, 0.1);
        let num_steps_to_find = 4;

        binary_search_between(
            super::SearchMode::Full,
            true,
            num_steps_to_find,
            p1,
            p2,
            &mut classifier,
        )
        .expect("Failed to find envelope with the correct max_samples.");
    }
}
