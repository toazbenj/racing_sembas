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
