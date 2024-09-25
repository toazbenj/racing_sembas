use nalgebra::SVector;

use crate::{extensions::Queue, structs::Classifier};

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
