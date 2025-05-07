use crate::prelude::{Boundary, Classifier, Domain, Halfspace, Result, Sample};

/// Acquires the EXACT boundary for a given outdated halfspace.
///
/// This is used when the FUT has undergone some transformation, leading to
/// boundary data being invalidated.
///
/// ### Return
/// - Ok(Some(hs)) : The halfspace that was successfully reacquired
/// - Ok(None) : Failed to find the boundary.
/// - Err(SamplingError) : Classifier induced error, generally unexpected unless @domain is
///     incorret.
fn reacquire_hs_incremental<const N: usize, C>(
    classifier: &mut C,
    hs: &Halfspace<N>,
    domain: &Domain<N>,
    max_err: f64,
    max_samples: Option<u32>,
) -> Result<Option<Halfspace<N>>>
where
    C: Classifier<N>,
{
    let mut prev_sample = classifier.classify(*hs.b)?;
    let init_cls = prev_sample.class();

    let s = (if init_cls { 1.0 } else { -1.0 }) * max_err * hs.n;
    let mut sample = classifier.classify(prev_sample.into_inner() + s)?;

    let mut i = 0;

    while max_samples.is_none_or(|m| i < m)
        && domain.contains(&sample)
        && sample.class() == init_cls
    {
        prev_sample = sample;
        sample = classifier.classify(sample.into_inner() + s)?;
        i += 1;
    }

    let boundary_exists = sample.class() != init_cls;

    match (boundary_exists, domain.contains(&sample), sample) {
        (true, true, Sample::WithinMode(b)) => Ok(Some(Halfspace { b: b, n: hs.n })),
        (true, _, _) => {
            if let Sample::WithinMode(b) = prev_sample {
                Ok(Some(Halfspace { b, n: hs.n }))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Attempts to reacquire the EXACT boundary after the FUT has changed in some way.
///
/// "Precise" means find_opposing_boundary is used. This requires more computation time to provide greater
/// confidence that the boundary that is found is for certain the same as @boundary.
///
/// ### Return
/// - new_boundary : The resultant boundary
/// - displacements : corresponding displacements for each halfspace in the @boundary
pub fn reacquire_precise<const N: usize, C>(
    classifier: &mut C,
    boundary: &Boundary<N>,
    domain: Domain<N>,
    max_err: f64,
    num_checks: u32,
    num_iter: u32,
) -> Result<(Vec<Halfspace<N>>, Vec<f64>)>
where
    C: Classifier<N>,
{
    let mut new_boundary = vec![];
    let mut displacements = vec![];

    for hs in boundary {
        let b = find_opposing_boundary(
            max_err, hs.b, hs.n, &domain, classifier, num_checks, num_iter,
        )?;
        let s = (b - hs.b).norm();
        new_boundary.push(Halfspace { b, n: hs.n });
        displacements.push(s);
    }

    Ok((new_boundary, displacements))
}
