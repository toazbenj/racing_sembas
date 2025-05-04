use crate::{
    prelude::{Boundary, Classifier, Domain, Halfspace, Result},
    search::find_opposing_boundary,
};

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
