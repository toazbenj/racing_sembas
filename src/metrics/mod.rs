use nalgebra::{Const, OMatrix};

use crate::{
    search::find_opposing_boundary,
    structs::{BoundaryPair, Classifier, Domain, SamplingError, Span},
};

/// Finds the diameter for ndim number of dimensions. 1 <= ndim <= N
/// # Arguments
/// * d : The maximum error (distance) allowed for the diameter to have from the
///   boundary.
/// * initial_pair : Describes where the known boundary is
/// * ndim : How many dimensions to find the diameter for. A value of 0 will search
///   only in the direction @initial_pair.t() - @initial_pair.x()
/// * domain : The region of the search space to limit the exploration to.
/// # Return (Ok)
/// * diameters : Of size ndim, represents the diameter for each dimension
///   of the envelope. Each dimension is independent and orthogonal to each other.
/// # Error (Err)
/// * Returns a OutOfBounds exception if either sample of the initial pair is outside
///   of the domain.
pub fn find_diameter<const N: usize>(
    d: f64,
    initial_pair: &BoundaryPair<N>,
    ndim: usize,
    domain: Domain<N>,
    classifier: &mut Box<dyn Classifier<N>>,
) -> Result<Vec<f64>, SamplingError<N>> {
    let basis_vectors = OMatrix::<f64, Const<N>, Const<N>>::identity();

    // let v: SVector<f64, N> = SVector::new_random();
    let s = initial_pair.t() - initial_pair.x();
    let v0 = basis_vectors.column(0).into();
    // let v = span.v();

    let angle = s.angle(&v0);
    let span = Span::new(s, v0);

    let rot = (span.get_rotater())(angle);
    let basis_vectors = rot * basis_vectors;
    let v0 = s.normalize();

    let p1 = find_opposing_boundary(d, **initial_pair.t(), v0, &domain, classifier, 10, 10)?;
    let p2 = find_opposing_boundary(d, **initial_pair.t(), -v0, &domain, classifier, 10, 10)?;

    let mid = p1 + (p2 - p1) / 2.0;
    let mut result = vec![(p2 - p1).magnitude()];

    for i in 1..ndim {
        let vi = basis_vectors.column(i).into_owned();

        let b1 = find_opposing_boundary(d, mid, vi, &domain, classifier, 10, 10)?;

        let b2 = find_opposing_boundary(d, mid, -vi, &domain, classifier, 10, 10)?;

        result.push((b2 - b1).magnitude());
    }

    Ok(result)
}

}
