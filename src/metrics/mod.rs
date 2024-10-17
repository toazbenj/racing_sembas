use nalgebra::{Const, OMatrix};

use crate::{
    prelude::{Halfspace, WithinMode},
    search::find_opposing_boundary,
    structs::{BoundaryPair, Classifier, Domain, Result, Span},
};

pub mod boundary_metrics;
pub mod bs_adherer_metrics;
pub mod const_adherer_metrics;

pub type Chord<const N: usize> = (Halfspace<N>, Halfspace<N>);

/// Finds @ndim number of chords through the (estimated) center of the envelope.
/// ## Arguments
/// * max_err : The maximum error (distance) allowed for boundary points to be from
///   the boundary.
/// * initial_pair : Describes where the known boundary is.
/// * ndim : How many dimensions to find the diameter for.
///   1 <= ndim <= N    
///   A value of 1 will search
///   only in the direction @initial_pair.t() - @initial_pair.x(). 0 and negative
///   numbers are invalid.
/// * domain : The region of the search space to limit the exploration to.
/// ## Return (Ok)
/// * diameters : Of size ndim, represents the diameter for each dimension
///   of the envelope. Each dimension is independent and orthogonal to each other.
/// ## Error (Err)
/// * Returns a OutOfBounds exception if either sample of the initial pair is outside
///   of the domain.
pub fn find_chords<const N: usize, C: Classifier<N>>(
    max_err: f64,
    initial_pair: &BoundaryPair<N>,
    ndim: usize,
    domain: &Domain<N>,
    classifier: &mut C,
) -> Result<Vec<Chord<N>>> {
    assert!(
        ndim >= 1,
        "Invalid number for ndim, must be positive non-zero! Got: {ndim}"
    );
    let basis_vectors = OMatrix::<f64, Const<N>, Const<N>>::identity();

    let s = initial_pair.t() - initial_pair.x();
    let v0 = basis_vectors.column(0).into();

    let angle = s.angle(&v0);
    let span = Span::new(s, v0);

    let rot = (span.get_rotater())(angle);
    let basis_vectors = rot * basis_vectors;
    let v0 = s.normalize();

    let p1 = find_opposing_boundary(max_err, *initial_pair.t(), v0, domain, classifier, 10, 10)?;
    let p2 = find_opposing_boundary(max_err, *initial_pair.t(), -v0, domain, classifier, 10, 10)?;

    let mid = p1 + (p2 - p1) / 2.0;
    let mut result = vec![(Halfspace { b: p1, n: v0 }, Halfspace { b: p2, n: -v0 })];

    for i in 1..ndim {
        let vi = basis_vectors.column(i).into_owned();

        let b1 = find_opposing_boundary(max_err, WithinMode(mid), vi, domain, classifier, 10, 10)?;

        let b2 = find_opposing_boundary(max_err, WithinMode(mid), -vi, domain, classifier, 10, 10)?;

        result.push((Halfspace { b: b1, n: v0 }, Halfspace { b: b2, n: -v0 }));
    }

    Ok(result)
}

pub fn get_diameters_from_chords<const N: usize>(chords: &[Chord<N>]) -> Vec<f64> {
    chords.iter().map(|(h1, h2)| (h2.b - h1.b).norm()).collect()
}

#[cfg(test)]
mod find_diameter {
    use nalgebra::SVector;

    use crate::{
        sps::Sphere,
        structs::{OutOfMode, WithinMode},
    };

    use super::*;

    const RADIUS: f64 = 0.25;

    fn create_sphere<const N: usize>() -> Sphere<N> {
        let c: SVector<f64, N> = SVector::from_fn(|_, _| 0.5);

        Sphere::new(c, RADIUS, Some(Domain::normalized()))
    }

    #[test]
    fn finds_the_diameter_of_sphere() {
        let d = 0.01;

        let domain = Domain::normalized();
        let mut classifier = create_sphere::<10>();
        let t: SVector<f64, 10> =
            SVector::from_fn(|i, _| if i == 0 { 0.5 - RADIUS + d * 0.75 } else { 0.5 });
        let x = SVector::zeros();

        let chords = find_chords(
            d,
            &BoundaryPair::new(WithinMode(t), OutOfMode(x)),
            10,
            &domain,
            &mut classifier,
        )
        .expect("Unexpected error from find_diameter.");

        let diameters = get_diameters_from_chords(&chords);

        assert!(
            diameters.iter().all(|x| x - 2.0 * RADIUS <= 2.0 * d),
            "One or more diameters had excessive error."
        )
    }
}
