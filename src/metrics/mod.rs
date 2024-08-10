use nalgebra::SVector;

use crate::structs::BoundaryPair;

fn diameter<const N: usize>(initial_pair: BoundaryPair<N>, target_dims: Vec<f64>) -> Vec<f64> {
    let result = vec![];
    let basis_vectors = SVector::identity();

    let s = initial_pair.t() - initial_pair.x();

    result
}
