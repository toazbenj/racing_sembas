use nalgebra::{Const, OMatrix, SVector};

use crate::prelude::Halfspace;

/// Calculates a metric that describes how the surface is curved relative to the CoM,
/// -1 <= K <= 1.
/// A value of 1 means a concave sphere.
/// A value -1 means a convex sphere.
/// A value of 0 means a flat plane.
/// # Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// # Returns
/// * K : The curvature metric that describes how the surface is curved.
fn curvature<const N: usize>(boundary: &[Halfspace<N>]) -> f64 {
    let com = center_of_mass(boundary);
    // mean ( s . n )
    let mut total = 0.0;
    let mut count = 0.0;
    for hs in boundary.iter() {
        let s = hs.b - com;
        total += s.dot(&hs.n);
        count += 1.0;
    }

    total / count
}

/// Calculates the center of the geometry.
/// # Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// # Returns
/// * com : The mean position of the boundary.
fn center_of_mass<const N: usize>(boundary: &[Halfspace<N>]) -> SVector<f64, N> {
    let mut total = SVector::zeros();
    let mut count = 0.0;
    for hs in boundary.iter() {
        total = hs.b + total;
        count += 1.0;
    }

    total / count
}

/// Calculates the average direction that the boundary is facing, 0 <= v.norm() <= 1.
/// A magnitude of 0 suggests a perfect sphere, whereas a value of 1 suggests a
/// perfect plane.
/// # Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// # Returns
/// * v : The mean direction of the surface.
fn mean_direction<const N: usize>(boundary: &[Halfspace<N>]) -> SVector<f64, N> {
    let mut total = SVector::zeros();
    let mut count = 0.0;
    for hs in boundary.iter() {
        total = hs.n + total;
        count += 1.0;
    }

    total / count
}

/// Calculates how spread out the boundary is.
/// # Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// # Returns
/// * std_dev : The standard deviation of the
fn boundary_std_dev<const N: usize>(boundary: &[Halfspace<N>]) -> OMatrix<f64, Const<N>, Const<N>> {
    let com = center_of_mass(boundary);

    let mut cov = OMatrix::<f64, Const<N>, Const<N>>::zeros();
    let mut count = 0.0;

    for hs in boundary.iter() {
        let diff = hs.b - com;
        cov += diff * diff.transpose();
        count += 1.0;
    }

    cov / count
}
