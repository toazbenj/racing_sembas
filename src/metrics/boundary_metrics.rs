use nalgebra::{Const, OMatrix, SVector};

use crate::prelude::Halfspace;

/// Calculates K, a metric that describes how the surface is curved relative to the
/// CoM. Where -1 <= K <= 1.
/// ## Caveats
/// * If the CoM falls outside of the envelope, values close to 0 are incorrect due
///   to near and far boundaries canceling out and indicates a closed concave
///   topology for those dimensions.
/// * If CoM falls outside of the envelope, values far from zero indicate an open
///  concave topology for those dimensions.
/// ## Values
/// A value of 1 means a concave sphere. A value -1 means a convex sphere. A value of
/// 0 means a flat plane.
/// ## Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// ## Returns
/// * K : The curvature metric that describes how the surface is curved.
pub fn curvature<const N: usize>(boundary: &[Halfspace<N>]) -> f64 {
    let com = center_of_mass(boundary);
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
/// ## Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// ## Returns
/// * com : The mean position of the boundary.
pub fn center_of_mass<const N: usize>(boundary: &[Halfspace<N>]) -> SVector<f64, N> {
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
/// ## Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// ## Returns
/// * v : The mean direction of the surface.
pub fn mean_direction<const N: usize>(boundary: &[Halfspace<N>]) -> SVector<f64, N> {
    let mut total = SVector::zeros();
    let mut count = 0.0;
    for hs in boundary.iter() {
        total = hs.n + total;
        count += 1.0;
    }

    total / count
}

/// Calculates how spread out the boundary is.
/// ## Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// ## Returns
/// * std_dev : The standard deviation of the boundary point cloud.
pub fn boundary_std_dev<const N: usize>(
    boundary: &[Halfspace<N>],
) -> OMatrix<f64, Const<N>, Const<N>> {
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

/// Calculates the radius of the boundary.
/// ## Arguments
/// * boundary : The set of halfspaces describing the boundary.
/// ## Returns
/// * radius : The maximum distance from the CoM
pub fn boundary_radius<const N: usize>(boundary: &[Halfspace<N>]) -> f64 {
    let com = center_of_mass(boundary);
    boundary
        .iter()
        .map(|hs| (hs.b - com).norm())
        .min_by(|a, b| {
            a.partial_cmp(b)
                .expect("Unexpected NaN while finding min(dist from com).")
        })
        .expect("Must provide a non-empty boundary!")
}

#[cfg(test)]
mod test_metrics {
    use nalgebra::SVector;

    use crate::{
        metrics::boundary_metrics::curvature,
        prelude::{Halfspace, WithinMode},
    };

    use super::{boundary_radius, center_of_mass, mean_direction};

    fn get_simple_line<const N: usize>(n: u32, d: f64) -> Vec<Halfspace<N>> {
        let mut boundary = vec![];

        let offset = -d * (n - 1) as f64 / 2.0;
        let direction = SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.0 });

        let axis_mask = SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.0 });

        for i in 0..n {
            boundary.push(Halfspace {
                b: WithinMode(axis_mask * (offset + d * i as f64)),
                n: direction,
            });
        }

        boundary
    }

    #[test]
    fn average_direction_is_one_for_plane() {
        let boundary = get_simple_line::<10>(10, 0.1);
        let n = mean_direction(&boundary);
        assert!(n.norm() - 1.0 < 1e-10, "Incorrect mean direction.")
    }

    #[test]
    fn com_is_zero_for_plane() {
        let boundary = get_simple_line::<10>(10, 0.1);
        let com = center_of_mass(&boundary);
        assert!(com.norm() <= 1e-10, "Center of mass in incorrect location.")
    }

    #[test]
    fn correct_radius() {
        let n_points = 10;
        let d = 0.1;
        let correct_radius = d * n_points as f64 / 2.0;
        let boundary = get_simple_line::<10>(n_points, d);
        let r = boundary_radius(&boundary);

        assert!(r - correct_radius <= 1e-10, "Incorrect radius?");
    }

    #[test]
    fn curvature_is_zero_for_plane() {
        let boundary = get_simple_line::<10>(10, 0.1);
        let k = curvature(&boundary);
        assert!(k <= 1e-10, "Curvature was not 0 for a plane.")
    }
}
