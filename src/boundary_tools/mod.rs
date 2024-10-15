use crate::prelude::{Boundary, BoundaryRTree, Halfspace, KnnNode};
use rstar::RTree;

/// Converts a boundary into an RTree. This is useful when many K-nearest neighbor
/// searches are needed.
/// ## Arguments
/// * boundary : The boundary to be placed within a RTree structure.
/// ## Return
/// * boundary_rtree : BoundaryRTree (RTree<GeomWithData<[f64; N], usize>>)
pub fn get_rtree_from_boundary<const N: usize>(boundary: &Boundary<N>) -> BoundaryRTree<N> {
    RTree::bulk_load(
        boundary
            .iter()
            .enumerate()
            .map(|(i, hs)| KnnNode::new(hs.b.into(), i))
            .collect(),
    )
}

/// Returns true if the provided halfspace @hs is likely to be on the surface of
/// @boundary. This is an early implementation, and is more of a proof-of-concept
/// than a robust solution.
/// ## Warning
/// * Performs poorly on edges sharp corners of envelopes.
/// * Performs poorly with low resolution boundaries (i.e. with large  jumpst
///   distances @d).
/// * Performs poorly when @d <= min(envelope_diameters)
/// ## Arguments
/// * hs : The halfspace to query against @boundary.
/// * boundary : The known boundary to compare @hs to.
/// ## Returns
/// * is_on_boundary : true ifthe halfspace is likely to be on the boundary,
///   otherwise false. This is an approximation and may incur false positives and
///   negatives. Accuracy improves with density and completeness of @boundary.
pub fn falls_on_boundary<const N: usize>(
    d: f64,
    hs: &Halfspace<N>,
    boundary: &Boundary<N>,
    boundary_rtree: &BoundaryRTree<N>,
) -> bool {
    // The maximum distance between two points on the boundary.
    let max_dist = d * (N as f64).sqrt();

    let node = boundary_rtree
        .nearest_neighbor(&hs.b.into())
        .expect("Boundary RTree must not be empty");

    let b_index = node.data;
    let nearest_hs = boundary.get(b_index).expect("Boundary index from BoundaryRTree node was out of bounds. This can occur if &Boundary is not the boundary same as &BoundaryRTree");

    let dist = (nearest_hs.b - hs.b).norm();
    if dist > max_dist {
        false
    } else {
        hs.n.dot(&nearest_hs.n) >= 0.0
    }
}
