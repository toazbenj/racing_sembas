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
