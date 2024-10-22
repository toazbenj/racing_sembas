use crate::prelude::{Boundary, BoundaryRTree, Halfspace, KnnNode, Sample};
use nalgebra::SVector;
use rstar::RTree;

pub mod estimation;

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

pub fn bulk_insert_rtree<const N: usize>(btree: &mut BoundaryRTree<N>, boundary: &Boundary<N>) {
    let size = btree.size();
    for (i, hs) in boundary.iter().enumerate() {
        btree.insert(KnnNode::new(hs.b.into(), size + i));
    }
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

pub fn approx_prediction<const N: usize>(
    p: SVector<f64, N>,
    boundary: &Boundary<N>,
    btree: &BoundaryRTree<N>,
    k: u32,
) -> Sample<N> {
    let mut cls = true;
    for (_, neighbor) in (0..k).zip(btree.nearest_neighbor_iter(&p.into())) {
        let hs = boundary.get(neighbor.data).expect(
            "Invalid neighbor index used on @boundary. Often a result of @boundary being out of sync or entirely different from @btree."
        );

        let s = (p - *hs.b).normalize();
        if s.dot(&hs.n) > 0.0 {
            cls = false;
            break;
        }
    }

    Sample::from_class(p, cls)
}

#[cfg(test)]
mod falls_on_boundary_tests {
    use nalgebra::vector;

    use crate::prelude::WithinMode;

    use super::*;

    const JUMP_DIST: f64 = 0.1;

    fn get_plane() -> Vec<Halfspace<3>> {
        vec![
            Halfspace {
                b: WithinMode(vector![0.5, 0.5, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5 - JUMP_DIST, 0.5, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5 + JUMP_DIST, 0.5, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5, 0.5 - JUMP_DIST, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5, 0.5 + JUMP_DIST, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5 - JUMP_DIST, 0.5 - JUMP_DIST, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5 + JUMP_DIST, 0.5 + JUMP_DIST, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5 - JUMP_DIST, 0.5 + JUMP_DIST, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
            Halfspace {
                b: WithinMode(vector![0.5 + JUMP_DIST, 0.5 - JUMP_DIST, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
        ]
    }

    #[test]
    fn hs_on_boundary_returns_true() {
        // Boundary Plane:
        let boundary = get_plane();
        let brtree = get_rtree_from_boundary(&boundary);

        for hs in &boundary {
            assert!(
                falls_on_boundary(JUMP_DIST, hs, &boundary, &brtree),
                "falls_on_boundary for halfspace in @boundary returned false"
            );
        }
    }

    #[test]
    fn hs_off_boundary_returns_false() {
        // Boundary Plane:
        let boundary = get_plane();
        let brtree = get_rtree_from_boundary(&boundary);

        let other_hs = vector![
            // Above facing toward @boundary
            Halfspace {
                b: WithinMode(vector![0.5, 0.5, 0.55]),
                n: vector![-1.0, 0.0, 0.0],
            },
            // Below facing away from @boundary
            Halfspace {
                b: WithinMode(vector![0.45, 0.45, 0.45]),
                n: vector![-1.0, 0.0, 0.0],
            },
            // Far from @boundary
            Halfspace {
                b: WithinMode(vector![5.0, 5.0, 0.5]),
                n: vector![1.0, 0.0, 0.0],
            },
        ];

        for hs in &other_hs {
            assert!(
                !falls_on_boundary(JUMP_DIST, hs, &boundary, &brtree),
                "falls_on_boundary returned true for halfspace not in @boundary."
            )
        }
    }
}
