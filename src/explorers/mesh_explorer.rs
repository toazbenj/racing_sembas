use crate::{
    adherer_core::{Adherer, AdhererError, AdhererFactory, AdhererState},
    explorer_core::Explorer,
    extensions::Queue,
    structs::{Halfspace, PointNode, Span},
    utils::{array_distance, svector_to_array},
};
use nalgebra::{self, Const, OMatrix, SVector};
use petgraph::{graph::NodeIndex, Graph};
use rstar::{primitives::GeomWithData, RTree};

// const BASIS_VECTORS = nalgebra::OMatrix::<f64, Const<N>, Const<N>>::identity();

type NodeID = usize;
type Path<const N: usize> = (NodeID, SVector<f64, N>);
type KnnNode<const N: usize> = GeomWithData<[f64; N], NodeID>;

struct MeshExplorer<const N: usize> {
    d: f64,
    boundary: Vec<Halfspace<N>>,
    margin: f64,
    basis_vectors: OMatrix<f64, Const<N>, Const<N>>,
    path_queue: Vec<Path<N>>,
    current_parent: NodeID,
    tree: Graph<Halfspace<N>, ()>,
    knn_index: RTree<KnnNode<N>>,
    next_id: usize,
    prev_id: usize,
    adherer: Option<Box<dyn Adherer<N>>>,
    adherer_f: Box<dyn AdhererFactory<N>>,
}

impl<const N: usize> MeshExplorer<N> {
    // Returns the
    fn select_parent(&mut self) -> Option<(Halfspace<N>, NodeID, SVector<f64, N>)> {
        while let Some((id, v)) = self.path_queue.dequeue() {
            let hs = &self.boundary[id];
            let p = hs.b + self.d * v;

            if !self.check_overlap(p) {
                self.prev_id = id;
                self.current_parent = id;

                return Some((*hs, id, v));
            }
        }

        return None;
    }

    fn add_child(&mut self, hs: Halfspace<N>, parent_id: NodeIndex) {
        self.path_queue
            .extend(self.get_next_paths_from(self.next_id));
        // TODO: We are going to need to add some KNN logic. We use RTree in python.

        let b = svector_to_array(hs.b);

        self.knn_index.insert(KnnNode::new(b, self.next_id));

        let next_id = self.tree.add_node(hs);
        self.tree.add_edge(parent_id, next_id, ());
    }

    fn get_next_paths_from(&self, id: NodeID) -> Vec<Path<N>> {
        let hs = &self.boundary[id];
        let next_paths = MeshExplorer::create_cardinals(hs.n, self.basis_vectors)
            .iter()
            .map(|&v| (id, v))
            .collect();

        return next_paths;
    }

    // Creates the cardinal vectors around a given boundary point's surface vector
    fn create_cardinals(
        n: SVector<f64, N>,
        basis_vectors: OMatrix<f64, Const<N>, Const<N>>,
    ) -> Vec<SVector<f64, N>> {
        let align_vector: SVector<f64, N> = basis_vectors.column(1).into();
        // let
        let span = Span::new(n, align_vector); // NOTE: Potential directional issue
        let angle = align_vector.angle(&n);
        let rot = span.get_rotater()(angle);

        let axes = rot * basis_vectors; //.columns(2, N);

        let mut cardinals = vec![];

        for i in 1..axes.ncols() {
            let column = axes.column(i).into();
            cardinals.push(column);
            cardinals.push(-&column);
        }
        return cardinals;
    }

    fn check_overlap(&self, p: SVector<f64, N>) -> bool {
        let p = svector_to_array(p);

        return if let Some(nearest) = self.knn_index.nearest_neighbor(&p) {
            array_distance(&p, nearest.geom()) < self.margin
        } else {
            false
        };
    }
}

impl<const N: usize> Explorer<N> for MeshExplorer<N> {
    fn step(&mut self) -> Result<Option<PointNode<N>>, AdhererError<N>> {
        let mut adherer = self.adherer.take().or_else(|| {
            if let Some((hs, id, v)) = self.select_parent() {
                // begin new adherence
                // self.update_adherer(&hs, &v);
                self.prev_id = id;
                Some(self.adherer_f.adhere_from(&hs, &v))
            } else {
                // End of exploration
                None
            }
        });

        let node;
        let state;
        if let Some(ref mut adh) = adherer.take() {
            node = adh.sample_next()?;
            state = adh.get_state();
        } else {
            // Ends exploration
            return Ok(None);
        }

        self.adherer = if let AdhererState::FoundBoundary(hs) = state {
            self.boundary.push(hs);
            None
        } else {
            adherer
        };

        return Ok(Some(node));
    }

    fn get_boundary(&self) -> &Vec<Halfspace<N>> {
        return &self.boundary;
    }

    fn get_boundary_count(&self) -> usize {
        return self.boundary.len();
    }
}
