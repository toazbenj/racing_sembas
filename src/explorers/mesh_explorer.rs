use crate::{
    adherer_core::{Adherer, AdhererFactory, AdhererState, SamplingError},
    explorer_core::Explorer,
    extensions::Queue,
    structs::{Classifier, Halfspace, Sample, Span},
    utils::{array_distance, svector_to_array},
};
use nalgebra::{self, Const, OMatrix, SVector};
use petgraph::{graph::NodeIndex, Graph};
use rstar::{primitives::GeomWithData, RTree};

// const BASIS_VECTORS = nalgebra::OMatrix::<f64, Const<N>, Const<N>>::identity();

type NodeID = usize;
type Path<const N: usize> = (NodeID, SVector<f64, N>);
type KnnNode<const N: usize> = GeomWithData<[f64; N], NodeID>;

/// Explores a surface uniformly by using a grid-search approach.
pub struct MeshExplorer<const N: usize> {
    d: f64,
    boundary: Vec<Halfspace<N>>,
    margin: f64,
    basis_vectors: OMatrix<f64, Const<N>, Const<N>>,
    path_queue: Vec<Path<N>>,
    current_parent: NodeID,
    tree: Graph<Halfspace<N>, ()>,
    knn_index: RTree<KnnNode<N>>,
    // next_id: usize,
    // prev_id: usize,
    adherer: Option<Box<dyn Adherer<N>>>,
    adherer_f: Box<dyn AdhererFactory<N>>,
}

impl<const N: usize> MeshExplorer<N> {
    /// Creates a MeshExplorer instance.
    /// ## Arguments
    /// * d: The jump distance between boundary points. Describes how far apart the
    ///   samples are taken.
    /// * root: The initial boundary halfspace to begin exploration from.
    /// * margin: 0 < margin < d, The minimum distance between a sample and a known
    ///   halfspace before a path along a cardinal direction is rejected.
    ///   Values that are closer to d improve efficiency at the cost of coverage.
    ///   If uncertain of what value to use, try 0.9 * d to start.
    pub fn new(
        d: f64,
        root: Halfspace<N>,
        margin: f64,
        adherer_f: Box<dyn AdhererFactory<N>>,
    ) -> Self {
        let boundary = vec![root];
        let basis_vectors = OMatrix::<f64, Const<N>, Const<N>>::identity();
        let path_queue = vec![];
        let current_parent = 0; // dunno
        let tree = Graph::new();
        let knn_index = RTree::new();

        let mut exp = MeshExplorer {
            d,
            boundary,
            margin,
            basis_vectors,
            path_queue,
            current_parent,
            tree,
            knn_index,
            adherer: None,
            adherer_f,
        };

        exp.add_child(root, None);

        exp
    }

    fn select_parent(&mut self) -> Option<(Halfspace<N>, NodeID, SVector<f64, N>)> {
        while let Some((id, v)) = self.path_queue.dequeue() {
            let hs = &self.boundary[id];
            let p = *hs.b + self.d * v;

            if !self.check_overlap(p) {
                return Some((*hs, id, v));
            }
        }

        None
    }

    fn add_child(&mut self, hs: Halfspace<N>, parent_id: Option<NodeIndex>) {
        let next_id = self.tree.add_node(hs);
        if let Some(pid) = parent_id {
            self.tree.add_edge(pid, next_id, ());
        }

        self.path_queue
            .extend(self.get_next_paths_from(next_id.index()));

        let b = svector_to_array(*hs.b);

        self.knn_index.insert(KnnNode::new(b, next_id.index()));
    }

    fn get_next_paths_from(&self, id: NodeID) -> Vec<Path<N>> {
        let hs = &self.boundary[id];
        let next_paths = MeshExplorer::create_cardinals(hs.n, self.basis_vectors)
            .iter()
            .map(|&v| (id, v))
            .collect();

        next_paths
    }

    fn create_cardinals(
        n: SVector<f64, N>,
        basis_vectors: OMatrix<f64, Const<N>, Const<N>>,
    ) -> Vec<SVector<f64, N>> {
        let align_vector: SVector<f64, N> = basis_vectors.column(0).into();
        let span = Span::new(n, align_vector);
        let angle = align_vector.angle(&n);

        let axes = if angle <= 1e-10 {
            basis_vectors
        } else {
            let rot = span.get_rotater()(angle);
            rot * basis_vectors
        };

        let mut cardinals = vec![];

        for i in 1..axes.ncols() {
            let column: SVector<f64, N> = axes.column(i).into();
            cardinals.push(column);
            cardinals.push(-column);
        }

        cardinals
    }

    fn check_overlap(&self, p: SVector<f64, N>) -> bool {
        let p = svector_to_array(p);

        if let Some(nearest) = self.knn_index.nearest_neighbor(&p) {
            array_distance(&p, nearest.geom()) < self.margin
        } else {
            false
        }
    }
}

impl<const N: usize> Explorer<N> for MeshExplorer<N> {
    fn step(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<Option<Sample<N>>, SamplingError<N>> {
        if self.adherer.is_none() {
            if let Some((hs, id, v)) = self.select_parent() {
                self.current_parent = id;
                self.adherer = Some(self.adherer_f.adhere_from(hs, v * self.d))
            }
        }

        let node = if let Some(ref mut adh) = self.adherer {
            let sample = adh.sample_next(classifier).inspect_err(|_| {})?;

            // Clone needed, lifespan of sample attached to adherer. Adherer will be
            // dropped when boundary found.
            let sample = sample.clone();
            let state = adh.get_state();

            if let AdhererState::FoundBoundary(hs) = state {
                self.boundary.push(hs);
                self.add_child(hs, Some(NodeIndex::new(self.current_parent)));
                self.adherer = None
            }

            Ok(Some(sample))
        } else {
            // Ends exploration
            Ok(None)
        };

        node.inspect_err(|_| self.adherer = None)
    }

    fn boundary(&self) -> &Vec<Halfspace<N>> {
        &self.boundary
    }

    fn boundary_count(&self) -> usize {
        self.boundary.len()
    }
}
