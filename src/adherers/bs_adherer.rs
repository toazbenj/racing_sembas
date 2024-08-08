use crate::{
    adherer_core::{Adherer, AdhererFactory, AdhererState},
    structs::{Classifier, Halfspace, OutOfMode, Sample, SamplingError, Span, WithinMode},
};
use nalgebra::{Const, OMatrix, SVector};
use std::f64::consts::PI;

/// Pivots around a known boundary halfspace by taking fixed-angle rotations until
/// the boundary is crossed.
pub struct BinarySearchAdherer<const N: usize> {
    pivot: Halfspace<N>,
    v: SVector<f64, N>,
    samples: Vec<Sample<N>>,
    n_iter: u32,
    angle: f64,
    prev_cls: Option<bool>,
    t: Option<WithinMode<N>>,
    x: Option<OutOfMode<N>>,
    rot_factory: Box<dyn Fn(f64) -> OMatrix<f64, Const<N>, Const<N>>>,
    pub state: AdhererState<N>,
}

/// Builds a ConstantAdherer instance.
pub struct BinarySearchAdhererFactory<const N: usize> {
    init_angle: f64,
    n_iter: u32,
}

impl<const N: usize> BinarySearchAdherer<N> {
    pub fn new(pivot: Halfspace<N>, v: SVector<f64, N>, init_angle: f64, n_iter: u32) -> Self {
        let rot_factory = Span::new(pivot.n, v).get_rotater();

        BinarySearchAdherer {
            pivot,
            v,
            samples: vec![],
            n_iter,
            angle: init_angle,
            prev_cls: None,
            t: None,
            x: None,
            rot_factory: Box::new(rot_factory),
            state: AdhererState::Searching,
        }
    }

    fn take_initial_sample(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<Sample<N>, SamplingError<N>> {
        let cur = self.pivot.b + self.v;
        let cls = classifier.classify(&cur)?;
        self.prev_cls = Some(cls);

        if cls {
            self.t = Some(cur.into());
        } else {
            self.x = Some(cur.into());
        }

        self.n_iter -= 1;

        Ok(Sample::from_class(cur, cls))
    }

    fn take_sample(
        &mut self,
        prev_cls: bool,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<Sample<N>, SamplingError<N>> {
        let cof = if prev_cls { 1.0 } else { -1.0 };
        let rot = (self.rot_factory)(cof * self.angle);
        self.v = rot * self.v;

        let cur = self.pivot.b + self.v;
        let cls = classifier.classify(&cur)?;

        self.angle /= 2.0;
        self.n_iter -= 1;

        if cls {
            self.t = Some(cur.into());
        } else {
            self.x = Some(cur.into());
        }

        self.prev_cls = Some(cls);

        Ok(Sample::from_class(cur, cls))
    }
}

impl<const N: usize> Adherer<N> for BinarySearchAdherer<N> {
    fn get_state(&self) -> AdhererState<N> {
        self.state
    }

    fn sample_next(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<&Sample<N>, SamplingError<N>> {
        let cur = if let Some(prev_cls) = self.prev_cls {
            self.take_sample(prev_cls, classifier)?
        } else {
            self.take_initial_sample(classifier)?
        };

        if self.n_iter == 0 {
            if let (Some(t), Some(_)) = (self.t, self.x) {
                let rot90 = (self.rot_factory)(PI / 2.0);
                let b = t;
                let s = b - self.pivot.b;
                let n = (rot90 * s).normalize();
                self.state = AdhererState::FoundBoundary(Halfspace { b, n })
            } else {
                return Err(SamplingError::BoundaryLost);
            }
        }

        self.samples.push(cur);

        Ok(self
            .samples
            .last()
            .expect("Invalid state, cur was not added to samples?"))
    }
}

impl<const N: usize> BinarySearchAdhererFactory<N> {
    pub fn new(init_angle: f64, n_iter: u32) -> Self {
        BinarySearchAdhererFactory { init_angle, n_iter }
    }
}

impl<const N: usize> AdhererFactory<N> for BinarySearchAdhererFactory<N> {
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> Box<dyn Adherer<N>> {
        Box::new(BinarySearchAdherer::new(
            hs,
            v,
            self.init_angle,
            self.n_iter,
        ))
    }
}
