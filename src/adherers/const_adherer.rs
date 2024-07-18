use crate::{
    adherer_core::{Adherer, AdhererFactory, AdhererState},
    structs::{Classifier, Halfspace, Sample, SamplingError, Span},
};
use nalgebra::{Const, OMatrix, SVector};
use std::f64::consts::PI;

/// Pivots around a known boundary halfspace by taking fixed-angle rotations until
/// the boundary is crossed.
#[derive(Debug)]
pub struct ConstantAdherer<const N: usize> {
    span: Span<N>,
    pivot: Halfspace<N>,
    v: SVector<f64, N>,
    samples: Vec<Sample<N>>,
    rot: Option<OMatrix<f64, Const<N>, Const<N>>>,
    angle: f64,
    delta_angle: f64,
    max_rotation: f64,
    pub state: AdhererState<N>,
}

/// Builds a ConstantAdherer instance.
pub struct ConstantAdhererFactory<const N: usize> {
    delta_angle: f64,
    max_rotation: Option<f64>,
}

impl<const N: usize> ConstantAdherer<N> {
    pub fn new(
        pivot: Halfspace<N>,
        v: SVector<f64, N>,
        delta_angle: f64,
        max_rotation: Option<f64>,
    ) -> Self {
        let span = Span::new(pivot.n, v);

        let max_rotation = max_rotation.unwrap_or(PI);

        ConstantAdherer {
            span,
            pivot,
            v,
            delta_angle,
            max_rotation,
            rot: None,
            samples: vec![],
            angle: 0.0,
            state: AdhererState::Searching,
        }
    }

    fn take_initial_sample(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<Sample<N>, SamplingError<N>> {
        let cur = self.pivot.b + self.v;
        let cls = classifier.classify(&cur)?;
        let delta_angle = if cls {
            self.delta_angle
        } else {
            -self.delta_angle
        };
        self.rot = Some(self.span.get_rotater()(delta_angle));
        Ok(Sample::from_class(cur, cls))
    }

    fn take_sample(
        &mut self,
        rot: OMatrix<f64, Const<N>, Const<N>>,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<Sample<N>, SamplingError<N>> {
        self.v = rot * self.v;
        let cur = self.pivot.b + self.v;
        let cls = classifier.classify(&cur)?;

        self.angle += self.delta_angle;

        Ok(Sample::from_class(cur, cls))
    }
}

impl<const N: usize> Adherer<N> for ConstantAdherer<N> {
    fn get_state(&self) -> AdhererState<N> {
        self.state
    }

    fn sample_next(
        &mut self,
        classifier: &mut Box<dyn Classifier<N>>,
    ) -> Result<&Sample<N>, SamplingError<N>> {
        let cur = if let Some(rot) = self.rot {
            self.take_sample(rot, classifier)?
        } else {
            self.take_initial_sample(classifier)?
        };

        if let Some(prev) = self.samples.last() {
            match (&cur, &prev) {
                // <- Move occurs here
                (Sample::WithinMode(t), Sample::OutOfMode(_))
                | (Sample::OutOfMode(_), Sample::WithinMode(t)) => {
                    let b = *t;
                    let s = b - self.pivot.b;
                    let rot90 = self.span.get_rotater()(PI / 2.0);
                    let n = (rot90 * s).normalize();
                    self.state = AdhererState::FoundBoundary(Halfspace { b, n });
                }
                _ => {}
            }
        }

        if matches!(self.state, AdhererState::Searching {}) && self.angle > self.max_rotation {
            return Err(SamplingError::BoundaryLost);
        }

        self.samples.push(cur); // <- use of moved value occurs here

        Ok(self
            .samples
            .last()
            .expect("Invalid state, cur was not added to samples?"))
    }
}

impl<const N: usize> ConstantAdhererFactory<N> {
    pub fn new(delta_angle: f64, max_rotation: Option<f64>) -> Self {
        ConstantAdhererFactory {
            delta_angle,
            max_rotation,
        }
    }
}

impl<const N: usize> AdhererFactory<N> for ConstantAdhererFactory<N> {
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> Box<dyn Adherer<N>> {
        Box::new(ConstantAdherer::new(
            hs,
            v,
            self.delta_angle,
            self.max_rotation,
        ))
    }
}
