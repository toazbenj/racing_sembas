use crate::{
    adherer_core::{Adherer, AdhererError, AdhererState},
    structs::{Classifier, Domain, Halfspace, PointNode, Span},
};
use nalgebra::{Const, OMatrix, SVector};
use std::f64::consts::PI;

struct ConstantAdherer<const N: usize> {
    span: Span<N>,
    pivot: Halfspace<N>,
    v: SVector<f64, N>,
    samples: Vec<PointNode<N>>,
    rot: Option<OMatrix<f64, Const<N>, Const<N>>>,
    domain: Domain<N>,
    angle: f64,
    delta_angle: f64,
    max_rotation: f64,
    pub state: AdhererState<N>,
    classify: fn(SVector<f64, N>) -> Result<bool, AdhererError<N>>,
}

impl<const N: usize> ConstantAdherer<N> {
    fn new(
        pivot: Halfspace<N>,
        v: SVector<f64, N>,
        delta_angle: f64,
        max_rotation: Option<f64>,
        domain: Domain<N>,
        classifier: fn(SVector<f64, N>) -> Result<bool, AdhererError<N>>,
    ) -> Self {
        let span = Span::new(pivot.n, v);
        let rotater = span.get_rotater();
        let rot = rotater(delta_angle);
        let v = rotater(-PI / 2.0) * pivot.n;

        let max_rotation = max_rotation.unwrap_or(PI);

        return ConstantAdherer {
            span,
            pivot,
            v,
            domain,
            delta_angle,
            max_rotation,
            rot: None,
            samples: vec![],
            angle: 0.0,
            state: AdhererState::Searching,
            classify: classifier,
        };
    }

    fn take_initial_sample(&mut self) -> Result<PointNode<N>, AdhererError<N>> {
        let cur = self.pivot.b + self.v;
        let cls = (self.classify)(cur)?;
        let delta_angle = if cls {
            self.delta_angle
        } else {
            -self.delta_angle
        };
        self.rot = Some(self.span.get_rotater()(delta_angle));
        return Ok(PointNode { p: cur, class: cls });
    }

    fn take_sample(
        &mut self,
        rot: OMatrix<f64, Const<N>, Const<N>>,
    ) -> Result<PointNode<N>, AdhererError<N>> {
        self.v = rot * self.v;
        let cur = self.pivot.b + self.v;
        let cls = (self.classify)(cur)?;

        self.angle += self.delta_angle;

        return Ok(PointNode { p: cur, class: cls });
    }
}

impl<const N: usize> Adherer<N> for ConstantAdherer<N> {
    fn get_state(&self) -> AdhererState<N> {
        return self.state;
    }

    fn sample_next(&mut self) -> Result<PointNode<N>, AdhererError<N>> {
        let cur;
        let cls;
        if let Some(rot) = self.rot {
            PointNode { p: cur, class: cls } = self.take_sample(rot)?;
        } else {
            PointNode { p: cur, class: cls } = self.take_initial_sample()?;
        }

        if let Some(&PointNode {
            p: prev,
            class: prev_cls,
        }) = self.samples.last()
        {
            if cls != prev_cls {
                let b = if cls { cur } else { prev };
                let s = b - self.pivot.b;
                let rot90 = self.span.get_rotater()(PI / 2.0);
                let n = (rot90 * s).normalize();
                self.state = AdhererState::FoundBoundary(Halfspace { b, n })
            }
        }

        if matches!(self.state, AdhererState::Searching {}) && self.angle > self.max_rotation {
            return Err(AdhererError::BoundaryLostError(
                cur,
                "Max rotation exceeded".to_string(),
            ));
        }

        self.samples.push(PointNode { p: cur, class: cls });

        return Ok(PointNode { p: cur, class: cls });
    }
}
