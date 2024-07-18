use core::panic;

use nalgebra::{vector, SVector};
use sembas::{
    adherer_core::{Adherer, AdhererState},
    adherers,
    structs::{Classifier, Domain, Halfspace, SamplingError, WithinMode},
};

struct Cube<const N: usize> {
    pub domain: Domain<N>,
    pub shape: Domain<N>,
}

impl<const N: usize> Cube<N> {
    pub fn new(size: f64, center: SVector<f64, N>, domain: Domain<N>) -> Self {
        let low = center - SVector::from_fn(|_, _| size / 2.0);
        let high = center + SVector::from_fn(|_, _| size / 2.0);
        let shape = Domain::new(low, high);
        Cube { shape, domain }
    }
}

impl<const N: usize> Classifier<N> for Cube<N> {
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool, SamplingError<N>> {
        if !self.domain.contains(p) {
            return Err(SamplingError::OutOfBounds);
        }

        Ok(self.shape.contains(p))
    }
}

#[test]
fn finds_boundary_when_near() {
    let dist = 0.1;

    let b = WithinMode(vector![0.5, 0.5, 0.71]);
    let n = vector![0.0, 0.0, 1.0];
    let pivot = Halfspace { b, n };
    let v = dist * vector![1.0, 0.0, 0.0];
    let delta_angle = 15.0f64.to_radians();
    let max_rotation = 180.0f64.to_radians();

    let cube = Cube::new(0.25, vector![0.5, 0.5, 0.5], Domain::normalized());

    let z_dist = (*b - cube.shape.high())[2];
    let angle_to_boundary = (z_dist / dist).asin();
    let n_steps_to_boundary = (angle_to_boundary / delta_angle).ceil() as i32;

    let mut classifier: Box<dyn Classifier<3>> = Box::new(cube);

    let mut adh = adherers::ConstantAdherer::new(pivot, v, delta_angle, Some(max_rotation));

    let mut i = 0;

    while let AdhererState::Searching = adh.get_state() {
        adh.sample_next(&mut classifier)
            .inspect_err(|e| println!("Unexpected sampling error? {e:?}"))
            .unwrap();
        i += 1;
    }

    assert!(i <= n_steps_to_boundary + 1);
}

#[test]
fn loses_boundary_when_out_of_reach() {
    let dist = 0.1;

    let b = WithinMode(vector![0.5, 0.5, 0.5]);
    let n = vector![0.0, 0.0, 1.0];
    let pivot = Halfspace { b, n };
    let v = dist * vector![1.0, 0.0, 0.0];
    let delta_angle = 15.0f64.to_radians();
    let max_rotation = 180.0f64.to_radians();
    let max_steps = (max_rotation / delta_angle).ceil() as i32;

    let mut classifier: Box<dyn Classifier<3>> = Box::new(Cube::new(
        0.25,
        vector![0.5, 0.5, 0.5],
        Domain::normalized(),
    ));

    let mut adh = adherers::ConstantAdherer::new(pivot, v, delta_angle, Some(max_rotation));

    let mut i = 0;
    while adh.get_state() == AdhererState::Searching {
        if let Err(e) = adh.sample_next(&mut classifier) {
            assert_eq!(
                e,
                SamplingError::BoundaryLost,
                "Unexpected error type? Expected BSE got {e:?}"
            );
            return;
        }
        if i > max_steps + 1 {
            println!("Failed adherer state: {adh:?}");
            panic!("Failed to lose boundary, exceeded max steps without returning error!")
        }
        i += 1;
    }
}
