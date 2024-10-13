#![cfg(feature = "sps")]
use core::panic;

use nalgebra::vector;
use sembas::{
    adherer_core::{Adherer, AdhererState},
    adherers,
    sps::Cube,
    structs::{Classifier, Domain, Halfspace, SamplingError, WithinMode},
};

#[test]
fn const_adh_finds_boundary_when_near() {
    let dist = 0.1;

    let b = WithinMode(vector![0.5, 0.5, 0.71]);
    let n = vector![0.0, 0.0, 1.0];
    let pivot = Halfspace { b, n };
    let v = dist * vector![1.0, 0.0, 0.0];
    let delta_angle = 15.0f64.to_radians();
    let max_rotation = 180.0f64.to_radians();

    let cube = Cube::from_size(0.25, vector![0.5, 0.5, 0.5], Some(Domain::normalized()));

    let z_dist = (*b - cube.shape().high())[2];
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
fn const_adh_loses_boundary_when_out_of_reach() {
    let dist = 0.1;

    let b = WithinMode(vector![0.5, 0.5, 0.5]);
    let n = vector![0.0, 0.0, 1.0];
    let pivot = Halfspace { b, n };
    let v = dist * vector![1.0, 0.0, 0.0];
    let delta_angle = 15.0f64.to_radians();
    let max_rotation = 180.0f64.to_radians();
    let max_steps = (max_rotation / delta_angle).ceil() as i32;

    let mut classifier: Box<dyn Classifier<3>> = Box::new(Cube::from_size(
        0.25,
        vector![0.5, 0.5, 0.5],
        Some(Domain::normalized()),
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

#[test]
fn bs_adh_finds_boundary_when_near() {
    let dist = 0.1;

    let b = WithinMode(vector![0.5, 0.5, 0.71]);
    let n = vector![0.0, 0.0, 1.0];
    let pivot = Halfspace { b, n };
    let v = dist * vector![1.0, 0.0, 0.0];
    let initial_angle = 90.0f64.to_radians();
    let n = 4;

    let cube = Cube::from_size(0.25, vector![0.5, 0.5, 0.5], Some(Domain::normalized()));

    let mut classifier: Box<dyn Classifier<3>> = Box::new(cube);

    let mut adh = adherers::bs_adherer::BinarySearchAdherer::new(pivot, v, initial_angle, n);

    let mut i = 0;

    while let AdhererState::Searching = adh.get_state() {
        adh.sample_next(&mut classifier)
            .inspect_err(|e| println!("Unexpected sampling error? {e:?}"))
            .unwrap();
        i += 1;
    }

    assert_eq!(i, n, "Took too many samples!");
}

#[test]
fn bs_adh_loses_boundary_when_out_of_reach() {
    let dist = 0.1;

    let b = WithinMode(vector![0.5, 0.5, 0.5]);
    let n = vector![0.0, 0.0, 1.0];
    let pivot = Halfspace { b, n };
    let v = dist * vector![1.0, 0.0, 0.0];
    let init_angle = 90.0f64.to_radians();
    let n_iter = 4;

    let mut classifier: Box<dyn Classifier<3>> = Box::new(Cube::from_size(
        0.25,
        vector![0.5, 0.5, 0.5],
        Some(Domain::normalized()),
    ));

    let mut adh = adherers::bs_adherer::BinarySearchAdherer::new(pivot, v, init_angle, n_iter);

    while adh.get_state() == AdhererState::Searching {
        if let Err(e) = adh.sample_next(&mut classifier) {
            assert_eq!(
                e,
                SamplingError::BoundaryLost,
                "Unexpected error type? Expected BSE got {e:?}"
            );
            return;
        }
    }
}
