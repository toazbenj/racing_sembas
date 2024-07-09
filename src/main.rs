use std::{f64::consts::PI, fmt::format, thread::sleep, time::Duration};

use adherer_core::{Adherer, AdhererState, SamplingError};
use adherers::const_adherer::{ConstantAdherer, ConstantAdhererFactory};
use explorer_core::Explorer;
use explorers::mesh_explorer::MeshExplorer;
use nalgebra::{coordinates::X, vector, SVector};
use plotters::prelude::*;
// use plotters::{
//     backend::BitMapBackend,
//     chart::ChartBuilder,
//     drawing::IntoDrawingArea,
//     element::{Circle, EmptyElement, Text},
//     series::PointSeries,
//     style::{BLACK, BLUE, WHITE},
// };
use structs::{Domain, Halfspace, Span};
use utils::{array_distance, svector_to_array};

mod adherer_core;
mod adherers;
mod explorer_core;
mod explorers;
mod extensions;
mod search;
mod structs;
mod utils;

const RADIUS: f64 = 1.0;
const CENTER: SVector<f64, 3> = vector![0.0, 0.0, 0.0];
const DOMAIN: Domain<3> = Domain {
    low: vector![-2.0, -2.0, -2.0],
    high: vector![2.0, 2.0, 2.0],
};
fn classify(p: SVector<f64, 3>) -> Result<bool, SamplingError<3>> {
    if !DOMAIN.contains(p) {
        return Err(SamplingError::OutOfBoundsError(
            p,
            "Out of bounds.".to_string(),
        ));
    }
    Ok((CENTER - p).norm() <= RADIUS)
}

fn main() {
    // let v: SVector<f64, 4> = SVector::from([1.0, 2.0, 3.0, 4.0]);
    test_visual();
}

fn convert_boundary_to_vec(boundary: &Vec<Halfspace<3>>) -> Vec<(f64, f64, f64)> {
    return boundary
        .iter()
        .map(|&hs| (hs.b[0], hs.b[1], hs.b[2]))
        .collect();
}

fn test_visual() {
    let d = 0.25;
    let b0 = vector![0.0, 1.0, 0.0];
    let n0 = vector![0.0, 1.0, 0.0];

    let pivot = Halfspace { b: b0, n: n0 };
    let delta_angle = 10.0f64.to_radians();

    // let mut adh = ConstantAdherer::new(d, pivot, v, delta_angle, None, DOMAIN, classify);

    let adhf = ConstantAdhererFactory::new(delta_angle, Some(PI), classify);

    let mut exp = MeshExplorer::new(d, pivot, d * 0.9, Box::new(adhf));

    let mut frame_number = 0;

    let root_area = BitMapBackend::gif("images/output.gif", (800, 600), 100)
        .unwrap()
        .into_drawing_area();

    // loop {
    //     match exp.step() {
    //         Ok(x) => match x {
    //             Some(node) => println!("Node: {}", node),
    //             None => {
    //                 println!("Done?");
    //                 break;
    //             }
    //         },
    //         Err(e) => {
    //             println!("Error: {}", e);
    //             break;
    //         }
    //     }
    // }

    while let Ok(Some(node)) = exp.step() {
        // println!("Sample: {}", node)
        root_area.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root_area)
            .caption("Adherence Process", ("sans-serif", 50))
            .build_cartesian_3d(-2.0..2.0, -2.0..2.0, -2.0..2.0)
            .unwrap();

        // chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(PointSeries::of_element(
                convert_boundary_to_vec(exp.boundary()),
                5,
                &RED,
                &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
            ))
            .unwrap();

        chart
            .draw_series(PointSeries::of_element(
                vec![(b0[0], b0[1], b0[2]), (node.p[0], node.p[1], node.p[2])],
                5,
                &BLUE,
                &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
            ))
            .unwrap();

        root_area.present().unwrap();
        if frame_number > 500 {
            break;
        }
        // sleep(Duration::from_millis(500));
        frame_number += 1;
    }
}

fn test_rot() {
    let n0 = vector![0.0, 1.0];
    let v = vector![1.0, 0.0];
    let span = Span::new(n0, v);
    let rot = span.get_rotater()(30.0f64.to_radians());

    println!("Span: {}", span);

    let mut a = vector![1.0, 1.0];

    for _ in 0..5 {
        a = rot * a;
        println!("v: {}", a);
    }
}
