use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
};

use sembas::{
    api::RemoteClassifier,
    boundary_tools::{
        bulk_insert_rtree,
        estimation::{approx_mc_volume_intersection, approx_surface},
        falls_on_boundary, get_rtree_from_boundary,
    },
    metrics::find_chords,
    prelude::*,
    search::{global_search::*, surfacing::binary_surface_search},
    structs::{
        messagse::{MSG_PHASE_BOUNDARY_EXPL, MSG_PHASE_GLOBAL_SEARCH, MSG_PHASE_SURFACE_SEARCH},
        Classifier, Halfspace,
    },
};
use serde::{Deserialize, Serialize};

const NDIM: usize = 4;
const JUMP_DIST: f64 = 0.01;
const ANGLE: f64 = 0.0873; // 5 deg
const MSG_REACQUIRE: &str = "REACQ";
const MSG_CONTINUE: &str = "CONT";

#[derive(Serialize, Deserialize)]
struct BoundaryData {
    boundary_points: Vec<Vec<f64>>,
    boundary_surface: Vec<Vec<f64>>,
}

fn main() {
    let domain = Domain::normalized();
    let mut classifier = RemoteClassifier::<NDIM>::bind("127.0.0.1:2000".to_string()).unwrap();

    println!("Finding initial pair...");
    classifier.send_msg(MSG_PHASE_GLOBAL_SEARCH).unwrap();
    let bp = find_initial_boundary_pair(&mut classifier, 1000).unwrap();

    println!("Establishing roots...");
    classifier.send_msg(MSG_PHASE_SURFACE_SEARCH).unwrap();

    let roots: Vec<Halfspace<NDIM>> =
        find_chords(JUMP_DIST * 0.25, &bp, NDIM, &domain, &mut classifier)
            .unwrap()
            .into_iter()
            .flat_map(|(a, b)| vec![a, b])
            .collect();

    println!("Initial bp: {bp:?}");
    println!("Roots: {roots:?}");

    classifier.send_msg(MSG_PHASE_BOUNDARY_EXPL).unwrap();
    let mut classifier = RemoteClassifier::<NDIM>::bind("127.0.0.1:2000".to_string()).unwrap();

    println!("Finding initial pair...");
    let bp = find_initial_boundary_pair(&mut classifier, 1000).unwrap();
    println!("Establishing root...");
    // focusing on just a single starting point
    let root = binary_surface_search(JUMP_DIST, &bp, 100, &mut classifier).unwrap();

    // let roots: Vec<Halfspace<NDIM>> =
    //     find_chords(JUMP_DIST * 0.25, &bp, NDIM, &domain, &mut classifier)
    //         .unwrap()
    //         .into_iter()
    //         .flat_map(|(a, b)| vec![a, b])
    //         .collect();

    let adh_f = ConstantAdhererFactory::new(ANGLE, None);

    let hs = match approx_surface(JUMP_DIST, root, &adh_f, &mut classifier) {
        Ok((hs, _, _)) => hs,
        Err(_) => root,
    };

    let mut expl = MeshExplorer::new(JUMP_DIST, hs, JUMP_DIST * 0.8, adh_f);
    while let Ok(Some(_)) = expl.step(&mut classifier) {}
}

fn find_initial_boundary_pair<const N: usize, C: Classifier<N>>(
    classifier: &mut C,
    max_samples: i32,
) -> Result<BoundaryPair<N>> {
    let mut search = MonteCarloSearch::new(Domain::normalized(), 1);
    let mut take_sample = move || {
        let p = search.sample();
        classifier
            .classify(p)
            .expect("Invalid sample. Bad global search domain?")
    };

    let mut t0 = None;
    let mut x0 = None;
    let mut i = 0;

    while (t0.is_none() || x0.is_none()) && i < max_samples {
        let sample = take_sample();
        match sample {
            Sample::WithinMode(t) => {
                if t0.is_none() {
                    println!("Found target");
                    t0 = Some(t)
                }
            }
            Sample::OutOfMode(x) => {
                if x0.is_none() {
                    println!("Found non-target");
                    x0 = Some(x)
                }
            }
        }

        i += 1;
    }

    if let (Some(t), Some(x)) = (t0, x0) {
        Ok(BoundaryPair::new(t, x))
    } else {
        Err(SamplingError::MaxSamplesExceeded)
    }
}

fn save_boundary<const N: usize>(boundary: &Boundary<N>, path: &str) -> io::Result<()> {
    let path = Path::new(path);
    if let Some(prefix) = path.parent() {
        std::fs::create_dir_all(prefix).unwrap();
    }
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let (boundary_points, boundary_surface): (Vec<Vec<f64>>, Vec<Vec<f64>>) = boundary
        .iter()
        .map(|hs| {
            (
                (*hs.b).iter().copied().collect(),
                hs.n.iter().copied().collect(),
            )
        })
        .unzip();

    f.write_all(
        serde_json::to_string_pretty(&BoundaryData {
            boundary_points,
            boundary_surface,
        })?
        .as_bytes(),
    )?;
    Ok(())
}
