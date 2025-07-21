use std::{
    f64::consts::PI,
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
};

use sembas::{
    api::SembasSession,
    boundary_tools::estimation::{approx_mc_volume, approx_surface, PredictionMode},
    prelude::{bs_adherer::BinarySearchAdhererFactory, *},
    search::{global_search::*, surfacing::binary_surface_search},
    structs::{
        messagse::{MSG_PHASE_GLOBAL_SEARCH, MSG_PHASE_SURFACE_SEARCH},
        Classifier,
    },
};
use serde::{Deserialize, Serialize};

const NDIM: usize = 2;
const NUM_BPOINTS: usize = 1000;
// const JUMP_DIST: f64 = 0.075;
const JUMP_DIST: f64 = 0.02;

#[derive(Serialize, Deserialize)]
struct BoundaryData {
    boundary_points: Vec<Vec<f64>>,
    boundary_surface: Vec<Vec<f64>>,
}

fn main() {
    let domain = Domain::<NDIM>::normalized();
    // let mut classifier = RemoteClassifier::<NDIM>::bind("127.0.0.1:2000".to_string()).unwrap();
    let mut classifier =
        SembasSession::<NDIM>::bind("127.0.0.1:2000".to_string(), MSG_PHASE_GLOBAL_SEARCH).unwrap();


    loop {
        println!("Running new test");
        run_test(&domain, &mut classifier);
    }
    

}

fn run_test<const N: usize>(domain: &Domain<N>, classifier: &mut SembasSession<N>) {
    
    println!("Finding initial pair...");
    // classifier
    let bp = find_initial_boundary_pair(classifier, 1000).unwrap();
    
    println!("Establishing roots...");
    classifier.update_phase(MSG_PHASE_SURFACE_SEARCH);
    
    let root = binary_surface_search(JUMP_DIST, &bp, 100, classifier).unwrap();
    
    let adh_f = BinarySearchAdhererFactory::new(PI / 2.0, 3);
    let root = match approx_surface(JUMP_DIST, root, &adh_f, classifier) {
        Ok((hs, _, _)) => hs,
        Err(_) => root,
    };
    
    let mut expl = MeshExplorer::new(JUMP_DIST, root, JUMP_DIST * 0.8, adh_f);
    
    for _ in 0..NUM_BPOINTS {
        match expl.step(classifier) {
            Ok(None) => println!("Ran out of boundary, ending exploration early."),
            Err(e) => println!("Got error: {e:?}"),
            _ => (),
        }
    }
    
    println!("Saving boundary");
    save_boundary(expl.boundary(), ".results/boundary.json")
        .expect("Unexpected failure while saving boundary");
    
    let volume = approx_mc_volume(
        PredictionMode::Union,
        &[(expl.boundary().as_slice(), expl.knn_index())],
        1000,
        1,
        42,
        Some(domain),
    );
    
    println!("Volume: {volume}");
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
