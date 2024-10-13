use std::{
    fs::File,
    io::{BufWriter, Write},
};

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

use crate::prelude::AdhererFactory;

use super::{Halfspace, Sample, WithinMode};

#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
pub struct ExplorationSummary<const N: usize, F>
where
    F: AdhererFactory<N>,
{
    title: String,
    adherer_type: String,
    adherer_parameters: F,
    boundary_points: Vec<Vec<f64>>,
    boundary_surface: Vec<Vec<f64>>,
    non_boundary_points: Vec<(Vec<f64>, bool)>,
    notes: Option<String>,
}

impl<const N: usize, A> ExplorationSummary<N, A>
where
    A: AdhererFactory<N>,
{
    pub fn new(
        title: &str,
        adherer_type: &str,
        adherer_parameters: A,
        boundary: &[Halfspace<N>],
        non_boundary_points: &[Sample<N>],
        notes: Option<&str>,
    ) -> Self {
        let mut b_points: Vec<Vec<f64>> = vec![];
        let mut n_points: Vec<Vec<f64>> = vec![];
        let nonb_points: Vec<(Vec<f64>, bool)> = non_boundary_points
            .iter()
            .map(|s| match s {
                Sample::WithinMode(p) => (p.iter().copied().collect(), true),
                Sample::OutOfMode(p) => (p.iter().copied().collect(), false),
            })
            .collect();

        for hs in boundary {
            b_points.push(hs.b.iter().copied().collect());
            n_points.push(hs.n.iter().copied().collect());
        }

        ExplorationSummary {
            title: title.to_string(),
            adherer_type: adherer_type.to_string(),
            adherer_parameters,
            boundary_points: b_points,
            boundary_surface: n_points,
            non_boundary_points: nonb_points,
            notes: notes.map(|s| s.to_string()),
        }
    }

    pub fn as_state(self) -> (Vec<Halfspace<N>>, Vec<Sample<N>>, A) {
        let boundary = self
            .boundary_points
            .iter()
            .zip(self.boundary_surface.iter())
            .map(|(b, n)| Halfspace {
                b: WithinMode(SVector::from_column_slice(b)),
                n: SVector::from_column_slice(n),
            })
            .collect();

        let non_bsamples = self
            .non_boundary_points
            .iter()
            .map(|(p, cls)| Sample::from_class(SVector::from_column_slice(p), *cls))
            .collect();

        (boundary, non_bsamples, self.adherer_parameters)
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn adherer_type(&self) -> &str {
        &self.adherer_type
    }

    pub fn adherer_parameters(&self) -> &A {
        &self.adherer_parameters
    }

    pub fn boundary_points(&self) -> &[Vec<f64>] {
        &self.boundary_points
    }

    pub fn boundary_surface(&self) -> &[Vec<f64>] {
        &self.boundary_surface
    }

    pub fn non_boundary_points(&self) -> &[(Vec<f64>, bool)] {
        &self.non_boundary_points
    }

    pub fn notes(&self) -> Option<&String> {
        self.notes.as_ref()
    }
}

#[cfg(feature = "io")]
impl<const N: usize, A> ExplorationSummary<N, A>
where
    A: AdhererFactory<N> + Serialize + for<'a> Deserialize<'a>,
{
    pub fn load(path: &str) -> Self {
        let f = File::open(path).expect("File not found?");
        serde_json::from_reader(f)
            .expect("Incorrect ExplorationSummary JSON or Improper JSON format.")
    }

    pub fn save(&self, path: &str) {
        let f = File::create(path).expect("Full path does not exist.");
        let mut writer = BufWriter::new(f);
        serde_json::to_writer(&mut writer, &self).expect("Serialize derive failed?");
        writer.flush().expect("Failed to flush.");
    }
}
