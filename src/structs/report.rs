use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufWriter, Write},
};

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

use crate::prelude::AdhererFactory;

use super::{Halfspace, WithinMode};

#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
pub struct ExplorationStatus<const N: usize, F>
where
    F: AdhererFactory<N>,
{
    explorer_type: String,
    adherer_type: String,
    explorer_parameters: HashMap<String, f64>,
    adherer_parameters: F,
    b_count: usize,
    boundary_points: Vec<Vec<f64>>,
    boundary_surface: Vec<Vec<f64>>,
    notes: Option<String>,
}

impl<const N: usize, A> ExplorationStatus<N, A>
where
    A: AdhererFactory<N>,
{
    pub fn new(
        explorer_type: &str,
        adherer_type: &str,
        explorer_parameters: HashMap<String, f64>,
        adherer_parameters: A,
        boundary: &[Halfspace<N>],
        notes: Option<&str>,
    ) -> Self {
        let mut b_points: Vec<Vec<f64>> = vec![];
        let mut n_points: Vec<Vec<f64>> = vec![];

        for hs in boundary {
            b_points.push(hs.b.iter().copied().collect());
            n_points.push(hs.n.iter().copied().collect());
        }

        ExplorationStatus {
            explorer_type: explorer_type.to_string(),
            adherer_type: adherer_type.to_string(),
            explorer_parameters,
            adherer_parameters,
            b_count: b_points.len(),
            boundary_points: b_points,
            boundary_surface: n_points,
            notes: notes.map(|s| s.to_string()),
        }
    }

    pub fn as_state(self) -> (Vec<Halfspace<N>>, A) {
        let boundary = self
            .boundary_points
            .iter()
            .zip(self.boundary_surface.iter())
            .map(|(b, n)| Halfspace {
                b: WithinMode(SVector::from_column_slice(b)),
                n: SVector::from_column_slice(n),
            })
            .collect();

        (boundary, self.adherer_parameters)
    }

    pub fn title(&self) -> &str {
        &self.explorer_type
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

    pub fn notes(&self) -> Option<&String> {
        self.notes.as_ref()
    }
}

#[cfg(feature = "io")]
impl<const N: usize, A> ExplorationStatus<N, A>
where
    A: AdhererFactory<N> + Serialize + for<'a> Deserialize<'a>,
{
    pub fn load(path: &str) -> io::Result<Self> {
        let f = File::open(path)?;
        let status = serde_json::from_reader(f)?;
        Ok(status)
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let f = File::create(path)?;
        let mut writer = BufWriter::new(f);
        serde_json::to_writer(&mut writer, &self)?;
        writer.flush()?;
        Ok(())
    }
}
