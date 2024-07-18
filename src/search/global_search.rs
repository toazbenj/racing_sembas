use nalgebra::SVector;
use rand::{rngs::ThreadRng, Rng};

use crate::structs::Domain;

pub struct MonteCarloSearch<const N: usize> {
    rng: ThreadRng,
    domain: Domain<N>,
}

pub trait SearchFactory<const N: usize> {
    fn sample(&mut self) -> SVector<f64, N>;
    fn get_domain(&self) -> &Domain<N>;
}

impl<const N: usize> MonteCarloSearch<N> {
    pub fn new(domain: Domain<N>) -> Self {
        let rng = rand::thread_rng();
        MonteCarloSearch { rng, domain }
    }
}

impl<const N: usize> SearchFactory<N> for MonteCarloSearch<N> {
    fn sample(&mut self) -> SVector<f64, N> {
        let v: SVector<f64, N> = SVector::from_fn(|_, _| self.rng.gen());
        v.component_mul(&self.domain.dimensions()) + self.domain.low()
    }

    fn get_domain(&self) -> &Domain<N> {
        &self.domain
    }
}
