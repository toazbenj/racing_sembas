use nalgebra::SVector;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::structs::Domain;

/// Random exploration of the search domain.
pub struct MonteCarloSearch<const N: usize> {
    rng: ChaCha20Rng,
    domain: Domain<N>,
}

/// A system that produces points to be sampled for the purpose of exploring a
/// domain, also referred to as global search.
pub trait SearchFactory<const N: usize> {
    fn sample(&mut self) -> SVector<f64, N>;
    fn get_domain(&self) -> &Domain<N>;
}

impl<const N: usize> MonteCarloSearch<N> {
    pub fn new(domain: Domain<N>, seed: u64) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);
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

#[cfg(test)]
mod test_monte_carlo {
    use crate::structs::Domain;

    use super::{MonteCarloSearch, SearchFactory};

    #[test]
    fn all_points_fall_in_domain() {
        //
        let domain = Domain::<10>::normalized();
        let mut mc = MonteCarloSearch::new(domain.clone(), 1);

        assert!(
            (0..10000).all(|_| domain.contains(&mc.sample())),
            "MonteCarlo resulted in invalid samples - out of bounds?"
        )
    }
}
