use nalgebra::SVector;
use rand_chacha::rand_core::le;

use crate::structs::{Classifier, Domain};

pub struct Sphere<const N: usize> {
    center: SVector<f64, N>,
    radius: f64,
    domain: Option<Domain<N>>,
}

impl<const N: usize> Sphere<N> {
    pub fn boxed(
        center: SVector<f64, N>,
        radius: f64,
        domain: Option<Domain<N>>,
    ) -> Box<Sphere<N>> {
        Box::new(Sphere {
            center,
            radius,
            domain,
        })
    }

    pub fn new(center: SVector<f64, N>, radius: f64, domain: Option<Domain<N>>) -> Sphere<N> {
        Sphere {
            center,
            radius,
            domain,
        }
    }
}

impl<const N: usize> Classifier<N> for Sphere<N> {
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool, crate::prelude::SamplingError<N>> {
        if let Some(domain) = &self.domain {
            if !domain.contains(p) {
                return Err(crate::structs::SamplingError::OutOfBounds);
            }
        }

        Ok((self.center - p).norm() <= self.radius)
    }
}

pub struct Cube<const N: usize> {
    shape: Domain<N>,
    domain: Option<Domain<N>>,
}

impl<const N: usize> Cube<N> {
    pub fn boxed(shape: Domain<N>, domain: Option<Domain<N>>) -> Box<Cube<N>> {
        Box::new(Cube { shape, domain })
    }

    pub fn new(shape: Domain<N>, domain: Option<Domain<N>>) -> Cube<N> {
        Cube { shape, domain }
    }
}

impl<const N: usize> Classifier<N> for Cube<N> {
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool, crate::prelude::SamplingError<N>> {
        if let Some(domain) = &self.domain {
            if !domain.contains(p) {
                return Err(crate::structs::SamplingError::OutOfBounds);
            }
        }

        Ok(self.shape.contains(p))
    }
}

struct SphereCluster<const N: usize> {
    spheres: Vec<Sphere<N>>,
    domain: Option<Domain<N>>,
}

impl<const N: usize> SphereCluster<N> {
    pub fn boxed(spheres: Vec<Sphere<N>>, domain: Option<Domain<N>>) -> Box<SphereCluster<N>> {
        Box::new(SphereCluster { spheres, domain })
    }

    pub fn new(spheres: Vec<Sphere<N>>, domain: Option<Domain<N>>) -> Self {
        SphereCluster { spheres, domain }
    }
}

impl<const N: usize> Classifier<N> for SphereCluster<N> {
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool, crate::prelude::SamplingError<N>> {
        if let Some(domain) = &self.domain {
            if !domain.contains(p) {
                return Err(crate::structs::SamplingError::OutOfBounds);
            }
        }

        for sphere in self.spheres.iter_mut() {
            if sphere.classify(p)? {
                return Ok(true);
            }
        }

        Ok(false)
    }
}
