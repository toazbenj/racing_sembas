use nalgebra::SVector;

use crate::{
    structs::{OutOfMode, Sample, WithinMode},
    utils::svector_to_array,
};

pub trait Queue<T> {
    fn enqueue(&mut self, x: T);
    fn dequeue(&mut self) -> Option<T>;
}

impl<T> Queue<T> for Vec<T> {
    fn enqueue(&mut self, x: T) {
        self.push(x);
    }

    fn dequeue(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove(0))
        }
    }
}

impl<const N: usize> From<Sample<N>> for SVector<f64, N> {
    fn from(value: Sample<N>) -> Self {
        value.into_inner()
    }
}

impl<const N: usize> From<WithinMode<N>> for SVector<f64, N> {
    fn from(value: WithinMode<N>) -> Self {
        value.0
    }
}

impl<const N: usize> From<WithinMode<N>> for [f64; N] {
    fn from(value: WithinMode<N>) -> Self {
        svector_to_array(value.0)
    }
}

impl<const N: usize> From<OutOfMode<N>> for SVector<f64, N> {
    fn from(value: OutOfMode<N>) -> Self {
        value.0
    }
}

impl<const N: usize> From<OutOfMode<N>> for [f64; N] {
    fn from(value: OutOfMode<N>) -> Self {
        svector_to_array(value.0)
    }
}
