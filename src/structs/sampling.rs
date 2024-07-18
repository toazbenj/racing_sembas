use std::{fmt, ops::Deref};

use nalgebra::SVector;

use crate::adherer_core::SamplingError;

/// A system under test whose output can be classified as "target" or "non-target"
/// behavior. For example, safe/unsafe.
pub trait Classifier<const N: usize> {
    fn classify(&mut self, p: SVector<f64, N>) -> Result<bool, SamplingError<N>>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WithinMode<const N: usize>(pub SVector<f64, N>);
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutOfMode<const N: usize>(pub SVector<f64, N>);

/// A sample from the system under test's input space with a corresponding target
/// performance classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sample<const N: usize> {
    WithinMode(WithinMode<N>),
    OutOfMode(OutOfMode<N>),
}

impl<const N: usize> Sample<N> {
    pub fn from_class(p: SVector<f64, N>, cls: bool) -> Self {
        if cls {
            Sample::WithinMode(WithinMode(p))
        } else {
            Sample::OutOfMode(OutOfMode(p))
        }
    }

    pub fn into_inner(self) -> SVector<f64, N> {
        match self {
            Sample::WithinMode(WithinMode(p)) => p,
            Sample::OutOfMode(OutOfMode(p)) => p,
        }
    }
}

impl<const N: usize> Deref for Sample<N> {
    type Target = SVector<f64, N>;

    fn deref(&self) -> &Self::Target {
        match self {
            Sample::WithinMode(WithinMode(t)) => t,
            Sample::OutOfMode(OutOfMode(x)) => x,
        }
    }
}

impl<const N: usize> Deref for WithinMode<N> {
    type Target = SVector<f64, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<const N: usize> Deref for OutOfMode<N> {
    type Target = SVector<f64, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> fmt::Display for Sample<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Sample::WithinMode(t) => write!(f, "Target(p: {:?})", t),
            Sample::OutOfMode(x) => write!(f, "Non-Target(p: {:?})", x),
        }
    }
}
