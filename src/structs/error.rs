/// An error that occurred from sampling an system under test's input space.
#[derive(Clone, PartialEq)]
pub enum SamplingError<const N: usize> {
    BoundaryLost,
    OutOfBounds,
    MaxSamplesExceeded,
    InvalidClassifierResponse(String),
}
