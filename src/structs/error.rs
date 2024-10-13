/// An error that occurred from sampling an system under test's input space.
#[derive(Clone, PartialEq)]
pub enum SamplingError {
    BoundaryLost,
    OutOfBounds,
    MaxSamplesExceeded,
    InvalidClassifierResponse(String),
}
