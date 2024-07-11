use serde::{Deserialize, Serialize};
use std::io::Write;
use std::io::{self, Read};
use std::net;

use crate::adherer_core::SamplingError;
use crate::structs::Domain;
use crate::{
    structs::Classifier,
    utils::{svector_to_array, vector_to_string},
};

const BUFFER_SIZE: usize = 2usize.pow(10);

#[derive(Serialize)]
struct ClassifierRequest {
    pub p: Vec<f64>,
}

#[derive(Deserialize)]
struct ClassifierResponse {
    pub cls: bool,
}

/// A means to allow an external function under test to connect to SEMBAS and request
/// where to sample next. The classifier can then be called just like any other
/// classifier.
pub struct RemoteClassifier<const N: usize> {
    stream: net::TcpStream,
    domain: Domain<N>,
}

impl<const N: usize> RemoteClassifier<N> {
    pub fn bind(addr: String) -> Result<Self, io::Error> {
        let listener = net::TcpListener::bind(addr)?;
        println!("Listening for client connection...");
        let (stream, _) = listener.accept()?;
        println!("Connection established.");
        let domain = Domain::<N>::normalized();
        Ok(RemoteClassifier { stream, domain })
    }
}

impl<const N: usize> From<io::Error> for SamplingError<N> {
    fn from(value: io::Error) -> Self {
        SamplingError::InvalidClassifierResponse(format!(
            "Invalid client response message. IO Error: {value}"
        ))
    }
}

impl<const N: usize> Classifier<N> for RemoteClassifier<N> {
    fn classify(
        &mut self,
        p: nalgebra::SVector<f64, N>,
    ) -> Result<bool, crate::adherer_core::SamplingError<N>> {
        if !self.domain.contains(&p) {
            return Err(SamplingError::OutOfBounds);
        }

        // Send request
        let v: Vec<f64> = svector_to_array(p).to_vec();
        let r = ClassifierRequest { p: v };
        let json = serde_json::to_string(&r).expect("Invalid ClassifierRequest serialization?");
        self.stream
            .write_all(json.as_bytes())
            .expect("Failed to send classification request to client.");

        // Receive response
        let mut buffer = [0u8; BUFFER_SIZE];
        let n = self.stream.read(&mut buffer)?;
        let ClassifierResponse { cls } = serde_json::from_slice(&buffer[..n]).map_err(|_| {
            SamplingError::InvalidClassifierResponse(
                "Received invalid response from client.".to_string(),
            )
        })?;

        Ok(cls)
    }
}
