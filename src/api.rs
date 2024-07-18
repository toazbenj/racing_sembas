use crate::structs::SamplingError;
use nalgebra::SVector;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::io::{self, Read};
use std::net;

use crate::structs::Domain;
use crate::{structs::Classifier, utils::svector_to_array};

const BUFFER_SIZE: usize = 2usize.pow(10);

#[derive(Serialize)]
struct ClassifierRequest {
    pub p: Vec<f64>,
}

#[derive(Deserialize)]
struct ClassifierResponse {
    pub cls: bool,
}

#[derive(Deserialize)]
struct Config {
    num_params: usize,
}

/// A means to allow an external function under test to connect to SEMBAS and request
/// where to sample next. The classifier can then be called just like any other
/// classifier.
pub struct RemoteClassifier<const N: usize> {
    stream: net::TcpStream,
    domain: Domain<N>,
}

impl<const N: usize> RemoteClassifier<N> {
    /// Opens a socket to be connected to by a remote function under test (FUT).  
    /// Once a connection is established, the RemoteClassifier will send the points
    /// to the FUT to be classified, and the FUT will return the resulting class
    /// (bool).
    /// # Connection Sequence
    /// 1. RemoteClassifier binds to TcpListener.
    /// 2. FUT connects to socket.
    /// 3. RemoteClassifier accepts connection.
    /// 4. FUT sends config containing { num_params } config info
    /// 5. RemoteClassifier accepts configuration, throwing error if N != num_params
    /// 6. RemoteClassifier returns OK
    /// 7. RemoteClassifier setup complete, ready to classify.
    pub fn bind(addr: String) -> Result<Self, io::Error> {
        let listener = net::TcpListener::bind(addr)?;
        println!("Listening for client connection...");
        let (mut stream, _) = listener.accept()?;
        println!("Connection established.");

        println!("Waiting for sim config...");
        let mut buffer = [0u8; BUFFER_SIZE];
        let n = stream.read(&mut buffer)?;
        let config: Config = serde_json::from_slice(&buffer[..n]).map_err(|_| {
            stream
                .write_all("ERROR\n".as_bytes())
                .expect("Invalid 'ERROR' write to stream?");
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Received malformed config data.",
            )
        })?;

        if config.num_params != N {
            stream
                .write_all(format!("{N}\n").as_bytes())
                .expect("Invalid N write to stream?");

            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Invalid number of param names! Expected {N}, Got {}",
                    config.num_params
                ),
            ));
        }

        stream
            .write_all("OK\n".as_bytes())
            .expect("Invalid 'OK' write to stream?");

        println!("Got config. Ready");

        let domain = Domain::<N>::normalized();
        Ok(RemoteClassifier { stream, domain })
    }
}

impl<const N: usize> Drop for RemoteClassifier<N> {
    fn drop(&mut self) {
        let buffer = "end\n".as_bytes();
        self.stream
            .write_all(buffer)
            .expect("Invalid 'end' write to stream?")
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
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool, SamplingError<N>> {
        if !self.domain.contains(p) {
            return Err(SamplingError::OutOfBounds);
        }

        // Send request
        let v: Vec<f64> = svector_to_array(*p).to_vec();
        let r = ClassifierRequest { p: v };
        let json = serde_json::to_string(&r).expect("Invalid ClassifierRequest serialization?");
        self.stream
            .write_all((json + "\n").as_bytes())
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

// #[cfg(test)]
// mod temporary {
//     use nalgebra::vector;

//     use super::*;

//     #[test]
//     fn manual_remote() {
//         let mut classifier = RemoteClassifier::<6>::bind("127.0.0.1:2000".to_string()).unwrap();
//         let r = classifier
//             .classify(vector![0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
//             .unwrap();
//         println!("Got: {r}");
//     }
// }
