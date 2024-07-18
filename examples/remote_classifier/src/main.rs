use std::{
    io::{Read, Write},
    net::TcpStream,
    thread::{self, JoinHandle},
};

use nalgebra::SVector;
use sembas::{
    api::RemoteClassifier,
    structs::{Classifier, Domain, SamplingError},
};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// The number of dimensions that the UAS simulation has.
const D: usize = 7;

fn main() {
    // First start your SEMBAS program:
    let local_program_handle = setup_local();

    // Then begin you FUT (Pretend this is a different program)
    let other_program_handle = setup_remote::<D>();

    // Once they connect we can simply begin sending requests by using classifier as

    // We wait for the SEMBAS program to terminate first, since it is handling the
    // setup and teardown process.
    local_program_handle.join().unwrap();
    // The other program will then be ready to terminate.
    other_program_handle.join().unwrap();
}

fn setup_local() -> JoinHandle<()> {
    thread::spawn(|| {
        // Begin RemoteClassifier server
        let mut classifier = setup_classifier();

        let p1 = SVector::from_fn(|_, _| 0.5);
        println!("[Local] Sending {p1:?} request");
        let r = classifier
            .classify(&p1)
            .inspect_err(|e| println!("Error: {e:?}"))
            .unwrap();
        println!("[Local] Received result of {r}");

        let p2 = SVector::zeros();
        println!("[Local] Sending {p1:?} request");
        let r = classifier
            .classify(&p2)
            .inspect_err(|e| println!("Error: {e:?}"))
            .unwrap();
        println!("[Local] Received result of {r}");
    })
}

fn setup_remote<const N: usize>() -> JoinHandle<()> {
    thread::spawn(|| {
        println!("[Remote] Beginning connection...");
        let mut socket = TcpStream::connect("127.0.0.1:2000").unwrap();
        println!("[Remote] Connection established!");

        println!("[Remote] Sending config...");
        let config = json!({ "num_params": N });
        socket.write_all(config.to_string().as_bytes()).unwrap();

        let mut buffer = [0u8; 1024];
        let n = socket.read(&mut buffer).unwrap();

        let ack = std::str::from_utf8(&buffer[..n])
            .expect("Got bad message?")
            .to_string();

        println!("Got acknowledgement of {ack} back.");

        if ack.trim() != "OK" {
            println!("[Remote] Invalid configuration, exiting.");
            return;
        }

        // This would be some function that you're testing that has some well-defined
        // performance mode. Example: an AV test scenario that runs, a sim summary is
        // returned, which you parse and return true if a collision occurred
        // otherwise false.
        let mut classifier: Box<dyn Classifier<N>> = Box::new(Sphere {
            center: SVector::<f64, N>::from_fn(|_, _| 0.5),
            radius: 0.25,
            domain: Domain::normalized(),
        });

        println!("[Remote] Configuration accepted. Waiting for requests.");

        loop {
            let mut buffer = [0u8; 1024];
            let n = socket.read(&mut buffer).unwrap();
            match serde_json::from_slice(&buffer[..n]) {
                Ok(ClassifierRequest { p }) => {
                    let p = SVector::<f64, N>::from_fn(|i, _| p[i]);
                    let cls = classifier.classify(&p).unwrap_or(false);

                    let msg = serde_json::to_string(&ClassifierResponse { cls }).unwrap();
                    socket
                        .write_all((msg + "\n").as_bytes())
                        .expect("Failed to send classification result.");
                }
                Err(_) => return, // terminate when 'end' is received instead of request
            }
        }
    })
}

fn setup_classifier() -> Box<dyn Classifier<D>> {
    Box::new(RemoteClassifier::bind("127.0.0.1:2000".to_string()).expect("Failed to connect"))
}

struct Sphere<const N: usize> {
    pub radius: f64,
    pub center: SVector<f64, N>,
    pub domain: Domain<N>,
}

impl<const N: usize> Classifier<N> for Sphere<N> {
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool, SamplingError<N>> {
        if !self.domain.contains(p) {
            return Err(SamplingError::OutOfBounds);
        }

        Ok((p - self.center).norm() <= self.radius)
    }
}

#[derive(Deserialize)]
struct ClassifierRequest {
    pub p: Vec<f64>,
}

#[derive(Serialize)]
struct ClassifierResponse {
    pub cls: bool,
}
