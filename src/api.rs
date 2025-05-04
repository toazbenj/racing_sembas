use crate::prelude::messagse::{MSG_END, MSG_OK};
use crate::prelude::Sample;
use crate::structs::SamplingError;
use nalgebra::SVector;
use std::io::{self, Read};
use std::io::{BufRead, BufReader, Write};
use std::net;

use crate::structs::error;
use crate::structs::Classifier;
use crate::structs::Domain;

const BUFFER_CONFIG_SIZE: usize = 8;

/// Allows an external function under test to connect to SEMBAS and request
/// where to sample next. The classifier can then be called just like any other
/// classifier.
pub struct RemoteClassifier<const N: usize> {
    stream: net::TcpStream,
    domain: Domain<N>,
}

impl<const N: usize> RemoteClassifier<N> {
    /// Constructs a RemoteClassifer. Prefer using `bind()` unless you need
    /// fine-grained control. This is used internally after socket setup.
    /// During construction, sends OK signal to client.
    fn new(stream: net::TcpStream) -> Self {
        let domain = Domain::<N>::normalized();
        let mut classifier = RemoteClassifier { stream, domain };
        classifier
            .send_msg(MSG_OK)
            .expect("Invalid 'OK' write to stream?");

        classifier
    }

    /// Opens a socket to be connected to by a remote function under test (FUT).  
    /// Once a connection is established, the RemoteClassifier will send the points
    /// to the FUT to be classified, and the FUT will return the resulting class
    /// (bool).
    /// ## Connection Sequence
    /// 1. RemoteClassifier binds to TcpListener.
    /// 2. FUT connects to socket.
    /// 3. RemoteClassifier accepts connection.
    /// 4. FUT sends config containing number of params info
    /// 5. RemoteClassifier accepts configuration, throwing error if N != num params
    /// 6. RemoteClassifier sends back 'OK\n'
    /// 7. RemoteClassifier setup complete, ready to classify.
    pub fn bind(addr: String) -> io::Result<Self> {
        let listener = net::TcpListener::bind(addr)?;
        println!("Listening for client connection...");
        let (mut stream, _) = listener.accept()?;
        println!("Connection established.");

        println!("Waiting for sim config...");
        let mut buffer = [0u8; BUFFER_CONFIG_SIZE];
        stream.read_exact(&mut buffer)?;
        let num_params = usize::from_be_bytes(buffer);

        if num_params != N {
            stream
                .write_all(format!("{N}\n").as_bytes())
                .expect("Invalid N write to stream?");

            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid number of param names! Expected {N}, Got {num_params}"),
            ));
        }

        println!("Got valid config. Ready.");

        Ok(RemoteClassifier::new(stream))
    }

    /// Send a message to the client.
    ///
    /// Assertion Error: @msg must not contain a newline character.
    ///     Do not include newline ('\n') characters in your messages,
    ///     which is used to determine the end of the line and is automatically
    ///     appended to your data. To prevent this from causing unexpected
    ///     runtime defects,
    ///
    /// Provides a means of sending custom signals to the client. You can
    /// send anything, but be sure that the client is prepared to receive these
    /// messages. Pre-defined messages exist within structs/constants, which are
    /// already implemented in the provided python api scripts (not yet on pep).
    pub fn send_msg(&mut self, msg: &str) -> io::Result<()> {
        assert!(!msg.contains("\n"));

        let message = format!("{}\n", msg);
        self.stream.write_all(message.as_bytes())?;
        self.stream.flush()?;

        Ok(())
    }

    /// Receive a message from the client.
    ///
    /// WARNING: Destructive, if recieve_msg is called on a classification response,
    /// the class will be converted to a ASCII String.
    ///
    /// Provides a means of receiving custom signals from the client.
    /// Pre-defined messages exist within structs/constants, which are
    /// already implemented in the provided python api scripts (not yet on pep).
    pub fn receive_msg(&mut self) -> io::Result<String> {
        let mut reader = BufReader::new(&mut self.stream);

        let mut line = String::new();
        reader.read_line(&mut line)?;

        Ok(line)
    }
}

impl<const N: usize> Drop for RemoteClassifier<N> {
    fn drop(&mut self) {
        self.send_msg(MSG_END)
            .expect("Invalid 'END' write to stream?");
    }
}

impl From<io::Error> for SamplingError {
    fn from(value: io::Error) -> Self {
        SamplingError::InvalidClassifierResponse(format!(
            "Invalid client response message. IO Error: {value}"
        ))
    }
}

impl<const N: usize> Classifier<N> for RemoteClassifier<N> {
    fn classify(&mut self, p: SVector<f64, N>) -> error::Result<Sample<N>> {
        if !self.domain.contains(&p) {
            return Err(SamplingError::OutOfBounds);
        }

        // Send request
        let bytes: &[u8] = bytemuck::cast_slice(p.as_slice());
        self.stream.write_all(bytes)?;

        let mut buffer = [0; 1];
        self.stream.read_exact(&mut buffer)?;
        if buffer[0] > 1 {
            Err(SamplingError::InvalidClassifierResponse(
                "Remote Classifier received non-bool response?".to_string(),
            ))
        } else {
            Ok(Sample::from_class(p, buffer[0] == 1))
        }
    }
}
