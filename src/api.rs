use crate::prelude::messagse::{MSG_CONTINUE, MSG_END, MSG_OK};
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

/// While SEMBAS is in a directed outbound mode, it will enter a "messaging"
/// state after each request.
pub enum InboundState {
    Idle,
    Messaging,
}

/// Determines how to handle communication to the client.
///
/// none: Sends no signals to the client, other than standard
///     requests (samples to be classified).
/// phased: Updates the client with the current phase after
///     each completed request.
pub enum ApiOutboundMode {
    None,
    Phased(String),
}

/// Determines how to handle messages from the client.
///
/// none: Sends expects no signals from the client, other than
///     standard responses (classification results).
/// directed: Expects one or more messages after each completed
///     request. Only continues AFTER a "CONT" message is received.
pub enum ApiInboundMode {
    None,
    Directed(InboundState),
}

/// Represents the communication session with the client FUT.
///
/// Allows for complex bi-directional communication through a standardized
/// communication protocol.
pub struct SembasSession<const N: usize> {
    classifier: RemoteClassifier<N>,
    inbound_mode: ApiInboundMode,
    outbound_mode: ApiOutboundMode,
}

impl<const N: usize> SembasSession<N> {
    pub fn new(
        classifier: RemoteClassifier<N>,
        inbound_mode: ApiInboundMode,
        outbound_mode: ApiOutboundMode,
    ) -> Self {
        Self {
            classifier,
            inbound_mode,
            outbound_mode,
        }
    }

    pub fn bind(
        addr: String,
        inbound_mode: ApiInboundMode,
        outbound_mode: ApiOutboundMode,
    ) -> io::Result<Self> {
        Ok(Self {
            classifier: RemoteClassifier::<N>::bind(addr)?,
            inbound_mode,
            outbound_mode,
        })
    }

    pub fn update_phase(&mut self, phase: &str) -> io::Result<()> {
        assert!(
            matches!(&self.outbound_mode, ApiOutboundMode::Phased(_)),
            "Attempted to send_phase while not in ApiOutboundMode::Phased mode!"
        );

        self.outbound_mode = ApiOutboundMode::Phased(phase.to_string());

        self.classifier.send_msg(phase)
    }

    pub fn send_phase(&mut self) -> io::Result<()> {
        match &self.outbound_mode {
            ApiOutboundMode::Phased(phase) => self.classifier.send_msg(phase.as_str()),
            ApiOutboundMode::None => panic!(
                "Must be in ApiOutboundMode::Phased in order to send phase info back to client!"
            ),
        }
    }

    pub fn receive_msg(&mut self) -> io::Result<String> {
        assert!(
            matches!(
                &self.inbound_mode,
                ApiInboundMode::Directed(InboundState::Messaging)
            ),
            "Attempted to receive_msg while not in ApiInboundMode::Directed(InboundState::Messaging) mode!"
        );

        let msg = self.classifier.receive_msg()?;
        if msg == MSG_CONTINUE {
            self.inbound_mode = ApiInboundMode::Directed(InboundState::Idle);
        }
        Ok(msg)
    }
}

impl<const N: usize> Classifier<N> for SembasSession<N> {
    fn classify(&mut self, p: SVector<f64, N>) -> crate::prelude::Result<Sample<N>> {
        match &mut self.inbound_mode {
            ApiInboundMode::None => self.classifier.classify(p),
            ApiInboundMode::Directed(InboundState::Idle) => {
                self.inbound_mode = ApiInboundMode::Directed(InboundState::Messaging);
                self.classifier.classify(p)
            }
            ApiInboundMode::Directed(InboundState::Messaging) => {
                let msg = self.receive_msg()?;
                if msg == MSG_CONTINUE {
                    self.inbound_mode = ApiInboundMode::Directed(InboundState::Messaging);
                    self.classifier.classify(p)
                } else {
                    Err(SamplingError::InvalidClassifierResponse(
                        format!("By initiating a classify() call while in an ApiInboundMode::Direted(InboundState::Messaging) state, a {MSG_CONTINUE} message from client was expected in order to initiate next request. However, received {msg}'"))
                    )
                }
            }
        }
    }
}

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
