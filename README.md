# SEMBAS

State space Exploration using Boundary Adherence Strategies (SEMBAS) is a technique for finding phase transitions in performance for a search space. The boundary between performance modes can provide useful insight into the characteristics of the system. For example, if the target performance mode is "Safety" (which may be defined by your specific use case) the boundary between safe and unsafe behavior can provide meaningful test cases, whereas the extremes (highly safe and unsafe) may provide no insight at all. Furthermore, these boundaries may be used to analyze probability of performance modes be sampled from (executed by the function under test.)

An example of the boundary exploration process can be found under images/output.gif. Red dots are the explored boundary, blue dots are the samples taken by the adherence algorithm to find the next boundary point. This example explores a simple 3-Dimensional sphere, but SEMBAS can be applied to arbitrarily high dimensions.

## State of v0.1.0

Currently, the implementation is fairly unstable (which can be seen in the commit
history) but the goal of v0.2.0 will to clear up any remaining design decisions.

## Road Map to stable v0.2.0

- [x] Add server to allow for external classifier.
- [ ] Add unit tests for search methods
  - [ ] Binary surface search
  - [ ] MonteCarlo global search
- [x] Add unit tests for implemented traits
  - [x] Domain
  - [x] Span
- [ ] Add integration tests
  - [x] ConstantAdherer
  - [ ] MeshExplorer
- [ ] Add examples
  - [ ] MeshExplorer for Sphere
  - [ ] RemoteClassifier
- [ ] Add description to all structs
  - [x] Domain
  - [x] Span
  - [x] PointNode
  - [x] Halfspace
- [x] Add docs to all methods
  - [x] Domain methods
  - [x] Span methods
  - [x] Adherer methods
  - [x] Explorer methods
  - [x] ConstantAdherer methods
  - [x] MeshExplorer methods
- [ ] Compile-time Conditional logging feature
- [ ] Compile-time error handling of target/nontarget samples.

## Road Map to v0.5.0 (incomplete)

- [ ] Improve MeshExplorer step() method
  - [ ] Create a Explorer state enum (Exploring, EarlyTerminating, Interim)
- [ ] Add ExponentialAdherer
- [ ] Add integration tests
  - [ ] ExponentialAdherer
