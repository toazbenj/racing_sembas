
![sembas_logo](https://github.com/user-attachments/assets/1296357d-e22f-42b4-83fa-06f93388c92f)

**State space Exploration of Multidimemsional Boundaries using Adherence Strategies (SEMBAS)** is a technique
for finding phase transitions in performance for a given search space. The boundary
between performance modes can provide useful insight into the characteristics of the
system. For example, if the target performance mode is "Safety" (which may be defined
by your specific use case) the boundary between safe and unsafe behavior can provide
meaningful test cases, whereas the extremes (highly safe and unsafe) may provide no
insight at all. Furthermore, these boundaries may be used to analyze probability of
performance modes be sampled from (executed by the function under test.)

An example of the boundary exploration process can be found under images/output.gif.
Red dots are the explored boundary and blue dots are the samples taken by the
adherence algorithm to find the next boundary point. This example explores a simple
3-Dimensional sphere, but SEMBAS can be applied to arbitrarily high dimensions.

## Running Examples
Some of the examples require certain features to be enabled. A common feature is
`sps` (synthetic parameter space) which are used to create mock functions under test.
You can easily run any example, however, by using the below command:
```
cargo run --example <example-name> --features all
```

## Adherence
An Adherence strategy is the process of finding a neighboring boundary point. This
requires both an initial boundary point, which acts as a pivot to rotate around, and
direction of travel, which indicates which direction along the boundary to search for
a new boundary point. A displacement vector can then be rotated towards and away from
the surface in order to identify new boundary points. 

This component of the system influences how efficiently the boundary is acquired, and
has the most impact on performance (i.e. the number of times the function under test
must be executed.)

## Exploration
An Exploration strategy is the process of picking the new directions along the
surface. This part of the system determines coverage of the surface, and utilizes an
Adherer in order to find the next boundary point.

## The FUT and Classifier
The Function Under Test (FUT) is the system whose input search space is being
explored. The classifier is a wrapper around this FUT which classifies the FUT's
output as either In-Mode or Out-of-Mode for the performance that is begin targeted.
In Mode simply refers to the system performing the target behavior of interest, such
as unsafe behavior, throwing an error, or breaking a given threshold. This binary
classification allows for the boundary between the In-Mode and Out-of-Mode behavior
to be explored.

## Halfspaces - How the boundary is represented

The boundary has two major components: a position and direction. The position is
obvious, it describes where the search space transitions from being within the
performance mode to being outside of the performance mode. The surface direction, on
the other hand, describes the orientation of the surface, and consequently which side
corresponds to the target performance mode. This surface direction is referred to as
the Orthonormal Surface Vector (OSV), or surface vector for short. 

A halfspace is a single boundary location and OSV pair, which defines the location
and orientation of a hyperplane that divides the search space into two halves (thus
the name). The boundary is composed of these halfspaces, which carves out a
hyper-dimensional polyhedra, where each halfspace represents a face of that
polyhedra. 

## Terminology 
| Term / Acronym / Synonyms                                     | Description                                                                                                                                                       |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Function Under Test, FUT**                        | The system that is being tested / explored / executed. |
| **Target Performance, In-Mode**                     | The output behavior of FUT that is being explored. For example, Autonomous Vehicle safety for a simulation, where "unsafe" is defined as Time-to-Collision < $\lambda$, where lambda is the threshold for "unsafe" behavior. |
| **Performance Class, Sample Class, Classification** | A boolean value, `true` or `false`, where `true` means that the given point when executed by the FUT resulted in Target Performance.                              |
| **Classifier**                                      | The wrapper for the FUT which interprets the FUT's output and returns a WithinMode or OutOfMode sample, depending on its classification. |
| **Input, Point**                                    | Both Input and Point refer to the same thing, the parameters to be  passed into the FUT.                                                                          |
| **Sample (verb)**                                   | To execute the FUT with a given input to acquire it's classification.                                                                                             |
| **Sample (noun)**                                   | An input that has been executed by the FUT and classified.                                                                                                        |
| **Envelope, Geometry, Hyper-Geometry**              | A discrete region of the search space that results in target performance. Two envelopes are said to be the same if they intersect each other.                     |

# Crate Overview
This crate has the goal of providing all the tools necessary for exploring boundaries
in high dimensions. This includes identifying the initial in-mode and out-of-mode
boundary pairs, finding the initial boundary halfspace, and of course exploring the
boundary itself. 

This section discusses in detail some of the code you will find in the `examples/`
directory. 

## Step 1) Initial Boundary Pair
A boundary pair is any two points in the search space that successfully identifies
the existance of a boundary, where one point falls within the target performance and
the other falls outside of the target performance. This process will be referred to
as Global Search, since the full (global) search domain will be explored.

The `sembas::search::global_search` module provides tools for this, specifically the
`MonteCarloSearch` Search Factory. This is a simple solution, where the entire search
space is randomly sampled. (In the future, more sophisticated global search solutions
will be provided, such as gradient descent.) 

Below is an example of how to find an initial boundary pair. This function takes in
the classifier, the domain to be explored, and the maximum samples cut-off threshold
for early termination. The function samples the space until an In-Mode and
Out-of-Mode sample has been acquired.

```rust
fn find_initial_boundary_pair<const N: usize>(
    classifier: &mut Box<dyn Classifier<N>>,
    domain::Domain<N>,
    max_samples: i32,
) -> Result<BoundaryPair<N>> {
    let mut search = MonteCarloSearch::new(domain, 1);

    // Shorthand for creating a sample and classifying it:
    let mut take_sample = move || {
        let p = search.sample();
        let cls = classifier
            .classify(&p)
            .expect("Invalid sample. Bad global search domain?");
        Sample::from_class(p, cls)
    };

    let mut t0 = None;
    let mut x0 = None;
    let mut i = 0;

    while (t0.is_none() || x0.is_none()) && i < max_samples {
        let sample = take_sample();
        match sample {
            Sample::WithinMode(t) => {
                if t0.is_none() {
                    t0 = Some(t)
                }
            }
            Sample::OutOfMode(x) => {
                if x0.is_none() {
                    x0 = Some(x)
                }
            }
        }

        i += 1;
    }

    if let (Some(t), Some(x)) = (t0, x0) {
        Ok(BoundaryPair::new(t, x))
    } else {
        Err(SamplingError::MaxSamplesExceeded)
    }
}
```

With this initial boundary pair, we know that a boundary exists. However, the
distance between these two points is arbitrarily large, and therefore the error is
likely to be very high. As a result, the surface must then be identified by reducing
this distance. 

## Step 2) Surfacing
This process takes an initial boundary pair and returns a halfspace that is within
the desired maximum error (i.e. distance from the boundary). The
`sembas::search::surfacing` module provides the `binary_surface_search` function for
this very purpose. This algorithm finds the mid-point between the samples in the
boundary pair and classifies it. This midpoint then acts as one of the samples in the
boundary pair, and the process repeats until the distance between the samples falls
under the $d$ distance, or until the maximum number of samples has been reached. The
point that falls within the envelope is the boundary's location vector and the
normalized displacement between the pairs is the surface's direction.

## Step 3) Setting up the Adherer
Within the `sembas::adherers` module, two implementations are provided:
`ConstantAdherer` and `BinarySearchAdherer`. The difference between these two is
unimportant to this introduction, but the ConstantAdherer is a good first approach. 

In order for an adherer to be used by an Explorer, its corresponding AdhererFactory
must be constructed. The factory pattern decouples the initialization of the Adherer
from its use, allowing for any Adherer to be used by any Explorer.

There are two parameters unique to the ConstantAdherer: `delta_angle: f64` and
`max_rotations: Option<f64>`. The `delta_angle` ($\Delta \theta$) is the angle to
rotate the displacement vector by to find the next boundary point. Experimentally, an
angle between $5^\circ$ and $20^\circ$ has been found to work well for most
circumstances. Note that the smaller the angle, the more samples it will take to find
the boundary. The max rotation parameter defines the maximum rotation can occur
before the Adherer fails, throwing a `structs::error::BoundaryLostError`. The default
max angle is $180^\circ$, but an angle smaller than this (between $90^\circ$ and
$180^\circ$) can work as well.

```rust
let delta_angle = 15.0f64.to_radians();
let max_rotation = 120.0f64.to_radians();
let adherer_f = Box::new(
    ConstantAdhererFactory::new(delta_angle, Some(max_rotation))
);
```

## Step 4) Creating a Classifier
Although the classifier's implementation heavily depends on your specific use case
(finding classification boundaries of an image classifier, find the boundary of a
failure mode, etc.) we can produce synthetic parameter spaces using geoemtry.

```rust
struct Sphere<const N: usize> {
    pub radius: f64,
    pub center: SVector<f64, N>,
    pub domain: Domain<N>,
}

impl<const N: usize> Classifier<N> for Sphere<N> {
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool> {
        if !self.domain.contains(p) {
            return Err(SamplingError::OutOfBounds);
        }

        Ok((p - self.center).norm() <= self.radius)
    }
}
```

This `Sphere` classifier will return `true` if the provided point falls within the
sphere, otherwise `false`.

Final note on classifiers: A `RemoteClassifier` is also provided with the `api`
feature under `sembas::api`, which allows you to setup a connection to another
process that will act as the FUT. The RemoteClassifier will send a normalized point
of a given number of dimensions across the TCP connection and wait for a response.
The remote FUT will then process the point and transmit a boolean value for the
RemoteClassifier to return as its classification result. (This is a work in progress,
there are some details that need to be ironed out, but it is in a working state.)

## Step 5) Creating the Explorer
Next, we must create the Explorer that will handle the exploration process. The only
implementation provided at this time is the `sembas::explorers:MeshExplorer`. This
solution samples each cardinal direction along the surface of the boundary, but
prunes branches that sample too close (within a `margin` of error) to existing
boundary points. 

The MeshExplorer takes a jump distance `d: f64` to describe how far apart to sample the
boundary points, a `root: Halfspace<N>` initial halfspace, a `margin: f64` to
describe the minimum distance between samples before pruning, and a 
`adherer_f: Box<dyn AdhererFactory>`. The margin cannot ever be greater than `d`, but
should be a little smaller than `d`.

Below is a complete 
```rust
use nalgebra::vector!;
use sembas::prelude::*;

// ...

let main() {
    let center = vector![0.5, 0.5, 0.5]
    let radius = 0.25;
    let domain = Domain::normalized();
    let mut classifier: Box<dyn Classifier<3>> = Sphere {
        center,
        radius, 
        domain,
    }

    let b_pair = find_initial_boundary_pair(&mut classifier, 32).unwrap();
    
    let d = 0.05
    let root = binary_surface_search(
        d, &b_pair, 256, &mut classifier).expect("Couldn't find in time?");

    let delta_angle = 15.0f64.to_radians();
    let max_rotation = 120.0f64.to_radians();
    let adherer_f = Box::new(
        ConstantAdhererFactory::new(delta_angle, Some(max_rotation))
    );

    let expl = MeshExplorer::new(d, root, d * 0.9, adherer_f);
}
```

## Step 6) Exploring
Finally, we can begin exploring our function under test. An Explorer has a
`.step(...)` method which takes in the classifier and returns a reference to the
Result containing the sample that it took. This result may be a BoundaryLost error or
a OutOfBounds error. Under most circumstances, these are not fatal, since the
boundary exploration process can proceed without any alteration. Losing the
Boundary is very costly, particularly when `max_rotation` for the
ConstantAdherer is large, but is rare.
```rust
// ...
fn main() {
    // ...

    let max_boundary_points = 500;
    let max_samples = 1000;
    let mut ble_count = 0;
    let mut oob_count = 0;

    for _ in 0..max_samples {
        // Take samples and handle results
        if let Err(e) = expl.step(&mut classifier) {
            match e {
                SamplingError::BoundaryLost => ble_count += 1,
                SamplingError::OutOfBounds => oob_count += 1,
                _ => (),
            }
        }

        if expl.boundary_count() >= max_boundary_points {
            break;
        }
    }
}
```

Once this process completes, the explorer will have at most $500$ boundary
halfspaces. You can retrieve these halfspaces by the Explorer's `.boundary()` getter.
