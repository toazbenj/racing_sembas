/// In this example, we will look at how we can use SEMBAS to identify complementary
/// neural networks for constructing an ensemble from a
/// Bayesian Neural Network (BNN).
/// * The BNN is defined in `fut.py` using pytorch. We train the BNN using a limited
///   number of training examples, and sample from the BNN's distribution several
///   concrete neural networks (NN).
/// * SEMBAS is used to explores the region of validity for these NNs to identify
///   where they perform well.
/// * We then use these boundaries to determine which NNs should be used to compose
///   our ensemble by preferring unique regions of validity over redundant ones (i.e.
///   minimizing their overlap).
/// * Finally, we use these boundaries from the perspective of FUT to place greater
///   confidence on inputs into the ensemble that are within their region of validity
///   over those that are outside of the region of validity, to further increase
///   model performance.
fn main() {}
