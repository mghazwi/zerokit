//! Types shared by the dynamic proving APIs.

use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// A partial assignment (witness).
///
/// `None` means "unknown" or changing part of the witness, `Some` means fixed and can be
/// precomputed.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialAssignment<F: PrimeField> {
    /// Assignment entries, ordered as (public inputs excluding 1) || (witness/aux)
    pub values: Vec<Option<F>>,
}

impl<F> PartialAssignment<F>
where
    F: PrimeField,
{
    /// Creates a new partial assignment.
    pub fn new(values: Vec<Option<F>>) -> Self {
        Self { values }
    }
}

/// Precomputed partial proof elements for a given `PartialAssignment`
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PartialProofV1<E: Pairing> {
    /// For each entry in `PartialAssignment::values`.
    pub mask: Vec<bool>,
    /// partial_pi_a = \[alpha\]_1 + sum z_j*\[A_j\]_1
    pub partial_pi_a: E::G1,
    /// partial_rho = \[beta\]_1  + sum z_j*\[B_j\]_1
    pub partial_rho: E::G1,
    /// partial_pi_b = \[beta\]_2  + sum z_j*\[B_j\]_2
    pub partial_pi_b: E::G2,
    /// partial_pi_c =             sum z_j*\[K_j\]_1
    pub partial_pi_c: E::G1,
}

/// Caches contributions from the known, fixed part of a witness.
/// Rebuild this object if a known value or the mask changes.
#[derive(Debug, Clone)]
pub struct PartialProofV2<E: Pairing> {
    // V2 reuses the experimental v1 linear partial proof.
    pub(super) linear: PartialProofV1<E>,
    pub(super) assignment: PartialAssignment<E::ScalarField>,
    pub(super) static_nonlinear: E::G1,
    pub(super) gamma: Vec<E::G1Affine>,
    pub(super) dynamic_indices: Vec<usize>,
    pub(super) a_rows: Vec<usize>,
    pub(super) b_rows: Vec<usize>,
}
