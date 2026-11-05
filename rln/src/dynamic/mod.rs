//! Groth16 partial proving:
//! v1 caches linear terms.
//! v2 also caches nonlinear terms.

pub mod types;
pub mod v1;
pub mod v2;

use ark_ec::pairing::Pairing;
use ark_groth16::{
    r1cs_to_qap::{LibsnarkReduction, R1CSToQAP},
    Proof, ProvingKey,
};
use ark_relations::r1cs::{ConstraintMatrices, ConstraintSynthesizer, SynthesisError};
use ark_std::{marker::PhantomData, rand::RngCore};
pub use types::{PartialAssignment, PartialProofV1, PartialProofV2};
pub use v2::{prove, setup, DynamicSetup};

/// Entry point (API) for v1 linear partial proving.
pub struct Groth16DynamicV1<E: Pairing, QAP: R1CSToQAP = LibsnarkReduction> {
    _marker: PhantomData<(E, QAP)>,
}

impl<E: Pairing, QAP: R1CSToQAP> Groth16DynamicV1<E, QAP> {
    /// Prepare v1's linear partial proof.
    pub fn preprocess(
        pk: &ProvingKey<E>,
        assignment: &PartialAssignment<E::ScalarField>,
    ) -> Result<PartialProofV1<E>, SynthesisError> {
        v1::create_partial_proof_from_assignment(pk, assignment)
    }

    /// Finish v1 by synthesizing a completed circuit.
    pub fn finish<C: ConstraintSynthesizer<E::ScalarField>, R: RngCore>(
        pk: &ProvingKey<E>,
        circuit: C,
        rng: &mut R,
        partial: &PartialProofV1<E>,
    ) -> Result<Proof<E>, SynthesisError> {
        v1::finish_proof::<E, QAP, C, R>(pk, circuit, rng, partial)
    }

    /// Finish v1 with existing matrices and explicit blinding scalars.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_with_matrices(
        pk: &ProvingKey<E>,
        partial: &PartialProofV1<E>,
        r: E::ScalarField,
        s: E::ScalarField,
        matrices: &ConstraintMatrices<E::ScalarField>,
        num_inputs: usize,
        num_constraints: usize,
        full_assignment: &[E::ScalarField],
    ) -> Result<Proof<E>, SynthesisError> {
        v1::finish_proof_with_reduction_and_matrices::<E, QAP>(
            pk,
            partial,
            r,
            s,
            matrices,
            num_inputs,
            num_constraints,
            full_assignment,
        )
    }
}

/// Entry point for full-domain dynamic proving using a Circom-compatible key.
pub struct Groth16DynamicV2<E: Pairing> {
    _marker: PhantomData<E>,
}

impl<E: Pairing> Groth16DynamicV2<E> {
    /// Derive the reusable v2 setup from a Circom-compatible key.
    pub fn setup(pk: &ProvingKey<E>) -> Result<DynamicSetup<E>, SynthesisError> {
        DynamicSetup::from_circom_key(pk)
    }

    /// Prepare v2 for one static partial assignment.
    pub fn preprocess(
        pk: &ProvingKey<E>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        setup: &DynamicSetup<E>,
        assignment: &PartialAssignment<E::ScalarField>,
    ) -> Result<PartialProofV2<E>, SynthesisError> {
        PartialProofV2::from_circom_assignment(pk, matrices, setup, assignment)
    }

    /// Finish v2 by synthesizing a completed circuit.
    pub fn finish<C: ConstraintSynthesizer<E::ScalarField>, R: RngCore>(
        pk: &ProvingKey<E>,
        circuit: C,
        setup: &DynamicSetup<E>,
        rng: &mut R,
        partial: &PartialProofV2<E>,
    ) -> Result<Proof<E>, SynthesisError> {
        partial.finish_proof(pk, circuit, setup, rng)
    }

    /// Finish v2 with existing matrices and a full assignment (including one).
    pub fn finish_with_matrices<R: RngCore>(
        pk: &ProvingKey<E>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        setup: &DynamicSetup<E>,
        full_assignment: &[E::ScalarField],
        rng: &mut R,
        partial: &PartialProofV2<E>,
    ) -> Result<Proof<E>, SynthesisError> {
        partial.finish_with_rng(pk, matrices, setup, full_assignment, rng)
    }

    /// Finish v2 with existing matrices and given blinding scalars.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_with_randomness(
        pk: &ProvingKey<E>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        setup: &DynamicSetup<E>,
        full_assignment: &[E::ScalarField],
        r: E::ScalarField,
        s: E::ScalarField,
        partial: &PartialProofV2<E>,
    ) -> Result<Proof<E>, SynthesisError> {
        partial.finish_with_randomness(pk, matrices, setup, full_assignment, r, s)
    }
}
