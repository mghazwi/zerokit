use ark_bn254::{Bn254, Fr};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_ff::Field;
use ark_groth16::{prepare_verifying_key, r1cs_to_qap::LibsnarkReduction, Groth16};
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError, SynthesisMode,
};
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng, UniformRand,
};
use rln::{
    dynamic::{
        CircomReduction, DynamicQap, Groth16DynamicV1, Groth16DynamicV2,
        PartialAssignment as DynamicAssignment,
    },
    partial_proof::{Groth16Partial, PartialAssignment},
};

// Simple dummy multiplication circuit.
#[derive(Clone, Copy)]
struct MulCircuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
}

impl<ConstraintF> ConstraintSynthesizer<ConstraintF> for MulCircuit<ConstraintF>
where
    ConstraintF: Field,
{
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;
            a *= &b;
            Ok(a)
        })?;

        cs.enforce_constraint(
            ark_relations::lc!() + a,
            ark_relations::lc!() + b,
            ark_relations::lc!() + c,
        )?;
        Ok(())
    }
}

fn build_cs<F: Field>(circuit: MulCircuit<F>) -> ConstraintSystemRef<F> {
    let cs = ConstraintSystem::new_ref();
    cs.set_mode(SynthesisMode::Prove {
        construct_matrices: true,
    });
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();
    assert!(cs.is_satisfied().unwrap());
    cs
}

#[test]
fn test_partial_proof() {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    let (pk, vk) =
        Groth16::<Bn254>::setup(MulCircuit::<Fr> { a: None, b: None }, &mut rng).unwrap();
    let pvk = prepare_verifying_key::<Bn254>(&vk);

    let a = Fr::rand(&mut rng);
    let b = Fr::rand(&mut rng);
    let mut c = a;
    c *= b;

    // Partial witness: fix public input (c) and one witness (a), leave b unknown.
    let mut partial_vals = vec![None; 3];
    partial_vals[0] = Some(c);
    partial_vals[1] = Some(a);
    let partial = Groth16Partial::<Bn254, LibsnarkReduction>::prove_partial(
        &pk,
        &PartialAssignment::new(partial_vals),
    )
    .unwrap();
    // Finish the proof
    let mut rng_partial = rng.clone();
    let finished_partial_proof = Groth16Partial::<Bn254, LibsnarkReduction>::finish_proof(
        &pk,
        MulCircuit::<Fr> {
            a: Some(a),
            b: Some(b),
        },
        &mut rng_partial,
        &partial,
    )
    .unwrap();

    // This is the proof with full witness
    let mut rng_full = rng.clone(); // Clone rng so we get the same randomness
    let proof_full = Groth16::<Bn254, LibsnarkReduction>::prove(
        &pk,
        MulCircuit::<Fr> {
            a: Some(a),
            b: Some(b),
        },
        &mut rng_full,
    )
    .unwrap();

    assert_eq!(finished_partial_proof, proof_full);
    assert!(
        Groth16::<Bn254>::verify_with_processed_vk(&pvk, &[c], &finished_partial_proof).unwrap()
    );
    assert!(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &[c], &proof_full).unwrap());
}

#[test]
fn test_partial_proof_with_matrices() {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    let (pk, vk) =
        ark_groth16::Groth16::<Bn254>::setup(MulCircuit::<Fr> { a: None, b: None }, &mut rng)
            .unwrap();
    let pvk = prepare_verifying_key::<Bn254>(&vk);

    let a = Fr::rand(&mut rng);
    let b = Fr::rand(&mut rng);
    let mut c = a;
    c *= b;

    let cs = build_cs(MulCircuit::<Fr> {
        a: Some(a),
        b: Some(b),
    });
    let matrices = cs.to_matrices().unwrap();
    let num_inputs = cs.num_instance_variables();
    let num_constraints = cs.num_constraints();
    let prover = cs.borrow().unwrap();
    let full_assignment_qap = [
        prover.instance_assignment.as_slice(),
        prover.witness_assignment.as_slice(),
    ]
    .concat();

    // Partial witness: fix public input (c) and one witness (a), leave b unknown.
    let mut partial_vals = vec![None; 3];
    partial_vals[0] = Some(c);
    partial_vals[1] = Some(a);
    // Partial proof
    let partial = Groth16Partial::<Bn254, LibsnarkReduction>::prove_partial(
        &pk,
        &PartialAssignment::new(partial_vals),
    )
    .unwrap();
    // finish the proof
    let r = Fr::rand(&mut rng);
    let s = Fr::rand(&mut rng);
    let finished_partial_proof =
        Groth16Partial::<Bn254, LibsnarkReduction>::finish_proof_with_matrices(
            &pk,
            &partial,
            r,
            s,
            &matrices,
            num_inputs,
            num_constraints,
            &full_assignment_qap,
        )
        .unwrap();

    // this is the proof with full witness
    let proof_full = Groth16::<Bn254>::create_proof_with_reduction_and_matrices(
        &pk,
        r,
        s,
        &matrices,
        num_inputs,
        num_constraints,
        &full_assignment_qap,
    )
    .unwrap();

    assert_eq!(finished_partial_proof, proof_full);
    assert!(
        Groth16::<Bn254>::verify_with_processed_vk(&pvk, &[c], &finished_partial_proof).unwrap()
    );
    assert!(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &[c], &proof_full).unwrap());
}

fn check_dynamic_partial_proofs<QAP: DynamicQap>() {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(42);
    let (pk, vk) =
        Groth16::<Bn254, QAP>::setup(MulCircuit::<Fr> { a: None, b: None }, &mut rng).unwrap();
    let a = Fr::from(3u64);
    let cs = build_cs(MulCircuit {
        a: Some(a),
        b: Some(Fr::from(7u64)),
    });
    let matrices = cs.to_matrices().unwrap();
    let setup = Groth16DynamicV2::<Bn254>::setup::<QAP>(&pk).unwrap();

    // Fix a; both b and the public output can change.
    let assignment = DynamicAssignment::new(vec![None, Some(a), None]);
    let v1 = Groth16DynamicV1::<Bn254, QAP>::preprocess(&pk, &assignment).unwrap();
    let v2 = Groth16DynamicV2::<Bn254>::preprocess(&pk, &matrices, &setup, &assignment).unwrap();

    for b in [Fr::from(0u64), Fr::from(7u64)] {
        let full_assignment = [Fr::from(1u64), a * b, a, b];
        let r = Fr::rand(&mut rng);
        let s = Fr::rand(&mut rng);
        let expected = Groth16::<Bn254, QAP>::create_proof_with_reduction_and_matrices(
            &pk,
            r,
            s,
            &matrices,
            cs.num_instance_variables(),
            cs.num_constraints(),
            &full_assignment,
        )
        .unwrap();
        let proof_v1 = Groth16DynamicV1::<Bn254, QAP>::finish_with_matrices(
            &pk,
            &v1,
            r,
            s,
            &matrices,
            cs.num_instance_variables(),
            cs.num_constraints(),
            &full_assignment,
        )
        .unwrap();
        let proof_v2 = Groth16DynamicV2::<Bn254>::finish_with_randomness(
            &pk,
            &matrices,
            &setup,
            &full_assignment,
            r,
            s,
            &v2,
        )
        .unwrap();

        assert_eq!(proof_v1, expected);
        assert_eq!(proof_v2, expected);
        assert!(Groth16::<Bn254>::verify_with_processed_vk(
            &prepare_verifying_key(&vk),
            &[a * b],
            &proof_v2,
        )
        .unwrap());

        // Circuit-based finishing should use the same randomness as a full proof.
        let circuit = MulCircuit {
            a: Some(a),
            b: Some(b),
        };
        let mut full_rng = rng.clone();
        let mut v1_rng = rng.clone();
        let mut v2_rng = rng.clone();
        let expected = Groth16::<Bn254, QAP>::prove(&pk, circuit, &mut full_rng).unwrap();
        assert_eq!(
            Groth16DynamicV1::<Bn254, QAP>::finish(&pk, circuit, &mut v1_rng, &v1).unwrap(),
            expected
        );
        assert_eq!(
            Groth16DynamicV2::<Bn254>::finish(&pk, circuit, &setup, &mut v2_rng, &v2).unwrap(),
            expected
        );
    }
}

#[test]
fn test_dynamic_partial_proofs_circom() {
    check_dynamic_partial_proofs::<CircomReduction>();
}

#[test]
fn test_dynamic_partial_proofs_libsnark() {
    check_dynamic_partial_proofs::<LibsnarkReduction>();
}
