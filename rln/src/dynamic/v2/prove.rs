//! Witness-dependent preprocessing and finishing for dynamic Groth16.
//!
//! Note: in formulas, `**` denotes length-N cyclic convolution
//! `*` denotes ordinary multiplication, matrix-vector multiplication, or
//! scalar multiplication of a group point.

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{One, Zero};
use ark_groth16::{Proof, ProvingKey};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisError,
    SynthesisMode,
};
use ark_std::{rand::RngCore, UniformRand};

use super::setup::DynamicSetup;
use crate::dynamic::{
    types::{PartialAssignment, PartialProofV1, PartialProofV2},
    v1,
};

impl<E: Pairing> PartialProofV2<E> {
    /// Compute the simple linear partial proof, static nonlinear point,
    /// and unified mixed/linear C points.
    /// `assignment` excludes constant one, just like the simple partial API.
    pub(in crate::dynamic) fn from_assignment(
        pk: &ProvingKey<E>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        setup: &DynamicSetup<E>,
        assignment: &PartialAssignment<E::ScalarField>,
    ) -> Result<Self, SynthesisError> {
        // 1. Check that the assignment, matrices, key and prepared domain have
        // compatible dimensions.
        validate_inputs(pk, matrices, setup, assignment)?;
        let p = matrices.num_instance_variables;
        let m = p + pk.l_query.len();
        let domain = setup.domain();

        // 2. Reuse the v1 linear partial proof: A_pre, B_pre,
        // rho_pre and C_linear. We'll store the nonlinear contribution separately.
        let linear = v1::create_partial_proof_from_assignment(pk, assignment)?;
        // Sorted dynamic witness indices (sup') in the full assignment.
        // Index zero is constant one and is not dynamic.
        let dynamic_indices: Vec<_> = assignment
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.is_none().then_some(i + 1))
            .collect();

        // 3. Compute vectors a = A * z0 and b = B * z0, where z0 contains
        // the known values, zero for unknown entries, and constant one at index 0.
        // Support masks mark rows touched by unknown variables, independently
        // of their values. U needs B's affected rows; V needs A's affected rows.
        let n = domain.size();
        let mut a = vec![E::ScalarField::zero(); n];
        let mut b = a.clone();
        let mut a_support = vec![false; n]; // A*sup'
        let mut b_support = a_support.clone(); // B*sup'

        for (matrix, values, support) in [
            (&matrices.a, &mut a, &mut a_support),
            (&matrices.b, &mut b, &mut b_support),
        ] {
            for (i, row) in matrix.iter().enumerate() {
                for &(coefficient, j) in row {
                    if !coefficient.is_zero() {
                        let value = if j == 0 {
                            Some(E::ScalarField::one()) // this is because matrix includes the 1 and assignment doesn't
                        } else {
                            assignment.values[j - 1]
                        };
                        match value {
                            Some(value) => values[i] += coefficient * value,
                            None => support[i] = true,
                        }
                    }
                }
            }
        }

        // public-input (dummy coefficients) including the constant 1 in A rows; B remains zero.
        // This relates to the malleability attack
        let start = matrices.num_constraints;
        a[start] = E::ScalarField::one(); // Constant one
        for j in 1..p {
            match assignment.values[j - 1] {
                Some(value) => a[start + j] = value,
                None => a_support[start + j] = true,
            }
        }

        // 4. Compute Wbar ** a and Wbar ** b. These are the scalar-valued
        // convolution terms in U/V, also used for the static nonlinear point.
        let wbar_a = convolve_field_with_wbar(domain, &a, setup.weight_fft());
        let wbar_b = convolve_field_with_wbar(domain, &b, setup.weight_fft());

        // 5. compute and cache Q0 = A0*B0, where
        // Q0 = sum_i [a_i * (Wbar ** b)_i + b_i * (Wbar ** a)_i
        //      - (Wbar ** (a * b))_i - sigma * a_i * b_i] * Lambda_i
        let static_nonlinear = compute_static_nonlinear(setup, &a, &b, &wbar_a, &wbar_b);

        // 6. Compute the row-indexed projection points:
        // U = a * Psi + (Wbar ** a) * Lambda + W ** (a * Lambda),
        // V = b * Psi + (Wbar ** b) * Lambda + W ** (b * Lambda).
        let u = projections(setup, &a, &wbar_a, &b_support);
        let v = projections(setup, &b, &wbar_b, &a_support);

        // slots[j] maps a full witness index j to its position in the Gamma array.
        let mut slots = vec![None; m];
        for (slot, &j) in dynamic_indices.iter().enumerate() {
            slots[j] = Some(slot);
        }

        // 7. Combine row-indexed U/V into one point per dynamic variable j:
        // Gamma_j = K_j + sum_i(A_ij * V_i + B_ij * U_i).
        let mut gamma: Vec<E::G1> = dynamic_indices
            .iter()
            .map(|&j| {
                if j < p {
                    E::G1::zero()
                } else {
                    pk.l_query[j - p].into_group()
                }
            })
            .collect();
        for (matrix, projection) in [(&matrices.a, &v), (&matrices.b, &u)] {
            for (i, row) in matrix.iter().enumerate() {
                for &(coefficient, j) in row {
                    if let Some(slot) = slots[j] {
                        gamma[slot] += projection[i] * coefficient;
                    }
                }
            }
        }

        // Include the appended public-input A rows in the same Gamma sum.
        for j in 1..p {
            if let Some(slot) = slots[j] {
                gamma[slot] += v[matrices.num_constraints + j];
            }
        }

        // 8. return all components of the partial proof
        Ok(Self {
            assignment: assignment.clone(),
            linear,
            static_nonlinear,
            gamma: E::G1::normalize_batch(&gamma),
            dynamic_indices,
            a_rows: a_support
                .iter()
                .enumerate()
                .filter_map(|(i, &yes)| yes.then_some(i))
                .collect(),
            b_rows: b_support
                .iter()
                .enumerate()
                .filter_map(|(i, &yes)| yes.then_some(i))
                .collect(),
        })
    }

    /// Synthesize a completed circuit and finish
    pub(in crate::dynamic) fn finish_proof<C: ConstraintSynthesizer<E::ScalarField>, R: RngCore>(
        &self,
        pk: &ProvingKey<E>,
        circuit: C,
        setup: &DynamicSetup<E>,
        rng: &mut R,
    ) -> Result<Proof<E>, SynthesisError> {
        let cs = ConstraintSystem::new_ref();
        cs.set_optimization_goal(OptimizationGoal::Constraints);
        cs.set_mode(SynthesisMode::Prove {
            construct_matrices: true,
        });
        circuit.generate_constraints(cs.clone())?;
        cs.finalize();
        let matrices = cs.to_matrices().ok_or(SynthesisError::MissingCS)?;
        let full_assignment = {
            let prover = cs.borrow().ok_or(SynthesisError::MissingCS)?;
            [
                prover.instance_assignment.as_slice(),
                prover.witness_assignment.as_slice(),
            ]
            .concat()
        };
        self.finish_with_rng(pk, &matrices, setup, &full_assignment, rng)
    }

    /// Finish with blinding scalars from RNG.
    /// The full assignment includes constant one.
    pub(in crate::dynamic) fn finish_with_rng<R: RngCore>(
        &self,
        pk: &ProvingKey<E>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        setup: &DynamicSetup<E>,
        full_assignment: &[E::ScalarField],
        rng: &mut R,
    ) -> Result<Proof<E>, SynthesisError> {
        let r = E::ScalarField::rand(rng);
        let s = E::ScalarField::rand(rng);
        self.finish_with_randomness(pk, matrices, setup, full_assignment, r, s)
    }

    /// finish with given r and s
    #[allow(clippy::too_many_arguments)]
    pub(in crate::dynamic) fn finish_with_randomness(
        &self,
        pk: &ProvingKey<E>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        setup: &DynamicSetup<E>,
        full_assignment: &[E::ScalarField],
        r: E::ScalarField,
        s: E::ScalarField,
    ) -> Result<Proof<E>, SynthesisError> {
        // 1. Validate full assignment
        // we can skip this if this is already checked
        validate_inputs(pk, matrices, setup, &self.assignment)?;
        if full_assignment.len() != self.assignment.values.len() + 1
            || full_assignment.first() != Some(&E::ScalarField::one())
        {
            return Err(SynthesisError::AssignmentMissing);
        }
        for (known, actual) in self.assignment.values.iter().zip(&full_assignment[1..]) {
            if known.is_some_and(|value| value != *actual) {
                return Err(SynthesisError::Unsatisfiable);
            }
        }

        // 2. Extract z' using dynamic indices; static entries, including one, are zero
        let dynamic_values: Vec<_> = self
            .dynamic_indices
            .iter()
            .map(|&j| full_assignment[j])
            .collect();
        let mut z_prime = vec![E::ScalarField::zero(); full_assignment.len()];
        for (&j, &value) in self.dynamic_indices.iter().zip(&dynamic_values) {
            z_prime[j] = value;
        }

        // 3. Evaluate a'=A z', b'=B z',and append public values to A rows.
        let domain = setup.domain();
        let mut a_prime = vec![E::ScalarField::zero(); domain.size()];
        let mut b_prime = a_prime.clone();
        for (matrix, evaluations) in [(&matrices.a, &mut a_prime), (&matrices.b, &mut b_prime)] {
            for (i, row) in matrix.iter().enumerate() {
                for &(coefficient, j) in row {
                    evaluations[i] += coefficient * z_prime[j];
                }
            }
        }
        for j in 0..matrices.num_instance_variables {
            a_prime[matrices.num_constraints + j] = z_prime[j];
        }

        // 4. Finish A and B using dynamic MSMs and fresh blinding scalars.
        let a_points: Vec<_> = self
            .dynamic_indices
            .iter()
            .map(|&j| pk.a_query[j])
            .collect();
        let b_points: Vec<_> = self
            .dynamic_indices
            .iter()
            .map(|&j| pk.b_g2_query[j])
            .collect();
        let proof_a = self.linear.partial_pi_a
            + E::G1::msm_unchecked(&a_points, &dynamic_values)
            + pk.delta_g1 * r;
        let proof_b = self.linear.partial_pi_b
            + E::G2::msm_unchecked(&b_points, &dynamic_values)
            + pk.vk.delta_g2 * s;

        // 5. Prepare weights for A'*B':
        // wbar_a = Wbar ** a'
        // wbar_b = Wbar ** b'
        let wbar_a = convolve_field_with_wbar(domain, &a_prime, setup.weight_fft());
        let wbar_b = convolve_field_with_wbar(domain, &b_prime, setup.weight_fft());

        // 6. Concatenate all C MSMs into one big MSM:
        // c_update = Gamma*z' + B1*(r*z') + Lambda*e + Psi*d

        // sum_j z'_j * Gamma_j
        let mut c_points = self.gamma.clone();
        let mut c_scalars = dynamic_values.clone();
        // sum_j (r * z'_j)
        for (&j, &value) in self.dynamic_indices.iter().zip(&dynamic_values) {
            c_points.push(pk.b_g1_query[j]);
            c_scalars.push(r * value);
        }
        // sum_k e_k * Lambda_k, where:
        // e_k = a'_k * (Wbar ** b')_k + b'_k * (Wbar ** a')_k.
        // we use the union because: either a'_k or b'_k may contribute
        // outside both supports, a'_k = b'_k = 0, so e_k = 0
        for k in union(&self.a_rows, &self.b_rows) {
            c_points.push(setup.lambda()[k]);
            c_scalars.push(a_prime[k] * wbar_b[k] + b_prime[k] * wbar_a[k]);
        }
        // sum_k d_k * Psi_k, where d_k = a'_k * b'_k.
        // use the intersection: both factors may be nonzero
        // if one factor is zero, then d_k = 0
        for k in intersection(&self.a_rows, &self.b_rows) {
            c_points.push(setup.psi()[k]);
            c_scalars.push(a_prime[k] * b_prime[k]);
        }
        // one big MSM
        let c_update = E::G1::msm_unchecked(&c_points, &c_scalars);

        // 7. update C:
        // proof_c = proof_c0 + s * proof_a + r * rho_0 + c_update
        let proof_c = self.partial_c() + proof_a * s + self.linear.partial_rho * r + c_update;

        Ok(Proof {
            a: proof_a.into_affine(),
            b: proof_b.into_affine(),
            c: proof_c.into_affine(),
        })
    }

    /// Experimental v1 linear partial proof
    pub fn linear_partial(&self) -> &PartialProofV1<E> {
        &self.linear
    }
    /// Static quotient contribution Q0 = A0 * B0
    pub fn static_nonlinear(&self) -> &E::G1 {
        &self.static_nonlinear
    }
    /// Static linear C + Q0, before fresh proof randomization.
    pub fn partial_c(&self) -> E::G1 {
        self.linear.partial_pi_c + self.static_nonlinear
    }
    /// Unified mixed-term plus private linear-C bases, ordered by dynamic_indices.
    /// Gamma_j = K_j + sum_i(A_ij * V_i + B_ij * U_i).
    pub fn gamma(&self) -> &[E::G1Affine] {
        &self.gamma
    }
    /// Sorted dynamic witness indices in the full assignment (sup').
    /// Index 0 is constant one and is never included.
    pub fn dynamic_indices(&self) -> &[usize] {
        &self.dynamic_indices
    }
    /// Sorted, duplicate-free rows affected by dynamic A columns.
    pub fn a_rows(&self) -> &[usize] {
        &self.a_rows
    }
    /// Sorted, duplicate-free rows affected by dynamic B columns.
    pub fn b_rows(&self) -> &[usize] {
        &self.b_rows
    }
}

fn validate_inputs<E: Pairing>(
    pk: &ProvingKey<E>,
    matrices: &ConstraintMatrices<E::ScalarField>,
    setup: &DynamicSetup<E>,
    assignment: &PartialAssignment<E::ScalarField>,
) -> Result<(), SynthesisError> {
    let p = matrices.num_instance_variables;

    let m = p
        .checked_add(pk.l_query.len())
        .ok_or(SynthesisError::MalformedVerifyingKey)?;
    let rows = matrices
        .num_constraints
        .checked_add(p)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let domain = setup.domain();
    let expected_domain = Radix2EvaluationDomain::<E::ScalarField>::new(rows)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    if p == 0
        || domain.size() != expected_domain.size()
        || assignment.values.len() != m - 1
        || pk.a_query.len() != m
        || pk.b_g1_query.len() != m
        || pk.b_g2_query.len() != m
        || pk.l_query.len() != m - p
        || pk.vk.gamma_abc_g1.len() != p
        || matrices.a.len() != matrices.num_constraints
        || matrices.b.len() != matrices.num_constraints
        || matrices
            .a
            .iter()
            .chain(&matrices.b)
            .flatten()
            .any(|(_, j)| *j >= m)
    {
        return Err(SynthesisError::MalformedVerifyingKey);
    }
    Ok(())
}

/// Given `d = a * b`, `sigma = Sum(W)` ,the Lambda scalars `c` are:
/// `c_k = a_k (Wbar**b)_k + b_k (Wbar**a)_k - (Wbar**d)_k - sigma*d_k`
/// returns `A_0 * B_0 = MSM(c, lambda)`
fn compute_static_nonlinear<E: Pairing>(
    setup: &DynamicSetup<E>,
    a: &[E::ScalarField],
    b: &[E::ScalarField],
    wbar_a: &[E::ScalarField],
    wbar_b: &[E::ScalarField],
) -> E::G1 {
    let product: Vec<_> = a.iter().zip(b).map(|(a, b)| *a * b).collect();
    let wbar_product = convolve_field_with_wbar(setup.domain(), &product, setup.weight_fft());
    // sigma = sum(W) == weight_fft , because:
    // weight_fft = \sum_{i=0}^{N-1}\omega^{ik}W_i
    // at k=0 :
    // weight_fft[0] = \sum_{i=0}^{N-1} W_i
    // we can also compute it as = (1 - N) / (2 * N)
    let sigma = setup.weight_fft()[0];
    let lambda_scalars: Vec<_> = (0..setup.domain().size())
        .map(|k| a[k] * wbar_b[k] + b[k] * wbar_a[k] - wbar_product[k] - sigma * product[k])
        .collect();
    E::G1::msm_unchecked(setup.lambda(), &lambda_scalars)
}

/// Length-N cyclic convolution `Wbar ** values`, where `Wbar[k] = W[(-k) mod N]`.
/// The input `weight_fft` is cached FFT(W), NOT FFT(Wbar). Reversal in the original
/// domain reverses Fourier indices too: `FFT(Wbar)[k] = FFT(W)[(-k) mod N]`.
fn convolve_field_with_wbar<F: ark_ff::FftField>(
    domain: &Radix2EvaluationDomain<F>,
    values: &[F],
    weight_fft: &[F],
) -> Vec<F> {
    let mut result = domain.fft(values);
    let n = domain.size();
    debug_assert_eq!(values.len(), n);
    debug_assert_eq!(weight_fft.len(), n);
    for (k, value) in result.iter_mut().enumerate() {
        let reversed_k = if k == 0 { 0 } else { n - k };
        *value *= weight_fft[reversed_k];
    }
    domain.ifft_in_place(&mut result);
    result
}

/// Compute U (values=a) or V (values=b):
/// `result[k] = values[k] * Psi[k] + reverse_convolution[k] * Lambda[k]
///             + (W ** (values * Lambda))[k]`.
/// `reverse_convolution = Wbar ** values` comes from convolve_field_with_wbar.
fn projections<E: Pairing>(
    setup: &DynamicSetup<E>,
    values: &[E::ScalarField],
    reverse_convolution: &[E::ScalarField],
    support: &[bool],
) -> Vec<E::G1> {
    let domain = setup.domain();
    // Group convolution: W ** (values * Lambda).
    let mut result: Vec<_> = setup
        .lambda()
        .iter()
        .zip(values)
        .map(|(point, value)| *point * value)
        .collect();
    domain.fft_in_place(&mut result);
    for (point, weight) in result.iter_mut().zip(setup.weight_fft()) {
        *point *= *weight;
    }
    domain.ifft_in_place(&mut result);
    // Add the diagonal and field-convolution contributions on required rows.
    for k in 0..result.len() {
        if support[k] {
            result[k] += setup.psi()[k] * values[k] + setup.lambda()[k] * reverse_convolution[k];
        } else {
            result[k] = E::G1::zero();
        }
    }
    result
}

/// just a generic intersection function for two sets (Vec<usize>)
/// we expect the sets to be ordered and contain no duplicates
fn intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    result
}

/// just a generic union function for two sets (Vec<usize>)
/// we expect the sets to be ordered and contain no duplicates
fn union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);

    result
}
