//! Circuit-specific preparation for dynamic Groth16 with an existing proving key.
//!

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{Field, One, Zero};
use ark_groth16::{
    r1cs_to_qap::{LibsnarkReduction, R1CSToQAP},
    ProvingKey,
};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_relations::r1cs::SynthesisError;

use crate::circuit::qap::CircomReduction;

/// A reduction whose H-query convention and public-row layout are compatible
/// with this v2 implementation.
pub trait DynamicQap: R1CSToQAP {
    /// get and validate the full domain size from the H-query length.
    fn h_query_domain_size(query_len: usize) -> Result<usize, SynthesisError>;
    /// Return N Jens points, with the unused last monomial point set to zero.
    fn h_query_jens<E: Pairing>(pk: &ProvingKey<E>) -> Result<Vec<E::G1>, SynthesisError>;
    /// Return N Jordi points.
    fn h_query_jordi<E: Pairing>(pk: &ProvingKey<E>) -> Result<Vec<E::G1>, SynthesisError>;
}

impl DynamicQap for CircomReduction {
    fn h_query_domain_size(query_len: usize) -> Result<usize, SynthesisError> {
        check_domain_size(query_len)
    }

    fn h_query_jens<E: Pairing>(pk: &ProvingKey<E>) -> Result<Vec<E::G1>, SynthesisError> {
        Self::h_query_domain_size(pk.h_query.len())?;
        let points: Vec<_> = pk.h_query.iter().map(|p| p.into_group()).collect();
        let mut jens = jordi_to_jens::<E>(&points)?;
        // The degree N-1 point is unused by any Groth16 quotient.
        *jens.last_mut().unwrap() = E::G1::zero();
        Ok(jens)
    }

    fn h_query_jordi<E: Pairing>(pk: &ProvingKey<E>) -> Result<Vec<E::G1>, SynthesisError> {
        Self::h_query_domain_size(pk.h_query.len())?;
        Ok(pk.h_query.iter().map(|p| p.into_group()).collect())
    }
}
impl DynamicQap for LibsnarkReduction {
    fn h_query_domain_size(query_len: usize) -> Result<usize, SynthesisError> {
        check_domain_size(
            query_len
                .checked_add(1)
                .ok_or(SynthesisError::MalformedVerifyingKey)?,
        )
    }

    fn h_query_jens<E: Pairing>(pk: &ProvingKey<E>) -> Result<Vec<E::G1>, SynthesisError> {
        Self::h_query_domain_size(pk.h_query.len())?;
        let mut jens: Vec<_> = pk.h_query.iter().map(|p| p.into_group()).collect();
        // The h query omits the unused top monomial point.
        jens.push(E::G1::zero());
        Ok(jens)
    }

    fn h_query_jordi<E: Pairing>(pk: &ProvingKey<E>) -> Result<Vec<E::G1>, SynthesisError> {
        jens_to_jordi::<E>(&Self::h_query_jens(pk)?)
    }
}

fn check_domain_size(n: usize) -> Result<usize, SynthesisError> {
    if n < 2 || !n.is_power_of_two() {
        return Err(SynthesisError::MalformedVerifyingKey);
    }
    Ok(n)
}

/// Convert N Jordi points to N Jens monomial points
pub fn jordi_to_jens<E: Pairing>(points: &[E::G1]) -> Result<Vec<E::G1>, SynthesisError> {
    let n = check_domain_size(points.len())?;
    let domain = Radix2EvaluationDomain::<E::ScalarField>::new(n)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let doubled = n
        .checked_mul(2)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let double_domain = Radix2EvaluationDomain::<E::ScalarField>::new(doubled)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let eta = double_domain.element(1);
    let mut jens = points.to_vec();
    domain.fft_in_place(&mut jens);
    let mut factor = -E::ScalarField::from(2u64);
    for point in &mut jens {
        *point *= factor;
        factor *= eta;
    }
    Ok(jens)
}

/// Convert N Jens monomial points to N Jordi points
pub fn jens_to_jordi<E: Pairing>(points: &[E::G1]) -> Result<Vec<E::G1>, SynthesisError> {
    let n = check_domain_size(points.len())?;
    let domain = Radix2EvaluationDomain::<E::ScalarField>::new(n)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let doubled = n
        .checked_mul(2)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let double_domain = Radix2EvaluationDomain::<E::ScalarField>::new(doubled)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let mut factor = -E::ScalarField::from(2u64)
        .inverse()
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let mut jordi = points.to_vec();
    for point in &mut jordi {
        *point *= factor;
        factor *= double_domain.group_gen_inv;
    }
    domain.ifft_in_place(&mut jordi);
    Ok(jordi)
}

/// precomputed params, these are cached to be used later.
/// The main two vecs we need to compute are `lambda` and `psi`
/// `domain` and `weight_fft` are just cached for convenience.
pub struct DynamicSetup<E: Pairing> {
    domain: Radix2EvaluationDomain<E::ScalarField>,
    lambda: Vec<E::G1Affine>,
    psi: Vec<E::G1Affine>,
    weight_fft: Vec<E::ScalarField>,
}

impl<E: Pairing> Clone for DynamicSetup<E> {
    fn clone(&self) -> Self {
        Self {
            domain: self.domain,
            lambda: self.lambda.clone(),
            psi: self.psi.clone(),
            weight_fft: self.weight_fft.clone(),
        }
    }
}

impl<E: Pairing> std::fmt::Debug for DynamicSetup<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicSetup")
            .field("domain", &self.domain)
            .field("lambda", &self.lambda)
            .field("psi", &self.psi)
            .field("weight_fft", &self.weight_fft)
            .finish()
    }
}

impl<E: Pairing> DynamicSetup<E> {
    /// Prepare a key using the given pk with H-query convention of `QAP`.
    pub fn new<QAP: DynamicQap>(pk: &ProvingKey<E>) -> Result<Self, SynthesisError> {
        let n = check_domain_size(QAP::h_query_domain_size(pk.h_query.len())?)?;
        let domain =
            Radix2EvaluationDomain::new(n).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let mut lambda = QAP::h_query_jens(pk)?;
        if lambda.len() != n || !lambda[n - 1].is_zero() {
            return Err(SynthesisError::MalformedVerifyingKey);
        }
        // get the Lagrange-based points.
        domain.ifft_in_place(&mut lambda);

        // compute the weights first before psi
        // W_0 = 0; W_k = 1 / (N (omega^(-k) - 1)). Batch inversion
        // also leaves W_0 zero.
        let mut weights = Vec::with_capacity(n);
        let mut omega_inv_power = E::ScalarField::one();
        let omega_inv = domain.group_gen_inv;
        let n_field = E::ScalarField::from(n as u64);
        for _ in 0..n {
            weights.push(n_field * (omega_inv_power - E::ScalarField::one()));
            omega_inv_power *= omega_inv;
        }
        ark_ff::batch_inversion(&mut weights);
        // sigma = |W| = (1 - N) / (2N).
        let two_n_inv = (n_field + n_field)
            .inverse()
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let sigma = (E::ScalarField::one() - n_field) * two_n_inv;
        // TODO: see if we can optimize this using properties of the weights
        let weight_fft = domain.fft(&weights);

        // compute Psi = -(W ** Lambda) - sigma * Lambda. (** is cyclic convolution)
        let mut psi = domain.fft(&lambda);
        for (point, weight) in psi.iter_mut().zip(&weight_fft) {
            *point *= *weight;
        }
        domain.ifft_in_place(&mut psi);

        for (point, base) in psi.iter_mut().zip(&lambda) {
            *point = -*point - *base * sigma;
        }

        Ok(Self {
            domain,
            lambda: E::G1::normalize_batch(&lambda),
            psi: E::G1::normalize_batch(&psi),
            weight_fft,
        })
    }

    pub fn domain(&self) -> &Radix2EvaluationDomain<E::ScalarField> {
        &self.domain
    }

    pub fn lambda(&self) -> &[E::G1Affine] {
        &self.lambda
    }

    pub fn psi(&self) -> &[E::G1Affine] {
        &self.psi
    }

    /// FFT(W), reusable for cyclic convolution.
    /// FFT(Wbar)[k] = FFT(W)[(-k) mod N]
    pub fn weight_fft(&self) -> &[E::ScalarField] {
        &self.weight_fft
    }
}
