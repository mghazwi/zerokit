//! Circuit-specific preparation for dynamic Groth16 with an existing proving key.
//!

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{One, Zero};
use ark_groth16::ProvingKey;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_relations::r1cs::SynthesisError;

/// precomputed params, these are cached to be used later.
/// The main two vecs we need to compute are `lambda` and `psi`
/// `domain` and `weight_fft` are just cached for convenience.
#[derive(Debug, Clone)]
pub struct DynamicSetup<E: Pairing> {
    domain: Radix2EvaluationDomain<E::ScalarField>,
    lambda: Vec<E::G1Affine>,
    psi: Vec<E::G1Affine>,
    weight_fft: Vec<E::ScalarField>,
}

impl<E: Pairing> DynamicSetup<E> {
    /// Derive public setup data from a Circom/snarkjs proving key.
    pub fn from_circom_key(pk: &ProvingKey<E>) -> Result<Self, SynthesisError> {
        let n = pk.h_query.len();
        if !n.is_power_of_two() {
            return Err(SynthesisError::MalformedVerifyingKey);
        }
        let domain =
            Radix2EvaluationDomain::new(n).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let doubled = n
            .checked_mul(2)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let double_domain = Radix2EvaluationDomain::<E::ScalarField>::new(doubled)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let eta = double_domain.element(1);

        // convert from Jordi to Jens
        // H_i = [L_{2i+1}^{(2N)}(tau)/delta] Convert to
        // [Z(tau) tau^k/delta] = -2 eta^k FFT(H)_k
        let mut lambda: Vec<E::G1> = pk.h_query.iter().map(|p| p.into_group()).collect();
        domain.fft_in_place(&mut lambda);
        let mut factor = -E::ScalarField::from(2u64);
        for point in &mut lambda {
            *point *= factor;
            factor *= eta;
        }
        // A Groth16 quotient has degree at most N-2. Normalize the unused
        // top power to zero
        lambda[n - 1] = E::G1::zero();
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
        // sigma = |W| = (1 - N) / (2N); reuse the doubled domain's inverse size.
        let sigma = (E::ScalarField::one() - n_field) * double_domain.size_inv;
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
