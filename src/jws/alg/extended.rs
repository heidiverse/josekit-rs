//! Parameterized RustCrypto signature algorithms.
//!
//! This module is opt-in through the `extended-algorithms` Cargo feature. The
//! algorithms in this module are intended for combinations outside JOSE's
//! algorithm registry. Existing JOSE constants remain the recommended API for
//! registered JOSE algorithms.

use std::fmt::Debug;
use std::marker::PhantomData;

use anyhow::anyhow;
use digest::{const_oid::AssociatedOid, Digest, FixedOutputReset};

#[cfg(feature = "kapun-provider")]
use kapun_crypto_provider::{
    KeyEncoding as KapunKeyEncoding, Metadata as KapunMetadata, Signer as KapunSigner,
    Signing as KapunSigning, Verifier as KapunVerifier, Verifying as KapunVerifying,
};

use crate::jwe::alg::ecdh_es::{PrivateKey as EcPrivateKey, PublicKey as EcPublicKey};
use crate::jwk::alg::ec::{EcCurve, EcKeyPair};
use crate::jwk::alg::rsa::RsaKeyPair;
use crate::jwk::{Jwk, KeyPair};
use crate::jws::{JwsAlgorithm, JwsSigner, JwsVerifier};
use crate::util;
use crate::JoseError;

/// Digest types accepted by the extended RustCrypto algorithms.
///
/// The bound ensures the digest can be used by RSA-PSS and has an associated
/// object identifier for ASN.1 DigestInfo encoding.
pub trait ExtendedHash:
    Digest + FixedOutputReset + AssociatedOid + Clone + Debug + Send + Sync + 'static
{
}

impl<T> ExtendedHash for T where
    T: Digest + FixedOutputReset + AssociatedOid + Clone + Debug + Send + Sync + 'static
{
}

/// RSA signature padding supported by [`RsaJwsAlgorithm`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RsaPadding {
    /// RSASSA-PKCS1-v1_5, with the digest's ASN.1 OID encoded in DigestInfo.
    Pkcs1v15,
    /// RSASSA-PSS with MGF1 using the same digest and an explicit salt length.
    Pss { salt_len: usize },
}

/// An RSA JWS algorithm parameterized by a RustCrypto digest implementation.
///
/// For unknown JOSE combinations, the default JWS `alg` name is derived from
/// the padding mode and digest OID. Call [`with_jws_name`](Self::with_jws_name)
/// to use a different private algorithm name. Digest types must provide an
/// associated ASN.1 OID; that OID is used by PKCS#1 v1.5 and for JOSE profile
/// matching.
#[derive(Debug, Clone)]
pub struct RsaJwsAlgorithm<D> {
    name: String,
    padding: RsaPadding,
    jose_name: Option<&'static str>,
    _digest: PhantomData<D>,
}

impl<D> RsaJwsAlgorithm<D>
where
    D: ExtendedHash,
{
    /// Create an RSA PKCS#1 v1.5 algorithm using `D`.
    pub fn pkcs1v15() -> Self {
        Self::new(RsaPadding::Pkcs1v15)
    }

    /// Create an RSA-PSS algorithm using `D` for both the message digest and
    /// MGF1, with the supplied salt length in bytes.
    pub fn pss(salt_len: usize) -> Self {
        Self::new(RsaPadding::Pss { salt_len })
    }

    fn new(padding: RsaPadding) -> Self {
        let jose_name = rsa_jose_name::<D>(padding);
        let name = jose_name
            .map(str::to_owned)
            .unwrap_or_else(|| match padding {
                RsaPadding::Pkcs1v15 => format!("RSA-PKCS1v1_5-{}", D::OID),
                RsaPadding::Pss { salt_len } => {
                    format!("RSA-PSS-{}-SALT-{}", D::OID, salt_len)
                }
            });
        Self {
            name,
            padding,
            jose_name,
            _digest: PhantomData,
        }
    }

    /// Override the `alg` value used in JWS serialization.
    ///
    /// A registered JOSE name is accepted only when the selected digest and
    /// padding exactly match that name. This prevents custom combinations from
    /// being mislabeled as standard JOSE algorithms.
    pub fn with_jws_name(mut self, name: impl Into<String>) -> Result<Self, JoseError> {
        let name = name.into();
        if name.is_empty() {
            return Err(JoseError::UnsupportedSignatureAlgorithm(anyhow!(
                "A JWS algorithm name cannot be empty"
            )));
        }
        if self.jose_name.is_some_and(|expected| expected != name) {
            return Err(unsupported_jwk_export(
                "the selected RSA parameters have a different JOSE name",
            ));
        }
        if self.jose_name.is_none() && is_reserved_jose_name(&name) {
            return Err(unsupported_jwk_export(
                "the selected RSA parameters do not match the requested JOSE name",
            ));
        }
        self.name = name;
        Ok(self)
    }

    /// Generate an RSA key pair using the same entry point as the JOSE
    /// algorithm presets.
    pub fn generate_key_pair(&self, bits: u32) -> Result<ExtendedRsaKeyPair<D>, JoseError> {
        if bits < 2048 {
            return Err(invalid_rsa_size());
        }
        let mut inner = RsaKeyPair::generate(bits)?;
        if let Some(name) = self.jose_name {
            inner.set_algorithm(Some(name));
        }
        Ok(ExtendedRsaKeyPair {
            inner,
            algorithm_name: self.name.clone(),
            padding: self.padding,
            jose: self.jose_name.is_some(),
            _digest: PhantomData,
        })
    }

    /// Load an RSA private key from DER.
    pub fn key_pair_from_der(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedRsaKeyPair<D>, JoseError> {
        let mut inner = RsaKeyPair::from_der(input)?;
        if inner.key_len() * 8 < 2048 {
            return Err(invalid_rsa_size());
        }
        if let Some(name) = self.jose_name {
            inner.set_algorithm(Some(name));
        }
        Ok(ExtendedRsaKeyPair {
            inner,
            algorithm_name: self.name.clone(),
            padding: self.padding,
            jose: self.jose_name.is_some(),
            _digest: PhantomData,
        })
    }

    /// Load an RSA private key from PEM.
    pub fn key_pair_from_pem(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedRsaKeyPair<D>, JoseError> {
        let mut inner = RsaKeyPair::from_pem(input)?;
        if inner.key_len() * 8 < 2048 {
            return Err(invalid_rsa_size());
        }
        if let Some(name) = self.jose_name {
            inner.set_algorithm(Some(name));
        }
        Ok(ExtendedRsaKeyPair {
            inner,
            algorithm_name: self.name.clone(),
            padding: self.padding,
            jose: self.jose_name.is_some(),
            _digest: PhantomData,
        })
    }

    /// Create a signer from a key pair produced or loaded by this algorithm.
    pub fn signer_from_key_pair(
        &self,
        key_pair: &ExtendedRsaKeyPair<D>,
    ) -> Result<ExtendedRsaSigner<D>, JoseError> {
        if key_pair.algorithm_name != self.name || key_pair.padding != self.padding {
            return Err(JoseError::InvalidKeyFormat(anyhow!(
                "The RSA key pair was created for a different signature algorithm"
            )));
        }
        Ok(ExtendedRsaSigner {
            algorithm: self.clone(),
            private_key: key_pair.inner.clone().into_private_key(),
            key_id: key_pair.inner.key_id().map(str::to_owned),
        })
    }

    /// Create a signer from a DER-encoded private key.
    pub fn signer_from_der(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedRsaSigner<D>, JoseError> {
        let key_pair = self.key_pair_from_der(input)?;
        self.signer_from_key_pair(&key_pair)
    }

    /// Create a signer from a PEM-encoded private key.
    pub fn signer_from_pem(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedRsaSigner<D>, JoseError> {
        let key_pair = self.key_pair_from_pem(input)?;
        self.signer_from_key_pair(&key_pair)
    }

    /// Create a verifier from a DER SubjectPublicKeyInfo or PKCS#1 public key.
    pub fn verifier_from_der(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedRsaVerifier<D>, JoseError> {
        use rsa::pkcs8::DecodePublicKey;
        use rsa::traits::PublicKeyParts;

        let input = input.as_ref();
        let spki = if RsaKeyPair::detect_pkcs8(input, true).is_some() {
            input.to_vec()
        } else {
            RsaKeyPair::to_pkcs8(input, true)
        };
        let public_key = rsa::RsaPublicKey::from_public_key_der(&spki)
            .map_err(|err| JoseError::InvalidKeyFormat(anyhow!(err)))?;
        if public_key.size() * 8 < 2048 {
            return Err(invalid_rsa_size());
        }
        Ok(ExtendedRsaVerifier {
            algorithm: self.clone(),
            public_key,
            key_id: None,
        })
    }

    /// Create a verifier from the public half of a key pair generated or
    /// loaded by this algorithm.
    pub fn verifier_from_key_pair(
        &self,
        key_pair: &ExtendedRsaKeyPair<D>,
    ) -> Result<ExtendedRsaVerifier<D>, JoseError> {
        if key_pair.algorithm_name != self.name || key_pair.padding != self.padding {
            return Err(JoseError::InvalidKeyFormat(anyhow!(
                "The RSA key pair was created for a different signature algorithm"
            )));
        }
        let mut verifier = self.verifier_from_der(key_pair.to_der_public_key())?;
        verifier.key_id = key_pair.inner.key_id().map(str::to_owned);
        Ok(verifier)
    }

    /// Create a verifier from a PEM SubjectPublicKeyInfo or PKCS#1 public key.
    pub fn verifier_from_pem(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedRsaVerifier<D>, JoseError> {
        let (label, data) =
            util::parse_pem(input.as_ref()).map_err(|err| JoseError::InvalidKeyFormat(err))?;
        if label != "PUBLIC KEY" && label != "RSA PUBLIC KEY" {
            return Err(JoseError::InvalidKeyFormat(anyhow!(
                "Expected a PUBLIC KEY or RSA PUBLIC KEY PEM block"
            )));
        }
        self.verifier_from_der(data)
    }
}

impl<D> JwsAlgorithm for RsaJwsAlgorithm<D>
where
    D: ExtendedHash,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn box_clone(&self) -> Box<dyn JwsAlgorithm> {
        Box::new(self.clone())
    }
}

/// An RSA key pair tied to a parameterized algorithm.
///
/// It intentionally does not implement the legacy infallible `KeyPair`
/// export trait. Use the `try_to_jwk_*` methods, which reject combinations
/// without a JOSE algorithm definition.
#[derive(Debug, Clone)]
pub struct ExtendedRsaKeyPair<D> {
    inner: RsaKeyPair,
    algorithm_name: String,
    padding: RsaPadding,
    jose: bool,
    _digest: PhantomData<D>,
}

impl<D> ExtendedRsaKeyPair<D> {
    pub fn to_der_private_key(&self) -> Vec<u8> {
        self.inner.to_der_private_key()
    }

    pub fn to_der_public_key(&self) -> Vec<u8> {
        self.inner.to_der_public_key()
    }

    pub fn to_pem_private_key(&self) -> Vec<u8> {
        self.inner.to_pem_private_key()
    }

    pub fn to_pem_public_key(&self) -> Vec<u8> {
        self.inner.to_pem_public_key()
    }

    pub fn key_id(&self) -> Option<&str> {
        self.inner.key_id()
    }

    pub fn set_key_id(&mut self, key_id: impl Into<String>) {
        self.inner.set_key_id(Some(key_id));
    }

    pub fn try_to_jwk_private_key(&self) -> Result<Jwk, JoseError> {
        self.ensure_jose()?;
        Ok(self.inner.to_jwk_private_key())
    }

    pub fn try_to_jwk_public_key(&self) -> Result<Jwk, JoseError> {
        self.ensure_jose()?;
        Ok(self.inner.to_jwk_public_key())
    }

    pub fn try_to_jwk_key_pair(&self) -> Result<Jwk, JoseError> {
        self.ensure_jose()?;
        Ok(self.inner.to_jwk_key_pair())
    }

    fn ensure_jose(&self) -> Result<(), JoseError> {
        if self.jose {
            Ok(())
        } else {
            Err(unsupported_jwk_export(&format!(
                "{} is not a JOSE-defined algorithm",
                self.algorithm_name
            )))
        }
    }
}

/// RSA JWS signer using a parameterized digest and padding mode.
#[derive(Debug, Clone)]
pub struct ExtendedRsaSigner<D> {
    algorithm: RsaJwsAlgorithm<D>,
    private_key: rsa::RsaPrivateKey,
    key_id: Option<String>,
}

impl<D> ExtendedRsaSigner<D> {
    pub fn set_key_id(&mut self, key_id: impl Into<String>) {
        self.key_id = Some(key_id.into());
    }

    pub fn remove_key_id(&mut self) {
        self.key_id = None;
    }
}

impl<D> JwsSigner for ExtendedRsaSigner<D>
where
    D: ExtendedHash,
{
    fn algorithm(&self) -> &dyn JwsAlgorithm {
        &self.algorithm
    }

    fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    fn signature_len(&self) -> usize {
        use rsa::traits::PublicKeyParts;
        self.private_key.size()
    }

    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, JoseError> {
        self.sign_prehashed(&D::digest(message))
    }

    fn sign_prehashed(&self, digest: &[u8]) -> Result<Vec<u8>, JoseError> {
        use rsa::traits::SignatureScheme;

        let mut rng = rand::rng();
        let signature = match self.algorithm.padding {
            RsaPadding::Pkcs1v15 => {
                rsa::Pkcs1v15Sign::new::<D>().sign(Some(&mut rng), &self.private_key, digest)
            }
            RsaPadding::Pss { salt_len } => rsa::Pss::<D>::new_with_salt(salt_len).sign(
                Some(&mut rng),
                &self.private_key,
                digest,
            ),
        }
        .map_err(|err| JoseError::InvalidSignature(anyhow!(err)))?;
        Ok(signature)
    }

    fn box_clone(&self) -> Box<dyn JwsSigner> {
        Box::new(self.clone())
    }
}

/// RSA JWS verifier using a parameterized digest and padding mode.
#[derive(Debug, Clone)]
pub struct ExtendedRsaVerifier<D> {
    algorithm: RsaJwsAlgorithm<D>,
    public_key: rsa::RsaPublicKey,
    key_id: Option<String>,
}

impl<D> ExtendedRsaVerifier<D> {
    pub fn set_key_id(&mut self, key_id: impl Into<String>) {
        self.key_id = Some(key_id.into());
    }

    pub fn remove_key_id(&mut self) {
        self.key_id = None;
    }

    pub fn try_to_jwk_public_key(&self) -> Result<Jwk, JoseError> {
        let mut jwk = crate::jwk::PublicKey::to_jwk_public_key(&self.public_key);
        let Some(name) = self.algorithm.jose_name else {
            return Err(unsupported_jwk_export(&format!(
                "{} is not a JOSE-defined algorithm",
                self.algorithm.name
            )));
        };
        jwk.set_algorithm(name);
        if let Some(key_id) = &self.key_id {
            jwk.set_key_id(key_id);
        }
        Ok(jwk)
    }
}

impl<D> JwsVerifier for ExtendedRsaVerifier<D>
where
    D: ExtendedHash,
{
    fn algorithm(&self) -> &dyn JwsAlgorithm {
        &self.algorithm
    }

    fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), JoseError> {
        self.verify_prehashed(&D::digest(message), signature)
    }

    fn verify_prehashed(&self, digest: &[u8], signature: &[u8]) -> Result<(), JoseError> {
        use rsa::traits::SignatureScheme;

        let result = match self.algorithm.padding {
            RsaPadding::Pkcs1v15 => {
                rsa::Pkcs1v15Sign::new::<D>().verify(&self.public_key, digest, signature)
            }
            RsaPadding::Pss { salt_len } => {
                rsa::Pss::<D>::new_with_salt(salt_len).verify(&self.public_key, digest, signature)
            }
        };
        result.map_err(|err| JoseError::InvalidSignature(anyhow!(err)))
    }

    fn box_clone(&self) -> Box<dyn JwsVerifier> {
        Box::new(self.clone())
    }
}

/// ECDSA JWS algorithm parameterized by an existing supported curve and a
/// RustCrypto digest implementation.
#[derive(Debug, Clone)]
pub struct EcdsaJwsAlgorithm<D> {
    curve: EcCurve,
    name: String,
    jose_name: Option<&'static str>,
    _digest: PhantomData<D>,
}

impl<D> EcdsaJwsAlgorithm<D>
where
    D: ExtendedHash,
{
    pub fn new(curve: EcCurve) -> Self {
        let jose_name = ecdsa_jose_name::<D>(curve);
        let name = jose_name
            .map(str::to_owned)
            .unwrap_or_else(|| format!("ECDSA-{}-{}", curve.name(), D::OID));
        Self {
            curve,
            name,
            jose_name,
            _digest: PhantomData,
        }
    }

    /// Override the JWS `alg` value, while preventing an unregistered
    /// parameter combination from claiming a registered JOSE name.
    pub fn with_jws_name(mut self, name: impl Into<String>) -> Result<Self, JoseError> {
        let name = name.into();
        if name.is_empty() {
            return Err(JoseError::UnsupportedSignatureAlgorithm(anyhow!(
                "A JWS algorithm name cannot be empty"
            )));
        }
        if self.jose_name.is_some_and(|expected| expected != name) {
            return Err(unsupported_jwk_export(
                "the selected ECDSA parameters have a different JOSE name",
            ));
        }
        if self.jose_name.is_none() && is_reserved_jose_name(&name) {
            return Err(unsupported_jwk_export(
                "the selected ECDSA parameters do not match the requested JOSE name",
            ));
        }
        self.name = name;
        Ok(self)
    }

    pub fn curve(&self) -> EcCurve {
        self.curve
    }

    pub fn generate_key_pair(&self) -> Result<ExtendedEcKeyPair<D>, JoseError> {
        let mut inner = EcKeyPair::generate(self.curve)?;
        if let Some(name) = self.jose_name {
            inner.set_algorithm(Some(name));
        }
        Ok(ExtendedEcKeyPair {
            inner,
            algorithm_name: self.name.clone(),
            curve: self.curve,
            jose: self.jose_name.is_some(),
            _digest: PhantomData,
        })
    }

    pub fn key_pair_from_der(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedEcKeyPair<D>, JoseError> {
        let mut inner = EcKeyPair::from_der(input, Some(self.curve))?;
        if let Some(name) = self.jose_name {
            inner.set_algorithm(Some(name));
        }
        Ok(ExtendedEcKeyPair {
            inner,
            algorithm_name: self.name.clone(),
            curve: self.curve,
            jose: self.jose_name.is_some(),
            _digest: PhantomData,
        })
    }

    pub fn key_pair_from_pem(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedEcKeyPair<D>, JoseError> {
        let mut inner = EcKeyPair::from_pem(input, Some(self.curve))?;
        if let Some(name) = self.jose_name {
            inner.set_algorithm(Some(name));
        }
        Ok(ExtendedEcKeyPair {
            inner,
            algorithm_name: self.name.clone(),
            curve: self.curve,
            jose: self.jose_name.is_some(),
            _digest: PhantomData,
        })
    }

    pub fn signer_from_key_pair(
        &self,
        key_pair: &ExtendedEcKeyPair<D>,
    ) -> Result<ExtendedEcdsaSigner<D>, JoseError> {
        if key_pair.algorithm_name != self.name || key_pair.curve != self.curve {
            return Err(JoseError::InvalidKeyFormat(anyhow!(
                "The EC key pair was created for a different signature algorithm"
            )));
        }
        Ok(ExtendedEcdsaSigner {
            algorithm: self.clone(),
            private_key: key_pair.inner.clone().into_private_key(),
            key_id: key_pair.inner.key_id().map(str::to_owned),
        })
    }

    pub fn signer_from_der(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedEcdsaSigner<D>, JoseError> {
        let key_pair = self.key_pair_from_der(input)?;
        self.signer_from_key_pair(&key_pair)
    }

    pub fn signer_from_pem(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedEcdsaSigner<D>, JoseError> {
        let key_pair = self.key_pair_from_pem(input)?;
        self.signer_from_key_pair(&key_pair)
    }

    pub fn verifier_from_der(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedEcdsaVerifier<D>, JoseError> {
        let input = input.as_ref();
        let curve = EcKeyPair::detect_pkcs8(input, true).ok_or_else(|| {
            JoseError::InvalidKeyFormat(anyhow!(
                "An ECDSA public key must be wrapped in SubjectPublicKeyInfo"
            ))
        })?;
        if curve != self.curve {
            return Err(JoseError::InvalidKeyFormat(anyhow!(
                "The EC curve is mismatched: expected {}, got {}",
                self.curve,
                curve
            )));
        }
        let public_key =
            crate::jwe::alg::ecdh_es::PublicKey::from_pkcs8_der_with_ec_curve(self.curve, input)
                .map_err(|err| JoseError::InvalidKeyFormat(err))?;
        Ok(ExtendedEcdsaVerifier {
            algorithm: self.clone(),
            public_key,
            key_id: None,
        })
    }

    pub fn verifier_from_key_pair(
        &self,
        key_pair: &ExtendedEcKeyPair<D>,
    ) -> Result<ExtendedEcdsaVerifier<D>, JoseError> {
        if key_pair.algorithm_name != self.name || key_pair.curve != self.curve {
            return Err(JoseError::InvalidKeyFormat(anyhow!(
                "The EC key pair was created for a different signature algorithm"
            )));
        }
        let mut verifier = self.verifier_from_der(key_pair.to_der_public_key())?;
        verifier.key_id = key_pair.inner.key_id().map(str::to_owned);
        Ok(verifier)
    }

    pub fn verifier_from_pem(
        &self,
        input: impl AsRef<[u8]>,
    ) -> Result<ExtendedEcdsaVerifier<D>, JoseError> {
        let (label, data) =
            util::parse_pem(input.as_ref()).map_err(|err| JoseError::InvalidKeyFormat(err))?;
        if label != "PUBLIC KEY" {
            return Err(JoseError::InvalidKeyFormat(anyhow!(
                "Expected a PUBLIC KEY PEM block"
            )));
        }
        self.verifier_from_der(data)
    }
}

impl<D> JwsAlgorithm for EcdsaJwsAlgorithm<D>
where
    D: ExtendedHash,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn box_clone(&self) -> Box<dyn JwsAlgorithm> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ExtendedEcKeyPair<D> {
    inner: EcKeyPair,
    algorithm_name: String,
    curve: EcCurve,
    jose: bool,
    _digest: PhantomData<D>,
}

impl<D> ExtendedEcKeyPair<D> {
    pub fn to_der_private_key(&self) -> Vec<u8> {
        self.inner.to_der_private_key()
    }

    pub fn to_der_public_key(&self) -> Vec<u8> {
        self.inner.to_der_public_key()
    }

    pub fn to_pem_private_key(&self) -> Vec<u8> {
        self.inner.to_pem_private_key()
    }

    pub fn to_pem_public_key(&self) -> Vec<u8> {
        self.inner.to_pem_public_key()
    }

    pub fn key_id(&self) -> Option<&str> {
        self.inner.key_id()
    }

    pub fn set_key_id(&mut self, key_id: impl Into<String>) {
        self.inner.set_key_id(Some(key_id));
    }

    pub fn try_to_jwk_private_key(&self) -> Result<Jwk, JoseError> {
        self.ensure_jose()?;
        Ok(self.inner.to_jwk_private_key())
    }

    pub fn try_to_jwk_public_key(&self) -> Result<Jwk, JoseError> {
        self.ensure_jose()?;
        Ok(self.inner.to_jwk_public_key())
    }

    pub fn try_to_jwk_key_pair(&self) -> Result<Jwk, JoseError> {
        self.ensure_jose()?;
        Ok(self.inner.to_jwk_key_pair())
    }

    fn ensure_jose(&self) -> Result<(), JoseError> {
        if self.jose {
            Ok(())
        } else {
            Err(unsupported_jwk_export(&format!(
                "{} is not a JOSE-defined algorithm",
                self.algorithm_name
            )))
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtendedEcdsaSigner<D> {
    algorithm: EcdsaJwsAlgorithm<D>,
    private_key: EcPrivateKey,
    key_id: Option<String>,
}

impl<D> ExtendedEcdsaSigner<D> {
    pub fn set_key_id(&mut self, key_id: impl Into<String>) {
        self.key_id = Some(key_id.into());
    }

    pub fn remove_key_id(&mut self) {
        self.key_id = None;
    }

    fn sign_digest(&self, digest: &[u8]) -> anyhow::Result<Vec<u8>> {
        use p256::ecdsa::signature::hazmat::PrehashSigner;

        Ok(match &self.private_key {
            EcPrivateKey::P256(key) => {
                let signing_key = p256::ecdsa::SigningKey::from(key.clone());
                PrehashSigner::<p256::ecdsa::Signature>::sign_prehash(&signing_key, digest)?
                    .to_bytes()
                    .to_vec()
            }
            EcPrivateKey::P384(key) => {
                let signing_key = p384::ecdsa::SigningKey::from(key.clone());
                PrehashSigner::<p384::ecdsa::Signature>::sign_prehash(&signing_key, digest)?
                    .to_bytes()
                    .to_vec()
            }
            EcPrivateKey::P521(key) => {
                let signing_key = p521::ecdsa::SigningKey::from_slice(
                    key.to_nonzero_scalar().to_bytes().as_slice(),
                )?;
                PrehashSigner::<p521::ecdsa::Signature>::sign_prehash(&signing_key, digest)?
                    .to_bytes()
                    .to_vec()
            }
            EcPrivateKey::K256(key) => {
                let signing_key = k256::ecdsa::SigningKey::from(key.clone());
                PrehashSigner::<k256::ecdsa::Signature>::sign_prehash(&signing_key, digest)?
                    .to_bytes()
                    .to_vec()
            }
            _ => return Err(anyhow!("The selected key is not an ECDSA key")),
        })
    }
}

impl<D> JwsSigner for ExtendedEcdsaSigner<D>
where
    D: ExtendedHash,
{
    fn algorithm(&self) -> &dyn JwsAlgorithm {
        &self.algorithm
    }

    fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    fn signature_len(&self) -> usize {
        self.algorithm.curve.coordinate_size() * 2
    }

    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, JoseError> {
        self.sign_prehashed(&D::digest(message))
    }

    fn sign_prehashed(&self, digest: &[u8]) -> Result<Vec<u8>, JoseError> {
        self.sign_digest(digest)
            .map_err(|err| JoseError::InvalidSignature(anyhow!(err)))
    }

    fn box_clone(&self) -> Box<dyn JwsSigner> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ExtendedEcdsaVerifier<D> {
    algorithm: EcdsaJwsAlgorithm<D>,
    public_key: EcPublicKey,
    key_id: Option<String>,
}

impl<D> ExtendedEcdsaVerifier<D> {
    pub fn set_key_id(&mut self, key_id: impl Into<String>) {
        self.key_id = Some(key_id.into());
    }

    pub fn remove_key_id(&mut self) {
        self.key_id = None;
    }

    pub fn try_to_jwk_public_key(&self) -> Result<Jwk, JoseError>
    where
        D: ExtendedHash,
    {
        let Some(name) = self.algorithm.jose_name else {
            return Err(unsupported_jwk_export(&format!(
                "{} is not a JOSE-defined algorithm",
                self.algorithm.name
            )));
        };
        let mut jwk = crate::jwk::PublicKey::to_jwk_public_key(&self.public_key);
        jwk.set_algorithm(name);
        if let Some(key_id) = &self.key_id {
            jwk.set_key_id(key_id);
        }
        Ok(jwk)
    }

    fn verify_digest(&self, digest: &[u8], signature: &[u8]) -> anyhow::Result<()> {
        use p256::ecdsa::signature::hazmat::PrehashVerifier;

        match &self.public_key {
            EcPublicKey::P256(key) => {
                let sig = p256::ecdsa::Signature::from_slice(signature)?;
                p256::ecdsa::VerifyingKey::from(key.clone()).verify_prehash(digest, &sig)?;
            }
            EcPublicKey::P384(key) => {
                let sig = p384::ecdsa::Signature::from_slice(signature)?;
                p384::ecdsa::VerifyingKey::from(key.clone()).verify_prehash(digest, &sig)?;
            }
            EcPublicKey::P521(key) => {
                let sig = p521::ecdsa::Signature::from_slice(signature)?;
                p521::ecdsa::VerifyingKey::from_affine(key.as_affine().to_owned())?
                    .verify_prehash(digest, &sig)?;
            }
            EcPublicKey::K256(key) => {
                let sig = k256::ecdsa::Signature::from_slice(signature)?;
                k256::ecdsa::VerifyingKey::from(key.clone()).verify_prehash(digest, &sig)?;
            }
            _ => return Err(anyhow!("The selected key is not an ECDSA key")),
        }
        Ok(())
    }
}

impl<D> JwsVerifier for ExtendedEcdsaVerifier<D>
where
    D: ExtendedHash,
{
    fn algorithm(&self) -> &dyn JwsAlgorithm {
        &self.algorithm
    }

    fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), JoseError> {
        self.verify_prehashed(&D::digest(message), signature)
    }

    fn verify_prehashed(&self, digest: &[u8], signature: &[u8]) -> Result<(), JoseError> {
        self.verify_digest(digest, signature)
            .map_err(|err| JoseError::InvalidSignature(anyhow!(err)))
    }

    fn box_clone(&self) -> Box<dyn JwsVerifier> {
        Box::new(self.clone())
    }
}

fn rsa_jose_name<D: AssociatedOid>(padding: RsaPadding) -> Option<&'static str> {
    let oid = D::OID.to_string();
    match (padding, oid.as_str()) {
        (RsaPadding::Pkcs1v15, "2.16.840.1.101.3.4.2.1") => Some("RS256"),
        (RsaPadding::Pkcs1v15, "2.16.840.1.101.3.4.2.2") => Some("RS384"),
        (RsaPadding::Pkcs1v15, "2.16.840.1.101.3.4.2.3") => Some("RS512"),
        (RsaPadding::Pss { salt_len: 32 }, "2.16.840.1.101.3.4.2.1") => Some("PS256"),
        (RsaPadding::Pss { salt_len: 48 }, "2.16.840.1.101.3.4.2.2") => Some("PS384"),
        (RsaPadding::Pss { salt_len: 64 }, "2.16.840.1.101.3.4.2.3") => Some("PS512"),
        _ => None,
    }
}

fn ecdsa_jose_name<D: AssociatedOid>(curve: EcCurve) -> Option<&'static str> {
    let oid = D::OID.to_string();
    match (curve, oid.as_str()) {
        (EcCurve::P256, "2.16.840.1.101.3.4.2.1") => Some("ES256"),
        (EcCurve::P384, "2.16.840.1.101.3.4.2.2") => Some("ES384"),
        (EcCurve::P521, "2.16.840.1.101.3.4.2.3") => Some("ES512"),
        (EcCurve::Secp256k1, "2.16.840.1.101.3.4.2.1") => Some("ES256K"),
        _ => None,
    }
}

fn is_reserved_jose_name(name: &str) -> bool {
    matches!(
        name,
        "none"
            | "HS256"
            | "HS384"
            | "HS512"
            | "RS1"
            | "RS256"
            | "RS384"
            | "RS512"
            | "PS256"
            | "PS384"
            | "PS512"
            | "ES256"
            | "ES384"
            | "ES512"
            | "ES256K"
            | "EdDSA"
            | "ML-DSA-44"
            | "ML-DSA-65"
            | "ML-DSA-87"
    )
}

fn unsupported_jwk_export(message: &str) -> JoseError {
    JoseError::UnsupportedSignatureAlgorithm(anyhow!(message.to_owned()))
}

fn invalid_rsa_size() -> JoseError {
    JoseError::InvalidKeyFormat(anyhow!("RSA key length must be 2048 bits or more."))
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunSigner for ExtendedRsaSigner<D> where D: ExtendedHash {}

#[cfg(feature = "kapun-provider")]
impl<D> KapunSigning for ExtendedRsaSigner<D>
where
    D: ExtendedHash,
{
    fn kapun_sign(
        &self,
        data: Vec<u8>,
    ) -> Result<Vec<u8>, crate::kapun_crypto_provider::SigningProblem> {
        self.sign(&data)
            .map_err(|_| crate::kapun_crypto_provider::SigningProblem::SigningFailed)
    }

    fn kapun_sign_hash(
        &self,
        hash: Vec<u8>,
    ) -> Result<Vec<u8>, crate::kapun_crypto_provider::SigningProblem> {
        self.sign_prehashed(&hash)
            .map_err(|_| crate::kapun_crypto_provider::SigningProblem::SigningFailed)
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunKeyEncoding for ExtendedRsaSigner<D>
where
    D: ExtendedHash,
{
    fn kapun_private_jwk(&self) -> Option<String> {
        let name = self.algorithm.jose_name?;
        let mut key_pair = RsaKeyPair::from_private_key(self.private_key.clone());
        key_pair.set_algorithm(Some(name));
        key_pair.set_key_id(self.key_id.clone());
        serde_json::to_string(&key_pair.to_jwk_private_key()).ok()
    }

    fn kapun_public_jwk(&self) -> Option<String> {
        let name = self.algorithm.jose_name?;
        let mut jwk = crate::jwk::PublicKey::to_jwk_public_key(&self.private_key.to_public_key());
        jwk.set_algorithm(name);
        if let Some(key_id) = &self.key_id {
            jwk.set_key_id(key_id);
        }
        serde_json::to_string(&jwk).ok()
    }

    fn kapun_private_pkcs8_der(&self) -> Option<Vec<u8>> {
        use rsa::pkcs8::EncodePrivateKey;

        Some(self.private_key.to_pkcs8_der().ok()?.as_bytes().to_vec())
    }

    fn kapun_private_pkcs8_pem(&self) -> Option<String> {
        use rsa::pkcs8::{EncodePrivateKey, LineEnding};

        Some(
            self.private_key
                .to_pkcs8_pem(LineEnding::CRLF)
                .ok()?
                .to_string(),
        )
    }

    fn kapun_public_spki_der(&self) -> Option<Vec<u8>> {
        use rsa::pkcs8::EncodePublicKey;

        Some(
            self.private_key
                .to_public_key()
                .to_public_key_der()
                .ok()?
                .to_vec(),
        )
    }

    fn kapun_public_spki_pem(&self) -> Option<String> {
        use rsa::pkcs8::{EncodePublicKey, LineEnding};

        Some(
            self.private_key
                .to_public_key()
                .to_public_key_pem(LineEnding::CRLF)
                .ok()?,
        )
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunMetadata for ExtendedRsaSigner<D>
where
    D: ExtendedHash,
{
    fn kapun_jose_alg(&self) -> Option<String> {
        Some(self.algorithm.name().to_owned())
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunVerifier for ExtendedRsaVerifier<D> where D: ExtendedHash {}

#[cfg(feature = "kapun-provider")]
impl<D> KapunVerifying for ExtendedRsaVerifier<D>
where
    D: ExtendedHash,
{
    fn kapun_verify(
        &self,
        data: Vec<u8>,
        signature: Vec<u8>,
    ) -> Result<(), crate::kapun_crypto_provider::VerificationProblem> {
        self.verify(&data, &signature)
            .map_err(|_| crate::kapun_crypto_provider::VerificationProblem::SignatureInvalid)
    }

    fn kapun_verify_hash(
        &self,
        hash: Vec<u8>,
        signature: Vec<u8>,
    ) -> Result<(), crate::kapun_crypto_provider::VerificationProblem> {
        self.verify_prehashed(&hash, &signature)
            .map_err(|_| crate::kapun_crypto_provider::VerificationProblem::SignatureInvalid)
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunKeyEncoding for ExtendedRsaVerifier<D>
where
    D: ExtendedHash,
{
    fn kapun_public_jwk(&self) -> Option<String> {
        let name = self.algorithm.jose_name?;
        let mut jwk = crate::jwk::PublicKey::to_jwk_public_key(&self.public_key);
        jwk.set_algorithm(name);
        if let Some(key_id) = &self.key_id {
            jwk.set_key_id(key_id);
        }
        serde_json::to_string(&jwk).ok()
    }

    fn kapun_public_spki_der(&self) -> Option<Vec<u8>> {
        use rsa::pkcs8::EncodePublicKey;

        Some(self.public_key.to_public_key_der().ok()?.to_vec())
    }

    fn kapun_public_spki_pem(&self) -> Option<String> {
        use rsa::pkcs8::{EncodePublicKey, LineEnding};

        Some(self.public_key.to_public_key_pem(LineEnding::CRLF).ok()?)
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunMetadata for ExtendedRsaVerifier<D>
where
    D: ExtendedHash,
{
    fn kapun_jose_alg(&self) -> Option<String> {
        Some(self.algorithm.name().to_owned())
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunSigner for ExtendedEcdsaSigner<D> where D: ExtendedHash {}

#[cfg(feature = "kapun-provider")]
impl<D> KapunSigning for ExtendedEcdsaSigner<D>
where
    D: ExtendedHash,
{
    fn kapun_sign(
        &self,
        data: Vec<u8>,
    ) -> Result<Vec<u8>, crate::kapun_crypto_provider::SigningProblem> {
        self.sign(&data)
            .map_err(|_| crate::kapun_crypto_provider::SigningProblem::SigningFailed)
    }

    fn kapun_sign_hash(
        &self,
        hash: Vec<u8>,
    ) -> Result<Vec<u8>, crate::kapun_crypto_provider::SigningProblem> {
        self.sign_prehashed(&hash)
            .map_err(|_| crate::kapun_crypto_provider::SigningProblem::SigningFailed)
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunKeyEncoding for ExtendedEcdsaSigner<D>
where
    D: ExtendedHash,
{
    fn kapun_private_jwk(&self) -> Option<String> {
        let name = self.algorithm.jose_name?;
        let mut jwk: Jwk = serde_json::from_str(&self.private_key.ec_key_jwk().ok()?).ok()?;
        jwk.set_algorithm(name);
        if let Some(key_id) = &self.key_id {
            jwk.set_key_id(key_id);
        }
        serde_json::to_string(&jwk).ok()
    }

    fn kapun_public_jwk(&self) -> Option<String> {
        let name = self.algorithm.jose_name?;
        let mut jwk: Jwk =
            serde_json::from_str(&self.private_key.public_key().ec_key_jwk().ok()?).ok()?;
        jwk.set_algorithm(name);
        if let Some(key_id) = &self.key_id {
            jwk.set_key_id(key_id);
        }
        serde_json::to_string(&jwk).ok()
    }

    fn kapun_private_pkcs8_der(&self) -> Option<Vec<u8>> {
        let sec1_der = self.private_key.ec_key_der().ok()?;
        Some(EcKeyPair::to_pkcs8(&sec1_der, false, self.algorithm.curve))
    }

    fn kapun_private_pkcs8_pem(&self) -> Option<String> {
        let der = self.kapun_private_pkcs8_der()?;
        Some(pem::encode(&pem::Pem::new("PRIVATE KEY", der)))
    }

    fn kapun_public_spki_der(&self) -> Option<Vec<u8>> {
        self.private_key.public_key().ec_key_der().ok()
    }

    fn kapun_public_spki_pem(&self) -> Option<String> {
        self.private_key.public_key().ec_key_pem().ok()
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunMetadata for ExtendedEcdsaSigner<D>
where
    D: ExtendedHash,
{
    fn kapun_jose_alg(&self) -> Option<String> {
        Some(self.algorithm.name().to_owned())
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunVerifier for ExtendedEcdsaVerifier<D> where D: ExtendedHash {}

#[cfg(feature = "kapun-provider")]
impl<D> KapunVerifying for ExtendedEcdsaVerifier<D>
where
    D: ExtendedHash,
{
    fn kapun_verify(
        &self,
        data: Vec<u8>,
        signature: Vec<u8>,
    ) -> Result<(), crate::kapun_crypto_provider::VerificationProblem> {
        self.verify(&data, &signature)
            .map_err(|_| crate::kapun_crypto_provider::VerificationProblem::SignatureInvalid)
    }

    fn kapun_verify_hash(
        &self,
        hash: Vec<u8>,
        signature: Vec<u8>,
    ) -> Result<(), crate::kapun_crypto_provider::VerificationProblem> {
        self.verify_prehashed(&hash, &signature)
            .map_err(|_| crate::kapun_crypto_provider::VerificationProblem::SignatureInvalid)
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunKeyEncoding for ExtendedEcdsaVerifier<D>
where
    D: ExtendedHash,
{
    fn kapun_public_jwk(&self) -> Option<String> {
        let name = self.algorithm.jose_name?;
        let mut jwk: Jwk = serde_json::from_str(&self.public_key.ec_key_jwk().ok()?).ok()?;
        jwk.set_algorithm(name);
        if let Some(key_id) = &self.key_id {
            jwk.set_key_id(key_id);
        }
        serde_json::to_string(&jwk).ok()
    }

    fn kapun_public_spki_der(&self) -> Option<Vec<u8>> {
        self.public_key.ec_key_der().ok()
    }

    fn kapun_public_spki_pem(&self) -> Option<String> {
        self.public_key.ec_key_pem().ok()
    }
}

#[cfg(feature = "kapun-provider")]
impl<D> KapunMetadata for ExtendedEcdsaVerifier<D>
where
    D: ExtendedHash,
{
    fn kapun_jose_alg(&self) -> Option<String> {
        Some(self.algorithm.name().to_owned())
    }
}
