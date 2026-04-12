//! Implementation of the scattering matrix method for multilayer structures.
//!
//! # Numerical stability
//!
//! The scattering matrix formulation is preferred over the transfer matrix when
//! the in-plane wavevector `k` is complex (lossy media or leaky modes), because
//! it never multiplies exponentially growing and decaying terms in the same matrix
//! element.  Each propagation step only accumulates a phase; overflow is avoided
//! even for large `|Im(k)|`.
//!
//! # Branch-cut convention
//!
//! All transverse wavevectors are computed via [`kz_physical`] (re-exported from
//! `transfer_matrix`), which enforces `Im(kz) ≥ 0`.  See that module's
//! documentation for details.
extern crate itertools;
extern crate num_complex;

use itertools::izip;
use num_complex::Complex;

use crate::enums::Polarization;
use crate::layer::Layer;
use crate::transfer_matrix::kz_physical;

/// Struct representing the scattering matrix.
#[derive(Debug)]
pub struct ScatteringMatrix {
    /// s11 element of the scattering matrix.
    s11: Complex<f64>,
    /// s12 element of the scattering matrix.
    s12: Complex<f64>,
    /// s21 element of the scattering matrix.
    s21: Complex<f64>,
    /// s22 element of the scattering matrix.
    s22: Complex<f64>,
}

impl ScatteringMatrix {
    /// Compose two scattering matrices.
    ///
    /// Returns the scattering matrix for the cascade `self` followed by `other`.
    ///
    /// # Arguments
    /// * `other` - The scattering matrix of the downstream sub-system.
    ///
    /// # Returns
    /// The composed scattering matrix.
    pub fn compose(self, other: ScatteringMatrix) -> ScatteringMatrix {
        let denominator = 1.0 - self.s12 * other.s21;
        ScatteringMatrix {
            s11: self.s11 * other.s11 / denominator,
            s12: other.s12 + other.s11 * self.s12 * other.s22 / denominator,
            s21: self.s21 + self.s22 * other.s21 * self.s11 / denominator,
            s22: self.s22 * other.s22 / denominator,
        }
    }

    /// Calculate the determinant of the scattering matrix.
    ///
    /// # Returns
    /// The determinant of the scattering matrix.
    pub fn determinant(&self) -> Complex<f64> {
        self.s11 * self.s22 - self.s12 * self.s21
    }

    /// Creates the identity scattering matrix.
    ///
    /// # Returns
    /// The identity scattering matrix.
    pub fn matrix_start() -> ScatteringMatrix {
        ScatteringMatrix {
            s11: Complex::new(1.0, 0.0),
            s12: Complex::new(0.0, 0.0),
            s21: Complex::new(0.0, 0.0),
            s22: Complex::new(1.0, 0.0),
        }
    }

    /// Creates a scattering matrix for propagation through a homogeneous layer.
    ///
    /// The forward and backward amplitudes both accumulate the phase
    /// `exp(i·kz·d)`, where `kz = kz_physical(k0, n, k)`.
    ///
    /// # Arguments
    /// * `n`  - The refractive index of the layer (complex).
    /// * `d`  - The thickness of the layer.
    /// * `k0` - The vacuum wavevector (real, passed as `Complex<f64>`).
    /// * `k`  - The in-plane wavevector (complex for leaky / lossy modes).
    ///
    /// # Returns
    /// The scattering matrix for propagation in the layer.
    pub fn matrix_propagation(
        n: Complex<f64>,
        d: f64,
        k0: Complex<f64>,
        k: Complex<f64>,
    ) -> ScatteringMatrix {
        let d = Complex::new(d, 0.0);
        let kz = kz_physical(k0, n, k);
        let phase = Complex::new(0.0, 1.0) * kz * d;
        let p = phase.exp();
        ScatteringMatrix {
            s11: p,
            s12: Complex::new(0.0, 0.0),
            s21: Complex::new(0.0, 0.0),
            s22: p,
        }
    }

    /// Creates a scattering matrix for the interface between two layers — TE polarisation.
    ///
    /// Enforces continuity of the tangential electric field (`Ey`) and its
    /// transverse derivative at the interface.
    ///
    /// # Arguments
    /// * `n1` - The refractive index of the layer on the left (complex).
    /// * `n2` - The refractive index of the layer on the right (complex).
    /// * `k0` - The vacuum wavevector (complex).
    /// * `k`  - The in-plane wavevector (complex).
    ///
    /// # Returns
    /// The TE scattering matrix for the interface.
    pub fn matrix_interface_te(
        n1: Complex<f64>,
        n2: Complex<f64>,
        k0: Complex<f64>,
        k: Complex<f64>,
    ) -> ScatteringMatrix {
        let k1 = kz_physical(k0, n1, k);
        let k2 = kz_physical(k0, n2, k);
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }

    /// Creates a scattering matrix for the interface between two layers — TM polarisation.
    ///
    /// Enforces continuity of the tangential magnetic field (`Hy`) and the
    /// normal displacement field at the interface.
    ///
    /// # Arguments
    /// * `n1` - The refractive index of the layer on the left (complex).
    /// * `n2` - The refractive index of the layer on the right (complex).
    /// * `k0` - The vacuum wavevector (complex).
    /// * `k`  - The in-plane wavevector (complex).
    ///
    /// # Returns
    /// The TM scattering matrix for the interface.
    pub fn matrix_interface_tm(
        n1: Complex<f64>,
        n2: Complex<f64>,
        k0: Complex<f64>,
        k: Complex<f64>,
    ) -> ScatteringMatrix {
        let k1 = n2.powi(2) * kz_physical(k0, n1, k);
        let k2 = n1.powi(2) * kz_physical(k0, n2, k);
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }

    /// Creates a scattering matrix for the interface between two layers.
    ///
    /// Dispatches to [`matrix_interface_te`] or [`matrix_interface_tm`] based on
    /// the requested polarisation.
    ///
    /// # Arguments
    /// * `n1`           - The refractive index of the layer on the left (complex).
    /// * `n2`           - The refractive index of the layer on the right (complex).
    /// * `k0`           - The vacuum wavevector (complex).
    /// * `k`            - The in-plane wavevector (complex).
    /// * `polarization` - The polarisation of the light.
    ///
    /// # Returns
    /// The scattering matrix for the interface.
    pub fn matrix_interface(
        n1: Complex<f64>,
        n2: Complex<f64>,
        k0: Complex<f64>,
        k: Complex<f64>,
        polarization: Polarization,
    ) -> ScatteringMatrix {
        match polarization {
            Polarization::TE => ScatteringMatrix::matrix_interface_te(n1, n2, k0, k),
            Polarization::TM => ScatteringMatrix::matrix_interface_tm(n1, n2, k0, k),
        }
    }
}

/// Calculates the scattering matrix for a multilayer system.
///
/// # Arguments
/// * `layers`       - The layers of the system.
/// * `k0`           - The vacuum wavevector (complex).
/// * `k`            - The in-plane wavevector (complex).
/// * `polarization` - The polarisation of the light.
///
/// # Returns
/// The scattering matrix for the multilayer system.
pub fn calculate_s_matrix(
    layers: &[Layer],
    k0: Complex<f64>,
    k: Complex<f64>,
    polarization: Polarization,
) -> ScatteringMatrix {
    let mut result =
        ScatteringMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    for (layer1, layer2) in izip!(layers.iter().skip(1), layers.iter().skip(2)) {
        let matrix = ScatteringMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        result = result.compose(matrix);
        let matrix = ScatteringMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization);
        result = result.compose(matrix);
    }
    result
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    fn real(v: f64) -> Complex<f64> {
        c(v, 0.0)
    }

    /// The identity (start) matrix composed with itself gives itself.
    #[test]
    fn identity_compose() {
        let a = ScatteringMatrix::matrix_start();
        let b = ScatteringMatrix::matrix_start();
        let r = a.compose(b);
        assert!((r.s11 - c(1.0, 0.0)).norm() < 1e-12);
        assert!((r.s22 - c(1.0, 0.0)).norm() < 1e-12);
        assert!(r.s12.norm() < 1e-12);
        assert!(r.s21.norm() < 1e-12);
    }

    /// For a lossless layer the propagation matrix must satisfy |s11| = |s22| = 1.
    #[test]
    fn propagation_unimodular() {
        let k0 = real(1.0);
        let n = real(1.5);
        let k = real(1.0);
        let d = 0.5;
        let m = ScatteringMatrix::matrix_propagation(n, d, k0, k);
        assert!((m.s11.norm() - 1.0).abs() < 1e-12);
        assert!((m.s22.norm() - 1.0).abs() < 1e-12);
        assert!(m.s12.norm() < 1e-12);
        assert!(m.s21.norm() < 1e-12);
    }

    /// For a zero-thickness layer the propagation matrix is the identity.
    #[test]
    fn propagation_zero_thickness() {
        let k0 = real(1.0);
        let n = real(2.0);
        let k = real(0.5);
        let m = ScatteringMatrix::matrix_propagation(n, 0.0, k0, k);
        assert!((m.s11 - c(1.0, 0.0)).norm() < 1e-12);
        assert!((m.s22 - c(1.0, 0.0)).norm() < 1e-12);
    }

    /// TE interface at normal incidence (k = 0): check Fresnel coefficients.
    /// r = (n1 - n2)/(n1 + n2), t = 2*n1/(n1 + n2).
    #[test]
    fn te_interface_normal_incidence() {
        let k0 = real(1.0);
        let n1 = real(1.0);
        let n2 = real(1.5);
        let k = real(0.0);
        let m = ScatteringMatrix::matrix_interface_te(n1, n2, k0, k);
        // s12 = (k2 - k1)/(k1 + k2) = (n2 - n1)/(n1 + n2) = 0.5/2.5 = 0.2
        let expected_s12 = (1.5 - 1.0) / (1.0 + 1.5);
        assert!((m.s12 - real(expected_s12)).norm() < 1e-12);
        // s21 = (k1 - k2)/(k1 + k2) = -0.2
        assert!((m.s21 + real(expected_s12)).norm() < 1e-12);
    }

    /// TM interface at normal incidence (k = 0): check Fresnel coefficients.
    ///
    /// For TM the scattering matrix uses kz weighted by n²:
    ///   k1_tm = n2² · kz1,  k2_tm = n1² · kz2
    /// At normal incidence kz1 = n1·k0, kz2 = n2·k0, so:
    ///   k1_tm = n2²·n1·k0,  k2_tm = n1²·n2·k0
    ///   s12_tm = (k2_tm - k1_tm)/(k1_tm + k2_tm)
    ///          = (n1²·n2 - n2²·n1)/(n2²·n1 + n1²·n2)
    ///          = n1·n2·(n1 - n2) / (n1·n2·(n2 + n1))
    ///          = (n1 - n2)/(n1 + n2)
    /// which is the same as TE: (n1-n2)/(n1+n2). So at normal incidence
    /// the reflection coefficients (s12, s21) match TE, but the transmission
    /// coefficients (s11, s22) differ because they also carry the n² ratio.
    #[test]
    fn tm_interface_normal_incidence() {
        let k0 = real(1.0);
        let n1 = real(1.0);
        let n2 = real(1.5);
        let k = real(0.0);
        let te = ScatteringMatrix::matrix_interface_te(n1, n2, k0, k);
        let tm = ScatteringMatrix::matrix_interface_tm(n1, n2, k0, k);

        // At normal incidence the TM convention uses k1_tm = n2²·kz1, k2_tm = n1²·kz2,
        // so (k2_tm - k1_tm) = k0·n1·n2·(n1 - n2), giving s12_tm = (n1-n2)/(n1+n2)
        // while s12_te = (n2-n1)/(n1+n2).  The reflection coefficients have
        // opposite signs — which is the physically correct Fresnel convention.
        assert!(
            (te.s12 + tm.s12).norm() < 1e-12,
            "s12 TE={} should equal -TM={}",
            te.s12,
            tm.s12
        );
        assert!(
            (te.s21 + tm.s21).norm() < 1e-12,
            "s21 TE={} should equal -TM={}",
            te.s21,
            tm.s21
        );

        // Transmission coefficients differ: TM carries an extra n²/n² factor.
        // s11_tm = 2·k2_tm/(k1_tm+k2_tm) = 2·n1²·n2·k0 / (n1·n2·k0·(n1+n2))
        //        = 2·n1/(n1+n2)
        // s11_te = 2·kz2/(kz1+kz2) = 2·n2·k0/(n1·k0+n2·k0) = 2·n2/(n1+n2)
        let expected_s11_tm = 2.0 * 1.0 / (1.0 + 1.5); // 2·n1/(n1+n2)
        let expected_s11_te = 2.0 * 1.5 / (1.0 + 1.5); // 2·n2/(n1+n2)
        assert!(
            (tm.s11 - real(expected_s11_tm)).norm() < 1e-12,
            "TM s11={} expected={}",
            tm.s11,
            expected_s11_tm
        );
        assert!(
            (te.s11 - real(expected_s11_te)).norm() < 1e-12,
            "TE s11={} expected={}",
            te.s11,
            expected_s11_te
        );
    }

    /// For a symmetric interface (n1 == n2) the S-matrix must be the identity.
    #[test]
    fn interface_symmetric_is_identity() {
        let k0 = real(1.0);
        let n = real(1.5);
        let k = real(0.5);
        for pol in [Polarization::TE, Polarization::TM] {
            let m = ScatteringMatrix::matrix_interface(n, n, k0, k, pol);
            assert!((m.s11 - c(1.0, 0.0)).norm() < 1e-12);
            assert!((m.s22 - c(1.0, 0.0)).norm() < 1e-12);
            assert!(m.s12.norm() < 1e-12);
            assert!(m.s21.norm() < 1e-12);
        }
    }

    /// Complex refractive index: propagation matrix must still have bounded entries.
    #[test]
    fn propagation_complex_n() {
        // n = 1.5 + 0.01i  (slightly lossy)
        let k0 = real(1.0);
        let n = c(1.5, 0.01);
        let k = real(1.0);
        let d = 1.0;
        let m = ScatteringMatrix::matrix_propagation(n, d, k0, k);
        // With Im(n) > 0 and Im(k) = 0, kz has Im(kz) > 0, so |phase| < 1.
        assert!(m.s11.norm() <= 1.0 + 1e-12);
        assert!(m.s22.norm() <= 1.0 + 1e-12);
    }
}
