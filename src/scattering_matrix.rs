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

use crate::enums::BoundaryCondition;
use crate::enums::Polarization;
use crate::layer::Layer;
use crate::transfer_matrix::{kz_outgoing, kz_physical};

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

    /// Creates a TE scattering matrix for an interface using pre-computed transverse
    /// wavevectors.
    ///
    /// Use this variant when a non-standard kz convention (e.g. [`kz_outgoing`]) is
    /// required for one or both sides of the interface, rather than the default
    /// `kz_physical` used by [`matrix_interface_te`].
    ///
    /// # Arguments
    /// * `kz1` - Transverse wavevector in the left medium.
    /// * `kz2` - Transverse wavevector in the right medium.
    pub fn matrix_interface_te_from_kz(kz1: Complex<f64>, kz2: Complex<f64>) -> ScatteringMatrix {
        ScatteringMatrix {
            s11: 2.0 * kz2 / (kz1 + kz2),
            s12: (kz2 - kz1) / (kz1 + kz2),
            s21: (kz1 - kz2) / (kz1 + kz2),
            s22: 2.0 * kz1 / (kz1 + kz2),
        }
    }

    /// Creates a TM scattering matrix for an interface using pre-computed transverse
    /// wavevectors.
    ///
    /// # Arguments
    /// * `n1`  - Refractive index of the left medium.
    /// * `n2`  - Refractive index of the right medium.
    /// * `kz1` - Transverse wavevector in the left medium.
    /// * `kz2` - Transverse wavevector in the right medium.
    pub fn matrix_interface_tm_from_kz(
        n1: Complex<f64>,
        n2: Complex<f64>,
        kz1: Complex<f64>,
        kz2: Complex<f64>,
    ) -> ScatteringMatrix {
        let k1 = n2.powi(2) * kz1;
        let k2 = n1.powi(2) * kz2;
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }

    /// Creates a scattering matrix for an interface using pre-computed transverse
    /// wavevectors, dispatching on polarisation.
    ///
    /// # Arguments
    /// * `n1`           - Refractive index of the left medium.
    /// * `n2`           - Refractive index of the right medium.
    /// * `kz1`          - Transverse wavevector in the left medium.
    /// * `kz2`          - Transverse wavevector in the right medium.
    /// * `polarization` - Polarisation.
    pub fn matrix_interface_from_kz(
        n1: Complex<f64>,
        n2: Complex<f64>,
        kz1: Complex<f64>,
        kz2: Complex<f64>,
        polarization: Polarization,
    ) -> ScatteringMatrix {
        match polarization {
            Polarization::TE => ScatteringMatrix::matrix_interface_te_from_kz(kz1, kz2),
            Polarization::TM => ScatteringMatrix::matrix_interface_tm_from_kz(n1, n2, kz1, kz2),
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

/// Calculates the scattering matrix for a multilayer system using
/// **outgoing-wave** boundary conditions on the semi-infinite cladding layers.
///
/// Unlike [`calculate_s_matrix`], which uses `kz_physical` (`Im(kz) ≥ 0`) for all
/// layers, this function applies [`kz_outgoing`] (`Im(kz) ≤ 0`) to the first layer
/// (left cladding) and the last layer (right cladding).  All interior layers
/// continue to use `kz_physical`.
///
/// The resulting S-matrix has zeros of its determinant at **quasi-normal mode (QNM)**
/// eigenvalues — the complex effective indices at which outgoing waves exist in both
/// claddings simultaneously with no incoming excitation.  QNM poles therefore live in
/// the *lower* half of the complex `neff` plane (`Im(neff) < 0`).
///
/// # Arguments
/// * `layers`       - The layers of the system (first and last treated as claddings).
/// * `k0`           - The vacuum wavevector (complex, but typically real).
/// * `k`            - The in-plane wavevector (complex for QNMs, `Im(k) < 0`).
/// * `polarization` - The polarisation of the light.
///
/// # Returns
/// The scattering matrix built with outgoing-wave boundary conditions.
pub fn calculate_s_matrix_qnm(
    layers: &[Layer],
    k0: Complex<f64>,
    k: Complex<f64>,
    polarization: Polarization,
) -> ScatteringMatrix {
    debug_assert!(layers.len() >= 2, "QNM S-matrix requires at least 2 layers");

    // ── First interface (left cladding → first interior layer) ────────────────
    // The left cladding always uses kz_outgoing (outgoing radiation to the left).
    // If the structure has only two layers (no interior layers at all) the right
    // layer is also a cladding and likewise uses kz_outgoing; otherwise it is an
    // interior layer and uses kz_physical.
    let kz0 = kz_outgoing(k0, layers[0].n, k);
    let kz1 = if layers.len() == 2 {
        kz_outgoing(k0, layers[1].n, k)
    } else {
        kz_physical(k0, layers[1].n, k)
    };
    let mut result = ScatteringMatrix::matrix_interface_from_kz(
        layers[0].n,
        layers[1].n,
        kz0,
        kz1,
        polarization,
    );

    // ── Interior propagation + interfaces ─────────────────────────────────────
    // Iterate over consecutive pairs starting at layers[1]:
    //   (layers[1], layers[2]), …, (layers[n-2], layers[n-1]).
    // The last pair terminates at the right cladding (layers[n-1]), which uses
    // kz_outgoing; all earlier pairs are between interior layers and use kz_physical.
    let pairs: Vec<_> = layers.windows(2).skip(1).collect();
    let n_pairs = pairs.len();
    for (i, window) in pairs.iter().enumerate() {
        let layer1 = &window[0];
        let layer2 = &window[1];

        // Propagation through layer1 (interior): always kz_physical.
        let prop = ScatteringMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        result = result.compose(prop);

        // Interface: last pair reaches the right cladding (kz_outgoing);
        // all others are interior-to-interior (kz_physical on both sides).
        let intf = if i + 1 == n_pairs {
            let kz_left = kz_physical(k0, layer1.n, k);
            let kz_right = kz_outgoing(k0, layer2.n, k);
            ScatteringMatrix::matrix_interface_from_kz(
                layer1.n,
                layer2.n,
                kz_left,
                kz_right,
                polarization,
            )
        } else {
            ScatteringMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization)
        };
        result = result.compose(intf);
    }

    result
}

/// Calculates the scattering matrix for a multilayer system using a
/// **one-sided leaky** boundary condition.
///
/// The left cladding (first layer) uses [`kz_physical`] — the standard
/// evanescent-decay condition, identical to the ordinary guided-mode solver.
/// The right cladding (last layer) uses [`kz_outgoing`] — the outgoing-wave
/// condition, so energy radiates into the substrate.  All interior layers
/// continue to use [`kz_physical`].
///
/// This boundary condition models a mode that is evanescently confined on the
/// low-index side (typically air) and radiates into a higher-index substrate on
/// the other side.  The zeros of the resulting S-matrix determinant are the
/// complex effective indices of those one-sided leaky modes; they sit in the
/// **lower** half of the complex `neff` plane (`Im(neff) < 0`), and their real
/// part stays close to the guided-mode value of the isolated core.
///
/// Compare with:
/// * [`calculate_s_matrix`]     — `kz_physical` everywhere (guided modes).
/// * [`calculate_s_matrix_qnm`] — `kz_outgoing` on **both** claddings (full QNMs).
///
/// # Arguments
/// * `layers`       - The layers of the system (first and last treated as claddings).
/// * `k0`           - The vacuum wavevector (complex, but typically real).
/// * `k`            - The in-plane wavevector (complex, `Im(k) < 0` for leaky modes).
/// * `polarization` - The polarisation of the light.
///
/// # Returns
/// The scattering matrix built with a one-sided leaky boundary condition.
pub fn calculate_s_matrix_leaky_right(
    layers: &[Layer],
    k0: Complex<f64>,
    k: Complex<f64>,
    polarization: Polarization,
) -> ScatteringMatrix {
    debug_assert!(
        layers.len() >= 2,
        "leaky-right S-matrix requires at least 2 layers"
    );

    // Left cladding always uses kz_physical (evanescent / guided-mode BC).
    // If there are only two layers the right layer is the right cladding and
    // therefore uses kz_outgoing; otherwise it is an interior layer and uses
    // kz_physical.
    let kz0 = kz_physical(k0, layers[0].n, k);
    let kz1 = if layers.len() == 2 {
        kz_outgoing(k0, layers[1].n, k)
    } else {
        kz_physical(k0, layers[1].n, k)
    };
    let mut result = ScatteringMatrix::matrix_interface_from_kz(
        layers[0].n,
        layers[1].n,
        kz0,
        kz1,
        polarization,
    );

    // Interior propagation + interfaces.
    // The last pair terminates at the right cladding (kz_outgoing);
    // all earlier pairs are interior-to-interior (kz_physical on both sides).
    let pairs: Vec<_> = layers.windows(2).skip(1).collect();
    let n_pairs = pairs.len();
    for (i, window) in pairs.iter().enumerate() {
        let layer1 = &window[0];
        let layer2 = &window[1];

        // Propagation through layer1 (interior): always kz_physical.
        let prop = ScatteringMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        result = result.compose(prop);

        // Interface: last pair reaches the right cladding (kz_outgoing);
        // all others are interior-to-interior (kz_physical on both sides).
        let intf = if i + 1 == n_pairs {
            let kz_left = kz_physical(k0, layer1.n, k);
            let kz_right = kz_outgoing(k0, layer2.n, k);
            ScatteringMatrix::matrix_interface_from_kz(
                layer1.n,
                layer2.n,
                kz_left,
                kz_right,
                polarization,
            )
        } else {
            ScatteringMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization)
        };
        result = result.compose(intf);
    }

    result
}

/// Calculates the scattering matrix with configurable boundary conditions.
///
/// | `left_bc`    | `right_bc`   | Mode type                            |
/// |--------------|--------------|--------------------------------------|
/// | SemiInfinite | SemiInfinite | Guided modes (`Im(neff) ≈ 0`)        |
/// | Outgoing     | Outgoing     | Quasi-normal modes (`Im(neff) < 0`)  |
/// | SemiInfinite | Outgoing     | One-sided leaky (`Im(neff) > 0`)     |
/// | Outgoing     | SemiInfinite | One-sided leaky (mirrored)           |
///
/// `PEC` is treated identically to `SemiInfinite` (wall BCs do not apply to
/// the S-matrix; the complex solver always uses the S-matrix).
pub fn calculate_s_matrix_with_bc(
    layers: &[Layer],
    k0: Complex<f64>,
    k: Complex<f64>,
    polarization: Polarization,
    left_bc: BoundaryCondition,
    right_bc: BoundaryCondition,
) -> ScatteringMatrix {
    debug_assert!(layers.len() >= 2, "S-matrix requires at least 2 layers");

    let kz_for_bc = |n: Complex<f64>, bc: BoundaryCondition| -> Complex<f64> {
        match bc {
            BoundaryCondition::Outgoing => kz_outgoing(k0, n, k),
            _ => kz_physical(k0, n, k),
        }
    };

    let kz0 = kz_for_bc(layers[0].n, left_bc);
    let kz1 = if layers.len() == 2 {
        kz_for_bc(layers[1].n, right_bc)
    } else {
        kz_physical(k0, layers[1].n, k)
    };
    let mut result = ScatteringMatrix::matrix_interface_from_kz(
        layers[0].n,
        layers[1].n,
        kz0,
        kz1,
        polarization,
    );

    let pairs: Vec<_> = layers.windows(2).skip(1).collect();
    let n_pairs = pairs.len();
    for (i, window) in pairs.iter().enumerate() {
        let layer1 = &window[0];
        let layer2 = &window[1];
        let prop = ScatteringMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        result = result.compose(prop);
        let intf = if i + 1 == n_pairs {
            let kz_left = kz_physical(k0, layer1.n, k);
            let kz_right = kz_for_bc(layer2.n, right_bc);
            ScatteringMatrix::matrix_interface_from_kz(
                layer1.n,
                layer2.n,
                kz_left,
                kz_right,
                polarization,
            )
        } else {
            ScatteringMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization)
        };
        result = result.compose(intf);
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
