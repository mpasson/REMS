//! Implementation of the transfer matrix method for multilayer structures.
//!
//! # Transfer matrix convention
//!
//! All matrices in this module use the **physical forward-propagation** convention.
//! The full system transfer matrix for layers `[L0, L1, …, L_{n-1}]` is defined as
//!
//! ```text
//! T = T_int(L_{n-2}, L_{n-1}) · T_prop(L_{n-2}) · … · T_prop(L1) · T_int(L0, L1)
//! ```
//!
//! Applied to the column vector `[a_in, b_in]` at the left boundary, it produces
//! `[a_out, b_out]` at the right boundary:
//!
//! ```text
//! [a_out, b_out]^T = T · [a_in, b_in]^T
//! ```
//!
//! The standard semi-infinite mode condition (`b_out = 0` for input `[0, 1]`) therefore
//! corresponds to `T[1,1] = 0`, i.e. `t22 = 0`.
//!
//! The propagation-coefficient helpers walk left-to-right in the same physical order,
//! so the accumulated matrix and the step-by-step propagation are always consistent.
//!
//! # Branch-cut convention for `kz`
//!
//! All transverse wavevectors are computed via [`kz_physical`], which enforces
//! `Im(kz) ≥ 0` (and `Re(kz) ≥ 0` when `Im(kz) = 0`).  This corresponds to the
//! physical sheet on which evanescent fields decay away from the guiding region and
//! propagating fields carry energy in the positive-x direction.
//!
//! For **leaky modes** the outer-cladding layers must use the *outgoing* sheet
//! (`Im(kz) ≤ 0`), i.e. `-kz_physical(…)`.  The callers in `multilayer.rs` are
//! responsible for applying that sign flip where appropriate.
extern crate itertools;
extern crate num_complex;

use crate::enums::Polarization;
use crate::layer::{Layer, LayerCoefficientVector};
use num_complex::Complex;
use std::iter::zip;

// ─── Branch-cut-safe transverse wavevector ────────────────────────────────────

/// Returns the transverse wavevector
///
/// ```text
/// kz = sqrt( (k0 · n)² − k² )
/// ```
///
/// on the **physical sheet**: `Im(kz) ≥ 0`, and when `Im(kz) = 0` also
/// `Re(kz) ≥ 0`.
///
/// The Rust / `num-complex` principal square root already satisfies `Im(sqrt(z)) ≥ 0`
/// for all `z` not on the negative real axis.  The only problematic case is when the
/// argument lands exactly on the negative real axis (the branch cut), where the
/// principal value has `Im = 0` and `Re < 0`.  We detect and correct that case
/// explicitly so that the function is continuous and correct everywhere in the
/// complex plane.
///
/// # Arguments
/// * `k0` - Vacuum wavevector (real, but passed as `Complex<f64>` for uniformity).
/// * `n`  - Refractive index of the layer (may be complex for lossy / gain media).
/// * `k`  - In-plane (propagation) wavevector (may be complex for leaky / lossy modes).
pub fn kz_physical(k0: Complex<f64>, n: Complex<f64>, k: Complex<f64>) -> Complex<f64> {
    let z = (k0 * n).powi(2) - k.powi(2);
    let s = z.sqrt(); // principal sqrt: Im(s) >= 0 by definition
                      // The principal sqrt satisfies Im(s) >= 0 everywhere *except* on the branch
                      // cut (negative real axis of z), where it returns Im(s) = 0 and Re(s) < 0.
                      // Flip the sign in that degenerate case so Re(s) >= 0 as well.
    if s.im < 0.0 || (s.im == 0.0 && s.re < 0.0) {
        -s
    } else {
        s
    }
}

/// Returns the transverse wavevector on the **outgoing-wave sheet**: `Im(kz) ≤ 0`.
///
/// This is exactly `-kz_physical(k0, n, k)` and places the computation on the
/// Riemann sheet required for **quasi-normal modes (QNMs)** and leaky resonances.
///
/// Physics: a QNM has energy flowing *away* from the structure in both semi-infinite
/// cladding layers.  This outgoing radiation condition requires `Im(kz) ≤ 0` in the
/// outermost layers (the evanescent tail grows spatially rather than decays, consistent
/// with a mode that decays in *time*).  The standard `kz_physical` convention
/// (`Im(kz) ≥ 0`) enforces the opposite — exponential decay — which is correct for
/// bound guided modes but places the solver on the wrong Riemann sheet for QNMs.
///
/// # Arguments
/// * `k0` - Vacuum wavevector (real, passed as `Complex<f64>` for uniformity).
/// * `n`  - Refractive index of the layer (may be complex for lossy / gain media).
/// * `k`  - In-plane (propagation) wavevector (may be complex).
pub fn kz_outgoing(k0: Complex<f64>, n: Complex<f64>, k: Complex<f64>) -> Complex<f64> {
    -kz_physical(k0, n, k)
}

// ─── Transfer matrix struct ───────────────────────────────────────────────────

/// Struct representing the transfer matrix.
#[derive(Debug)]
pub struct TransferMatrix {
    /// t11 element of the matrix.
    t11: Complex<f64>,
    /// t12 element of the matrix.
    t12: Complex<f64>,
    /// t21 element of the matrix.
    t21: Complex<f64>,
    /// t22 element of the matrix.
    pub t22: Complex<f64>,
}

impl TransferMatrix {
    /// Returns a new TransferMatrix which is the composition of two matrices.
    ///
    /// `self.compose(other)` computes `self · other` (matrix multiplication).
    /// Applied to a vector, `other` acts first, then `self`.
    ///
    /// # Arguments
    /// * `other` - The matrix to right-multiply by.
    ///
    /// # Returns
    /// The product `self · other`.
    pub fn compose(self, other: TransferMatrix) -> TransferMatrix {
        TransferMatrix {
            t11: self.t11 * other.t11 + self.t12 * other.t21,
            t12: self.t11 * other.t12 + self.t12 * other.t22,
            t21: self.t21 * other.t11 + self.t22 * other.t21,
            t22: self.t21 * other.t12 + self.t22 * other.t22,
        }
    }

    /// Creates the identity transfer matrix. Useful to start a recursion.
    ///
    /// # Returns
    /// The identity transfer matrix.
    pub fn matrix_start() -> TransferMatrix {
        TransferMatrix {
            t11: Complex::new(1.0, 0.0),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: Complex::new(1.0, 0.0),
        }
    }

    /// Creates the propagation transfer matrix for a homogeneous layer.
    ///
    /// The matrix propagates the forward/backward amplitudes `(a, b)` from the
    /// left edge to the right edge of the layer:
    ///
    /// ```text
    /// a_right = exp(+i·kz·d) · a_left
    /// b_right = exp(-i·kz·d) · b_left
    /// ```
    ///
    /// where `kz = kz_physical(k0, n, k)`.
    ///
    /// # Arguments
    /// * `n` - The refractive index of the layer (complex).
    /// * `d` - The thickness of the layer.
    /// * `k0` - The vacuum wavevector (real).
    /// * `k`  - The in-plane component of the wavevector (complex for leaky/lossy modes).
    ///
    /// # Returns
    /// The propagation transfer matrix for the layer.
    pub fn matrix_propagation(
        n: Complex<f64>,
        d: f64,
        k0: Complex<f64>,
        k: Complex<f64>,
    ) -> TransferMatrix {
        let d = Complex::new(d, 0.0);
        let kz = kz_physical(k0, n, k);
        let phase_positive = Complex::new(0.0, 1.0) * kz * d;
        let phase_negative = Complex::new(0.0, -1.0) * kz * d;
        TransferMatrix {
            t11: phase_positive.exp(),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: phase_negative.exp(),
        }
    }

    /// Creates the TE interface transfer matrix between two layers.
    ///
    /// Relates the forward/backward amplitudes `(a, b)` on the left side of the
    /// interface to those on the right side, enforcing continuity of the tangential
    /// electric field (`Ey`) and its derivative.
    ///
    /// # Arguments
    /// * `n1` - The refractive index of the layer on the left (complex).
    /// * `n2` - The refractive index of the layer on the right (complex).
    /// * `k0` - The vacuum wavevector (real).
    /// * `k`  - The in-plane component of the wavevector (complex).
    ///
    /// # Returns
    /// The TE interface transfer matrix.
    pub fn matrix_interface_te(
        n1: Complex<f64>,
        n2: Complex<f64>,
        k0: Complex<f64>,
        k: Complex<f64>,
    ) -> TransferMatrix {
        let k1 = kz_physical(k0, n1, k);
        let k2 = kz_physical(k0, n2, k);
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k2,
            t12: 0.5 * (k2 - k1) / k2,
            t21: 0.5 * (k2 - k1) / k2,
            t22: 0.5 * (k2 + k1) / k2,
        }
    }

    /// Creates the TM interface transfer matrix between two layers.
    ///
    /// Relates the forward/backward amplitudes `(a, b)` on the left side of the
    /// interface to those on the right side, enforcing continuity of the tangential
    /// magnetic field (`Hy`) and the normal displacement field.
    ///
    /// # Arguments
    /// * `n1` - The refractive index of the layer on the left (complex).
    /// * `n2` - The refractive index of the layer on the right (complex).
    /// * `k0` - The vacuum wavevector (real).
    /// * `k`  - The in-plane component of the wavevector (complex).
    ///
    /// # Returns
    /// The TM interface transfer matrix.
    pub fn matrix_interface_tm(
        n1: Complex<f64>,
        n2: Complex<f64>,
        k0: Complex<f64>,
        k: Complex<f64>,
    ) -> TransferMatrix {
        let k1 = n2.powi(2) * kz_physical(k0, n1, k);
        let k2 = n1.powi(2) * kz_physical(k0, n2, k);
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k1,
            t12: 0.5 * (k1 - k2) / k1,
            t21: 0.5 * (k1 - k2) / k1,
            t22: 0.5 * (k2 + k1) / k1,
        }
    }

    /// Creates the interface transfer matrix between two layers for the given polarization.
    ///
    /// Dispatches to [`matrix_interface_te`] or [`matrix_interface_tm`].
    ///
    /// # Arguments
    /// * `n1`          - The refractive index of the layer on the left (complex).
    /// * `n2`          - The refractive index of the layer on the right (complex).
    /// * `k0`          - The vacuum wavevector (real).
    /// * `k`           - The in-plane component of the wavevector (complex).
    /// * `polarization` - The polarization of the light.
    ///
    /// # Returns
    /// The interface transfer matrix.
    pub fn matrix_interface(
        n1: Complex<f64>,
        n2: Complex<f64>,
        k0: Complex<f64>,
        k: Complex<f64>,
        polarization: Polarization,
    ) -> TransferMatrix {
        match polarization {
            Polarization::TE => TransferMatrix::matrix_interface_te(n1, n2, k0, k),
            Polarization::TM => TransferMatrix::matrix_interface_tm(n1, n2, k0, k),
        }
    }

    /// Applies the transfer matrix to a modal coefficient vector.
    ///
    /// Returns the output `(a_out, b_out)` produced by left-multiplying the
    /// column vector `[a, b]^T` by this matrix.
    ///
    /// # Arguments
    /// * `coefficient_vector` - The input modal coefficient vector.
    ///
    /// # Returns
    /// The output modal coefficient vector.
    pub fn multiply(&self, coefficient_vector: &LayerCoefficientVector) -> LayerCoefficientVector {
        LayerCoefficientVector {
            a: self.t11 * coefficient_vector.a + self.t12 * coefficient_vector.b,
            b: self.t21 * coefficient_vector.a + self.t22 * coefficient_vector.b,
        }
    }

    /// Applies the transfer matrix to a raw `(a, b)` pair.
    ///
    /// Returns `(a_out, b_out)`. This is a convenience overload used when
    /// evaluating mode conditions directly from scalar amplitudes.
    ///
    /// # Arguments
    /// * `a` - The forward amplitude.
    /// * `b` - The backward amplitude.
    ///
    /// # Returns
    /// The output `(a_out, b_out)`.
    pub fn apply(&self, a: Complex<f64>, b: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
        (self.t11 * a + self.t12 * b, self.t21 * a + self.t22 * b)
    }
}

// ─── Full-stack transfer matrix ───────────────────────────────────────────────

/// Calculates the physical forward-propagation transfer matrix of a multilayer system.
///
/// For a system with layers `[L0, L1, …, L_{n-1}]` the matrix is assembled as:
///
/// ```text
/// T = T_int(L_{n-2}, L_{n-1}) · T_prop(L_{n-2}) · … · T_prop(L1) · T_int(L0, L1)
/// ```
///
/// Applied to `[a_in, b_in]` at the **left** boundary it produces `[a_out, b_out]`
/// at the **right** boundary.
///
/// For the standard semi-infinite boundary mode condition the relevant entry is
/// `T[1,1]` (`t22`): starting from `(a_in, b_in) = (0, 1)` (only the decaying wave
/// in the left cladding) and requiring `b_out = 0` in the right cladding gives
/// `T[1,1] · 1 = 0`, i.e. `t22 = 0`.
///
/// # Arguments
/// * `layers`        - The layers of the system.
/// * `k0`            - The vacuum wavevector (real, lifted to `Complex<f64>`).
/// * `k`             - The in-plane component of the wavevector (complex).
/// * `polarization`  - The polarization of the light.
///
/// # Returns
/// The physical forward-propagation transfer matrix.
pub fn calculate_t_matrix(
    layers: &[Layer],
    k0: Complex<f64>,
    k: Complex<f64>,
    polarization: Polarization,
) -> TransferMatrix {
    let n = layers.len();
    let mut result =
        TransferMatrix::matrix_interface(layers[n - 2].n, layers[n - 1].n, k0, k, polarization);
    for i in (0..n - 2).rev() {
        let prop = TransferMatrix::matrix_propagation(layers[i + 1].n, layers[i + 1].d, k0, k);
        result = result.compose(prop);
        let iface =
            TransferMatrix::matrix_interface(layers[i].n, layers[i + 1].n, k0, k, polarization);
        result = result.compose(iface);
    }
    result
}

// ─── Per-layer coefficient propagation ───────────────────────────────────────

/// Calculates the modal coefficients in each layer for a semi-infinite left boundary.
///
/// In the standard case `layers[0]` is the semi-infinite left cladding.  Its field
/// decays exponentially away from the guide, so no propagation step is needed
/// before crossing the first interface.  The supplied `(a, b)` are the amplitudes
/// at the reference plane `x = 0` (the right edge of the left cladding / left edge
/// of `layers[1]`).
///
/// Coefficients are stored in the order `[layer 0, layer 1, …]`, each referenced
/// to the **left edge** of the corresponding layer.
///
/// # Arguments
/// * `layers`       - The layers of the system (first element is the semi-infinite left cladding).
/// * `k0`           - The vacuum wavevector (complex).
/// * `k`            - The in-plane component of the wavevector (complex).
/// * `polarization` - The polarization of the light.
/// * `a`            - The forward amplitude in the left cladding (typically 0 for a guided mode).
/// * `b`            - The backward amplitude in the left cladding (typically 1, normalised later).
///
/// # Returns
/// The modal coefficient vector for each layer, referenced to the layer's left edge.
pub fn get_propagation_coefficients_transfer(
    layers: &[Layer],
    k0: Complex<f64>,
    k: Complex<f64>,
    polarization: Polarization,
    a: Complex<f64>,
    b: Complex<f64>,
) -> Vec<LayerCoefficientVector> {
    let mut coefficients: Vec<LayerCoefficientVector> = Vec::new();
    let mut current_coefficients = LayerCoefficientVector::new(a, b);
    // Layer 0: semi-infinite left cladding, coefficients at x = 0 (right edge / reference).
    coefficients.push(current_coefficients);
    // Cross the L0→L1 interface (no propagation inside L0 since it is semi-infinite).
    let transfer_matrix =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    current_coefficients = transfer_matrix.multiply(&current_coefficients);
    coefficients.push(current_coefficients);
    // For every subsequent layer: propagate to the right edge, then cross the interface.
    for (layer1, layer2) in zip(layers.iter().skip(1), layers.iter().skip(2)) {
        let propagation_matrix = TransferMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        current_coefficients = propagation_matrix.multiply(&current_coefficients);
        let interface_matrix =
            TransferMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization);
        current_coefficients = interface_matrix.multiply(&current_coefficients);
        coefficients.push(current_coefficients);
    }
    coefficients
}

/// Calculates the modal coefficients in each layer for a PEC left boundary.
///
/// Here `layers[0]` is the first *finite* dielectric layer, whose left edge sits
/// at the PEC wall (`x = 0`).  The supplied `(a, b)` are the field amplitudes
/// **at the PEC wall** and must already satisfy the PEC boundary condition
/// (e.g. `a = 1, b = −1` for TE so that `Ey(0) = a + b = 0`).
///
/// Unlike the semi-infinite case, `layers[0]` has finite thickness, so we must
/// propagate through it before crossing the first interface.
///
/// Coefficients are stored in the order `[layer 0, layer 1, …]`, each referenced
/// to the **left edge** of the corresponding layer.
///
/// # Arguments
/// * `layers`       - The layers of the system; `layers[0]` is adjacent to the PEC wall.
/// * `k0`           - The vacuum wavevector (complex).
/// * `k`            - The in-plane component of the wavevector (complex).
/// * `polarization` - The polarization of the light.
/// * `a`            - The forward amplitude at the PEC wall (left edge of `layers[0]`).
/// * `b`            - The backward amplitude at the PEC wall (left edge of `layers[0]`).
///
/// # Returns
/// The modal coefficient vector for each layer, referenced to the layer's left edge.
pub fn get_propagation_coefficients_pec_left(
    layers: &[Layer],
    k0: Complex<f64>,
    k: Complex<f64>,
    polarization: Polarization,
    a: Complex<f64>,
    b: Complex<f64>,
) -> Vec<LayerCoefficientVector> {
    let mut coefficients: Vec<LayerCoefficientVector> = Vec::new();
    let mut current_coefficients = LayerCoefficientVector::new(a, b);
    // Layer 0: coefficients at x = 0 (PEC wall / left edge of layer 0).
    coefficients.push(current_coefficients);
    // Propagate through layer 0 to its right edge, then cross into layer 1.
    let propagation_matrix = TransferMatrix::matrix_propagation(layers[0].n, layers[0].d, k0, k);
    current_coefficients = propagation_matrix.multiply(&current_coefficients);
    let interface_matrix =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    current_coefficients = interface_matrix.multiply(&current_coefficients);
    coefficients.push(current_coefficients);
    // For every subsequent layer: propagate to the right edge, then cross the interface.
    for (layer1, layer2) in zip(layers.iter().skip(1), layers.iter().skip(2)) {
        let propagation_matrix = TransferMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        current_coefficients = propagation_matrix.multiply(&current_coefficients);
        let interface_matrix =
            TransferMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization);
        current_coefficients = interface_matrix.multiply(&current_coefficients);
        coefficients.push(current_coefficients);
    }
    coefficients
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    // ── kz_physical ──────────────────────────────────────────────────────────

    /// For a real argument with (k0·n)² > k²  the result must be real and positive.
    #[test]
    fn kz_real_propagating() {
        let k0 = c(1.0, 0.0);
        let n = c(2.0, 0.0);
        let k = c(1.0, 0.0);
        // (2)² - (1)² = 3  →  sqrt(3) ≈ 1.732
        let kz = kz_physical(k0, n, k);
        assert!(kz.im == 0.0, "Im(kz) should be zero for real propagating");
        assert!(
            kz.re > 0.0,
            "Re(kz) should be positive for real propagating"
        );
        assert!((kz.re - 3.0_f64.sqrt()).abs() < 1e-12);
    }

    /// For a real argument with (k0·n)² < k²  the result must be purely imaginary
    /// and positive (evanescent decay in the +x direction).
    #[test]
    fn kz_real_evanescent() {
        let k0 = c(1.0, 0.0);
        let n = c(1.0, 0.0);
        let k = c(2.0, 0.0);
        // (1)² - (2)² = -3  →  sqrt(-3) = i·sqrt(3)
        let kz = kz_physical(k0, n, k);
        assert!(kz.re == 0.0, "Re(kz) should be zero for evanescent");
        assert!(
            kz.im > 0.0,
            "Im(kz) should be positive for evanescent decay"
        );
        assert!((kz.im - 3.0_f64.sqrt()).abs() < 1e-12);
    }

    /// The result must never have Im(kz) < 0 for any point in the complex k-plane.
    #[test]
    fn kz_im_nonnegative_across_plane() {
        let k0 = c(1.0, 0.0);
        let n = c(1.5, 0.0);
        // Sweep a grid of complex k values and verify Im(kz) >= 0 everywhere.
        for re in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            for im in [-2.0, -0.5, 0.0, 0.5, 2.0] {
                let k = c(re, im);
                let kz = kz_physical(k0, n, k);
                assert!(kz.im >= -1e-14, "Im(kz) < 0 at k = ({re}, {im}): kz = {kz}");
            }
        }
    }

    /// Verify continuity: kz_physical should not jump when k moves across the
    /// region near the branch cut.  We check that a small step in k gives a
    /// small step in kz.
    #[test]
    fn kz_continuous_near_branch_cut() {
        let k0 = c(1.0, 0.0);
        let n = c(1.0, 0.0);
        // The branch cut of sqrt(1 - k²) runs along the real axis for |Re(k)| > 1.
        // Approach from above and below.
        let k_above = c(1.5, 1e-9);
        let k_below = c(1.5, -1e-9);
        let kz_above = kz_physical(k0, n, k_above);
        let kz_below = kz_physical(k0, n, k_below);
        // Both must have Im >= 0.
        assert!(kz_above.im >= 0.0);
        assert!(kz_below.im >= 0.0);
        // They must be close to each other (continuity).
        let diff = (kz_above - kz_below).norm();
        assert!(
            diff < 1e-6,
            "kz discontinuous near branch cut: diff = {diff}"
        );
    }

    // ── Propagation matrix ────────────────────────────────────────────────────

    /// The propagation matrix for zero thickness must be the identity.
    #[test]
    fn propagation_matrix_zero_thickness() {
        let k0 = c(1.0, 0.0);
        let n = c(1.5, 0.0);
        let k = c(1.0, 0.0);
        let m = TransferMatrix::matrix_propagation(n, 0.0, k0, k);
        assert!((m.t11 - c(1.0, 0.0)).norm() < 1e-12);
        assert!((m.t22 - c(1.0, 0.0)).norm() < 1e-12);
    }

    /// For a lossless layer the propagation matrix must be unitary:
    /// |t11|² = |t22|² = 1, t12 = t21 = 0.
    #[test]
    fn propagation_matrix_unimodular() {
        let k0 = c(1.0, 0.0);
        let n = c(2.0, 0.0);
        let k = c(1.0, 0.0);
        let d = 0.5;
        let m = TransferMatrix::matrix_propagation(n, d, k0, k);
        assert!((m.t11.norm() - 1.0).abs() < 1e-12);
        assert!((m.t22.norm() - 1.0).abs() < 1e-12);
        assert!(m.t12.norm() < 1e-12);
        assert!(m.t21.norm() < 1e-12);
    }
}
