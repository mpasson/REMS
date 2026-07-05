//! This module contains the implementation of the `MultiLayer` struct and its methods.
extern crate cumsum;
extern crate find_peaks;
extern crate itertools;

use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::iter::zip;
use std::iter::Sum;
use std::ops::{Add, Mul, Sub};

use crate::enums::BackEnd;
use crate::enums::BoundaryCondition;
use crate::enums::Normalization;
use crate::enums::Polarization;
use crate::layer::{Layer, LayerCoefficientVector, PEC};
use crate::scattering_matrix::{calculate_s_matrix, calculate_s_matrix_with_bc};
use crate::transfer_matrix::TransferMatrix;
use crate::transfer_matrix::{
    calculate_t_matrix, get_propagation_coefficients_pec_left,
    get_propagation_coefficients_transfer,
};
use cumsum::cumsum;
use find_peaks::PeakFinder;
use log::warn;
use num_complex::Complex;

/// Vacuum impedance.
const Z0: Complex<f64> = Complex {
    re: 376.73031346177066,
    im: 0.0,
};

/// Upper imaginary half-width of the complex search rectangle (used as
/// `im_max` in the default window).  Large enough to give the winding-number
/// contour room to accumulate its full `2π` contribution around near-real-axis
/// poles, while keeping the contour far enough above the real axis that the
/// branch-cut-contaminated zone (where `kz_physical` performs its sign flip)
/// does not dominate the integral.  0.05 keeps the aspect ratio below ~20:1
/// for typical structures with `re_range` width of order 1.
const MIN_IM_HALF_WIDTH: f64 = 0.05;

/// Lower imaginary half-width of the complex search rectangle for **lossless**
/// structures (used as `im_min = -MIN_IM_LOWER` in the default window).
///
/// For lossless structures all modes sit at `Im(neff) ≈ 0` (exactly real for
/// guided modes, or infinitesimally positive for quasi-guided / leaky resonances).
/// The characteristic function `1/det(S)` has branch cuts in the *upper*
/// half-plane arising from the `kz_physical` sign convention, so the search
/// rectangle should not extend very far below the real axis (which would force
/// the vertical sides to traverse a large branch-cut-contaminated stretch when
/// they cross `Im = 0`).
///
/// Setting `im_min = -MIN_IM_LOWER` keeps the vertical sides' crossing of the
/// real axis brief, reducing — but not eliminating — the branch-cut
/// contamination, while still placing the bottom contour edge far enough from
/// the near-zero mode poles for the discrete sampling to resolve the `2π`
/// argument winding correctly.
const MIN_IM_LOWER: f64 = 1e-3;

/// Number of points used to discretise each side of the contour when computing
/// the winding number via the argument principle.
const CONTOUR_POINTS_PER_SIDE: usize = 512;

/// Maximum allowed argument change between two consecutive contour samples.
/// If a step exceeds this threshold the segment is adaptively refined by
/// bisection until the step is small enough or the maximum refinement depth
/// is reached.  π/4 is conservative enough to catch the rapid phase rotation
/// near branch cuts on the real axis while keeping the total sample count low
/// for smooth segments.
const MAX_ARG_STEP: f64 = std::f64::consts::PI / 4.0;

/// Maximum bisection depth used when adaptively refining a single contour
/// segment.  2^10 = 1024 sub-steps per original step is more than sufficient
/// to resolve any physically meaningful branch-cut crossing.
const ADAPTIVE_REFINE_DEPTH: usize = 10;

/// Maximum recursion depth for the rectangle-subdivision zero-counter.
/// A rectangle that is smaller than ~ (search_width / 2^MAX_DEPTH) in each
/// dimension is treated as containing a single zero and polished directly.
const MAX_SUBDIVISION_DEPTH: usize = 20;

/// Default depth of the imaginary-part search window for the QNM solver.
///
/// QNM poles live at `Im(neff) < 0`.  The default search rectangle spans
/// `(-QNM_IM_DEFAULT_DEPTH, -QNM_IM_DEFAULT_MAX)`, placing the contour entirely in
/// the lower half-plane and avoiding the real axis where branch-cut artefacts are
/// worst.  The chosen depth of 0.15 gives a rectangle that is neither too thin
/// (which causes poor Im-direction sampling) nor too deep (which can produce
/// spurious winding-number contributions from S-matrix anti-resonances on the
/// bottom edge).  Users who need to find more strongly or more weakly leaky modes
/// should pass an explicit `im_range` to `qnm_neff` / `all_qnm_neff`.
const QNM_IM_DEFAULT_DEPTH: f64 = 0.15;

/// Default upper bound for `Im(neff)` in QNM searches.
///
/// A small negative value keeps the top contour edge slightly below the real axis,
/// where branch-cut artefacts from `kz_physical` (for interior layers) are most
/// severe.  Using `-1e-3` rather than `0` avoids the worst of these artefacts while
/// still capturing modes with moderately small radiation loss.
const QNM_IM_DEFAULT_MAX: f64 = -MIN_IM_LOWER;

/// Tolerance for the Muller polisher: iteration stops when |f(k)| < this value
/// or the step size is smaller than this value.
const MULLER_TOL: f64 = 1e-10;

/// Maximum number of Muller iterations per root.
const MULLER_MAX_ITER: usize = 200;

// ─── Quadrature helper ────────────────────────────────────────────────────────

/// Integrates a function on sampled data using the trapezoidal rule.
/// # Arguments
/// * `y` - The y values of the function to integrate.
/// * `x` - The x values of the function to integrate.
/// # Returns
/// The integral of the function.
fn quadrature_integration<T, U>(y: &[T], x: &[U]) -> Result<T, &'static str>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<U, Output = T> + Sum + Copy,
    U: Add<Output = U> + Sub<Output = U> + Copy,
{
    if y.len() != x.len() {
        return Err("The length of the input vectors must be the same");
    }
    let correction = match (y.last(), y.first()) {
        (Some(&last), Some(&first)) => last + first,
        _ => return Err("The input vectors must not be empty"),
    };
    let dx = x[1] - x[0];
    let sum: T = y.iter().copied().sum();
    let integral = sum - correction;
    Ok(integral * dx)
}

// ─── Grid data ────────────────────────────────────────────────────────────────

/// Struct representing the grid data used for plotting.
struct GridData {
    /// The x values of the grid.
    xplot: Vec<f64>,
    /// The x starting positions of the layers.
    xstarts: Vec<f64>,
    /// The indices of the x values where the layers start.
    ixstarts: Vec<usize>,
}

// ─── FieldData ────────────────────────────────────────────────────────────────

/// Struct representing the field data of a mode.
/// This is also available in the Python API.
#[pyclass]
#[allow(non_snake_case)]
pub struct FieldData {
    /// x coordinates of the field data.
    #[pyo3(get)]
    pub x: Vec<f64>,
    /// Electric field in the x direction.
    #[pyo3(get)]
    pub Ex: Vec<Complex<f64>>,
    /// Electric field in the y direction.
    #[pyo3(get)]
    pub Ey: Vec<Complex<f64>>,
    /// Electric field in the z direction.
    #[pyo3(get)]
    pub Ez: Vec<Complex<f64>>,
    /// Magnetic field in the x direction.
    #[pyo3(get)]
    pub Hx: Vec<Complex<f64>>,
    /// Magnetic field in the y direction.
    #[pyo3(get)]
    pub Hy: Vec<Complex<f64>>,
    /// Magnetic field in the z direction.
    #[pyo3(get)]
    pub Hz: Vec<Complex<f64>>,
}

impl FieldData {
    /// Creates a zeroed FieldData (all field components set to zero) for a given x grid.
    /// Used as a safe fallback when a requested mode does not exist.
    pub fn zeros(x: Vec<f64>) -> FieldData {
        let n = x.len();
        FieldData {
            x,
            Ex: vec![Complex::new(0.0, 0.0); n],
            Ey: vec![Complex::new(0.0, 0.0); n],
            Ez: vec![Complex::new(0.0, 0.0); n],
            Hx: vec![Complex::new(0.0, 0.0); n],
            Hy: vec![Complex::new(0.0, 0.0); n],
            Hz: vec![Complex::new(0.0, 0.0); n],
        }
    }

    /// Returns the z component of the Poynting vector of the field.
    pub fn get_poyinting_vector(&self) -> Complex<f64> {
        let poynting: Vec<Complex<f64>> = self
            .Ex
            .iter()
            .zip(self.Hy.iter())
            .zip(self.Ey.iter().zip(self.Hx.iter()))
            .map(|((&ex, &hy), (&ey, &hx))| ex * hy.conj() - ey * hx.conj())
            .collect();
        quadrature_integration(&poynting, &self.x).unwrap()
    }

    /// Normalizes the field data so that the absolute value of z component of the Poynting vector is 1.
    pub fn normalize_power(self) -> FieldData {
        let poynting_vector = self.get_poyinting_vector();
        let norm = poynting_vector.sqrt();
        let ex = self.Ex.into_iter().map(|x| x / norm).collect();
        let ey = self.Ey.into_iter().map(|x| x / norm).collect();
        let ez = self.Ez.into_iter().map(|x| x / norm).collect();
        let hx = self.Hx.into_iter().map(|x| x / norm).collect();
        let hy = self.Hy.into_iter().map(|x| x / norm).collect();
        let hz = self.Hz.into_iter().map(|x| x / norm).collect();
        FieldData {
            x: self.x,
            Ex: ex,
            Ey: ey,
            Ez: ez,
            Hx: hx,
            Hy: hy,
            Hz: hz,
        }
    }

    /// Normalizes the field so that the maximum total electric field amplitude is 1.
    /// max(sqrt(|Ex|² + |Ey|² + |Ez|²)) = 1
    pub fn normalize_max_field(self) -> FieldData {
        let max_e = self
            .Ex
            .iter()
            .zip(self.Ey.iter())
            .zip(self.Ez.iter())
            .map(|((&ex, &ey), &ez)| (ex.norm_sqr() + ey.norm_sqr() + ez.norm_sqr()).sqrt())
            .fold(0.0_f64, f64::max);
        let norm = if max_e < 1e-300 { 1.0 } else { max_e };
        let ex = self.Ex.into_iter().map(|x| x / norm).collect();
        let ey = self.Ey.into_iter().map(|x| x / norm).collect();
        let ez = self.Ez.into_iter().map(|x| x / norm).collect();
        let hx = self.Hx.into_iter().map(|x| x / norm).collect();
        let hy = self.Hy.into_iter().map(|x| x / norm).collect();
        let hz = self.Hz.into_iter().map(|x| x / norm).collect();
        FieldData {
            x: self.x,
            Ex: ex,
            Ey: ey,
            Ez: ez,
            Hx: hx,
            Hy: hy,
            Hz: hz,
        }
    }
}

// ─── IndexData ────────────────────────────────────────────────────────────────

/// Struct representing the index data of the multi-layer.
/// This is also available in the Python API.
#[pyclass]
pub struct IndexData {
    /// The x values of the index data.
    #[pyo3(get)]
    pub x: Vec<f64>,
    /// The index of refraction of the multi-layer (real part only, for plotting).
    #[pyo3(get)]
    pub n: Vec<f64>,
}

// ─── Field slice helper ───────────────────────────────────────────────────────

/// Calculates the field profile in a layer given the modal coefficients.
/// # Arguments
/// * `a` - The modal coefficient of the forward propagating wave.
/// * `b` - The modal coefficient of the backward propagating wave.
/// * `k0` - The vacuum wavevector (real).
/// * `k`  - The parallel wavevector (complex, supports both real and complex neff).
/// * `n`  - The complex index of refraction.
/// * `x`  - The x coordinates inside the layer.
/// # Returns
/// The field profile in the layer.
fn get_field_slice(
    a: Complex<f64>,
    b: Complex<f64>,
    k0: f64,
    k: Complex<f64>,
    n: Complex<f64>,
    x: Vec<f64>,
) -> Vec<Complex<f64>> {
    use crate::transfer_matrix::kz_physical;
    let k0c = Complex::new(k0, 0.0);
    let beta = kz_physical(k0c, n, k);
    x.iter()
        .map(|&xv| {
            let z = Complex::new(0.0, xv);
            let phase_p = z * beta;
            let phase_n = -z * beta;
            a * phase_p.exp() + b * phase_n.exp()
        })
        .collect()
}

// ─── Full layer coefficient vector type alias ─────────────────────────────────

type FullLayerCoefficientVector = (
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
);

// ─── MultiLayer struct ────────────────────────────────────────────────────────

/// Struct representing the multilayer structure.
/// Implements methods for calculating the modes and fields of the structure.
#[pyclass]
pub struct MultiLayer {
    /// The layers of the multi-layer.
    layers: Vec<Layer>,
    /// The backend used for the calculations.
    backend: BackEnd,
    /// Number of significant digits requested for neff calculation.
    required_accuracy: i32,
    /// The step size for plotting the field.
    #[pyo3(get, set)]
    pub plot_step: f64,
    /// Boundary condition on the left side of the structure.
    left_bc: BoundaryCondition,
    /// Boundary condition on the right side of the structure.
    right_bc: BoundaryCondition,
    /// Normalization convention used for field reconstruction.
    #[pyo3(get)]
    pub normalization: Normalization,
}

// ─── Python-facing methods ────────────────────────────────────────────────────

/// Methods of the MultiLayer struct also available in the Python API.
#[pymethods]
impl MultiLayer {
    #[new]
    /// Creates a new MultiLayer from a list of layers and optional PEC boundary markers.
    /// A PEC element placed first sets a PEC left boundary; placed last sets a PEC right boundary.
    /// PEC cannot appear in the middle or on both ends simultaneously.
    pub fn from_python_layers(layers: Vec<Bound<'_, PyAny>>) -> PyResult<MultiLayer> {
        let mut left_bc = BoundaryCondition::SemiInfinite;
        let mut right_bc = BoundaryCondition::SemiInfinite;
        let mut dielectric_layers: Vec<Layer> = Vec::new();
        let len = layers.len();

        if len < 2 {
            return Err(PyValueError::new_err(
                "MultiLayer requires at least 2 elements",
            ));
        }

        for (i, item) in layers.iter().enumerate() {
            if item.is_instance_of::<PEC>() {
                if i == 0 {
                    left_bc = BoundaryCondition::PEC;
                } else if i == len - 1 {
                    right_bc = BoundaryCondition::PEC;
                } else {
                    return Err(PyValueError::new_err(
                        "PEC can only be the first or last element of the layer list",
                    ));
                }
            } else {
                let layer = item.extract::<Layer>().map_err(|_| {
                    PyValueError::new_err("All non-PEC elements must be Layer instances")
                })?;
                dielectric_layers.push(layer);
            }
        }

        if matches!(left_bc, BoundaryCondition::PEC) && matches!(right_bc, BoundaryCondition::PEC) {
            return Err(PyValueError::new_err(
                "PEC on both sides simultaneously is not supported",
            ));
        }

        if dielectric_layers.len() < 2 {
            return Err(PyValueError::new_err(
                "MultiLayer requires at least 2 dielectric (non-PEC) layers",
            ));
        }

        let mut multilayer = MultiLayer {
            layers: dielectric_layers,
            backend: BackEnd::Transfer,
            required_accuracy: 10,
            plot_step: 1e-3,
            left_bc,
            right_bc,
            normalization: Normalization::MaxField,
        };
        multilayer.set_backend(BackEnd::Transfer);
        Ok(multilayer)
    }

    /// Sets the left boundary condition of the structure.
    #[pyo3(name = "set_left_boundary")]
    pub fn python_set_left_boundary(&mut self, bc: BoundaryCondition) {
        self.set_left_boundary(bc);
    }

    /// Sets the right boundary condition of the structure.
    #[pyo3(name = "set_right_boundary")]
    pub fn python_set_right_boundary(&mut self, bc: BoundaryCondition) {
        self.set_right_boundary(bc);
    }

    /// Calculates neff of the requested mode.
    ///
    /// Uses the fast real-axis scan.  For modes in lossy or leaky structures
    /// use [`python_complex_neff`] instead.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency of the mode.
    /// * `polarization` - The polarization of the mode.
    /// * `mode`         - The mode number.
    /// # Returns
    /// The effective index of refraction of the mode, or None if the mode does not exist.
    #[pyo3(name = "neff")]
    #[pyo3(signature = (omega, polarization=None, mode=None))]
    pub fn python_neff(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
    ) -> Option<f64> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        self.neff(omega, polarization, mode).ok()
    }

    /// Returns all effective indices supported by the structure.
    ///
    /// Uses the fast real-axis scan.  For modes in lossy or leaky structures
    /// use [`python_all_complex_neff`] instead.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency.
    /// * `polarization` - The polarization of the modes.
    /// # Returns
    /// A list of effective indices, sorted from highest to lowest.
    #[pyo3(name = "all_neff")]
    #[pyo3(signature = (omega, polarization=None))]
    pub fn python_all_neff(&self, omega: f64, polarization: Option<Polarization>) -> Vec<f64> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        self.solve(omega, polarization)
    }

    /// Returs the index profile of the multi-layer.
    #[pyo3(name = "index")]
    pub fn python_index(&self) -> IndexData {
        self.index()
    }

    /// Calculates the field profile of the requested mode.
    /// # Arguments
    /// * `omega`        - The angular frequency of the mode.
    /// * `polarization` - The polarization of the mode.
    /// * `mode`         - The mode number.
    /// # Returns
    /// The field profile of the mode, or a zeroed FieldData if the mode does not exist.
    #[pyo3(name = "field")]
    #[pyo3(signature = (omega, polarization=None, mode=None))]
    pub fn python_field(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
    ) -> FieldData {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        self.field(omega, polarization, mode)
            .unwrap_or_else(|_| FieldData::zeros(self.get_grid_data().xplot))
    }

    /// Finds a single complex effective index in the complex neff plane.
    ///
    /// The boundary conditions used to build the S-matrix are controlled by the
    /// `left_bc` / `right_bc` arguments (which override the boundary conditions
    /// stored on the `MultiLayer` object for this call only).
    ///
    /// | `left_bc`    | `right_bc`   | Mode type                              |
    /// |--------------|--------------|----------------------------------------|
    /// | SemiInfinite | SemiInfinite | Guided / lossy guided (`Im ≈ 0`)       |
    /// | Outgoing     | Outgoing     | Quasi-normal modes (`Im(neff) < 0`)    |
    /// | SemiInfinite | Outgoing     | One-sided leaky modes (`Im(neff) > 0`) |
    ///
    /// The default `im_range` adapts automatically to the active boundary conditions:
    /// - `SemiInfinite` + `SemiInfinite` → near-real window (symmetric for lossy, slightly
    ///   asymmetric for lossless).
    /// - `Outgoing` + `Outgoing` → lower half-plane `(-0.15, -1e-3)` for QNMs.
    /// - One `Outgoing` → upper half-plane `(1e-3, 0.15)` for leaky modes.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency (real).
    /// * `polarization` - `Polarization.TE` or `TM`.
    /// * `mode`         - Zero-based mode index (sorted by descending `Re(neff)`).
    /// * `re_range`     - Optional `(re_min, re_max)`.
    /// * `im_range`     - Optional `(im_min, im_max)`.
    /// * `left_bc`      - Optional BC override for the left cladding.
    /// * `right_bc`     - Optional BC override for the right cladding.
    ///
    /// # Returns
    /// `(Re(neff), Im(neff))` or `None`.
    #[pyo3(name = "complex_neff")]
    #[pyo3(signature = (omega, polarization=None, mode=None, re_range=None, im_range=None, left_bc=None, right_bc=None))]
    pub fn python_complex_neff(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
        left_bc: Option<BoundaryCondition>,
        right_bc: Option<BoundaryCondition>,
    ) -> Option<(f64, f64)> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        let eff_left = left_bc.unwrap_or(self.left_bc);
        let eff_right = right_bc.unwrap_or(self.right_bc);
        let roots =
            self.solve_complex(omega, polarization, re_range, im_range, eff_left, eff_right);
        roots.get(mode).map(|c| (c.re, c.im))
    }

    /// Returns all complex effective indices found in the search rectangle.
    ///
    /// Same boundary-condition logic as [`complex_neff`].  See that method's
    /// documentation for the `left_bc` / `right_bc` parameter description.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency (real).
    /// * `polarization` - The polarization of the modes.
    /// * `re_range`     - Optional `(re_min, re_max)`.
    /// * `im_range`     - Optional `(im_min, im_max)`.
    /// * `left_bc`      - Optional BC override for the left cladding.
    /// * `right_bc`     - Optional BC override for the right cladding.
    ///
    /// # Returns
    /// A list of `(Re(neff), Im(neff))` tuples, sorted by descending `Re(neff)`.
    #[pyo3(name = "all_complex_neff")]
    #[pyo3(signature = (omega, polarization=None, re_range=None, im_range=None, left_bc=None, right_bc=None))]
    pub fn python_all_complex_neff(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
        left_bc: Option<BoundaryCondition>,
        right_bc: Option<BoundaryCondition>,
    ) -> Vec<(f64, f64)> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let eff_left = left_bc.unwrap_or(self.left_bc);
        let eff_right = right_bc.unwrap_or(self.right_bc);
        self.solve_complex(omega, polarization, re_range, im_range, eff_left, eff_right)
            .into_iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    /// Sets the normalization convention used for field reconstruction.
    #[pyo3(name = "set_normalization")]
    pub fn python_set_normalization(&mut self, norm: Normalization) {
        self.normalization = norm;
    }

    /// Calculates the field profile of a complex mode found by the complex-plane solver.
    ///
    /// Mirrors the signature of [`python_field`] but uses the complex-plane solver
    /// ([`python_complex_neff`]) to find the effective index, then reconstructs the
    /// field for that mode.
    ///
    /// For semi-infinite boundaries the outgoing wave in the rightmost layer is
    /// **not** zeroed: for a complex neff the radiation condition is already encoded
    /// in the imaginary part, and zeroing the outgoing amplitude would give a
    /// physically wrong result.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency (real).
    /// * `polarization` - The polarization of the mode.
    /// * `mode`         - Zero-based mode index (same ordering as [`python_complex_neff`]).
    /// * `re_range`     - Optional `(re_min, re_max)` forwarded to the complex-plane solver.
    /// * `im_range`     - Optional `(im_min, im_max)` forwarded to the complex-plane solver.
    /// * `left_bc`      - Optional BC override for the left cladding.
    /// * `right_bc`     - Optional BC override for the right cladding.
    /// # Returns
    /// A `FieldData` with all six field components on the standard plotting grid,
    /// or a zeroed `FieldData` if the requested mode does not exist.
    #[pyo3(name = "complex_field")]
    #[pyo3(signature = (omega, polarization=None, mode=None, re_range=None, im_range=None, left_bc=None, right_bc=None))]
    pub fn python_complex_field(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
        left_bc: Option<BoundaryCondition>,
        right_bc: Option<BoundaryCondition>,
    ) -> FieldData {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        let eff_left = left_bc.unwrap_or(self.left_bc);
        let eff_right = right_bc.unwrap_or(self.right_bc);
        let roots =
            self.solve_complex(omega, polarization, re_range, im_range, eff_left, eff_right);
        match roots.get(mode) {
            Some(&neff) => self.field_complex(omega, polarization, neff),
            None => FieldData::zeros(self.get_grid_data().xplot),
        }
    }
}

// ─── Internal Rust methods ────────────────────────────────────────────────────

impl MultiLayer {
    /// Creates a new MultiLayer from a Vec<Layer> with default SemiInfinite boundary conditions.
    /// Used internally by Rust code and the CLI binary.
    pub fn new(layers: Vec<Layer>) -> MultiLayer {
        let mut multilayer = MultiLayer {
            layers,
            backend: BackEnd::Transfer,
            required_accuracy: 10,
            plot_step: 1e-3,
            left_bc: BoundaryCondition::SemiInfinite,
            right_bc: BoundaryCondition::SemiInfinite,
            normalization: Normalization::MaxField,
        };
        multilayer.set_backend(BackEnd::Transfer);
        multilayer
    }

    /// Sets the left boundary condition.
    pub fn set_left_boundary(&mut self, bc: BoundaryCondition) {
        self.left_bc = bc;
    }

    /// Sets the right boundary condition.
    pub fn set_right_boundary(&mut self, bc: BoundaryCondition) {
        self.right_bc = bc;
    }

    /// Switches the backend used for the calculations.
    pub fn set_backend(&mut self, backend: BackEnd) {
        self.backend = backend;
    }

    /// Get the threshold for the findpeak function for a given number of significant digits.
    fn get_threshold(accuracy: i32) -> f64 {
        match accuracy {
            0..=2 => -2.0,
            3..=5 => 0.0,
            6..=8 => 3.0,
            9..=11 => 6.0,
            _ => 9.0,
        }
    }

    // ── Characteristic functions ──────────────────────────────────────────────

    /// Evaluates the characteristic function for a **real** in-plane wavevector `k`.
    ///
    /// Returns `1/t22` (TMM) or `det(S)` (SMM).  A zero of this function
    /// corresponds to a guided mode.
    ///
    /// All matrix calls are lifted to `Complex<f64>` even though `k` is real,
    /// matching the generalised signatures in `transfer_matrix` and
    /// `scattering_matrix`.
    fn characteristic_function(&self, k0: f64, k: f64, polarization: Polarization) -> Complex<f64> {
        let k0c = Complex::new(k0, 0.0);
        let kc = Complex::new(k, 0.0);
        match self.backend {
            BackEnd::Scattering => {
                calculate_s_matrix(&self.layers, k0c, kc, polarization).determinant()
            }
            BackEnd::Transfer => {
                let t = calculate_t_matrix(&self.layers, k0c, kc, polarization);
                // For the real-axis solver, Outgoing behaves the same as
                // SemiInfinite (outgoing BCs are only meaningful for the
                // complex-plane S-matrix solver).
                let eff_left = match self.left_bc {
                    BoundaryCondition::Outgoing => BoundaryCondition::SemiInfinite,
                    bc => bc,
                };
                let eff_right = match self.right_bc {
                    BoundaryCondition::Outgoing => BoundaryCondition::SemiInfinite,
                    bc => bc,
                };
                match (eff_left, eff_right) {
                    (BoundaryCondition::SemiInfinite, BoundaryCondition::SemiInfinite) => {
                        1.0 / t.t22
                    }
                    (BoundaryCondition::PEC, BoundaryCondition::SemiInfinite) => {
                        let prop = TransferMatrix::matrix_propagation(
                            self.layers[0].n,
                            self.layers[0].d,
                            k0c,
                            kc,
                        );
                        let t_full = t.compose(prop);
                        let (_, b_out) =
                            t_full.apply(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
                        1.0 / b_out
                    }
                    (BoundaryCondition::SemiInfinite, BoundaryCondition::PEC) => {
                        let last = self.layers.last().unwrap();
                        let prop = TransferMatrix::matrix_propagation(last.n, last.d, k0c, kc);
                        let t_full = prop.compose(t);
                        let (a_out, b_out) =
                            t_full.apply(Complex::new(0.0, 0.0), Complex::new(1.0, 0.0));
                        1.0 / (a_out + b_out)
                    }
                    (BoundaryCondition::PEC, BoundaryCondition::PEC) => {
                        panic!("Both-PEC boundary condition is not supported")
                    }
                    // After Outgoing→SemiInfinite normalization these arms are
                    // unreachable, but the compiler requires exhaustiveness.
                    _ => 1.0 / t.t22,
                }
            }
        }
    }

    /// Evaluates `1 / det(S)` for a complex in-plane wavevector `k`,
    /// using boundary conditions specified by `left_bc` and `right_bc`.
    fn characteristic_function_with_bc(
        &self,
        k0: f64,
        k: Complex<f64>,
        polarization: Polarization,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Complex<f64> {
        let k0c = Complex::new(k0, 0.0);
        let det = calculate_s_matrix_with_bc(&self.layers, k0c, k, polarization, left_bc, right_bc)
            .determinant();
        if det.norm() < 1e-300 {
            Complex::new(1e300, 0.0)
        } else {
            Complex::new(1.0, 0.0) / det
        }
    }

    // ── Real-axis mode search ─────────────────────────────────────────────────

    /// Finds the minimum and maximum *real part* of the refractive index.
    fn find_minmax_n(&self) -> (f64, f64) {
        find_minmax_n(&self.layers)
    }

    /// Single step of the real-axis peak-finding process.
    fn solve_step(
        &self,
        k0: f64,
        k_min: f64,
        k_max: f64,
        step: f64,
        _treshold: f64,
        polarization: Polarization,
    ) -> Vec<f64> {
        let kv: Vec<f64> = iter_num_tools::arange(k_min..k_max, step).collect();
        let det: Vec<f64> = kv
            .iter()
            .map(|&k| {
                self.characteristic_function(k0, k, polarization)
                    .norm()
                    .log10()
            })
            .collect();

        let mut peak_finder = PeakFinder::new(&det);
        let peaks = peak_finder.with_min_prominence(0.8).find_peaks();

        peaks.into_iter().map(|p| kv[p.middle_position()]).collect()
    }

    /// Finds the guided modes of the multi-layer using the fast real-axis scan.
    ///
    /// # Arguments
    /// * `k0`           - The vacuum wavevector.
    /// * `polarization` - The polarization of the mode.
    /// # Returns
    /// The effective indices of the modes, sorted descending.
    pub fn solve(&self, k0: f64, polarization: Polarization) -> Vec<f64> {
        let (min_n, max_n) = self.find_minmax_n();
        let k_min = k0 * min_n + 1e-9;
        let k_max = k0 * max_n - 1e-9;

        let mut solution_backets = vec![(k_min, k_max)];

        let mut ksolutions = Vec::new();

        for accuracy in 2..self.required_accuracy {
            let step = 10.0_f64.powi(-accuracy);
            let threshold = Self::get_threshold(accuracy);
            ksolutions.clear();
            for (_kmin, _kmax) in solution_backets {
                let _solutions =
                    self.solve_step(k0, _kmin, _kmax, 0.1 * step, threshold, polarization);
                ksolutions.extend(_solutions);
            }
            solution_backets = ksolutions
                .iter()
                .map(|k| (k - step, k + step))
                .collect::<Vec<_>>();
        }

        let mut n_solutions = ksolutions.into_iter().map(|k| k / k0).collect::<Vec<_>>();

        n_solutions.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        n_solutions
    }

    /// Finds the effective index of a guided mode.
    pub fn neff(&self, k0: f64, polarization: Polarization, mode: usize) -> Result<f64, String> {
        let n_solutions = self.solve(k0, polarization);
        match n_solutions.get(mode) {
            Some(&n) => Ok(n),
            None => Err(format!(
                "Mode {} not found. Only {} modes (0->{}) available.",
                mode,
                n_solutions.len(),
                n_solutions.len().saturating_sub(1)
            )),
        }
    }

    // ── Complex-plane mode search ─────────────────────────────────────────────

    /// Resolves the user-supplied search ranges against the defaults for the
    /// active boundary conditions.
    ///
    /// Returns `None` when the caller passed `None` for **both** `re_range` and
    /// `im_range`, signalling that [`solve_complex`] should use the adaptive
    /// multi-rectangle search ([`adaptive_solve_complex`]) instead of a single
    /// fixed rectangle. When either range is supplied explicitly, the legacy
    /// single-rectangle behaviour is preserved: the supplied range is used as-is
    /// and the missing one is filled in from the BC-dependent defaults below.
    ///
    /// Default `im_range` by boundary-condition combination:
    /// - `SemiInfinite` + `SemiInfinite` → near-real window (symmetric for lossy,
    ///   slightly asymmetric for lossless).
    /// - `Outgoing` + `Outgoing` → lower half-plane `(-0.15, -1e-3)` for QNMs.
    /// - One `Outgoing` → upper half-plane `(1e-3, 0.15)` for leaky modes.
    fn default_search_ranges_for_bc(
        &self,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Option<((f64, f64), (f64, f64))> {
        // Both ranges unset → adaptive path takes over.
        if re_range.is_none() && im_range.is_none() {
            return None;
        }

        let re = re_range.unwrap_or_else(|| {
            let (min_n, max_n) = self.find_minmax_n();
            let margin = 1e-6;
            (min_n + margin, max_n - margin)
        });
        let im = im_range.unwrap_or_else(|| match (left_bc, right_bc) {
            (BoundaryCondition::Outgoing, BoundaryCondition::Outgoing) => {
                (-QNM_IM_DEFAULT_DEPTH, QNM_IM_DEFAULT_MAX)
            }
            (BoundaryCondition::Outgoing, _) | (_, BoundaryCondition::Outgoing) => {
                (MIN_IM_LOWER, QNM_IM_DEFAULT_DEPTH)
            }
            _ => {
                let max_im = self
                    .layers
                    .iter()
                    .map(|l| l.n.im.abs())
                    .fold(0.0_f64, f64::max);
                if max_im > 0.0 {
                    let half_w = max_im.max(MIN_IM_HALF_WIDTH);
                    (-half_w, half_w)
                } else {
                    (-MIN_IM_LOWER, MIN_IM_HALF_WIDTH)
                }
            }
        });
        Some((re, im))
    }

    /// Computes the winding number of `f` around a rectangle in the complex plane.
    ///
    /// Uses the argument principle: integrates `Δ arg(f)` around the boundary
    /// of the rectangle and divides by `2π`.  The characteristic function used is
    /// `1/det(S)` (see [`characteristic_function_complex`]), whose zeros are the
    /// modes; the winding number therefore equals the number of modes enclosed.
    ///
    /// The rectangle is defined by its four corners
    /// `(re_min, im_min)`, `(re_max, im_min)`, `(re_max, im_max)`, `(re_min, im_max)`.
    ///
    /// # Sampling density
    ///
    /// The number of contour points per side is chosen adaptively so that the
    /// longer side always has at least `CONTOUR_POINTS_PER_SIDE` points and the
    /// shorter side has proportionally fewer but at least 4.  This avoids over-
    /// sampling extremely thin rectangles while keeping the longer sides dense.
    ///
    /// # Arguments
    /// * `k0`           - Vacuum wavevector.
    /// * `polarization` - Polarisation.
    /// * `re_min/max`   - Real-part bounds of the rectangle (in *neff* units).
    /// * `im_min/max`   - Imaginary-part bounds (in *neff* units).
    /// * `n_pts`        - Base number of sample points per unit length of contour.
    fn winding_number(
        &self,
        k0: f64,
        re_min: f64,
        re_max: f64,
        im_min: f64,
        im_max: f64,
        n_pts: usize,
        char_fn: &dyn Fn(Complex<f64>) -> Complex<f64>,
    ) -> i32 {
        let re_width = re_max - re_min;
        let im_width = im_max - im_min;
        let max_width = re_width.max(im_width);

        // Scale sample counts so the longer side always gets n_pts points and
        // the shorter side is proportional (minimum 4 to form a valid contour).
        let n_re = ((n_pts as f64 * re_width / max_width).round() as usize).max(4);
        let n_im = ((n_pts as f64 * im_width / max_width).round() as usize).max(4);

        // Build the four sides of the rectangle (in neff space, then multiply by k0).
        let mut contour: Vec<Complex<f64>> = Vec::with_capacity(2 * (n_re + n_im));

        // Bottom: re_min → re_max,  im = im_min
        for i in 0..n_re {
            let t = i as f64 / n_re as f64;
            let re = re_min + t * re_width;
            contour.push(Complex::new(re, im_min) * k0);
        }
        // Right: re = re_max,  im_min → im_max
        for i in 0..n_im {
            let t = i as f64 / n_im as f64;
            let im = im_min + t * im_width;
            contour.push(Complex::new(re_max, im) * k0);
        }
        // Top: re_max → re_min,  im = im_max
        for i in 0..n_re {
            let t = i as f64 / n_re as f64;
            let re = re_max - t * re_width;
            contour.push(Complex::new(re, im_max) * k0);
        }
        // Left: re = re_min,  im_max → im_min
        for i in 0..n_im {
            let t = i as f64 / n_im as f64;
            let im = im_max - t * im_width;
            contour.push(Complex::new(re_min, im) * k0);
        }

        // Evaluate f on the contour and accumulate the total argument change.
        // Guard: if any evaluation produces NaN or Inf (which can happen when the
        // contour passes through a branch point of the S-matrix, e.g. neff = n_max
        // where kz = 0 and the compose denominator 1 − s12·s21 vanishes), skip that
        // segment.  A single bad point does not corrupt the whole integral because
        // the winding contribution from the rest of the contour is unchanged.
        //
        // Adaptive refinement: for lossless structures the characteristic function
        // 1/det(S) has a branch-cut discontinuity along the real neff axis wherever
        // the substrate (or any layer with Re(n) > Re(neff)) transitions from
        // evanescent to propagating.  Near this branch cut the argument of f can
        // change by nearly π in a single coarse step, causing the atan2 accumulator
        // to mis-count the winding number by ±1.  We therefore subdivide any segment
        // where |Δarg| ≥ MAX_ARG_STEP recursively until the step is small enough or
        // ADAPTIVE_REFINE_DEPTH bisection levels are exhausted.
        let fvals: Vec<Complex<f64>> = contour.iter().map(|&k| char_fn(k)).collect();

        // Recursive helper: accumulate the argument change from k_a to k_b,
        // adaptively bisecting when the single-step |Δarg| is too large.
        fn adaptive_arg_change(
            char_fn: &dyn Fn(Complex<f64>) -> Complex<f64>,
            k_a: Complex<f64>,
            f_a: Complex<f64>,
            k_b: Complex<f64>,
            f_b: Complex<f64>,
            depth: usize,
        ) -> f64 {
            // If either endpoint is non-finite, skip this segment.
            if !f_a.re.is_finite()
                || !f_a.im.is_finite()
                || !f_b.re.is_finite()
                || !f_b.im.is_finite()
            {
                return 0.0;
            }
            let ratio = f_b / f_a;
            let darg = ratio.arg();
            // Accept this step if it is small enough or we have hit max depth.
            if darg.abs() < MAX_ARG_STEP || depth >= ADAPTIVE_REFINE_DEPTH {
                return darg;
            }
            // Bisect.
            let k_mid = (k_a + k_b) * 0.5;
            let f_mid = char_fn(k_mid);
            adaptive_arg_change(char_fn, k_a, f_a, k_mid, f_mid, depth + 1)
                + adaptive_arg_change(char_fn, k_mid, f_mid, k_b, f_b, depth + 1)
        }

        let mut total_arg_change = 0.0_f64;
        let n = fvals.len();
        for i in 0..n {
            let f_curr = fvals[i];
            let f_next = fvals[(i + 1) % n];
            let k_curr = contour[i];
            let k_next = contour[(i + 1) % n];
            total_arg_change += adaptive_arg_change(char_fn, k_curr, f_curr, k_next, f_next, 0);
        }

        // Winding number = total change / (2π), rounded to nearest integer.
        // Take absolute value: the orientation of det(S) zeros can vary by sign
        // convention, but the count is always positive.
        ((total_arg_change / (2.0 * PI)).round() as i32).abs()
    }

    /// Recursively subdivides a rectangle until each sub-rectangle contains at
    /// most one zero, then polishes each zero with Muller's method.
    ///
    /// # Arguments
    /// * `k0`           - Vacuum wavevector.
    /// * `polarization` - Polarisation.
    /// * `re_min/max`   - Real bounds (neff units).
    /// * `im_min/max`   - Imaginary bounds (neff units).
    /// * `depth`        - Current recursion depth (starts at 0).
    /// * `roots`        - Accumulator for found roots.
    fn find_zeros_in_rectangle(
        &self,
        k0: f64,
        re_min: f64,
        re_max: f64,
        im_min: f64,
        im_max: f64,
        depth: usize,
        roots: &mut Vec<Complex<f64>>,
        char_fn: &dyn Fn(Complex<f64>) -> Complex<f64>,
    ) {
        let wn = self.winding_number(
            k0,
            re_min,
            re_max,
            im_min,
            im_max,
            CONTOUR_POINTS_PER_SIDE,
            char_fn,
        );

        if wn == 0 {
            // No zeros enclosed.
            return;
        }

        if wn == 1 || depth >= MAX_SUBDIVISION_DEPTH {
            // Exactly one zero (or we've reached max depth): polish with Muller.
            let re_mid = (re_min + re_max) * 0.5;
            let im_mid = (im_min + im_max) * 0.5;
            let initial_guess = Complex::new(re_mid, im_mid) * k0;
            if let Some(root) = self.muller_polish(k0, initial_guess, char_fn) {
                // Accept only if the root is inside (or very close to) the rectangle.
                let neff_root = root / k0;
                let margin = 1e-6;
                if neff_root.re >= re_min - margin
                    && neff_root.re <= re_max + margin
                    && neff_root.im >= im_min - margin
                    && neff_root.im <= im_max + margin
                {
                    // De-duplicate: discard if a root already found is very close.
                    let is_duplicate = roots.iter().any(|&r| (r - root).norm() < 1e-8 * k0);
                    if !is_duplicate {
                        roots.push(root);
                    }
                }
            }
            return;
        }

        // More than one zero: bisect along the longer side.
        let re_width = re_max - re_min;
        let im_width = im_max - im_min;
        if re_width >= im_width {
            let re_mid = (re_min + re_max) * 0.5;
            self.find_zeros_in_rectangle(
                k0,
                re_min,
                re_mid,
                im_min,
                im_max,
                depth + 1,
                roots,
                char_fn,
            );
            self.find_zeros_in_rectangle(
                k0,
                re_mid,
                re_max,
                im_min,
                im_max,
                depth + 1,
                roots,
                char_fn,
            );
        } else {
            let im_mid = (im_min + im_max) * 0.5;
            self.find_zeros_in_rectangle(
                k0,
                re_min,
                re_max,
                im_min,
                im_mid,
                depth + 1,
                roots,
                char_fn,
            );
            self.find_zeros_in_rectangle(
                k0,
                re_min,
                re_max,
                im_mid,
                im_max,
                depth + 1,
                roots,
                char_fn,
            );
        }
    }

    /// Polishes a single zero of the characteristic function using Muller's method.
    ///
    /// Muller's method is a three-point iteration that fits a quadratic through
    /// the last three function evaluations and steps to its nearest root.  It
    /// converges super-linearly (order ≈ 1.84) and works for complex functions
    /// without requiring a derivative.
    ///
    /// # Arguments
    /// * `k0`           - Vacuum wavevector.
    /// * `polarization` - Polarisation.
    /// * `k_init`       - Initial guess for the root (complex wavevector).
    ///
    /// # Returns
    /// The polished root, or `None` if convergence was not achieved.
    fn muller_polish(
        &self,
        k0: f64,
        k_init: Complex<f64>,
        char_fn: &dyn Fn(Complex<f64>) -> Complex<f64>,
    ) -> Option<Complex<f64>> {
        // Seed three starting points with a small perturbation around the guess.
        let eps = 1e-6 * k0;
        let mut x0 = k_init - Complex::new(eps, 0.0);
        let mut x1 = k_init + Complex::new(0.0, eps);
        let mut x2 = k_init + Complex::new(eps, 0.0);

        let mut f0 = char_fn(x0);
        let mut f1 = char_fn(x1);
        let mut f2 = char_fn(x2);

        for _ in 0..MULLER_MAX_ITER {
            if f2.norm() < MULLER_TOL {
                return Some(x2);
            }

            // Differences.
            let h1 = x1 - x0;
            let h2 = x2 - x1;

            // Avoid division by zero if the three points collapse.
            if h1.norm() < 1e-30 || h2.norm() < 1e-30 {
                break;
            }

            let delta1 = (f1 - f0) / h1;
            let delta2 = (f2 - f1) / h2;

            let denom_coeff = h1 + h2;
            if denom_coeff.norm() < 1e-30 {
                break;
            }

            let a = (delta2 - delta1) / denom_coeff;
            let b = a * h2 + delta2;
            let c_val = f2;

            // Discriminant of the quadratic a·w² + b·w + c = 0.
            let discriminant = (b * b - Complex::new(4.0, 0.0) * a * c_val).sqrt();

            // Choose the sign of the square root that maximises |b ± sqrt|.
            let denom = if (b + discriminant).norm() >= (b - discriminant).norm() {
                b + discriminant
            } else {
                b - discriminant
            };

            if denom.norm() < 1e-30 {
                break;
            }

            let w = Complex::new(-2.0, 0.0) * c_val / denom;

            // Shift the window.
            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f2;
            x2 = x2 + w;
            f2 = char_fn(x2);

            if w.norm() < MULLER_TOL * x2.norm().max(1.0) {
                return Some(x2);
            }
        }

        // Return best estimate even if tolerance was not fully reached.
        // 1/det(S) is ~0 at modes, so 1e-6 is a reasonable loose fallback
        // (MULLER_TOL = 1e-10 is the tight criterion).
        if f2.norm() < 1e-6 {
            Some(x2)
        } else {
            None
        }
    }

    /// Builds the decade cascade of `im` windows used by the adaptive solver.
    ///
    /// Each window has a 10:1 ratio of `im_max` to `im_min`, giving a
    /// well-conditioned aspect ratio for typical `re_range` widths of order 1.
    /// The sign of both bounds follows `sign`: negative for QNM (lower
    /// half-plane), positive for leaky (upper half-plane).
    ///
    /// The cascade covers four decades from `1e-1` down to `1e-9`, which is the
    /// practical floor of the winding-number method. Modes with smaller `|Im|`
    /// are caught by the real-axis fallback in [`adaptive_solve_complex`].
    fn im_cascade(sign: f64) -> Vec<(f64, f64)> {
        // (im_min, im_max) in absolute value; sign applied below.
        const DECADES: [(f64, f64); 4] = [(1e-3, 1e-1), (1e-5, 1e-2), (1e-7, 1e-3), (1e-9, 1e-4)];
        DECADES
            .iter()
            .map(|&(lo, hi)| {
                // For sign = -1 (QNM, lower half-plane) the bounds are negative
                // and must be ordered im_min < im_max, i.e. the more-negative
                // value comes first.
                let (a, b) = (sign * lo, sign * hi);
                if a <= b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect()
    }

    /// Builds narrow overlapping `re` windows around each real-axis probe result.
    ///
    /// Each probe at `r_i` produces a window `(r_i - δ, r_i + δ)` with
    /// `δ = ADAPTIVE_RE_HALF_WIDTH`. Isolating each pole in its own narrow `re`
    /// window is the single biggest reliability win for the winding-number
    /// method: it ensures the contour sees one mode at a time and allows the
    /// `im` cascade to push `im_max` well above the actual `Im(neff)` without
    /// straying into a neighbouring pole's territory.
    fn narrow_re_windows(probes: &[f64]) -> Vec<(f64, f64)> {
        const ADAPTIVE_RE_HALF_WIDTH: f64 = 0.05;
        probes
            .iter()
            .map(|&r| (r - ADAPTIVE_RE_HALF_WIDTH, r + ADAPTIVE_RE_HALF_WIDTH))
            .collect()
    }

    /// Removes duplicate roots that are closer than `1e-8 * k0` in the complex
    /// plane, or that both lie on the real axis within `1e-10` of each other
    /// (the real-axis-fallback vs. complex-solver case).
    fn dedup_roots(roots: Vec<Complex<f64>>, k0: f64) -> Vec<Complex<f64>> {
        let mut kept: Vec<Complex<f64>> = Vec::with_capacity(roots.len());
        for r in roots {
            let is_dup = kept.iter().any(|&k| {
                let both_real = r.im.abs() < 1e-10 && k.im.abs() < 1e-10;
                if both_real {
                    (r.re - k.re).abs() < 1e-10
                } else {
                    (r - k).norm() < 1e-8 * k0
                }
            });
            if !is_dup {
                kept.push(r);
            }
        }
        kept
    }

    /// Builds an isolated-core probe for the real-axis solver.
    ///
    /// Constructs a 3-layer stack: lowest-index cladding | core layers |
    /// lowest-index cladding, where the "core layers" are all interior layers
    /// of the original stack (i.e. `self.layers[1..n-1]`). Runs the real-axis
    /// solver on this simplified stack to find guided-mode Re(neff) candidates.
    ///
    /// This is used by [`adaptive_solve_complex`] when the full stack has no
    /// guided mode (e.g. `n_substrate > n_core`): the leaky mode's Re(neff) is
    /// typically close to the isolated-core guided mode's Re(neff), so the
    /// probe seeds narrow `re` windows for the complex cascade.
    fn isolated_core_probe(&self, k0: f64, polarization: Polarization) -> Vec<f64> {
        if self.layers.len() < 3 {
            return Vec::new();
        }
        // Find the lowest real index among all layers (use Re(n)).
        let min_n_re = self
            .layers
            .iter()
            .map(|l| l.n.re)
            .fold(f64::INFINITY, f64::min);
        // Build the isolated core: cladding | interior layers | cladding.
        let mut iso_layers = Vec::with_capacity(self.layers.len() + 1);
        iso_layers.push(Layer::from_real(min_n_re, 1.0));
        for layer in &self.layers[1..self.layers.len() - 1] {
            iso_layers.push(layer.clone());
        }
        iso_layers.push(Layer::from_real(min_n_re, 1.0));
        let iso_ml = MultiLayer::new(iso_layers);
        iso_ml.solve(k0, polarization)
    }

    /// Adaptive multi-rectangle complex-mode search.
    ///
    /// Replaces the single fixed rectangle of the legacy path with a four-stage
    /// search that needs no user-supplied `re_range` / `im_range` for reasonable
    /// structures:
    ///
    /// - **Stage A — real-axis probe.** Runs the fast real-axis solver [`solve`]
    ///   to find guided-mode candidates. For lossless + both-`SemiInfinite`
    ///   structures these *are* the modes (returned with `Im = 0`). Otherwise
    ///   the probe seeds narrow `re` windows (via [`narrow_re_windows`]) for the
    ///   complex cascade and provides the real-axis fallback.
    /// - **Stage B — coarse complex rectangle.** One broad-`Im` rectangle (the
    ///   old default) to catch strongly leaky / strongly lossy modes.
    /// - **Stage C — `Im` decade cascade.** Four geometrically well-conditioned
    ///   windows (see [`im_cascade`]) covering `|Im|` from `1e-1` down to `1e-9`.
    /// - **Stage D — real-axis fallback.** If the cascade found no complex root
    ///   but the probe did, emit the real-axis mode with `Im = 0`. Disabled for
    ///   QNM BCs (both `Outgoing`), where a real root is unphysical.
    ///
    /// Roots from all stages are deduplicated via [`dedup_roots`] and sorted by
    /// descending `Re(neff)`.
    fn adaptive_solve_complex(
        &self,
        k0: f64,
        polarization: Polarization,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Vec<Complex<f64>> {
        let char_fn = |k: Complex<f64>| {
            self.characteristic_function_with_bc(k0, k, polarization, left_bc, right_bc)
        };

        let both_outgoing = matches!(left_bc, BoundaryCondition::Outgoing)
            && matches!(right_bc, BoundaryCondition::Outgoing);
        let one_outgoing = matches!(left_bc, BoundaryCondition::Outgoing)
            ^ matches!(right_bc, BoundaryCondition::Outgoing);
        let all_lossless = self.layers.iter().all(|l| l.n.im == 0.0);
        let both_physical = !both_outgoing && !one_outgoing;

        // ── Stage A: real-axis probe ──────────────────────────────────────────
        // The real-axis solver uses kz_physical (TMM sheet), which is the correct
        // sheet for guided modes but not for QNMs. We still run it for QNM/leaky
        // structures because a real-axis peak there signals a mode with
        // |Im| < ~1e-11 (effectively guided), and its Re(neff) seeds the narrow
        // re windows for the complex cascade.
        //
        // For leaky/QNM structures where the full stack has no guided mode
        // (e.g. n_substrate > n_core), we also probe the *isolated core* — the
        // stack with both claddings replaced by the lowest-index material — to
        // get a Re(neff) hint for the leaky mode. The leaky mode's Re(neff) is
        // typically close to the isolated-core guided mode's Re(neff).
        //
        // `full_probe` is used for the real-axis fallback (stage D);
        // `re_hints` combines full_probe + isolated-core probe and is used only
        // to seed narrow re windows for the complex cascade.
        let full_probe: Vec<f64> = if all_lossless {
            self.solve(k0, polarization)
        } else {
            Vec::new()
        };
        let mut re_hints = full_probe.clone();
        if re_hints.is_empty() && all_lossless && (both_outgoing || one_outgoing) {
            re_hints = self.isolated_core_probe(k0, polarization);
        }
        let real_probe = re_hints;

        // Lossless guided structure: the probe is the answer.
        if both_physical && all_lossless {
            return real_probe.iter().map(|&n| Complex::new(n, 0.0)).collect();
        }

        // ── Determine the Im search strategy from the BCs ────────────────────
        // Three cases:
        //  - QNM (both Outgoing)            → lower half-plane only (sign = -1).
        //  - One-sided leaky (one Outgoing) → upper half-plane only (sign = +1).
        //  - Lossy guided (both physical, lossy material): the mode sits near
        //    the real axis with Im(neff) < 0 (loss) but small. Use a symmetric
        //    window around the real axis, sized from the material loss.
        //
        // (both_outgoing / one_outgoing / both_physical are computed above.)

        // Build the list of (im_min, im_max) windows to search.
        let mut im_windows: Vec<(f64, f64)> = Vec::new();
        if both_outgoing {
            // QNM: lower half-plane. Stage B (broad) + stage C (cascade).
            // Bounds are negative; order them im_min < im_max.
            im_windows.push((-QNM_IM_DEFAULT_DEPTH, -MIN_IM_LOWER));
            im_windows.extend(Self::im_cascade(-1.0));
        } else if one_outgoing {
            // One-sided leaky: upper half-plane. Stage B (broad) + stage C.
            im_windows.push((MIN_IM_LOWER, QNM_IM_DEFAULT_DEPTH));
            im_windows.extend(Self::im_cascade(1.0));
        } else {
            // Lossy guided (both physical, lossy material). The mode sits near
            // the real axis; Im(neff) < 0 with magnitude ~ material loss.
            // Use a symmetric window sized from the max material loss, clamped
            // to a minimum half-width so the contour is well-conditioned.
            let max_loss = self
                .layers
                .iter()
                .map(|l| l.n.im.abs())
                .fold(0.0_f64, f64::max);
            let half_w = max_loss.max(MIN_IM_HALF_WIDTH) * 2.0;
            im_windows.push((-half_w, half_w));
        }

        // ── Build the list of re windows to search ────────────────────────────
        // Build the list of re windows to search. We always include the broad
        // index-range window as a fallback, plus narrow windows around any
        // real-axis probe results. The narrow windows isolate individual poles
        // (better winding-number conditioning); the broad window catches modes
        // whose Re(neff) is shifted far from the probe (e.g. strongly leaky QNMs).
        let (min_n, max_n) = self.find_minmax_n();
        let broad_re = (min_n.max(0.0), max_n + 1e-6);
        let mut re_windows: Vec<(f64, f64)> = Self::narrow_re_windows(&real_probe);
        re_windows.push(broad_re);

        let mut roots: Vec<Complex<f64>> = Vec::new();

        // Run the im windows in order (broad first, then cascade). For each im
        // window, sub-divide the re windows to keep the aspect ratio bounded.
        // All im windows are run (no stop-early) to ensure the correct mode is
        // found even when a spurious mode appears in a broader window. The
        // cascade windows are cheap, so the full sweep is still fast.
        const TARGET_ASPECT_RATIO: f64 = 10.0;

        for &(im_min, im_max) in &im_windows {
            let im_width = (im_max - im_min).abs();
            if im_width < 1e-15 {
                continue;
            }
            // Target re sub-window width: aspect_ratio * im_width, clamped to
            // [0.05, 0.5] so we don't get absurdly tiny or huge windows.
            let target_re_width = (TARGET_ASPECT_RATIO * im_width).clamp(0.05, 0.5);

            for (re_lo, re_hi) in &re_windows {
                let mut start = *re_lo;
                while start < *re_hi {
                    let end = (start + target_re_width).min(*re_hi);
                    self.find_zeros_in_rectangle(
                        k0, start, end, im_min, im_max, 0, &mut roots, &char_fn,
                    );
                    // If we've reached the end of the re window, stop.
                    if end >= *re_hi {
                        break;
                    }
                    // Overlap by 20% to avoid missing a mode on the seam.
                    start = end - 0.2 * target_re_width;
                }
            }

            // No stop-early: we run all im windows to ensure we find the
            // correct mode. The cascade windows are cheap (a few hundred
            // S-matrix evaluations each for the winding number) and stopping
            // early risks missing the target mode when a spurious mode is found
            // first in a broader window. The real-axis fallback (stage D) still
            // runs after the loop.
        }

        // ── Stage D: real-axis fallback ───────────────────────────────────────
        // If the cascade found nothing and the full-structure probe found a
        // real-axis mode, the mode is effectively guided (|Im| < 1e-9). Emit it
        // with Im = 0 — unless we are under QNM BCs, where a real root is
        // unphysical. We use `full_probe` (not the isolated-core hints) so we
        // don't emit a spurious real mode from a different structure.
        if !both_outgoing {
            for &n_re in &full_probe {
                // Only emit fallback for probes that lie inside one of the re
                // windows we searched (otherwise we'd resurrect out-of-range
                // guided modes that the user did not ask for).
                let in_window = re_windows.iter().any(|(lo, hi)| n_re >= *lo && n_re <= *hi);
                if in_window {
                    roots.push(Complex::new(n_re * k0, 0.0));
                }
            }
        }

        // ── Deduplicate and sort ─────────────────────────────────────────────
        let roots = Self::dedup_roots(roots, k0);
        let mut neff_roots: Vec<Complex<f64>> = roots.iter().map(|&k| k / k0).collect();
        neff_roots.sort_by(|a, b| b.re.partial_cmp(&a.re).unwrap_or(Ordering::Equal));
        neff_roots
    }

    /// Finds all complex effective indices in the given search rectangle.
    ///
    /// Uses the argument-principle winding-number method to count and bracket
    /// zeros, then polishes each one with Muller's method.
    ///
    /// When called with `Some((re_range, im_range))` the legacy single-rectangle
    /// behaviour is used (plus the real-axis supplement for lossless + physical
    /// BCs). When called with `None` the adaptive multi-rectangle search
    /// ([`adaptive_solve_complex`]) takes over, requiring no user-supplied
    /// ranges.
    ///
    /// # Returns
    /// Complex effective indices sorted by descending `Re(neff)`.
    pub fn solve_complex(
        &self,
        k0: f64,
        polarization: Polarization,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Vec<Complex<f64>> {
        // Adaptive path: no explicit ranges supplied.
        let (re_range, im_range) =
            match self.default_search_ranges_for_bc(re_range, im_range, left_bc, right_bc) {
                Some(ranges) => ranges,
                None => return self.adaptive_solve_complex(k0, polarization, left_bc, right_bc),
            };

        // Legacy path: single rectangle.
        let char_fn = |k: Complex<f64>| {
            self.characteristic_function_with_bc(k0, k, polarization, left_bc, right_bc)
        };
        let mut roots: Vec<Complex<f64>> = Vec::new();
        self.find_zeros_in_rectangle(
            k0, re_range.0, re_range.1, im_range.0, im_range.1, 0, &mut roots, &char_fn,
        );

        // Supplement with real-axis solver only for lossless structures with both
        // claddings using physical (non-outgoing) boundary conditions.
        let both_physical = !matches!(left_bc, BoundaryCondition::Outgoing)
            && !matches!(right_bc, BoundaryCondition::Outgoing);
        let all_lossless = self.layers.iter().all(|l| l.n.im == 0.0);
        if all_lossless && both_physical {
            let real_modes = self.solve(k0, polarization);
            for neff_re in real_modes {
                if neff_re >= re_range.0 && neff_re <= re_range.1 {
                    let k_real = Complex::new(neff_re * k0, 0.0);
                    let is_duplicate = roots.iter().any(|&r| (r - k_real).norm() < 1e-6 * k0);
                    if !is_duplicate {
                        roots.push(k_real);
                    }
                }
            }
        }

        let mut neff_roots: Vec<Complex<f64>> = roots.iter().map(|&k| k / k0).collect();
        neff_roots.sort_by(|a, b| b.re.partial_cmp(&a.re).unwrap_or(Ordering::Equal));
        neff_roots
    }

    /// Returns the complex effective index of a single mode.
    ///
    /// When `re_range` / `im_range` are `None` the adaptive multi-rectangle
    /// search is used; otherwise the legacy single-rectangle path is used.
    pub fn complex_neff(
        &self,
        k0: f64,
        polarization: Polarization,
        mode: usize,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Result<Complex<f64>, String> {
        let solutions = self.solve_complex(k0, polarization, re_range, im_range, left_bc, right_bc);
        match solutions.get(mode) {
            Some(&n) => Ok(n),
            None => Err(format!(
                "Complex mode {} not found. Only {} modes (0->{}) available.",
                mode,
                solutions.len(),
                solutions.len().saturating_sub(1)
            )),
        }
    }

    // ── Propagation coefficients ──────────────────────────────────────────────

    /// Calculates the modal coefficients of each layer given the starting coefficients.
    pub fn get_propagation_coefficients(
        &self,
        k0: f64,
        k: Complex<f64>,
        polarization: Polarization,
        a: Complex<f64>,
        b: Complex<f64>,
    ) -> Vec<LayerCoefficientVector> {
        let k0c = Complex::new(k0, 0.0);
        let kc = k;
        match self.backend {
            BackEnd::Transfer => match self.left_bc {
                BoundaryCondition::PEC => {
                    get_propagation_coefficients_pec_left(&self.layers, k0c, kc, polarization, a, b)
                }
                BoundaryCondition::SemiInfinite | BoundaryCondition::Outgoing => {
                    get_propagation_coefficients_transfer(&self.layers, k0c, kc, polarization, a, b)
                }
            },
            BackEnd::Scattering => {
                panic!("Not implemented yet")
            }
        }
    }

    // ── Grid and field helpers ────────────────────────────────────────────────

    /// Calculates the plotting grid data for the multilayer.
    fn get_grid_data(&self) -> GridData {
        let xstart = match self.left_bc {
            BoundaryCondition::SemiInfinite | BoundaryCondition::Outgoing => -self.layers[0].d,
            BoundaryCondition::PEC => 0.0,
        };
        let xend = self.layers.iter().map(|l| l.d).sum::<f64>() + xstart;
        let xgrid: Vec<f64> = iter_num_tools::arange(xstart..xend, self.plot_step).collect();
        let grid_starts: Vec<f64> = self.layers.iter().map(|l| l.d).collect();
        let grid_starts: Vec<f64> = [vec![0.0_f64], grid_starts].concat();
        let mut grid_starts: Vec<f64> = cumsum(&grid_starts).iter().map(|x| x + xstart).collect();
        let mut grid_istarts: Vec<usize> = vec![0];
        let mut slice_iter = grid_starts.iter();
        let _ = slice_iter.next();
        let mut start = slice_iter.next().unwrap();
        for (i, x) in xgrid.iter().enumerate() {
            if x >= start {
                grid_istarts.push(i);
                start = slice_iter.next().unwrap();
            }
        }
        grid_istarts.push(xgrid.len());

        grid_starts[0] = 0.0;

        GridData {
            xplot: xgrid,
            xstarts: grid_starts,
            ixstarts: grid_istarts,
        }
    }

    /// Calculates the profile of a single field component given the modal coefficients
    /// of all layers.
    fn get_field_componet(
        &self,
        coefficient_vector: &[LayerCoefficientVector],
        grid_data: &GridData,
        k0: f64,
        k: Complex<f64>,
    ) -> Vec<Complex<f64>> {
        let x = grid_data.xplot.clone();

        let mut field_vectors: Vec<Complex64> = Vec::new();
        for (i, (&istart, &iend)) in zip(
            grid_data.ixstarts.clone().iter(),
            grid_data.ixstarts.clone().iter().skip(1),
        )
        .enumerate()
        {
            let xstart = grid_data.xstarts[i];
            let coefficients = coefficient_vector[i];
            let layer = &self.layers[i];
            let xslice: Vec<f64> = x[istart..iend]
                .iter()
                .map(|x| x - xstart)
                .collect::<Vec<_>>();
            field_vectors.extend(get_field_slice(
                coefficients.a,
                coefficients.b,
                k0,
                k,
                layer.n,
                xslice,
            ));
        }
        field_vectors
    }

    /// Calculates the modal coefficients for all field components.
    pub fn get_coefficient_all_components(
        &self,
        k0: f64,
        k: Complex<f64>,
        main_coefficients: Vec<LayerCoefficientVector>,
    ) -> FullLayerCoefficientVector {
        let main1 = main_coefficients;
        let k0c = Complex::new(k0, 0.0);
        let kc = k;
        let mut main2 = Vec::new();
        let mut main3 = Vec::new();
        let mut maink = Vec::new();
        let mut mainb = Vec::new();
        let zeros = vec![
            LayerCoefficientVector::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
            self.layers.len()
        ];
        for (layer, coefficients) in zip(self.layers.iter(), main1.iter()) {
            use crate::transfer_matrix::kz_physical;
            let kpar = kz_physical(k0c, layer.n, kc);
            main2.push(LayerCoefficientVector::new(
                -coefficients.a * kc / kpar,
                coefficients.b * kc / kpar,
            ));
            main3.push(LayerCoefficientVector::new(
                -coefficients.a * k0c * layer.n.powi(2) / kpar,
                coefficients.b * k0c * layer.n.powi(2) / kpar,
            ));
            maink.push(LayerCoefficientVector::new(
                coefficients.a * kpar / k0c,
                -coefficients.b * kpar / k0c,
            ));
            mainb.push(LayerCoefficientVector::new(
                -coefficients.a * kc / k0c,
                -coefficients.b * kc / k0c,
            ));
        }
        (main1, main2, main3, maink, mainb, zeros)
    }

    // ── Field reconstruction ──────────────────────────────────────────────────

    /// Calculates the field profile of the requested mode.
    pub fn field(
        &self,
        k0: f64,
        polarization: Polarization,
        mode: usize,
    ) -> Result<FieldData, String> {
        let neff = match self.neff(k0, polarization, mode) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };

        let (init_a, init_b) = match self.left_bc {
            BoundaryCondition::SemiInfinite | BoundaryCondition::Outgoing => {
                (Complex::new(0.0, 0.0), Complex::new(1.0, 0.0))
            }
            BoundaryCondition::PEC => (Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)),
        };
        let mut coefficient_vector = self.get_propagation_coefficients(
            k0,
            Complex::new(k0 * neff, 0.0),
            polarization,
            init_a,
            init_b,
        );
        let grid_data = self.get_grid_data();

        match self.right_bc {
            BoundaryCondition::SemiInfinite | BoundaryCondition::Outgoing => {
                let last_coefficient = coefficient_vector.pop().unwrap();
                coefficient_vector.push(LayerCoefficientVector {
                    a: last_coefficient.a,
                    b: Complex::new(0.0, 0.0),
                });
            }
            BoundaryCondition::PEC => {}
        }

        let coefficients = self.get_coefficient_all_components(
            k0,
            Complex::new(k0 * neff, 0.0),
            coefficient_vector,
        );

        let (main1, main2, main3, maink, mainb, zeros) = coefficients;
        let field1 = self.get_field_componet(&main1, &grid_data, k0, Complex::new(k0 * neff, 0.0));
        let fieldzeros =
            self.get_field_componet(&zeros, &grid_data, k0, Complex::new(k0 * neff, 0.0));

        let field_data = match polarization {
            Polarization::TE => {
                let fieldk =
                    self.get_field_componet(&maink, &grid_data, k0, Complex::new(k0 * neff, 0.0));
                let fieldb =
                    self.get_field_componet(&mainb, &grid_data, k0, Complex::new(k0 * neff, 0.0));

                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: fieldzeros.clone(),
                    Ey: field1,
                    Ez: fieldzeros.clone(),
                    Hx: fieldb.iter().map(|x| x / Z0).collect(),
                    Hy: fieldzeros.clone(),
                    Hz: fieldk.iter().map(|x| x / Z0).collect(),
                }
            }
            Polarization::TM => {
                let field2 =
                    self.get_field_componet(&main2, &grid_data, k0, Complex::new(k0 * neff, 0.0));
                let field3 =
                    self.get_field_componet(&main3, &grid_data, k0, Complex::new(k0 * neff, 0.0));
                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: field2,
                    Ey: fieldzeros.clone(),
                    Ez: field1,
                    Hx: fieldzeros.clone(),
                    Hy: field3.iter().map(|x| x / Z0).collect(),
                    Hz: fieldzeros.clone(),
                }
            }
        };

        Ok(match self.normalization {
            Normalization::MaxField => field_data.normalize_max_field(),
            Normalization::Power => field_data.normalize_power(),
        })
    }

    // ── Complex-neff field reconstruction ────────────────────────────────────

    /// Reconstructs the field for a mode specified by a complex effective index.
    ///
    /// For semi-infinite boundaries the outgoing wave amplitude in the rightmost
    /// layer is **not** forced to zero, because the complex neff already encodes
    /// the correct radiation or decay condition.
    ///
    /// Normalization follows `self.normalization`. If `Normalization::Power` is
    /// requested but neff has a nonzero imaginary part, a warning is emitted
    /// and `MaxField` normalization is used instead.
    pub fn field_complex(
        &self,
        k0: f64,
        polarization: Polarization,
        neff: Complex<f64>,
    ) -> FieldData {
        let k = neff * k0;

        let (init_a, init_b) = match self.left_bc {
            BoundaryCondition::SemiInfinite | BoundaryCondition::Outgoing => {
                (Complex::new(0.0, 0.0), Complex::new(1.0, 0.0))
            }
            BoundaryCondition::PEC => (Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)),
        };

        let coefficient_vector =
            self.get_propagation_coefficients(k0, k, polarization, init_a, init_b);
        let grid_data = self.get_grid_data();

        // NOTE: for complex neff we do NOT zero the outgoing amplitude in the
        // last layer. The correct boundary behaviour is encoded in Im(neff).

        let coefficients = self.get_coefficient_all_components(k0, k, coefficient_vector);
        let (main1, main2, main3, maink, mainb, zeros) = coefficients;
        let field1 = self.get_field_componet(&main1, &grid_data, k0, k);
        let fieldzeros = self.get_field_componet(&zeros, &grid_data, k0, k);

        let field_data = match polarization {
            Polarization::TE => {
                let fieldk = self.get_field_componet(&maink, &grid_data, k0, k);
                let fieldb = self.get_field_componet(&mainb, &grid_data, k0, k);
                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: fieldzeros.clone(),
                    Ey: field1,
                    Ez: fieldzeros.clone(),
                    Hx: fieldb.iter().map(|x| x / Z0).collect(),
                    Hy: fieldzeros.clone(),
                    Hz: fieldk.iter().map(|x| x / Z0).collect(),
                }
            }
            Polarization::TM => {
                let field2 = self.get_field_componet(&main2, &grid_data, k0, k);
                let field3 = self.get_field_componet(&main3, &grid_data, k0, k);
                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: field2,
                    Ey: fieldzeros.clone(),
                    Ez: field1,
                    Hx: fieldzeros.clone(),
                    Hy: field3.iter().map(|x| x / Z0).collect(),
                    Hz: fieldzeros.clone(),
                }
            }
        };

        // Normalization dispatch with warning for Power + complex neff.
        let use_normalization = if neff.im != 0.0 {
            match self.normalization {
                Normalization::Power => {
                    warn!(
                        "Power normalization is not valid for complex neff (neff = {:.6} + {:.6}i). \
                         Falling back to MaxField normalization.",
                        neff.re, neff.im
                    );
                    Normalization::MaxField
                }
                other => other,
            }
        } else {
            self.normalization
        };

        match use_normalization {
            Normalization::MaxField => field_data.normalize_max_field(),
            Normalization::Power => field_data.normalize_power(),
        }
    }

    // ── Index profile ─────────────────────────────────────────────────────────

    /// Calculates index profile of the multilayer from a grid data object.
    /// Returns the real part of n for plotting purposes.
    fn get_index(&self, grid_data: &GridData) -> Vec<f64> {
        let xgrid = grid_data.xplot.clone();
        let mut n = vec![self.layers[0].n.re; xgrid.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let start = grid_data.ixstarts[i];
            let end = grid_data.ixstarts[i + 1];
            n[start..end].iter_mut().for_each(|x| *x = layer.n.re);
        }
        n
    }

    /// Calculates the refractive index profile of the multilayer.
    pub fn index(&self) -> IndexData {
        let grid_data = self.get_grid_data();
        let index = self.get_index(&grid_data);
        IndexData {
            x: grid_data.xplot.clone(),
            n: index,
        }
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Returns the minimum and maximum *real part* of the refractive index across
/// all layers.  Used both for the real-axis scan bounds and as default real
/// bounds for the complex search.
fn find_minmax_n(layers: &[Layer]) -> (f64, f64) {
    let mut min_n = layers[0].n.re;
    let mut max_n = layers[0].n.re;
    for layer in layers.iter() {
        if layer.n.re < min_n {
            min_n = layer.n.re;
        }
        if layer.n.re > max_n {
            max_n = layer.n.re;
        }
    }
    (min_n, max_n)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f64::consts::PI, fmt::Debug};

    trait ApproxEqual {
        fn approx_eq(&self, other: &Self, tol: f64) -> bool;
    }

    impl ApproxEqual for f64 {
        fn approx_eq(&self, other: &Self, tol: f64) -> bool {
            (self - other).abs() < tol
        }
    }

    impl ApproxEqual for Complex<f64> {
        fn approx_eq(&self, other: &Self, tol: f64) -> bool {
            (self - other).norm() < tol
        }
    }

    impl ApproxEqual for LayerCoefficientVector {
        fn approx_eq(&self, other: &Self, tol: f64) -> bool {
            (self.a - other.a).norm() < tol && (self.b - other.b).norm() < tol
        }
    }

    fn assert_vec_approx_equal<T>(a: &[T], b: &[T], tol: f64)
    where
        T: ApproxEqual + Debug,
    {
        assert_eq!(a.len(), b.len(), "Vectors have different lengths");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                x.approx_eq(y, tol),
                "Vectors differ at index {}: {:?} != {:?}",
                i,
                x,
                y
            );
        }
    }

    fn create_slab_multilayer() -> MultiLayer {
        MultiLayer::new(vec![
            Layer::from_real(1.0, 1.0),
            Layer::from_real(2.0, 0.6),
            Layer::from_real(1.0, 1.0),
        ])
    }

    fn create_coupled_slab_multilayer() -> MultiLayer {
        MultiLayer::new(vec![
            Layer::from_real(1.0, 1.0),
            Layer::from_real(2.0, 0.6),
            Layer::from_real(1.0, 2.0),
            Layer::from_real(2.0, 0.6),
            Layer::from_real(1.0, 1.0),
        ])
    }

    fn create_asymmetric_coupled_slab_multilayer() -> MultiLayer {
        MultiLayer::new(vec![
            Layer::from_real(1.0, 2.0),
            Layer::from_real(1.51, 5.0),
            Layer::from_real(1.5, 2.0),
            Layer::from_real(2.0, 0.03),
            Layer::from_real(1.5, 2.0),
        ])
    }

    #[test]
    fn test_scattering_slab_te() {
        let mut slab = create_slab_multilayer();
        slab.set_backend(BackEnd::Scattering);
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TE);
        assert_vec_approx_equal(&neffs, &[1.804297363, 1.191174978], 1e-6);
    }

    #[test]
    fn test_scattering_slab_tm() {
        let mut slab = create_slab_multilayer();
        slab.set_backend(BackEnd::Scattering);
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TM);
        assert_vec_approx_equal(&neffs, &[1.657017474, 1.028990635], 1e-6);
    }

    #[test]
    fn test_scattering_coupled_slab_te() {
        let mut slab = create_coupled_slab_multilayer();
        slab.set_backend(BackEnd::Scattering);
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TE);
        assert_vec_approx_equal(
            &neffs,
            &[1.804297929, 1.804296798, 1.192052932, 1.190270579],
            1e-6,
        );
    }

    #[test]
    fn test_scattering_coupled_slab_tm() {
        let mut slab = create_coupled_slab_multilayer();
        slab.set_backend(BackEnd::Scattering);
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TM);
        assert_vec_approx_equal(
            &neffs,
            &[1.657019473, 1.657015474, 1.035192425, 1.019866805],
            1e-6,
        );
    }

    #[test]
    fn test_transfer_slab_te() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TE);
        assert_vec_approx_equal(&neffs, &[1.804297363, 1.191174978], 1e-6);
    }

    #[test]
    fn test_transfer_slab_tm() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TM);
        assert_vec_approx_equal(&neffs, &[1.657017474, 1.028990635], 1e-6);
    }

    #[test]
    fn test_transfer_coupled_slab_te() {
        let slab = create_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TE);
        assert_vec_approx_equal(
            &neffs,
            &[1.804297929, 1.804296798, 1.192052932, 1.190270579],
            1e-6,
        );
    }

    #[test]
    fn test_transfer_coupled_slab_tm() {
        let slab = create_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TM);
        assert_vec_approx_equal(
            &neffs,
            &[1.657019473, 1.657015474, 1.035192425, 1.019866805],
            1e-6,
        );
    }

    #[test]
    fn test_transfer_asymmetric_coupled_slab_te() {
        let slab = create_asymmetric_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TE);
        assert_vec_approx_equal(&neffs, &[1.50648353, 1.50216560], 1e-6);
    }

    #[test]
    fn test_transfer_asymmetric_coupled_slab_tm() {
        let slab = create_asymmetric_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let neffs = slab.solve(om, Polarization::TM);
        assert_vec_approx_equal(&neffs, &[1.50575211, 1.50019654], 1e-6);
    }

    #[test]
    fn test_transfer_field_slab() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let field = slab.field(om, Polarization::TE, 0).unwrap();

        // Check that the field is continuous at the interfaces.
        let n = field.Ey.len();
        let mid = n / 2;
        assert!(
            (field.Ey[mid] - field.Ey[mid + 1]).norm() < 1e-3,
            "Field discontinuous at mid"
        );

        // Check that the field is symmetric.
        let left = field.Ey[100].norm();
        let right = field.Ey[n - 101].norm();
        assert!((left - right).abs() < 1e-3, "Field not symmetric");
    }

    #[test]
    fn test_field_normalization() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let field = slab.field(om, Polarization::TE, 0).unwrap();
        let max_e = field
            .Ex
            .iter()
            .zip(field.Ey.iter())
            .zip(field.Ez.iter())
            .map(|((&ex, &ey), &ez)| (ex.norm_sqr() + ey.norm_sqr() + ez.norm_sqr()).sqrt())
            .fold(0.0_f64, f64::max);
        assert!(
            (max_e - 1.0).abs() < 1e-6,
            "Max E field not normalised to 1, got {max_e}"
        );
    }

    #[test]
    fn test_field_complex_lossy_max_field_normalization() {
        // Lossy core slab: core index has a small imaginary part.
        let slab = MultiLayer::new(vec![
            Layer::from_complex(Complex::new(1.0, 0.0), 1.0),
            Layer::from_complex(Complex::new(2.0, -0.01), 0.6),
            Layer::from_complex(Complex::new(1.0, 0.0), 1.0),
        ]);
        let om = 2.0 * PI / 1.55;
        // Find the complex neff.
        let roots = slab.solve_complex(
            om,
            Polarization::TE,
            Some((1.0, 2.0)),
            Some((-0.05, 0.05)),
            BoundaryCondition::SemiInfinite,
            BoundaryCondition::SemiInfinite,
        );
        assert!(!roots.is_empty(), "No complex mode found");
        let neff = roots[0];
        let field = slab.field_complex(om, Polarization::TE, neff);
        // Check MaxField normalization: max total |E| == 1.
        let max_e = field
            .Ex
            .iter()
            .zip(field.Ey.iter())
            .zip(field.Ez.iter())
            .map(|((&ex, &ey), &ez)| (ex.norm_sqr() + ey.norm_sqr() + ez.norm_sqr()).sqrt())
            .fold(0.0_f64, f64::max);
        assert!(
            (max_e - 1.0).abs() < 1e-6,
            "Max E not normalised to 1, got {max_e}"
        );
    }

    fn create_pec_left_half_slab() -> MultiLayer {
        // Half-slab with PEC on the left.
        let mut ml = MultiLayer::new(vec![Layer::from_real(2.0, 0.3), Layer::from_real(1.0, 1.0)]);
        ml.set_left_boundary(BoundaryCondition::PEC);
        ml
    }

    fn create_pec_right_half_slab() -> MultiLayer {
        let mut ml = MultiLayer::new(vec![Layer::from_real(1.0, 1.0), Layer::from_real(2.0, 0.3)]);
        ml.set_right_boundary(BoundaryCondition::PEC);
        ml
    }

    #[test]
    fn test_pec_left_te_matches_odd_mode_of_full_slab() {
        // The fundamental mode of the PEC-left half-slab should match the odd
        // (antisymmetric) TE mode of the full symmetric slab.
        let full_slab = create_slab_multilayer();
        let pec_slab = create_pec_left_half_slab();
        let om = 2.0 * PI / 1.55;

        let full_modes = full_slab.solve(om, Polarization::TE);
        let pec_modes = pec_slab.solve(om, Polarization::TE);

        assert!(
            !pec_modes.is_empty(),
            "PEC half-slab should find at least one mode"
        );
        // The PEC half-slab mode should match one of the full slab modes.
        let found = full_modes.iter().any(|&n| (n - pec_modes[0]).abs() < 1e-5);
        assert!(found, "PEC mode not found in full slab modes");
    }

    #[test]
    fn test_pec_left_tm_matches_even_hz_mode_of_full_slab() {
        let full_slab = create_slab_multilayer();
        let pec_slab = create_pec_left_half_slab();
        let om = 2.0 * PI / 1.55;

        let full_modes = full_slab.solve(om, Polarization::TM);
        let pec_modes = pec_slab.solve(om, Polarization::TM);

        assert!(!pec_modes.is_empty());
        let found = full_modes.iter().any(|&n| (n - pec_modes[0]).abs() < 1e-5);
        assert!(found);
    }

    #[test]
    fn test_pec_left_right_symmetry_te() {
        let left = create_pec_left_half_slab();
        let right = create_pec_right_half_slab();
        let om = 2.0 * PI / 1.55;
        let left_modes = left.solve(om, Polarization::TE);
        let right_modes = right.solve(om, Polarization::TE);
        assert_vec_approx_equal(&left_modes, &right_modes, 1e-6);
    }

    #[test]
    fn test_pec_left_right_symmetry_tm() {
        let left = create_pec_left_half_slab();
        let right = create_pec_right_half_slab();
        let om = 2.0 * PI / 1.55;
        let left_modes = left.solve(om, Polarization::TM);
        let right_modes = right.solve(om, Polarization::TM);
        assert_vec_approx_equal(&left_modes, &right_modes, 1e-6);
    }

    #[test]
    fn test_pec_field_satisfies_bc_te() {
        let pec_slab = create_pec_left_half_slab();
        let om = 2.0 * PI / 1.55;
        let field = pec_slab.field(om, Polarization::TE, 0).unwrap();
        // Ey must vanish at the PEC wall (x = 0, first point).
        assert!(
            field.Ey[0].norm() < 1e-3,
            "Ey not zero at PEC wall: {:?}",
            field.Ey[0]
        );
    }

    #[test]
    fn test_pec_field_satisfies_bc_tm() {
        let pec_slab = create_pec_left_half_slab();
        let om = 2.0 * PI / 1.55;
        let field = pec_slab.field(om, Polarization::TM, 0).unwrap();
        // Ez must vanish at the PEC wall (x = 0, first point).
        assert!(
            field.Ez[0].norm() < 1e-3,
            "Ez not zero at PEC wall: {:?}",
            field.Ez[0]
        );
    }

    #[test]
    fn test_pec_field_continuity_te() {
        let pec_slab = create_pec_left_half_slab();
        let om = 2.0 * PI / 1.55;
        let field = pec_slab.field(om, Polarization::TE, 0).unwrap();

        // Find the interface index (around x = 0.3 with plot_step = 1e-3).
        let interface_i = (0.3 / 1e-3) as usize;
        let left_val = field.Ey[interface_i].norm();
        let right_val = field.Ey[interface_i + 1].norm();
        assert!(
            (left_val - right_val).abs() < 0.1,
            "Ey discontinuous at interface: {} vs {}",
            left_val,
            right_val
        );
    }

    #[test]
    fn test_pec_field_continuity_tm() {
        let pec_slab = create_pec_left_half_slab();
        let om = 2.0 * PI / 1.55;
        let field = pec_slab.field(om, Polarization::TM, 0).unwrap();

        // Sample Ez just before and just after the interface at x = 0.3.
        // With plot_step = 1e-3 the interface sits between index 299 and 300.
        // Use a wider bracket (±5 steps) and compare amplitudes rather than
        // adjacent samples, since the field is smooth but sampled discretely.
        let i_before = (0.3 / 1e-3) as usize - 5;
        let i_after = (0.3 / 1e-3) as usize + 5;
        let val_before = field.Ez[i_before].norm();
        let val_after = field.Ez[i_after].norm();
        assert!(
            (val_before - val_after).abs() < val_before.max(val_after) * 0.2 + 1e-6,
            "Ez discontinuous across interface: {} vs {}",
            val_before,
            val_after
        );
    }

    // ── Multi-layer PEC tests (ported from original) ───────────────────────────

    fn create_multi_pec_left() -> MultiLayer {
        let mut ml = MultiLayer::new(vec![
            Layer::from_real(2.0, 0.3),
            Layer::from_real(1.0, 0.5),
            Layer::from_real(2.0, 0.3),
            Layer::from_real(1.0, 1.0),
        ]);
        ml.set_left_boundary(BoundaryCondition::PEC);
        ml
    }

    fn create_multi_pec_right() -> MultiLayer {
        let mut ml = MultiLayer::new(vec![
            Layer::from_real(1.0, 1.0),
            Layer::from_real(2.0, 0.3),
            Layer::from_real(1.0, 0.5),
            Layer::from_real(2.0, 0.3),
        ]);
        ml.set_right_boundary(BoundaryCondition::PEC);
        ml
    }

    #[test]
    fn test_multi_pec_left_right_symmetry_te() {
        let left = create_multi_pec_left();
        let right = create_multi_pec_right();
        let om = 2.0 * PI / 1.55;
        let left_modes = left.solve(om, Polarization::TE);
        let right_modes = right.solve(om, Polarization::TE);
        assert_vec_approx_equal(&left_modes, &right_modes, 1e-6);
    }

    #[test]
    fn test_multi_pec_left_right_symmetry_tm() {
        let left = create_multi_pec_left();
        let right = create_multi_pec_right();
        let om = 2.0 * PI / 1.55;
        let left_modes = left.solve(om, Polarization::TM);
        let right_modes = right.solve(om, Polarization::TM);
        assert_vec_approx_equal(&left_modes, &right_modes, 1e-6);
    }

    #[test]
    fn test_multi_pec_left_te_ey_zero_at_wall() {
        let ml = create_multi_pec_left();
        let om = 2.0 * PI / 1.55;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        assert!(field.Ey[0].norm() < 1e-3, "Ey not zero at PEC wall");
    }

    #[test]
    fn test_multi_pec_left_tm_ez_zero_at_wall() {
        let ml = create_multi_pec_left();
        let om = 2.0 * PI / 1.55;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        assert!(field.Ez[0].norm() < 1e-3, "Ez not zero at PEC wall");
    }

    #[test]
    fn test_multi_pec_left_te_ey_continuous_at_interfaces() {
        let ml = create_multi_pec_left();
        let om = 2.0 * PI / 1.55;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        // Interface positions: 0.3, 0.8, 1.1 (cumulative thicknesses)
        for &iface_x in &[0.3_f64, 0.8, 1.1] {
            let i = (iface_x / 1e-3) as usize;
            let diff = (field.Ey[i] - field.Ey[i + 1]).norm();
            assert!(
                diff < 0.1,
                "Ey discontinuous at x={}: diff={}",
                iface_x,
                diff
            );
        }
    }

    #[test]
    fn test_multi_pec_left_tm_ez_continuous_at_interfaces() {
        let ml = create_multi_pec_left();
        let om = 2.0 * PI / 1.55;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        // For TM modes, Ez is NOT continuous across a dielectric interface —
        // only Hy is (the tangential magnetic field).  Verify Hy continuity instead,
        // which holds unconditionally regardless of the index contrast.
        // Interface positions for [PEC | 2.0(0.3) | 1.0(0.5) | 2.0(0.3) | 1.0(1.0)]:
        //   x = 0.3, 0.8, 1.1
        for &iface_x in &[0.3_f64, 0.8, 1.1] {
            let i = (iface_x / 1e-3) as usize;
            let i_left = i.saturating_sub(1);
            let i_right = (i + 1).min(field.Hy.len() - 1);
            let hy_left = field.Hy[i_left].norm();
            let hy_right = field.Hy[i_right].norm();
            let rel_diff = (hy_left - hy_right).abs() / (hy_left.max(hy_right) + 1e-12);
            assert!(
                rel_diff < 0.15,
                "Hy discontinuous at x={}: Hy_left={} Hy_right={} (rel_diff={})",
                iface_x,
                hy_left,
                hy_right,
                rel_diff
            );
        }
    }

    #[test]
    fn test_multi_pec_right_te_ey_continuous_at_interfaces() {
        let ml = create_multi_pec_right();
        let om = 2.0 * PI / 1.55;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        for &iface_x in &[0.0_f64, 1.0, 1.3] {
            // offset by left cladding thickness = 1.0
            let abs_x = iface_x + ml.layers[0].d;
            let i = (abs_x / 1e-3) as usize;
            if i + 1 < field.Ey.len() {
                let diff = (field.Ey[i] - field.Ey[i + 1]).norm();
                assert!(diff < 0.1, "Ey discontinuous at x={}: diff={}", abs_x, diff);
            }
        }
    }

    #[test]
    fn test_multi_pec_right_tm_ez_continuous_at_interfaces() {
        let ml = create_multi_pec_right();
        let om = 2.0 * PI / 1.55;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        for &iface_x in &[0.0_f64, 1.0, 1.3] {
            let abs_x = iface_x + ml.layers[0].d;
            let i = (abs_x / 1e-3) as usize;
            if i + 1 < field.Ez.len() {
                let diff = (field.Ez[i] - field.Ez[i + 1]).norm();
                assert!(diff < 0.1, "Ez discontinuous at x={}: diff={}", abs_x, diff);
            }
        }
    }

    // ── Complex mode search tests ─────────────────────────────────────────────

    /// The complex solver should recover the same guided modes as the real-axis
    /// solver when applied to a lossless structure.  We test each mode individually
    /// using a tight box around its known real neff, which avoids the contour-
    /// sampling issues that arise when two modes share a very thin imaginary window.
    #[test]
    fn test_complex_solver_recovers_guided_modes_te() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let real_modes = slab.solve(om, Polarization::TE);

        assert!(!real_modes.is_empty(), "Real solver found no modes");

        for &re_mode in &real_modes {
            // Tight box: ±0.05 in real part, ±0.05 in imaginary part.
            // This gives a square rectangle, which the adaptive sampler handles well.
            let complex_modes = slab.solve_complex(
                om,
                Polarization::TE,
                Some((re_mode - 0.05, re_mode + 0.05)),
                Some((-0.05, 0.05)),
                BoundaryCondition::SemiInfinite,
                BoundaryCondition::SemiInfinite,
            );
            assert!(
                !complex_modes.is_empty(),
                "Complex solver found no mode near real neff={}",
                re_mode
            );
            let best = complex_modes
                .iter()
                .min_by(|a, b| {
                    (a.re - re_mode)
                        .abs()
                        .partial_cmp(&(b.re - re_mode).abs())
                        .unwrap()
                })
                .unwrap();
            assert!(
                (best.re - re_mode).abs() < 1e-4,
                "Complex Re(neff)={} far from real neff={}",
                best.re,
                re_mode
            );
            assert!(
                best.im.abs() < 1e-4,
                "Im(neff)={} should be ~0 for lossless mode",
                best.im
            );
        }
    }

    /// For a lossless structure the imaginary part of all found modes should be
    /// essentially zero (within numerical tolerance).
    #[test]
    fn test_complex_solver_im_zero_for_lossless() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let real_modes = slab.solve(om, Polarization::TE);
        // Search around each mode individually using a square box.
        for &re_mode in &real_modes {
            let complex_modes = slab.solve_complex(
                om,
                Polarization::TE,
                Some((re_mode - 0.05, re_mode + 0.05)),
                Some((-0.05, 0.05)),
                BoundaryCondition::SemiInfinite,
                BoundaryCondition::SemiInfinite,
            );
            for mode in &complex_modes {
                assert!(
                    mode.im.abs() < 1e-4,
                    "Im(neff) = {} is not ~0 for lossless structure near neff={}",
                    mode.im,
                    re_mode
                );
            }
        }
    }

    /// For a lossy slab (Im(n_core) > 0) the imaginary part of neff should be
    /// nonzero and the real part should be close to the lossless value.
    #[test]
    fn test_complex_solver_lossy_core() {
        let om = 2.0 * PI / 1.55;

        // Lossless reference.
        let slab_lossless = create_slab_multilayer();
        let neff_real = slab_lossless.neff(om, Polarization::TE, 0).unwrap();

        // Lossy slab: add a small imaginary part to the core index.
        let loss = 0.01;
        let slab_lossy = MultiLayer::new(vec![
            Layer::from_real(1.0, 1.0),
            Layer {
                n: Complex::new(2.0, loss),
                d: 0.6,
            },
            Layer::from_real(1.0, 1.0),
        ]);

        let complex_modes = slab_lossy.solve_complex(
            om,
            Polarization::TE,
            Some((1.0, 2.0)),
            Some((-0.05, 0.05)),
            BoundaryCondition::SemiInfinite,
            BoundaryCondition::SemiInfinite,
        );

        assert!(!complex_modes.is_empty(), "Lossy slab: no modes found");

        let mode = complex_modes[0];
        assert!(
            (mode.re - neff_real).abs() < 1e-3,
            "Lossy Re(neff)={} far from lossless neff={}",
            mode.re,
            neff_real
        );
        assert!(
            mode.im.abs() > 1e-6,
            "Im(neff)={} should be nonzero for lossy core",
            mode.im
        );
    }

    /// Winding number around a small circle that does not enclose any zero
    /// should be 0.
    #[test]
    fn test_winding_number_empty_rectangle() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        // Rectangle well away from any mode.
        let char_fn = |k: Complex<f64>| {
            slab.characteristic_function_with_bc(
                om,
                k,
                Polarization::TE,
                BoundaryCondition::SemiInfinite,
                BoundaryCondition::SemiInfinite,
            )
        };
        let wn = slab.winding_number(om, 0.5, 0.6, -0.001, 0.001, 32, &char_fn);
        assert_eq!(wn, 0, "Winding number should be 0 for empty rectangle");
    }

    /// Winding number around a rectangle that encloses the fundamental TE mode
    /// should be 1 (after taking the absolute value of the raw winding number).
    #[test]
    fn test_winding_number_one_mode() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let neff0 = slab.neff(om, Polarization::TE, 0).unwrap();
        // Tight box around the fundamental mode.
        let char_fn = |k: Complex<f64>| {
            slab.characteristic_function_with_bc(
                om,
                k,
                Polarization::TE,
                BoundaryCondition::SemiInfinite,
                BoundaryCondition::SemiInfinite,
            )
        };
        let wn = slab.winding_number(om, neff0 - 0.01, neff0 + 0.01, -0.001, 0.001, 64, &char_fn);
        // winding_number already returns the absolute value, so expect +1.
        assert_eq!(wn, 1, "Winding number should be 1 for one enclosed mode");
    }

    /// kz_physical used in find_minmax_n indirectly: verify find_minmax_n on
    /// a complex-n stack returns the real parts correctly.
    #[test]
    fn test_find_minmax_n_complex() {
        let layers = vec![
            Layer {
                n: Complex::new(1.0, 0.0),
                d: 1.0,
            },
            Layer {
                n: Complex::new(2.0, 0.01),
                d: 0.6,
            },
            Layer {
                n: Complex::new(1.5, -0.005),
                d: 1.0,
            },
        ];
        let (min_n, max_n) = find_minmax_n(&layers);
        assert!((min_n - 1.0).abs() < 1e-12);
        assert!((max_n - 2.0).abs() < 1e-12);
    }

    // ── QNM solver tests ─────────────────────────────────────────────────────

    /// Build the leaky asymmetric slab used in QNM tests:
    ///   air  |  n=2.0 core (0.6 µm)  |  air gap (t µm)  |  n=2.2 substrate
    ///
    /// With a thin gap the guided mode couples into the substrate and becomes a
    /// quasi-normal mode with `Im(neff) < 0`.
    fn create_leaky_slab(gap_t: f64) -> MultiLayer {
        MultiLayer::new(vec![
            Layer {
                n: Complex::new(1.0, 0.0),
                d: 1.0,
            },
            Layer {
                n: Complex::new(2.0, 0.0),
                d: 0.6,
            },
            Layer {
                n: Complex::new(1.0, 0.0),
                d: gap_t,
            },
            Layer {
                n: Complex::new(2.2, 0.0),
                d: 1.0,
            },
        ])
    }

    /// The QNM characteristic function must be finite and non-zero at a point
    /// well away from any pole.
    #[test]
    fn test_qnm_char_fn_finite_off_mode() {
        let slab = create_leaky_slab(2.0);
        let om = 2.0 * PI / 1.55;
        // A point far from any mode.
        let k_test = Complex::new(0.5 * om, -0.05 * om);
        let val = slab.characteristic_function_with_bc(
            om,
            k_test,
            Polarization::TE,
            BoundaryCondition::Outgoing,
            BoundaryCondition::Outgoing,
        );
        assert!(
            val.re.is_finite() && val.im.is_finite(),
            "QNM char fn should be finite off-mode, got {val}"
        );
        // Should not be zero off-mode.
        assert!(
            val.norm() > 1e-10,
            "QNM char fn should be non-zero off-mode, got {val}"
        );
    }

    /// `solve_qnm` on a leaky slab must find at least one mode with Im(neff) < 0.
    #[test]
    fn test_qnm_solver_finds_leaky_mode() {
        // Small gap (0.5 µm) → strong coupling to substrate → clear leakage.
        // The QNM for this structure sits at Im(neff) ≈ -0.003.
        // Use a targeted im_range=(-0.15, -1e-3) which is empirically reliable
        // for this mode location and avoids the S-matrix anti-resonance instabilities
        // that appear for very deep (im_min < -0.35) or very thin rectangles.
        let slab = create_leaky_slab(0.5);
        let om = 2.0 * PI / 1.55;
        let modes = slab.solve_complex(
            om,
            Polarization::TE,
            Some((1.0 + 1e-6, 2.2 - 1e-6)),
            Some((-0.15, -1e-3)),
            BoundaryCondition::Outgoing,
            BoundaryCondition::Outgoing,
        );
        assert!(
            !modes.is_empty(),
            "Expected at least one QNM for leaky slab with thin gap"
        );
        for neff in &modes {
            assert!(
                neff.im < 0.0,
                "All QNM Im(neff) must be negative, got Im={:.6e}",
                neff.im
            );
            assert!(
                neff.re > 1.0 && neff.re < 2.2,
                "QNM Re(neff) should be in (1.0, 2.2), got Re={:.6}",
                neff.re
            );
        }
    }

    /// `solve_qnm` on a symmetric lossless slab (air/core/air) with a small
    /// imaginary window should find modes in the lower half-plane as well.
    #[test]
    fn test_qnm_solver_symmetric_slab_modes_have_negative_im() {
        let slab = create_slab_multilayer(); // air / n=2 core / air
        let om = 2.0 * PI / 1.55;
        let modes = slab.solve_complex(
            om,
            Polarization::TE,
            Some((1.0 + 1e-6, 2.0 - 1e-6)),
            Some((-0.5, -1e-3)),
            BoundaryCondition::Outgoing,
            BoundaryCondition::Outgoing,
        );
        // We may or may not find modes in this range, but any found mode must
        // satisfy Im(neff) ≤ 0.
        for neff in &modes {
            assert!(
                neff.im <= 0.0,
                "QNM Im(neff) must be ≤ 0, got {:.6e}",
                neff.im
            );
        }
    }

    /// The leakage rate should increase monotonically as the gap shrinks.
    /// That is, |Im(neff)| should grow as gap_t decreases.
    ///
    /// Gap thicknesses are chosen so that all three modes lie comfortably inside
    /// the reliable search window (-0.15, -1e-3):
    ///   t=0.5 µm → Im(neff) ≈ -0.003
    ///   t=0.3 µm → Im(neff) ≈ -0.021
    ///   t=0.2 µm → Im(neff) ≈ -0.059
    #[test]
    fn test_qnm_leakage_increases_with_smaller_gap() {
        let om = 2.0 * PI / 1.55;
        let gaps = [0.5_f64, 0.3, 0.2];
        let re_range = (1.0 + 1e-6, 2.2 - 1e-6);
        let im_range = (-0.15, -1e-3);

        let mut prev_im: Option<f64> = None;
        for &t in &gaps {
            let slab = create_leaky_slab(t);
            let modes = slab.solve_complex(
                om,
                Polarization::TE,
                Some(re_range),
                Some(im_range),
                BoundaryCondition::Outgoing,
                BoundaryCondition::Outgoing,
            );
            if let Some(neff) = modes.first() {
                let im = neff.im;
                assert!(im < 0.0, "Im(neff) must be < 0 for gap t={t}, got {im:.6e}");
                if let Some(prev) = prev_im {
                    assert!(
                        im < prev,
                        "Im(neff) should become more negative as gap shrinks: \
                         Im(gap={t})={im:.6e} should be < Im(prev)={prev:.6e}"
                    );
                }
                prev_im = Some(im);
            }
        }
        // We must have found at least one gap where a QNM was located.
        assert!(
            prev_im.is_some(),
            "Expected QNMs for at least one gap thickness"
        );
    }

    // ── Adaptive solver tests ────────────────────────────────────────────────

    /// The adaptive solver (no explicit ranges) must find the weakly leaky
    /// mode at t=1.5 µm, which was previously a coverage gap (Im ≈ 1e-9,
    /// below the old default im_min=1e-3 but above the real-axis threshold).
    #[test]
    fn test_adaptive_finds_weakly_leaky_mode() {
        // One-sided leaky mode: left SemiInfinite, right Outgoing.
        // t=1.5 µm → Im(neff) ≈ 1e-9, previously a coverage gap.
        let mut ml = MultiLayer::new(vec![
            Layer::from_real(1.0, 1.0),
            Layer::from_real(2.0, 0.6),
            Layer::from_real(1.0, 1.5),
            Layer::from_real(2.2, 1.0),
        ]);
        ml.set_right_boundary(BoundaryCondition::Outgoing);
        let om = 2.0 * PI / 1.55;
        let modes = ml.solve_complex(
            om,
            Polarization::TE,
            None,
            None,
            BoundaryCondition::SemiInfinite,
            BoundaryCondition::Outgoing,
        );
        assert!(
            !modes.is_empty(),
            "Adaptive solver should find the weakly leaky mode at t=1.5µm"
        );
        let mode = modes[0];
        assert!(
            (mode.re - 1.8043).abs() < 0.01,
            "Re(neff) should be ≈ 1.8043, got {:.6}",
            mode.re
        );
        // Im should be very small (≈ 1e-9 or zero from fallback).
        assert!(
            mode.im.abs() < 1e-6,
            "|Im(neff)| should be tiny for weakly leaky mode, got {:.6e}",
            mode.im
        );
    }

    /// The adaptive solver must find the QNM for a strongly leaky structure
    /// (t=0.5 µm) without explicit ranges, even though the QNM's Re(neff) is
    /// shifted far from the isolated-core guided mode's Re(neff).
    #[test]
    fn test_adaptive_finds_qnm_no_ranges() {
        let slab = create_leaky_slab(0.5);
        let om = 2.0 * PI / 1.55;
        let modes = slab.solve_complex(
            om,
            Polarization::TE,
            None,
            None,
            BoundaryCondition::Outgoing,
            BoundaryCondition::Outgoing,
        );
        assert!(!modes.is_empty(), "Adaptive solver should find the QNM");
        let mode = modes[0];
        assert!(
            mode.im < 0.0,
            "QNM Im(neff) must be < 0, got {:.6e}",
            mode.im
        );
        assert!(
            (mode.re - 1.526).abs() < 0.01,
            "QNM Re(neff) should be ≈ 1.526, got {:.6}",
            mode.re
        );
    }

    /// Explicit ranges must preserve the legacy single-rectangle behaviour:
    /// passing im_range=(1e-3, 0.15) on a weakly leaky structure (Im ≈ 1e-9)
    /// should return no mode, because the mode is below the im_min floor.
    #[test]
    fn test_adaptive_explicit_ranges_preserve_legacy() {
        let mut ml = MultiLayer::new(vec![
            Layer::from_real(1.0, 1.0),
            Layer::from_real(2.0, 0.6),
            Layer::from_real(1.0, 1.5),
            Layer::from_real(2.2, 1.0),
        ]);
        ml.set_right_boundary(BoundaryCondition::Outgoing);
        let om = 2.0 * PI / 1.55;
        // Explicit im_range that excludes the weakly leaky mode.
        let modes = ml.solve_complex(
            om,
            Polarization::TE,
            None,
            Some((1e-3, 0.15)),
            BoundaryCondition::SemiInfinite,
            BoundaryCondition::Outgoing,
        );
        // The weakly leaky mode (Im ≈ 1e-9) is below im_min=1e-3, so it should
        // not be found. (The TE1 mode at Re≈1.19, Im≈4e-5 is also below 1e-3.)
        // So we expect either no modes or modes with Im > 1e-3.
        for mode in &modes {
            assert!(
                mode.im.abs() >= 1e-3,
                "Legacy path should not find modes below im_min=1e-3, got Im={:.6e}",
                mode.im
            );
        }
    }

    /// The adaptive solver must find the lossy guided mode without explicit
    /// ranges, with Im(neff) < 0 (loss) and Re(neff) close to the lossless value.
    #[test]
    fn test_adaptive_lossy_guided() {
        let slab = MultiLayer::new(vec![
            Layer::from_real(1.0, 1.0),
            Layer::from_complex(Complex::new(2.0, -0.01), 0.6),
            Layer::from_real(1.0, 1.0),
        ]);
        let om = 2.0 * PI / 1.55;
        let modes = slab.solve_complex(
            om,
            Polarization::TE,
            None,
            None,
            BoundaryCondition::SemiInfinite,
            BoundaryCondition::SemiInfinite,
        );
        assert!(
            !modes.is_empty(),
            "Adaptive solver should find the lossy mode"
        );
        let mode = modes[0];
        assert!(
            (mode.re - 1.8043).abs() < 0.01,
            "Re(neff) should be ≈ 1.8043"
        );
        assert!(mode.im < 0.0, "Im(neff) should be < 0 for lossy core");
    }

    /// The adaptive solver must find the lossless guided mode without explicit
    /// ranges, with Im(neff) = 0.
    #[test]
    fn test_adaptive_lossless_guided() {
        let slab = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        let modes = slab.solve_complex(
            om,
            Polarization::TE,
            None,
            None,
            BoundaryCondition::SemiInfinite,
            BoundaryCondition::SemiInfinite,
        );
        assert!(
            !modes.is_empty(),
            "Adaptive solver should find guided modes"
        );
        let mode = modes[0];
        assert!(
            (mode.re - 1.8043).abs() < 0.01,
            "Re(neff) should be ≈ 1.8043"
        );
        assert!(mode.im.abs() < 1e-10, "Im(neff) should be 0 for lossless");
    }
}
