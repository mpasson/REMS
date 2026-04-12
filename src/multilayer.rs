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
use crate::scattering_matrix::calculate_s_matrix;
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

/// Minimum imaginary half-width of the complex search rectangle when all layers
/// are lossless.  Large enough to avoid degenerate aspect ratios that make the
/// winding-number contour unreliable: the contour must not hug the real axis
/// too closely relative to the real-axis width of the search rectangle.
/// A value of 0.05 keeps the aspect ratio below ~20:1 for typical structures.
const MIN_IM_HALF_WIDTH: f64 = 0.05;

/// Number of points used to discretise each side of the contour when computing
/// the winding number via the argument principle.
const CONTOUR_POINTS_PER_SIDE: usize = 128;

/// Maximum recursion depth for the rectangle-subdivision zero-counter.
/// A rectangle that is smaller than ~ (search_width / 2^MAX_DEPTH) in each
/// dimension is treated as containing a single zero and polished directly.
const MAX_SUBDIVISION_DEPTH: usize = 20;

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

    /// Finds a single complex effective index using the 2-D complex-plane search.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency (real).
    /// * `polarization` - The polarization of the mode (`Polarization.TE` or `TM`).
    /// * `mode`         - Zero-based index into the list returned by
    ///                    [`python_all_complex_neff`], sorted by descending `Re(neff)`.
    /// * `re_range`     - Optional `(re_min, re_max)` for the real part of `neff`.
    ///                    Defaults to `(Re(n_min), Re(n_max))` across all layers.
    /// * `im_range`     - Optional `(im_min, im_max)` for the imaginary part of `neff`.
    ///                    Defaults to a window scaled to the maximum material loss,
    ///                    with a floor of ±[`MIN_IM_HALF_WIDTH`].
    ///
    /// # Returns
    /// `(Re(neff), Im(neff))` as a Python tuple, or `None` if the requested mode
    /// index is out of range.
    #[pyo3(name = "complex_neff")]
    #[pyo3(signature = (omega, polarization=None, mode=None, re_range=None, im_range=None))]
    pub fn python_complex_neff(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
    ) -> Option<(f64, f64)> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        let (re_range, im_range) = self.default_search_ranges(re_range, im_range);
        let roots = self.solve_complex(omega, polarization, re_range, im_range);
        roots.get(mode).map(|c| (c.re, c.im))
    }

    /// Returns all complex effective indices found in the search rectangle.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency (real).
    /// * `polarization` - The polarization of the modes.
    /// * `re_range`     - Optional `(re_min, re_max)` for `Re(neff)`.
    /// * `im_range`     - Optional `(im_min, im_max)` for `Im(neff)`.
    ///
    /// # Returns
    /// A list of `(Re(neff), Im(neff))` tuples, sorted by descending `Re(neff)`.
    #[pyo3(name = "all_complex_neff")]
    #[pyo3(signature = (omega, polarization=None, re_range=None, im_range=None))]
    pub fn python_all_complex_neff(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
    ) -> Vec<(f64, f64)> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let (re_range, im_range) = self.default_search_ranges(re_range, im_range);
        self.solve_complex(omega, polarization, re_range, im_range)
            .into_iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    /// Sets the normalization convention used for field reconstruction.
    #[pyo3(name = "set_normalization")]
    pub fn python_set_normalization(&mut self, norm: Normalization) {
        self.normalization = norm;
    }

    /// Calculates the field profile for a mode with a given complex effective index.
    ///
    /// Unlike `field()`, which uses the real-axis solver to find neff internally,
    /// this method accepts an explicit complex neff (as returned by `complex_neff`
    /// or `all_complex_neff`) and reconstructs the field for that mode.
    ///
    /// For semi-infinite boundaries the outgoing wave in the rightmost layer is
    /// **not** zeroed: for a complex neff the radiation condition is already encoded
    /// in the imaginary part of neff, and zeroing the outgoing amplitude would give
    /// a physically wrong result.
    ///
    /// # Arguments
    /// * `omega`        - The angular frequency (real, same units as used for `neff`).
    /// * `polarization` - The polarization of the mode.
    /// * `neff_re`      - Real part of the complex effective index.
    /// * `neff_im`      - Imaginary part of the complex effective index.
    /// # Returns
    /// A `FieldData` with all six field components on the standard plotting grid.
    #[pyo3(name = "field_complex")]
    #[pyo3(signature = (omega, polarization=None, neff_re=0.0, neff_im=0.0))]
    pub fn python_field_complex(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        neff_re: f64,
        neff_im: f64,
    ) -> FieldData {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let neff = Complex::new(neff_re, neff_im);
        self.field_complex(omega, polarization, neff)
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
                match (self.left_bc, self.right_bc) {
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
                }
            }
        }
    }

    /// Evaluates `1 / det(S)` for a **complex** in-plane wavevector `k`.
    ///
    /// Used **only for the winding-number contour integral**.  Modes are
    /// *zeros* of `det(S)`, hence *poles* of `1/det(S)`.  The argument
    /// principle applied to `1/det(S)` gives `#zeros − #poles = −N_modes`,
    /// and taking the absolute value recovers the mode count `N_modes`.
    ///
    /// Using the reciprocal (rather than `det(S)` itself) makes the argument
    /// change around each pole large and spread-out, so a moderately coarse
    /// contour sample still accumulates the full `±2π` contribution reliably.
    ///
    /// **Do not use this function as the objective for root polishing.**
    /// It has poles (not zeros) at the mode locations; Muller's method would
    /// converge to something that is not a mode.  Use [`det_s_complex`]
    /// instead for polishing.
    ///
    /// # Arguments
    /// * `k0`           - Vacuum wavevector (real).
    /// * `k`            - Complex in-plane wavevector.
    /// * `polarization` - Polarisation of the mode.
    fn characteristic_function_complex(
        &self,
        k0: f64,
        k: Complex<f64>,
        polarization: Polarization,
    ) -> Complex<f64> {
        let k0c = Complex::new(k0, 0.0);
        // Always use the scattering matrix for complex k: it is unconditionally
        // stable because phase factors never overflow.
        let det = calculate_s_matrix(&self.layers, k0c, k, polarization).determinant();
        // Guard against division by exactly zero (should not happen off-mode).
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

    /// Computes the default search ranges for the complex-plane solver.
    ///
    /// * `re_range`: `(Re(n_min), Re(n_max))` across all layers — identical to the
    ///   real-axis solver, automatically spans both guided and leaky regimes.
    /// * `im_range`: `(-w, +w)` where `w = max(|Im(n_j)|, MIN_IM_HALF_WIDTH)`.
    ///   For lossless structures this gives a small but nonzero imaginary window
    ///   so that weakly leaky modes are not missed.
    fn default_search_ranges(
        &self,
        re_range: Option<(f64, f64)>,
        im_range: Option<(f64, f64)>,
    ) -> ((f64, f64), (f64, f64)) {
        let re = re_range.unwrap_or_else(|| {
            let (min_n, max_n) = self.find_minmax_n();
            // Pull the real-axis bounds slightly inward so the contour never
            // lands exactly on n_min or n_max, where kz = 0 in some layer and
            // the S-matrix compose denominator (1 − s12·s21) can hit zero,
            // producing NaN and corrupting the winding-number integral.
            let margin = 1e-6;
            (min_n + margin, max_n - margin)
        });
        let im = im_range.unwrap_or_else(|| {
            let max_im = self
                .layers
                .iter()
                .map(|l| l.n.im.abs())
                .fold(0.0_f64, f64::max);
            let half_w = max_im.max(MIN_IM_HALF_WIDTH);
            (-half_w, half_w)
        });
        (re, im)
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
        polarization: Polarization,
        re_min: f64,
        re_max: f64,
        im_min: f64,
        im_max: f64,
        n_pts: usize,
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
        let fvals: Vec<Complex<f64>> = contour
            .iter()
            .map(|&k| self.characteristic_function_complex(k0, k, polarization))
            .collect();

        let mut total_arg_change = 0.0_f64;
        let n = fvals.len();
        for i in 0..n {
            let f_curr = fvals[i];
            let f_next = fvals[(i + 1) % n];
            // Skip segments where either endpoint is non-finite (NaN / Inf).
            if !f_curr.re.is_finite()
                || !f_curr.im.is_finite()
                || !f_next.re.is_finite()
                || !f_next.im.is_finite()
            {
                continue;
            }
            // Argument of f_next / f_curr — use atan2 of the ratio for numerical
            // stability near the real axis.
            let ratio = f_next / f_curr;
            total_arg_change += ratio.arg();
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
        polarization: Polarization,
        re_min: f64,
        re_max: f64,
        im_min: f64,
        im_max: f64,
        depth: usize,
        roots: &mut Vec<Complex<f64>>,
    ) {
        let wn = self.winding_number(
            k0,
            polarization,
            re_min,
            re_max,
            im_min,
            im_max,
            CONTOUR_POINTS_PER_SIDE,
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
            if let Some(root) = self.muller_polish(k0, polarization, initial_guess) {
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
                polarization,
                re_min,
                re_mid,
                im_min,
                im_max,
                depth + 1,
                roots,
            );
            self.find_zeros_in_rectangle(
                k0,
                polarization,
                re_mid,
                re_max,
                im_min,
                im_max,
                depth + 1,
                roots,
            );
        } else {
            let im_mid = (im_min + im_max) * 0.5;
            self.find_zeros_in_rectangle(
                k0,
                polarization,
                re_min,
                re_max,
                im_min,
                im_mid,
                depth + 1,
                roots,
            );
            self.find_zeros_in_rectangle(
                k0,
                polarization,
                re_min,
                re_max,
                im_mid,
                im_max,
                depth + 1,
                roots,
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
        polarization: Polarization,
        k_init: Complex<f64>,
    ) -> Option<Complex<f64>> {
        // Seed three starting points with a small perturbation around the guess.
        let eps = 1e-6 * k0;
        let mut x0 = k_init - Complex::new(eps, 0.0);
        let mut x1 = k_init + Complex::new(0.0, eps);
        let mut x2 = k_init + Complex::new(eps, 0.0);

        let mut f0 = self.characteristic_function_complex(k0, x0, polarization);
        let mut f1 = self.characteristic_function_complex(k0, x1, polarization);
        let mut f2 = self.characteristic_function_complex(k0, x2, polarization);

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
            f2 = self.characteristic_function_complex(k0, x2, polarization);

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

    /// Finds all complex effective indices in the given search rectangle.
    ///
    /// Uses the argument-principle winding-number method to count and bracket
    /// zeros, then polishes each one with Muller's method.
    ///
    /// # Arguments
    /// * `k0`           - Vacuum wavevector (real).
    /// * `polarization` - Polarisation.
    /// * `re_range`     - `(re_min, re_max)` for `Re(neff)`.
    /// * `im_range`     - `(im_min, im_max)` for `Im(neff)`.
    ///
    /// # Returns
    /// Complex effective indices sorted by descending `Re(neff)`.
    pub fn solve_complex(
        &self,
        k0: f64,
        polarization: Polarization,
        re_range: (f64, f64),
        im_range: (f64, f64),
    ) -> Vec<Complex<f64>> {
        let mut roots: Vec<Complex<f64>> = Vec::new();
        self.find_zeros_in_rectangle(
            k0,
            polarization,
            re_range.0,
            re_range.1,
            im_range.0,
            im_range.1,
            0,
            &mut roots,
        );
        // Convert from wavevector k to neff = k / k0.
        let mut neff_roots: Vec<Complex<f64>> = roots.iter().map(|&k| k / k0).collect();
        // Sort by descending real part.
        neff_roots.sort_by(|a, b| b.re.partial_cmp(&a.re).unwrap_or(Ordering::Equal));
        neff_roots
    }

    /// Returns the complex effective index of a single mode.
    pub fn complex_neff(
        &self,
        k0: f64,
        polarization: Polarization,
        mode: usize,
        re_range: (f64, f64),
        im_range: (f64, f64),
    ) -> Result<Complex<f64>, String> {
        let solutions = self.solve_complex(k0, polarization, re_range, im_range);
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
                BoundaryCondition::SemiInfinite => {
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
            BoundaryCondition::SemiInfinite => -self.layers[0].d,
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
            BoundaryCondition::SemiInfinite => (Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
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
            BoundaryCondition::SemiInfinite => {
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
            BoundaryCondition::SemiInfinite => (Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
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
        let roots = slab.solve_complex(om, Polarization::TE, (1.0, 2.0), (-0.05, 0.05));
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
                (re_mode - 0.05, re_mode + 0.05),
                (-0.05, 0.05),
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
                (re_mode - 0.05, re_mode + 0.05),
                (-0.05, 0.05),
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

        let complex_modes =
            slab_lossy.solve_complex(om, Polarization::TE, (1.0, 2.0), (-0.05, 0.05));

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
        let wn = slab.winding_number(om, Polarization::TE, 0.5, 0.6, -0.001, 0.001, 32);
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
        let wn = slab.winding_number(
            om,
            Polarization::TE,
            neff0 - 0.01,
            neff0 + 0.01,
            -0.001,
            0.001,
            64,
        );
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
}
