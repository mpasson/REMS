//! Implementation of the Layer and related structs
extern crate num_complex;

use num_complex::Complex;
use pyo3::prelude::*;

/// Struct representing the modal coefficient inside a layer.
#[derive(Debug, Copy, Clone)]
pub struct LayerCoefficientVector {
    /// Forward propagating wave.
    pub a: Complex<f64>,
    /// Backward propagating wave.
    pub b: Complex<f64>,
}

impl LayerCoefficientVector {
    /// Create a new LayerCoefficientVector.
    pub fn new(a: Complex<f64>, b: Complex<f64>) -> LayerCoefficientVector {
        LayerCoefficientVector { a, b }
    }
}

/// Struct representing a layer in the stack.
///
/// The refractive index `n` is stored as a complex number to support both
/// lossless (`Im(n) = 0`) and lossy / gain media.  From Python the index may
/// be supplied as either a plain `float` (backward-compatible) or as a Python
/// `complex`.
///
/// This class is also available in the Python API.
#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct Layer {
    /// Complex refractive index of the layer.
    /// For lossless media `Im(n) = 0`.
    pub n: Complex<f64>,
    /// Thickness of the layer.
    pub d: f64,
}

/// Implementation of the Python API for the Layer struct.
#[pymethods]
impl Layer {
    /// Create a new Layer.
    ///
    /// # Arguments
    /// * `n` - The refractive index of the layer.  May be a Python `float` or
    ///   `complex`.  A plain float is promoted to `Complex { re: n, im: 0.0 }`.
    /// * `d` - The thickness of the layer.
    #[new]
    pub fn new(n: &Bound<'_, PyAny>, d: f64) -> PyResult<Layer> {
        let n_complex = extract_complex(n)?;
        Ok(Layer { n: n_complex, d })
    }

    /// Define how a layer is printed in Python.
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    /// Define how a layer is printed in Python.
    fn __repr__(&self) -> PyResult<String> {
        if self.n.im == 0.0 {
            Ok(format!("Layer(n={}, d={})", self.n.re, self.d))
        } else {
            Ok(format!(
                "Layer(n={}+{}j, d={})",
                self.n.re, self.n.im, self.d
            ))
        }
    }
}

impl Layer {
    /// Convenience constructor for Rust code that takes a plain real index.
    pub fn from_real(n: f64, d: f64) -> Layer {
        Layer {
            n: Complex::new(n, 0.0),
            d,
        }
    }

    /// Convenience constructor for Rust code that takes a complex index.
    pub fn from_complex(n: Complex<f64>, d: f64) -> Layer {
        Layer { n, d }
    }
}

/// Extract a `Complex<f64>` from a Python object that is either a `float` or a
/// `complex`.  Returns a `PyValueError` if the object is neither.
pub fn extract_complex(obj: &Bound<'_, PyAny>) -> PyResult<Complex<f64>> {
    // Try plain float first (most common, backward-compatible path).
    if let Ok(re) = obj.extract::<f64>() {
        return Ok(Complex::new(re, 0.0));
    }
    // Try Python complex.
    if let Ok(c) = obj.extract::<num_complex::Complex<f64>>() {
        return Ok(c);
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "Refractive index must be a float or complex number",
    ))
}

/// Boundary marker struct representing a Perfect Electric Conductor (PEC).
/// Place as the first or last element of a MultiLayer list to apply a PEC boundary condition.
#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct PEC;

#[pymethods]
impl PEC {
    /// Create a new PEC boundary marker.
    #[new]
    pub fn new() -> PEC {
        PEC
    }

    /// Define how a PEC is printed in Python.
    fn __repr__(&self) -> PyResult<String> {
        Ok(String::from("PEC()"))
    }
}
