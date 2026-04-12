//!Rust module for calculation of electromagnetic modes in 1D multilayer structures.

use pyo3::prelude::*;

pub mod enums;
pub mod layer;
pub mod multilayer;
pub mod scattering_matrix;
pub mod transfer_matrix;

use enums::{BackEnd, BoundaryCondition, Normalization, Polarization};
use layer::{Layer, PEC};
use multilayer::{FieldData, IndexData, MultiLayer};

#[pymodule]
fn remsol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<BackEnd>()?;
    m.add_class::<Polarization>()?;
    m.add_class::<Normalization>()?;
    m.add_class::<Layer>()?;
    m.add_class::<MultiLayer>()?;
    m.add_class::<IndexData>()?;
    m.add_class::<FieldData>()?;
    m.add_class::<BoundaryCondition>()?;
    m.add_class::<PEC>()?;
    Ok(())
}
