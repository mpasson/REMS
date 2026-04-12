//! This modules implements the enums used in the library.
extern crate clap;
extern crate serde;

use clap::ValueEnum;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Enum for schosing the back end of the simulation.
#[pyclass]
#[derive(Copy, Clone)]
pub enum BackEnd {
    /// Implements the scattering matrix method.
    /// Only supports index finding and not the field plotting.
    Scattering,
    /// Implements the transfer matrix method.
    /// Supports index finding and field plotting.
    Transfer,
}

/// Enum for choosing the polarization of the light.
#[pyclass]
#[derive(Copy, Clone, Serialize, Deserialize, Debug, ValueEnum)]
pub enum Polarization {
    /// TE polarization.
    TE,
    /// TM polarization.
    TM,
}

/// Enum for choosing the boundary condition at the edges of the multilayer stack.
#[pyclass]
#[derive(Copy, Clone, Debug)]
pub enum BoundaryCondition {
    /// Semi-infinite boundary condition: the field decays exponentially into the cladding.
    SemiInfinite,
    /// Perfect Electric Conductor (PEC) boundary condition: the tangential electric field vanishes at the boundary.
    PEC,
}

/// Enum for choosing the field normalization convention.
#[pyclass]
#[derive(Copy, Clone, Debug, Default)]
pub enum Normalization {
    /// Normalize so that the maximum of the total electric field amplitude is 1.
    /// This is the default and works for all modes including leaky and lossy modes.
    #[default]
    MaxField,
    /// Normalize so that the absolute value of the integrated z-component of the
    /// Poynting vector equals 1. Only valid for lossless guided modes; using it
    /// for complex neff will emit a warning and fall back to MaxField.
    Power,
}
