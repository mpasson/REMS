extern crate clap;
extern crate serde;
extern crate toml;

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

use remsol::enums::Polarization;
use remsol::layer::Layer;
use remsol::multilayer::MultiLayer;

/// A TOML-friendly mirror of `Layer` that stores `n` as a plain `f64`.
/// After deserialisation it is converted to `Layer` (which stores `n` as
/// `Complex<f64>`) via `From<SerdeLayer>`.
#[derive(Serialize, Deserialize, Debug)]
struct SerdeLayer {
    n: f64,
    d: f64,
}

impl From<SerdeLayer> for Layer {
    fn from(sl: SerdeLayer) -> Layer {
        Layer::from_real(sl.n, sl.d)
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Settings {
    layers: Vec<SerdeLayer>,
    #[serde(default = "Settings::default_runs")]
    runs: Vec<RunSettings>,
}

impl Settings {
    fn default_runs() -> Vec<RunSettings> {
        vec![]
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct RunSettings {
    k0: f64,
    polarization: Polarization,
    mode: usize,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the TOML file
    #[arg(short, long, default_value = "./structure.toml")]
    file: String,
    /// Vacuum wavevector
    #[arg(short, long)]
    k0: Option<f64>,
    /// Polarization
    #[arg(short, long)]
    polarization: Option<Polarization>,
    /// Mode
    #[arg(short, long)]
    mode: Option<usize>,
}

fn parse_file(file_path: String) -> Result<Settings, Box<dyn Error>> {
    let contents = fs::read_to_string(file_path)?;
    let settings: Settings = toml::from_str(&contents)?;
    Ok(settings)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let file_path = cli.file;

    let mut settings = parse_file(file_path)?;

    // Convert SerdeLayer → Layer.
    let layers: Vec<Layer> = settings.layers.into_iter().map(Layer::from).collect();
    let multilayer = MultiLayer::new(layers);

    if let (Some(k0), Some(mode), Some(polarization)) = (cli.k0, cli.mode, cli.polarization) {
        settings.runs.push(RunSettings {
            k0,
            mode,
            polarization,
        });
    }

    for run in settings.runs {
        let result = multilayer.neff(run.k0, run.polarization, run.mode);
        println!("{:?}", result);
    }

    Ok(())
}
