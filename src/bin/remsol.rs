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

#[derive(Serialize, Deserialize, Debug)]
struct Settings {
    layers: Vec<Layer>,
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
    // Read the file contents into a string
    let contents = fs::read_to_string(file_path)?;
    let run_settings: Settings = toml::from_str(&contents)?;
    Ok(run_settings)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Specify the path to the file
    //
    let cli = Cli::parse();
    let file_path = cli.file;

    let mut settings = parse_file(file_path)?;

    let multilayer = MultiLayer::new(settings.layers);

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
