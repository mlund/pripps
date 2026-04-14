// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Definition of command line interfaces (CLI)

use crate::{DebyeIntensity, IntensityCalculator, Pripps, Vector3, cli, explicit::DirectTransform};
use anyhow::Result;
use clap::{Parser, Subcommand, builder::styling::AnsiColor};

use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use std::env::set_var;
use std::io::Write;
use std::path::PathBuf;

/// Commands
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Scattering intensity using the Debye formula
    #[clap()]
    Debye {
        /// Minimum q value (1/Å)
        #[clap(long, default_value = "0.0")]
        qmin: f64,
        /// Maximum q value (1/Å)
        #[clap(long, default_value = "0.75")]
        qmax: f64,
        /// Number of q values
        #[clap(long, default_value = "100")]
        nq: usize,
    },
    /// Scattering intensity using discrete q vectors
    Direct {
        /// Number of Miller index multiplications
        #[clap(short = 'p', long, default_value = "6")]
        pmax: usize,
        /// Cubic box side length (Å)
        #[clap(short = 'b', long = "box", default_value = "200.0")]
        side_length: f64,
    },
}
#[derive(Parser)]
#[clap(version, about, long_about = None, color = clap::ColorChoice::Auto)]
#[command(styles=clap::builder::Styles::styled()
    .header(AnsiColor::Green.on_default().bold())
    .usage(AnsiColor::Green.on_default().bold())
    .literal(AnsiColor::Cyan.on_default().bold())
    .placeholder(AnsiColor::Cyan.on_default()))]
pub struct Args {
    #[clap(subcommand)]
    pub command: Commands,

    /// Form factor definitions for species (atoms, residues, etc.)
    #[clap(long, short = 'a')]
    pub atomfile: PathBuf,

    /// Input structure (xyz, pdb, gro etc.)
    #[clap(short = 'i', long)]
    input: PathBuf,

    /// Output file with I(q)
    #[clap(short = 'o', long, default_value = "intensity.dat")]
    output: PathBuf,

    /// Optional .xtc trajectory file with structures to average over
    #[clap(short = 'x', long)]
    xtcfile: Option<PathBuf>,

    /// Optional file with weights for each frame in XTC file (one weight per line)
    #[clap(short = 'w', long, requires = "xtcfile")]
    weights: Option<PathBuf>,

    /// Verbose output. See more with e.g. RUST_LOG=Trace
    #[clap(long, short = 'v', action)]
    pub verbose: bool,
}

// Wrapper for main function to handle errors
pub fn do_main() -> Result<()> {
    let args = cli::Args::parse();
    // Configure logging
    if args.verbose {
        unsafe { set_var(DEFAULT_FILTER_ENV, "Debug") };
    } else if std::env::var(DEFAULT_FILTER_ENV).is_err() {
        unsafe { set_var(DEFAULT_FILTER_ENV, "Info") };
    }
    pretty_env_logger::init();

    let pripps = Pripps::from_yaml(&args.atomfile)?;
    let method = make_method(&pripps, args.command)?;

    let intensity = pripps.calc_intensities(
        method.as_ref(),
        &args.input,
        args.xtcfile.as_deref(),
        args.weights.as_deref(),
        None,
    )?;
    write_table(&args.output, &intensity, Some("# q/Å I(q)"))?;

    Ok(())
}

/// Construct an `IntensityCalculator` from parsed CLI arguments.
pub fn make_method(pripps: &Pripps, command: Commands) -> Result<Box<dyn IntensityCalculator>> {
    use Commands::*;
    match command {
        Debye { qmin, qmax, nq } => {
            log::info!("Using Debye formula to calculate 𝐼(𝑞) for {qmin} ≤ 𝑞 ≤ {qmax} Å⁻¹");
            let method = DebyeIntensity::new(pripps.formfactors().clone(), (qmin, qmax), nq)?;
            Ok(Box::new(method) as Box<dyn IntensityCalculator>)
        }
        Direct { pmax, side_length } => {
            let side_lengths = Vector3::new(side_length, side_length, side_length);
            let method = DirectTransform::new(pripps.formfactors().clone(), pmax, side_lengths);
            Ok(Box::new(method) as Box<dyn IntensityCalculator>)
        }
    }
}

/// Write f(x) to a file
fn write_table(path: &std::path::Path, table: &[(f64, f64)], header: Option<&str>) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("error writing intensity file: {e}"))?;

    log::info!("Writing {}", path.display());

    if let Some(header) = header {
        writeln!(file, "{header}")?;
    }

    for (x, y) in table {
        writeln!(file, "{x:.6} {y:.6}")?;
    }
    Ok(())
}
