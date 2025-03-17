// Copyright 2025 Isabel Vinterbladh, Rémi Vuillemot, Sergei Grudinin and Mikael Lund
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

use crate::{Pripps, cli};
use anyhow::Result;
use clap::{Parser, Subcommand, builder::styling::AnsiColor};

use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use std::env::set_var;
use std::io::Write;
use std::path::PathBuf;

/// LASR - Labbot Application Server
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Calculate scattering intensity using the Debye formula
    #[clap()]
    Debye {
        /// XYZ structure file
        #[clap(short, long)]
        input: PathBuf,
        /// Minimum q value (1/Å)
        #[clap(long, default_value = "0.0")]
        qmin: f64,
        /// Maximum q value (1/Å)
        #[clap(long, default_value = "0.75")]
        qmax: f64,
        /// Number of q values
        #[clap(long, default_value = "20")]
        nq: usize,
        /// Output file with I(q)
        #[clap(short = 'o', long, default_value = "intensity.dat")]
        output: PathBuf,
        /// Optional .xtc trajectory file with structures to average over
        #[clap(short = 'x', long)]
        xtcfile: Option<PathBuf>,
        /// Optional file with weights for each frame in XTC file (one weight per line)
        #[clap(short = 'w', long, requires = "xtcfile")]
        weights: Option<PathBuf>,
    },
}
#[derive(Parser)]
#[clap(version, about, long_about = None, color = clap::ColorChoice::Auto)]
#[command(styles=clap::builder::Styles::styled()
    .header(AnsiColor::Green.on_default().bold())
    .usage(AnsiColor::Green.on_default().bold())
    .literal(AnsiColor::Blue.on_default().bold())
    .placeholder(AnsiColor::Blue.on_default()))]
pub struct Args {
    #[clap(subcommand)]
    pub command: Commands,

    /// Atom definition file for form factors
    #[clap(long, short = 'a')]
    pub atomfile: Option<PathBuf>,

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

    let pripps = match &args.atomfile {
        Some(atompath) => Pripps::from_yaml(atompath)?,
        None => anyhow::bail!("missing formfactor file"),
    };

    match args.command {
        cli::Commands::Debye {
            input: xyzfile,
            qmin,
            qmax,
            nq,
            output,
            xtcfile,
            weights,
        } => {
            log::info!("Using Debye formula to calculate 𝐼(𝑞) for {qmin} ≤ 𝑞 ≤ {qmax} Å⁻¹");

            let q_values: Vec<_> = (0..nq)
                .map(|i| qmin + (qmax - qmin) * i as f64 / (nq - 1) as f64)
                .collect();
            let intensity = pripps.debye_intensities(
                &xyzfile,
                &q_values,
                xtcfile.as_deref(),
                weights.as_deref(),
            )?;
            let mut output = std::fs::File::create(output)?;

            writeln!(output, "# q/Å I(q)")?;
            for (q, i) in intensity {
                writeln!(output, "{:.6} {:.6}", q, i)?;
            }
        }
    };
    Ok(())
}
