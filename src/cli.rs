// Copyright 2025 Isabel Vinterbladh, Sergei Grudinin and Mikael Lund
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
use clap::{Parser, Subcommand};
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
        /// XYZ input file
        #[clap(short, long)]
        input: PathBuf,
        /// Minimum q value (1/Å)
        #[clap(long, default_value = "0.01")]
        qmin: f64,
        /// Maximum q value (1/Å)
        #[clap(long, default_value = "3.0")]
        qmax: f64,
        /// Number of q values
        #[clap(long, default_value = "20")]
        nq: usize,
        /// Output file with I(q)
        #[clap(short = 'o', long, default_value = "intensity.dat")]
        output: PathBuf,
    },
}

#[derive(Parser)]
#[clap(version, about, long_about = None)]
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
        } => {
            let intensity = pripps.debye_intensities(&xyzfile, (qmin, qmax), nq)?;
            let mut output = std::fs::File::create(output)?;
            writeln!(output, "# q/Å I(q)")?;
            for (q, i) in intensity {
                writeln!(output, "{:.6} {:.6}", q, i)?;
            }
        }
    };
    Ok(())
}
