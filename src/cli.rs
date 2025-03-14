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

use clap::{Parser, Subcommand};
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
