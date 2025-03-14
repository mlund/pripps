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
        #[clap(long, default_value = "10.0")]
        qmax: f64,
        /// Number of q values
        #[clap(long, default_value = "20")]
        nq: usize,
    },
}

#[derive(Parser)]
#[clap(version, about, long_about = None)]
pub struct Args {
    #[clap(subcommand)]
    pub command: Commands,

    /// Atom definition file
    #[clap(long, short = 'a')]
    pub atoms: Option<PathBuf>,

    /// Verbose output. See more with e.g. RUST_LOG=Trace
    #[clap(long, short = 'v', action)]
    pub verbose: bool,
}
