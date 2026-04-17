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

use crate::error::IoResultExt;
use crate::{
    ExcludedVolume, Hydration, Intensity, IntensityScheme, Pripps, Result, SolventConfig, cli,
};
use clap::{Parser, Subcommand, ValueEnum, builder::styling::AnsiColor};

use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use std::env::set_var;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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
    /// Scattering intensity via Svergun's spherical-harmonic multipole
    /// expansion (Acta Cryst. A51, 1995). Linear in atom count —
    /// preferred for large structures and iterative fitting.
    Svergun {
        /// Minimum q value (1/Å)
        #[clap(long, default_value = "0.0")]
        qmin: f64,
        /// Maximum q value (1/Å)
        #[clap(long, default_value = "0.75")]
        qmax: f64,
        /// Number of q values
        #[clap(long, default_value = "100")]
        nq: usize,
        /// Expansion order (multipole truncation). If unset,
        /// auto-computed from the structure's extent and qmax.
        #[clap(long)]
        lmax: Option<usize>,
        /// Excluded-volume model. `none` (default) skips the
        /// excluded-volume correction; `fraser` enables
        /// Fraser-MacRae-Suzuki Gaussian dummies.
        #[clap(long, value_enum, default_value_t = ExcludedKind::None)]
        excluded: ExcludedKind,
        /// Global scale on the excluded-volume displaced volumes.
        /// PepsiSAXS calls this `r0`; typical range is 0.95–1.05.
        #[clap(long, default_value = "1.0")]
        volume_scale: f64,
        /// Bulk solvent electron density in e/Å³. Pure water is 0.334.
        #[clap(long, default_value = "0.334")]
        bulk_electron_density: f64,
        /// Hydration-shell model. `none` (default) skips it;
        /// `sasa` places one water dummy per protein atom,
        /// weighted by solvent-accessible surface area
        /// (voronota-ltr tessellation).
        #[clap(long, value_enum, default_value_t = HydrationKind::None)]
        hydration: HydrationKind,
        /// Water probe radius (Å) used for the SASA tessellation.
        /// FoXS default is 1.8.
        #[clap(long, default_value = "1.8")]
        probe: f64,
        /// Dimensionless scale on the hydration-shell contribution.
        /// FoXS calls this `c₂`; 0.03 is a typical excess-water
        /// density baseline.
        #[clap(long, default_value = "0.03")]
        contrast_density: f64,
    },
}

/// Which excluded-volume generator to use. Mirrors
/// [`ExcludedVolume`] minus the `Grid` variant (Phase 2c).
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ExcludedKind {
    None,
    Fraser,
    Foxs,
}

/// Which hydration-shell generator to use. Mirrors the active
/// [`Hydration`] variants (Phase 2b ships `Sasa` only; `Grid` and
/// `Voronoi` land in later phases).
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum HydrationKind {
    None,
    Sasa,
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

    /// Output CSV file with the I(q) curve.
    #[clap(short = 'o', long, default_value = "intensity.csv")]
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
    // SAFETY: `set_var` is unsafe because it is not thread-safe.
    // This runs in `main` before rayon's global thread pool or any
    // other threads are spawned, so the single-threaded precondition
    // holds. `pretty_env_logger::init()` reads the variable
    // immediately after.
    if args.verbose {
        unsafe { set_var(DEFAULT_FILTER_ENV, "Debug") };
    } else if std::env::var(DEFAULT_FILTER_ENV).is_err() {
        unsafe { set_var(DEFAULT_FILTER_ENV, "Info") };
    }
    pretty_env_logger::init();

    let pripps = Pripps::from_yaml(&args.atomfile)?;
    let (scheme, box_override, solvent) = command_to_scheme(args.command);

    let intensity = pripps.intensity(
        scheme,
        &args.input,
        args.xtcfile.as_deref(),
        args.weights.as_deref(),
        box_override,
        solvent.as_ref(),
    )?;
    write_csv(&args.output, &intensity)?;

    Ok(())
}

/// Translate parsed CLI arguments into an [`IntensityScheme`], an
/// optional explicit box override (used by `Direct`), and an optional
/// [`SolventConfig`] (only produced by `Svergun` when an excluded or
/// hydration model is selected).
fn command_to_scheme(
    command: Commands,
) -> (IntensityScheme, Option<[f64; 3]>, Option<SolventConfig>) {
    match command {
        Commands::Debye { qmin, qmax, nq } => {
            (IntensityScheme::Debye { qmin, qmax, nq }, None, None)
        }
        Commands::Direct { pmax, side_length } => (
            IntensityScheme::Direct { pmax },
            Some([side_length; 3]),
            None,
        ),
        Commands::Svergun {
            qmin,
            qmax,
            nq,
            lmax,
            excluded,
            volume_scale,
            bulk_electron_density,
            hydration,
            probe,
            contrast_density,
        } => {
            let solvent = SolventConfig {
                excluded: match excluded {
                    ExcludedKind::None => ExcludedVolume::None,
                    ExcludedKind::Fraser => ExcludedVolume::Fraser { volume_scale },
                    ExcludedKind::Foxs => ExcludedVolume::FraserFoxs { volume_scale },
                },
                hydration: match hydration {
                    HydrationKind::None => Hydration::None,
                    HydrationKind::Sasa => Hydration::Sasa {
                        probe,
                        contrast_density,
                    },
                },
                bulk_electron_density,
            };
            // Drop the config if neither contribution is active, so
            // the vacuum fast path and its bit-identity stay intact
            // when no solvent flag was passed.
            let solvent = solvent.is_active().then_some(solvent);
            (
                IntensityScheme::Svergun {
                    qmin,
                    qmax,
                    nq,
                    l_max: lmax,
                },
                None,
                solvent,
            )
        }
    }
}

/// Write the intensity curve as CSV: two columns for the vacuum path,
/// five when the three-contribution Svergun path populated
/// `contributions`. The first row is a literal header; subsequent
/// rows are comma-separated `f64` values — safe raw-format CSV
/// because numeric fields never contain commas or quotes.
// `File` writes are unbuffered — each `writeln!` would be one
// `write(2)` without the `BufWriter` wrapper. For a trajectory
// averaging over thousands of frames × ~100 q points, that's
// hundreds of thousands of syscalls otherwise.
fn write_csv(path: &Path, curve: &Intensity) -> Result<()> {
    let file = std::fs::File::create(path).with_path(path)?;
    let mut out = BufWriter::new(file);
    log::info!("Writing {}", path.display());

    match &curve.contributions {
        None => {
            writeln!(out, "q,total")?;
            for (q, total) in curve.q.iter().zip(&curve.total) {
                writeln!(out, "{q:.6},{total:.6}")?;
            }
        }
        Some(c) => {
            writeln!(out, "q,total,atoms,excluded,hydration")?;
            for ((((q, total), atoms), excluded), hydration) in curve
                .q
                .iter()
                .zip(&curve.total)
                .zip(&c.atoms)
                .zip(&c.excluded)
                .zip(&c.hydration)
            {
                writeln!(
                    out,
                    "{q:.6},{total:.6},{atoms:.6},{excluded:.6},{hydration:.6}"
                )?;
            }
        }
    }
    Ok(())
}
