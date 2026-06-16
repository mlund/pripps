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
        /// Cubic box side length (Å). Omit when using a trajectory
        /// (the per-frame box from the XTC is used instead).
        #[clap(short = 'b', long = "box")]
        side_length: Option<f64>,
        #[clap(flatten)]
        solvent: SolventArgs,
    },
    /// Scattering intensity via spherical-harmonic multipole expansion, O(N).
    Multipole {
        /// Minimum q value (1/Å)
        #[clap(long, default_value = "0.0")]
        qmin: f64,
        /// Maximum q value (1/Å)
        #[clap(long, default_value = "0.75")]
        qmax: f64,
        /// Number of q values. If unset, auto-selected at the
        /// Nyquist rate `2·l_max + 1` (floored at 21) so the theory
        /// curve is sampled densely enough to cubic-spline-interpolate
        /// onto experimental q without aliasing.
        #[clap(long)]
        nq: Option<usize>,
        /// Expansion order (multipole truncation). If unset,
        /// auto-computed from the structure's extent and qmax.
        #[clap(long)]
        lmax: Option<usize>,
        #[clap(flatten)]
        solvent: SolventArgs,
    },
}

/// Solvent correction parameters shared by Multipole and Direct.
#[derive(Debug, Clone, clap::Args)]
pub struct SolventArgs {
    /// Excluded-volume model
    #[clap(long, value_enum, default_value_t = ExcludedKind::Disabled)]
    excluded: ExcludedKind,
    /// Scale on excluded-volume displaced volumes (PepsiSAXS `r0`)
    #[clap(long, default_value = "1.0")]
    volume_scale: f64,
    /// Grid spacing (Å) for `--excluded grid`
    #[clap(long, default_value = "1.0")]
    excluded_spacing: f64,
    /// Bulk solvent electron density (e/Å³)
    #[clap(long, default_value = "0.334")]
    bulk_electron_density: f64,
    /// Hydration-shell model
    #[clap(long, value_enum, default_value_t = HydrationKind::Disabled)]
    hydration: HydrationKind,
    /// Water probe radius (Å). Defaults: 1.8 for SASA/grid,
    /// 6.0 for voronoi (larger probe smooths the granular
    /// explicit-sphere shell into an envelope-like layer).
    #[clap(long)]
    probe: Option<f64>,
    /// Grid spacing (Å) for `--hydration grid`. Default is the
    /// bulk-water linear spacing (≈30 Å³/molecule), so each grid
    /// cell holds exactly one water molecule (weight = 1.0).
    #[clap(long, default_value = "3.10516")]
    spacing: f64,
    /// Shell thickness (Å) for `--hydration grid`
    #[clap(long, default_value = "3.0")]
    thickness: f64,
    /// Hydration shell excess electron density (e/Å³).
    /// ~0.03 for SASA, ~0.017 for grid
    #[clap(long, default_value = "0.03")]
    contrast_density: f64,
}

impl SolventArgs {
    /// Build a [`SolventConfig`], returning `None` when neither
    /// excluded volume nor hydration is active.
    fn to_config(&self) -> Option<SolventConfig> {
        let excluded = match self.excluded {
            ExcludedKind::Disabled => ExcludedVolume::Disabled,
            ExcludedKind::Fraser => ExcludedVolume::Fraser {
                volume_scale: self.volume_scale,
            },
            ExcludedKind::Foxs => ExcludedVolume::FraserFoxs {
                volume_scale: self.volume_scale,
            },
            ExcludedKind::Grid => ExcludedVolume::Grid {
                spacing: self.excluded_spacing,
                volume_scale: self.volume_scale,
            },
            ExcludedKind::Voronoi => ExcludedVolume::Voronoi {
                volume_scale: self.volume_scale,
            },
        };
        let hydration = match self.hydration {
            HydrationKind::Disabled => Hydration::Disabled,
            HydrationKind::Sasa => Hydration::Sasa {
                probe: self.probe.unwrap_or(1.8),
                contrast_density: self.contrast_density,
            },
            HydrationKind::Grid => Hydration::Grid {
                spacing: self.spacing,
                thickness: self.thickness,
                probe: self.probe.unwrap_or(1.8),
                contrast_density: self.contrast_density,
            },
            HydrationKind::Voronoi => Hydration::Voronoi {
                probe: self.probe.unwrap_or(6.0),
                contrast_density: self.contrast_density,
            },
        };
        let cfg = SolventConfig {
            excluded,
            hydration,
            bulk_electron_density: self.bulk_electron_density,
        };
        cfg.is_active().then_some(cfg)
    }
}

/// Which excluded-volume generator to use.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ExcludedKind {
    #[value(alias = "none")]
    Disabled,
    Fraser,
    Foxs,
    Grid,
    Voronoi,
}

/// Which hydration-shell generator to use.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum HydrationKind {
    #[value(alias = "none")]
    Disabled,
    Sasa,
    Grid,
    Voronoi,
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

    /// Experimental SAXS data (q, I, σ) for fitting c₁ and c₂
    #[clap(short = 'f', long)]
    fit: Option<PathBuf>,

    /// Optional .xtc trajectory file with structures to average over
    #[clap(short = 'x', long)]
    xtcfile: Option<PathBuf>,

    /// Optional stride for XTC frames (default: 1, i.e. use all frames)
    #[clap(short = 's', long, default_value = "1", requires = "xtcfile")]
    xtc_stride: std::num::NonZeroUsize,

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

    if let Some(fit_path) = &args.fit {
        // Fit mode: optimise (c1, c2) against experimental data
        let exp = crate::fit::ExpData::from_file(fit_path)?;
        let solvent = solvent.unwrap_or_default();

        match scheme {
            IntensityScheme::Multipole {
                qmin: _,
                qmax: _,
                nq,
                l_max,
            } => {
                let model =
                    pripps.multipole_model(&args.input, 0.0, exp.q_max(), nq, l_max, &solvent)?;
                let result = crate::fit::fit_parameters(&exp, |c1, c2| model.intensity(c1, c2))?;
                let out = model.intensity(result.volume_scale, result.contrast_density);
                crate::fit::write_fit_csv(&args.output, &exp, &out, result.scale)?;
            }
            _ => {
                return Err(crate::Error::usage(
                    "fitting is currently only supported by the Multipole scheme",
                ));
            }
        };
    } else {
        let xtc_opts = args
            .xtcfile
            .as_deref()
            .map(|xtcfile| crate::TrajectoryOptions {
                xtcfile,
                stride: args.xtc_stride,
                weights: args.weights.as_deref(),
            });
        // Normal mode: compute I(q) and write output
        let intensity = pripps.intensity(
            scheme,
            &args.input,
            xtc_opts,
            box_override,
            solvent.as_ref(),
        )?;
        write_csv(&args.output, &intensity)?;
    }

    Ok(())
}

/// Translate parsed CLI arguments into an [`IntensityScheme`], an
/// optional explicit box override (used by `Direct`), and an optional
/// [`SolventConfig`] (only produced by `Multipole` when an excluded or
/// hydration model is selected).
fn command_to_scheme(
    command: Commands,
) -> (IntensityScheme, Option<[f64; 3]>, Option<SolventConfig>) {
    match command {
        Commands::Debye { qmin, qmax, nq } => {
            (IntensityScheme::Debye { qmin, qmax, nq }, None, None)
        }
        Commands::Direct {
            pmax,
            side_length,
            solvent,
        } => (
            IntensityScheme::Direct { pmax },
            side_length.map(|s| [s; 3]),
            solvent.to_config(),
        ),
        Commands::Multipole {
            qmin,
            qmax,
            nq,
            lmax,
            solvent,
        } => (
            IntensityScheme::Multipole {
                qmin,
                qmax,
                nq,
                l_max: lmax,
            },
            None,
            solvent.to_config(),
        ),
    }
}

/// Write the intensity curve as CSV: two columns for the vacuum path,
/// five when the three-contribution Multipole path populated
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
