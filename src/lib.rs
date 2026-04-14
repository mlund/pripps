// Library-only builds (no `cli` feature) have no in-crate caller for the
// intensity calculators — they're only instantiated through `cli::make_method`.
// Suppress the per-item dead-code warnings in that configuration; external
// consumers (pyo3 bindings, other library users) reach them via `Pripps`.
#![cfg_attr(not(feature = "cli"), allow(dead_code))]

mod atomkind;
#[cfg(feature = "cli")]
pub mod cli;
mod debye;
mod explicit;
mod formfactor;
mod input;
mod structure;
mod trajectory;

use anyhow::{Result, ensure};
#[cfg(feature = "cli")]
use indicatif::ProgressBar;
use std::{
    collections::BTreeMap,
    iter::zip,
    path::{self, Path},
};
use structure::Structure;
use trajectory::{XtcIterator, get_side_lengths};

pub use atomkind::{AtomKind, FindByName};
pub use formfactor::{FormFactor, FormFactorMap, read_formfactors, write_formfactors};

/// Crate-internal short alias. Not re-exported; public signatures use
/// `[f64; 3]` instead so the API doesn't leak `nalgebra` types.
pub(crate) type Vector3 = nalgebra::Vector3<f64>;

/// Trait for calculating scattering intensities
pub trait IntensityCalculator {
    /// Calculate I(q) for a single structure
    fn calc_intensities(&self, structure: &Structure) -> Result<Vec<(f64, f64)>>;

    /// Set the side lengths of the box, if applicable
    fn set_side_lengths(&self, _side_lengths: [f64; 3]);

    /// Average I(q) over a trajectory of structures
    fn calc_intensities_xtc(
        &self,
        structure: &Structure,
        xtc_path: &path::Path,
        weights: Option<&path::Path>,
    ) -> Result<Vec<(f64, f64)>> {
        let reader = XtcIterator::open(xtc_path, weights)?;
        let num_frames = reader.num_frames;
        #[cfg(feature = "cli")]
        let progress = ProgressBar::new(reader.num_frames as u64);

        log::info!("Averaging 𝐼(𝑞) over {num_frames} frames");

        let mut mean_intensity = Vec::new();
        let mut structure = structure.clone();

        for (frame, weight) in reader {
            structure.set_positions(&frame)?;

            if let Some(side_lengths) = get_side_lengths(&frame) {
                self.set_side_lengths(side_lengths.into());
            }
            let intensity = self.calc_intensities(&structure)?;

            if mean_intensity.is_empty() {
                mean_intensity = intensity.iter().map(|(q, _)| (*q, 0.0)).collect();
            }

            for ((_, mean), (_, intensity)) in zip(&mut mean_intensity, intensity) {
                *mean += weight * intensity / num_frames as f64;
            }

            #[cfg(feature = "cli")]
            progress.inc(1);
        }

        Ok(mean_intensity)
    }
}

/// High-level interface to the Pripps library
pub struct Pripps {
    formfactors: BTreeMap<AtomKind, FormFactor>,
    atomkinds: Vec<AtomKind>,
}

impl Pripps {
    pub fn formfactors(&self) -> &FormFactorMap {
        &self.formfactors
    }

    /// Construct from a YAML form-factor definition file.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let formfactors = read_formfactors(path)?;
        let atomkinds = formfactors.keys().cloned().collect::<Vec<AtomKind>>();

        if std::env::var("FF_TABLE").is_ok() {
            write_formfactors(&formfactors, Path::new("tabulated_formfactors.yaml"))?;
        }

        Ok(Self {
            formfactors,
            atomkinds,
        })
    }

    /// Calculate I(q) for a structure, optionally averaged over an XTC
    /// trajectory. If `xtcfile` is provided, the single-frame structure is
    /// used only for atom names/ids; the frames are iterated from the XTC
    /// and averaged (the template structure is not part of the average).
    pub fn calc_intensities(
        &self,
        method: &dyn IntensityCalculator,
        xyzfile: &std::path::Path,
        xtcfile: Option<&path::Path>,
        weight_path: Option<&path::Path>,
        side_lengths: Option<[f64; 3]>,
    ) -> Result<Vec<(f64, f64)>> {
        let structure = Structure::from_file(xyzfile, &self.atomkinds)?;
        if let Some(xtcfile) = xtcfile {
            ensure!(
                side_lengths.is_none(),
                "manual box length cannot be used with trajectories"
            );
            method.calc_intensities_xtc(&structure, xtcfile, weight_path)
        } else {
            method.calc_intensities(&structure)
        }
    }
}
