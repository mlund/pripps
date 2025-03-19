pub mod cli;
pub mod debye;
pub mod explicit;
mod structure;
mod trajectory;
use anyhow::{Result, bail};
use indicatif::ProgressBar;
use std::collections::BTreeMap;
use std::iter::zip;
use std::path::{self, PathBuf};

pub use structure::Structure;
pub mod formfactor;
pub use faunus::Point as Vector3;
pub use faunus::topology::{AtomKind, AtomKindBuilder};
pub use formfactor::FormFactor;
use trajectory::{XtcIterator, get_side_lengths};

pub trait IntensityCalculator {
    /// Calculate I(q) for a single structure
    fn calc_intensities(&self, structure: &Structure) -> Result<Vec<(f64, f64)>>;

    /// Set the side lengths of the box if applicable
    fn set_side_lengths(&self, _side_lengths: Vector3) {}

    /// Average I(q) over a trajectory of structures
    fn calc_intensities_xtc(
        &self,
        structure: &Structure,
        xtc_path: &path::Path,
        weights: Option<&path::Path>,
    ) -> Result<Vec<(f64, f64)>> {
        let reader = XtcIterator::open(xtc_path, weights)?;
        let num_frames = reader.num_frames;
        let progress = ProgressBar::new(reader.num_frames as u64);

        log::info!("Averaging 𝐼(𝑞) over {num_frames} frames");

        let mut mean_intensity = Vec::new();
        let mut structure = structure.clone();

        for (frame, weight) in reader {
            structure.set_positions(&frame)?;

            if let Some(side_length) = get_side_lengths(&frame) {
                self.set_side_lengths(side_length);
            }
            let intensity = self.calc_intensities(&structure)?;

            if mean_intensity.is_empty() {
                mean_intensity = intensity.iter().map(|(q, _)| (*q, 0.0)).collect();
            }

            for ((_, m), (_, i)) in zip(mean_intensity.iter_mut(), intensity) {
                *m += weight * i / num_frames as f64;
            }

            progress.inc(1);
        }

        Ok(mean_intensity)
    }
}

/// High-level interface to the Pripps library
pub struct Pripps {
    /// Form factors for each atom kind
    formfactors: BTreeMap<AtomKind, FormFactor>,
    /// List of atom kinds derived from `self.formfactors`
    atomkinds: Vec<AtomKind>,
}

impl Pripps {
    /// Create a new a YAML file with form factors
    pub fn from_yaml(path: &PathBuf) -> Result<Self> {
        let formfactors = formfactor::read_formfactors(path)?;
        let atomkinds = formfactors.keys().cloned().collect::<Vec<AtomKind>>();
        Ok(Self {
            formfactors,
            atomkinds,
        })
    }

    /// Calculate I(q) for a given structure using the Debye formula
    ///
    /// If `xtcfile` is provided, the intensity is averaged over all frames in the trajectory
    /// (the original structure is not part of the average).
    pub fn calc_intensities(
        &self,
        method: &dyn IntensityCalculator,
        xyzfile: &std::path::Path,
        xtcfile: Option<&path::Path>,
        weight_path: Option<&path::Path>,
        side_lengths: Option<Vector3>,
    ) -> Result<Vec<(f64, f64)>> {
        let structure = Structure::from_xyz(xyzfile, &self.atomkinds)?;
        if let Some(xtcfile) = xtcfile {
            if side_lengths.is_some() {
                bail!("manual box length cannot be used with trajectories");
            }
            method.calc_intensities_xtc(&structure, xtcfile, weight_path)
        } else {
            method.calc_intensities(&structure)
        }
    }
}
