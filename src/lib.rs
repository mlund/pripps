pub mod cli;
pub mod debye;
pub mod explicit;
mod structure;
mod trajectory;
use anyhow::Result;
use indicatif::ProgressBar;
use std::collections::BTreeMap;
use std::iter::zip;
use std::path::{self, PathBuf};

pub use structure::Structure;
pub mod formfactor;
pub use faunus::Point as Vector3;
pub use faunus::topology::{AtomKind, AtomKindBuilder};
pub use formfactor::FormFactor;
pub use trajectory::XtcIterator;

pub trait IntensityCalculator {
    fn set_box(&mut self, box_lengths: &Vector3);

    fn calc_intensities(&self, structure: &Structure) -> Result<Vec<(f64, f64)>>;

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
    pub fn debye_intensities(
        &self,
        xyzfile: &std::path::Path,
        qrange: &[f64],
        xtcfile: Option<&path::Path>,
        weight_path: Option<&path::Path>,
    ) -> Result<Vec<(f64, f64)>> {
        let debye = debye::DebyeIntensity::new(
            self.formfactors.clone(),
            (qrange[0], *qrange.last().unwrap()),
            qrange.len(),
        )?;
        let structure = Structure::from_xyz(xyzfile, &self.atomkinds)?;
        if let Some(xtcfile) = xtcfile {
            debye.calc_intensities_xtc(&structure, xtcfile, weight_path)
        } else {
            debye.calc_intensities(&structure)
        }
    }

    pub fn direct_intensities(
        &self,
        xyzfile: &std::path::Path,
        pmax: usize,
        side_lengths: &Vector3,
    ) -> Result<Vec<(f64, f64)>> {
        let structure = Structure::from_xyz(xyzfile, &self.atomkinds)?;
        let direct = explicit::DirectTransform::new(pmax);
        Ok(direct.eval(&structure, &self.formfactors, side_lengths))
    }
}
