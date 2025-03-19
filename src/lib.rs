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
    /// Calculate I(q) for a single structure
    fn calc_intensities(
        &self,
        structure: &Structure,
        side_lengths: Option<Vector3>,
    ) -> Result<Vec<(f64, f64)>>;

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

            let side_lengths = get_side_lengths(&frame);
            let intensity = self.calc_intensities(&structure, side_lengths)?;

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

/// Get side lengths from a frame (Å)
///
/// This assumed that values are in nm and converts to Å
fn get_side_lengths(frame: &molly::Frame) -> Option<Vector3> {
    const ANGSTROM_PER_NM: f64 = 10.0;
    let b = frame.boxvec.to_cols_array_2d();
    let b = ANGSTROM_PER_NM * Vector3::new(b[0][0] as f64, b[1][1] as f64, b[2][2] as f64);
    log::trace!("frame side-length = {:?}", b);
    if b.norm() < 1e-6 { None } else { Some(b) }
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
            debye.calc_intensities(&structure, None)
        }
    }

    pub fn direct_intensities(
        &self,
        xyzfile: &std::path::Path,
        pmax: usize,
        side_lengths: &Vector3,
        xtcfile: Option<&path::Path>,
        weight_path: Option<&path::Path>,
    ) -> Result<Vec<(f64, f64)>> {
        let structure = Structure::from_xyz(xyzfile, &self.atomkinds)?;
        let direct = explicit::DirectTransform::new(self.formfactors.clone(), pmax);
        if let Some(xtcfile) = xtcfile {
            direct.calc_intensities_xtc(&structure, xtcfile, weight_path)
        } else {
            direct.calc_intensities(&structure, Some(*side_lengths))
        }
    }
}
