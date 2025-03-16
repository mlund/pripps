pub mod cli;
pub mod debye;
mod structure;
use anyhow::{Result, bail};
use itertools::Itertools;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub use structure::Structure;
pub mod formfactor;
pub use faunus::Point as Vector3;
pub use faunus::topology::{AtomKind, AtomKindBuilder};
pub use formfactor::FormFactor;

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

    fn read_weights(&self, path: &PathBuf) -> Result<Vec<f64>> {
        let read_error = |e| bail!("error reading weights: {e}");
        let parse_error = |e| bail!("error parsing weights: {e}");

        let weights: Vec<_> = std::fs::read_to_string(path)
            .or_else(read_error)?
            .lines()
            .map(|w| w.parse().or_else(parse_error))
            .try_collect()?;

        log::info!("Read {} weights from {}", weights.len(), path.display());
        Ok(weights)
    }

    /// Calculate I(q) for a given structure using the Debye formula
    ///
    /// If `xtcfile` is provided, the intensity is averaged over all frames in the trajectory
    /// (the original structure is not part of the average).
    pub fn debye_intensities(
        &self,
        xyzfile: &PathBuf,
        qrange: &[f64],
        xtcfile: Option<&PathBuf>,
        weight_path: Option<PathBuf>,
    ) -> Result<Vec<(f64, f64)>> {
        let structure = Structure::from_xyz(xyzfile, &self.atomkinds)?;

        if let Some(xtcfile) = xtcfile {
            let weights = match weight_path {
                Some(path) => Some(self.read_weights(&path)?),
                None => None,
            };
            debye::calc_intensity_xtc(&structure, &self.formfactors, qrange, xtcfile, weights)
        } else {
            debye::calc_intensity(&structure, &self.formfactors, qrange)
        }
    }
}
