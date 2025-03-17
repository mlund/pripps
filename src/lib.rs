pub mod cli;
pub mod debye;
pub mod explicit;
mod structure;
mod trajectory;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::{self, PathBuf};

pub use structure::Structure;
pub mod formfactor;
pub use faunus::Point as Vector3;
pub use faunus::topology::{AtomKind, AtomKindBuilder};
pub use formfactor::FormFactor;
pub use trajectory::XtcIterator;

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
        let structure = Structure::from_xyz(xyzfile, &self.atomkinds)?;

        if let Some(xtcfile) = xtcfile {
            debye::calc_intensity_xtc(&structure, &self.formfactors, qrange, xtcfile, weight_path)
        } else {
            debye::calc_intensity(&structure, &self.formfactors, qrange)
        }
    }
}
