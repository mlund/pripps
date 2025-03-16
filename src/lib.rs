pub mod cli;
pub mod debye;
mod structure;
use anyhow::Result;
use std::collections::BTreeMap;

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
    pub fn from_yaml(path: &std::path::PathBuf) -> Result<Self> {
        let formfactors = formfactor::read_formfactors(path)?;
        let atomkinds = formfactors.keys().cloned().collect::<Vec<AtomKind>>();
        Ok(Self {
            formfactors,
            atomkinds,
        })
    }

    /// Calculate I(q) for a given structure using the Debye formula
    pub fn debye_intensities(
        &self,
        xyzfile: &std::path::PathBuf,
        qrange: &[f64],
    ) -> Result<Vec<(f64, f64)>> {
        let structure = Structure::from_xyz(xyzfile, &self.atomkinds)?;
        debye::calc_intensity(&structure, &self.formfactors, qrange)
    }
}
