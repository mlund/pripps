// Copyright 2024 Mikael Lund
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

use crate::Vector3;
use anyhow::{Context, Result};
use faunus::topology::{AtomKind, FindByName};
use itertools::Itertools;
use log::debug;
use std::{
    fmt::{self, Display, Formatter},
    path::PathBuf,
};

/// Ad hoc molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug, Clone, Default)]
pub struct Structure {
    /// Particle positions
    pub pos: Vec<Vector3>,
    /// Atom kind ids
    pub kinds: Vec<AtomKind>,
}

impl Structure {
    /// Constructs a new structure from an XYZ file, centering the structure at the origin
    pub fn from_xyz(path: &PathBuf, atomkinds: &[AtomKind]) -> Result<Self> {
        let nxyz: Vec<(String, Vector3)> = std::fs::read_to_string(path)
            .context(format!("Could not read XYZ file {}", path.display()))?
            .lines()
            .skip(2) // skip header
            .map(from_xyz_line)
            .try_collect()?;

        let atom_ids = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .find_name(name)
                    .map(|kind| kind.clone())
                    .context(format!("Unknown atom name in structure file: {name}"))
            })
            .try_collect()?;

        let structure = Self {
            pos: nxyz.iter().map(|(_, pos)| *pos).collect(),
            kinds: atom_ids,
            ..Default::default()
        };
        debug!("Read {}: {}", path.display(), structure);
        Ok(structure)
    }
}

/// Display number of atoms, mass center etc.
impl Display for Structure {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "𝑁={}", self.pos.len())
    }
}

/// Parse a single line from an XYZ file
fn from_xyz_line(line: &str) -> Result<(String, Vector3)> {
    if let Some([name, x, y, z]) = line.split_whitespace().collect_array() {
        Ok((
            name.to_string(),
            Vector3::new(x.parse()?, y.parse()?, z.parse()?),
        ))
    } else {
        anyhow::bail!("'name x y z' expected in XYZ record: {}", line);
    }
}
