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

use crate::Vector3;
use crate::input::read_structure;
use anyhow::{Context, Result, bail};
use faunus::topology::{AtomKind, FindByName};
use molly::Frame;
use std::{
    fmt::{self, Display, Formatter},
    path::Path,
};

/// Ad hoc molecular structure: per-atom positions and kind ids.
#[derive(Debug, Clone, Default)]
pub struct Structure {
    /// Particle positions
    pub pos: Vec<Vector3>,
    /// Atom kind ids
    pub ids: Vec<usize>,
}

impl Structure {
    /// Number of atoms in the structure
    #[inline]
    pub fn len(&self) -> usize {
        self.pos.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }
    /// Construct a structure from a PDB or XYZ file, looking each atom name up
    /// in the provided [`AtomKind`] list for its id. Unknown atom names error.
    pub fn from_file(path: &Path, atomkinds: &[AtomKind]) -> Result<Self> {
        let atoms = read_structure(path)?;

        let mut pos = Vec::with_capacity(atoms.len());
        let mut ids = Vec::with_capacity(atoms.len());

        for atom in atoms {
            let kind = atomkinds
                .find_name(&atom.name)
                .with_context(|| format!("unknown atom name in file: {:?}", atom.name))?;
            pos.push(Vector3::new(
                atom.position[0],
                atom.position[1],
                atom.position[2],
            ));
            ids.push(kind.id());
        }

        Ok(Self { pos, ids })
    }

    /// Set positions from a trajectory frame
    pub fn set_positions(&mut self, frame: &Frame) -> Result<()> {
        if frame.natoms() != self.pos.len() {
            bail!("number of atoms in structure does not match trajectory frame");
        }
        const ANGSTROM_PER_NM: f64 = 10.0;
        for (pos, new_pos) in self.pos.iter_mut().zip(frame.coords()) {
            *pos = ANGSTROM_PER_NM
                * Vector3::new(new_pos[0] as f64, new_pos[1] as f64, new_pos[2] as f64);
        }
        Ok(())
    }
}

/// Display number of atoms, mass center etc.
impl Display for Structure {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "𝑁={}", self.pos.len())
    }
}
