// Copyright 2025 Isabel Vinterbladh, Rémi Vuillemot, Sergei Grudinin and Mikael Lund
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
use anyhow::{Context, Result, bail};
use faunus::topology::{AtomKind, FindByName};
use itertools::Itertools;
use molly::Frame;
use nalgebra::DMatrix;
use std::{
    fmt::{self, Display, Formatter},
    path::Path,
};

/// Ad hoc molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug, Clone, Default)]
pub struct Structure {
    /// Particle positions
    pub pos: Vec<Vector3>,
    /// Atom kind ids
    pub ids: Vec<usize>,
}

impl Structure {
    /// Constructs a new structure from an XYZ file, centering the structure at the origin
    pub fn from_xyz(path: &Path, atomkinds: &[AtomKind]) -> Result<Self> {
        let nxyz: Vec<_> = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("couldn't read XYZ file {}: {e}", path.display()))?
            .lines()
            .skip(2) // skip header
            .map(from_xyz_line)
            .try_collect()?;

        let atom_ids = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .find_name(name)
                    .map(|kind| kind.id())
                    .context(format!("Unknown atom name in structure file: {name}"))
            })
            .try_collect()?;

        let structure = Self {
            pos: nxyz.iter().map(|(_, pos)| *pos).collect(),
            ids: atom_ids,
        };
        log::info!("Read structure {}: {}", path.display(), structure);
        Ok(structure)
    }

    /// Calculate N x N distance matrix
    pub fn distance_matrix(&self) -> DMatrix<f64> {
        let n = self.pos.len();
        let mut matrix = DMatrix::<f64>::zeros(n, n);
        for i in 0..n - 1 {
            for j in i + 1..n {
                let r = (self.pos[i] - self.pos[j]).norm();
                matrix[(i, j)] = r;
                matrix[(j, i)] = r;
            }
        }
        matrix
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

/// Parse a single line from an XYZ file
fn from_xyz_line(line: &str) -> Result<(String, Vector3)> {
    if let Some([name, x, y, z]) = line.split_whitespace().collect_array() {
        Ok((
            name.to_string(),
            Vector3::new(x.parse()?, y.parse()?, z.parse()?),
        ))
    } else {
        bail!("'name x y z' expected in XYZ record: {}", line);
    }
}
