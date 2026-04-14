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
use voronota_ltr::{Ball, Results, compute_tessellation};

/// Ad hoc molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug, Clone, Default)]
pub struct Structure {
    /// Particle positions
    pub pos: Vec<Vector3>,
    /// Atom kind ids
    pub ids: Vec<usize>,
    /// Radii of each atom
    pub radii: Vec<f64>,
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
    /// in the provided [`AtomKind`] list for its id and radius (half its
    /// sigma). Unknown atom names error out.
    pub fn from_file(path: &Path, atomkinds: &[AtomKind]) -> Result<Self> {
        let atoms = read_structure(path)?;

        let mut pos = Vec::with_capacity(atoms.len());
        let mut ids = Vec::with_capacity(atoms.len());
        let mut radii = Vec::with_capacity(atoms.len());

        for atom in atoms {
            let kind = atomkinds
                .find_name(&atom.name)
                .with_context(|| format!("unknown atom name in file: {:?}", atom.name))?;
            let sigma = kind
                .sigma()
                .with_context(|| format!("atom kind {:?} has no radius", atom.name))?;
            pos.push(Vector3::new(
                atom.position[0],
                atom.position[1],
                atom.position[2],
            ));
            ids.push(kind.id());
            radii.push(0.5 * sigma);
        }

        Ok(Self { pos, ids, radii })
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
    /// Calculate SASA for each atom. Lonely atoms (no contacts) report 0.0.
    pub fn calc_sasa(&self, probe_radius: f64) -> Vec<f64> {
        assert_eq!(self.pos.len(), self.radii.len());

        log::info!(
            "Calculating SASA for {} atoms with probe radius {:.1} Å",
            self.pos.len(),
            probe_radius
        );

        let balls: Vec<Ball> = self
            .pos
            .iter()
            .zip(&self.radii)
            .map(|(p, r)| Ball::new(p.x, p.y, p.z, *r))
            .collect();

        compute_tessellation(&balls, probe_radius, None, None, false)
            .sas_areas()
            .into_iter()
            .map(|sas| sas.unwrap_or(0.0))
            .collect()
    }
}

/// Display number of atoms, mass center etc.
impl Display for Structure {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "𝑁={}", self.pos.len())
    }
}
