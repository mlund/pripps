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
use anyhow::{Context, Result, bail};
use faunus::topology::{AtomKind, FindByName};
use itertools::Itertools;
use molly::Frame;
use std::{
    fmt::{self, Display, Formatter},
    path::Path,
};
use voronota::{Ball, RadicalTessellation};

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
    /// Constructs a new structure from a file using chemfiles
    pub fn from_file(path: &Path, atomkinds: &[AtomKind]) -> Result<Self> {
        let mut traj = chemfiles::Trajectory::open(path, 'r')
            .context(format!("couldn't open file {}", path.display()))?;

        let to_vector3 = |pos: &[_; 3]| Vector3::new(pos[0], pos[1], pos[2]);

        let mut frame = chemfiles::Frame::new();
        traj.read(&mut frame)?;

        let pos = frame.positions().iter().map(to_vector3).collect();
        let ids = frame
            .iter_atoms()
            .map(|atom| {
                atomkinds
                    .find_name(&atom.name())
                    .map(|kind| kind.id())
                    .context(format!("unknown atom name in file: {:?}", atom.name()))
            })
            .try_collect()?;

        let radii = frame
            .iter_atoms()
            .map(|atom| {
                atomkinds
                    .find_name(&atom.name())
                    .map(|kind| kind.sigma().map(|sigma| 0.5 * sigma))
                    .unwrap_or_default()
                    .context(format!("unknown atom name in file: {:?}", atom.name()))
            })
            .try_collect()?;

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
    /// Calculate SASA for each atom
    pub fn calc_sasa(&self, probe_radius: f64) -> Vec<f64> {
        assert_eq!(self.pos.len(), self.radii.len());

        log::info!(
            "Calculating SASA for {} atoms with probe radius {:.1} Å",
            self.pos.len(),
            probe_radius
        );

        let balls = self
            .pos
            .iter()
            .zip(&self.radii)
            .map(|(p, r)| Ball {
                x: p.x,
                y: p.y,
                z: p.z,
                r: *r,
            })
            .collect();

        RadicalTessellation::from_balls(probe_radius, &balls, None, None, false)
            .cells
            .iter()
            .map(|c| c.sas_area)
            .collect()
    }
}

/// Display number of atoms, mass center etc.
impl Display for Structure {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "𝑁={}", self.pos.len())
    }
}
