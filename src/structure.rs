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

use crate::input::read_structure;
use crate::trajectory::get_side_lengths;
use crate::{AtomKind, Error, FindByName, Result, Vector3};
use molly::Frame;
use std::{
    fmt::{self, Display, Formatter},
    path::Path,
};

/// Ad hoc molecular structure: per-atom positions, kind ids, and optional
/// periodic box.
#[derive(Debug, Clone, Default)]
pub(crate) struct Structure {
    pub(crate) pos: Vec<Vector3>,
    pub(crate) ids: Vec<usize>,
    pub(crate) box_sides: Option<Vector3>,
}

/// Resolve a PDB atom to its united-atom form-factor name using
/// residue context. Returns `None` if the (residue, atom) pair
/// has no mapping — the caller falls back to atom-name-only lookup.
//
// The mapping is fixed biochemistry (standard amino acids), not
// user-configurable, so it's a const lookup table rather than a
// file. FoXS hardcodes the same table in
// `FormFactorTable.cpp:679-715`.
pub(crate) fn resolve_united_atom(residue: &str, atom: &str) -> Option<&'static str> {
    // Per-residue sidechain overrides checked first. When a residue
    // overrides a backbone atom (e.g. GLY:CA → CH2, PRO:N → N),
    // the match returns before the backbone fallback.
    let sidechain = match (residue, atom) {
        // Glycine — CA has two H
        ("GLY", "CA") => Some("CH2"),
        // Proline — backbone N has no H
        ("PRO", "N") => Some("N"),
        // Alanine
        ("ALA", "CB") => Some("CH3"),
        // Arginine
        ("ARG", "CB" | "CG" | "CD") => Some("CH2"),
        ("ARG", "NE") => Some("NH"),
        ("ARG", "CZ") => Some("C"),
        ("ARG", "NH1" | "NH2") => Some("NH2"),
        // Asparagine
        ("ASN", "CB") => Some("CH2"),
        ("ASN", "CG") => Some("C"),
        ("ASN", "OD1") => Some("O"),
        ("ASN", "ND2") => Some("NH2"),
        // Aspartate
        ("ASP", "CB") => Some("CH2"),
        ("ASP", "CG") => Some("C"),
        ("ASP", "OD1" | "OD2") => Some("O"),
        // Cysteine
        ("CYS", "CB") => Some("CH2"),
        ("CYS", "SG") => Some("SH"),
        // Glutamine
        ("GLN", "CB" | "CG") => Some("CH2"),
        ("GLN", "CD") => Some("C"),
        ("GLN", "OE1") => Some("O"),
        ("GLN", "NE2") => Some("NH2"),
        // Glutamate
        ("GLU", "CB" | "CG") => Some("CH2"),
        ("GLU", "CD") => Some("C"),
        ("GLU", "OE1" | "OE2") => Some("O"),
        // Histidine
        ("HIS", "CB") => Some("CH2"),
        ("HIS", "CG") => Some("C"),
        ("HIS", "ND1") => Some("N"),
        ("HIS", "CD2" | "CE1") => Some("CaromH"),
        ("HIS", "NE2") => Some("NH"),
        // Isoleucine
        ("ILE", "CB") => Some("CH"),
        ("ILE", "CG1") => Some("CH2"),
        ("ILE", "CG2" | "CD1") => Some("CH3"),
        // Leucine
        ("LEU", "CB") => Some("CH2"),
        ("LEU", "CG") => Some("CH"),
        ("LEU", "CD1" | "CD2") => Some("CH3"),
        // Lysine
        ("LYS", "CB" | "CG" | "CD" | "CE") => Some("CH2"),
        ("LYS", "NZ") => Some("NH3"),
        // Methionine
        ("MET", "CB" | "CG") => Some("CH2"),
        ("MET", "SD") => Some("S"),
        ("MET", "CE") => Some("CH3"),
        // Phenylalanine
        ("PHE", "CB") => Some("CH2"),
        ("PHE", "CG") => Some("C"),
        ("PHE", "CD1" | "CD2" | "CE1" | "CE2" | "CZ") => Some("CaromH"),
        // Proline sidechain
        ("PRO", "CB" | "CG" | "CD") => Some("CH2"),
        // Serine
        ("SER", "CB") => Some("CH2"),
        ("SER", "OG") => Some("O"),
        // Threonine
        ("THR", "CB") => Some("CH"),
        ("THR", "OG1") => Some("O"),
        ("THR", "CG2") => Some("CH3"),
        // Tryptophan
        ("TRP", "CB") => Some("CH2"),
        ("TRP", "CG" | "CD2" | "CE2") => Some("C"),
        ("TRP", "CD1" | "CE3" | "CZ2" | "CZ3" | "CH2") => Some("CaromH"),
        ("TRP", "NE1") => Some("NH"),
        // Tyrosine
        ("TYR", "CB") => Some("CH2"),
        ("TYR", "CG" | "CZ") => Some("C"),
        ("TYR", "CD1" | "CD2" | "CE1" | "CE2") => Some("CaromH"),
        ("TYR", "OH") => Some("O"),
        // Valine
        ("VAL", "CB") => Some("CH"),
        ("VAL", "CG1" | "CG2") => Some("CH3"),
        _ => None,
    };

    // Backbone atoms shared by all standard residues (except where
    // overridden above: GLY:CA, PRO:N).
    sidechain.or(match atom {
        "N" => Some("NH"),
        "CA" => Some("CH"),
        "C" => Some("C"),
        "O" | "OXT" => Some("O"),
        _ => None,
    })
}

impl Structure {
    /// Construct a structure from a PDB or XYZ file. Each atom is
    /// looked up by name in `atomkinds`. For PDB files, the lookup
    /// first tries [`resolve_united_atom`] with residue context to
    /// map heavy atoms to their united-atom form-factor type (e.g.
    /// LYS:CB → CH₂), falling back to atom-name-only lookup for
    /// atoms not in the table or for XYZ files (which have no
    /// residue context).
    pub fn from_file(path: &Path, atomkinds: &[AtomKind]) -> Result<Self> {
        let atoms = read_structure(path)?;

        let mut pos = Vec::with_capacity(atoms.len());
        let mut ids = Vec::with_capacity(atoms.len());

        for atom in atoms {
            let resolved_name = atom
                .residue
                .as_deref()
                .and_then(|res| resolve_united_atom(res, &atom.name))
                .unwrap_or(&atom.name);

            let id = atomkinds.find_name(resolved_name).ok_or_else(|| {
                Error::UnknownAtom(format!(
                    "{}; add an entry for this atom type to the form-factor YAML",
                    atom.name
                ))
            })?;
            pos.push(Vector3::new(
                atom.position[0],
                atom.position[1],
                atom.position[2],
            ));
            ids.push(id);
        }

        Ok(Self {
            pos,
            ids,
            box_sides: None,
        })
    }

    /// Axis-aligned bounding box of all atom positions.
    pub(crate) fn bounding_box(&self) -> (Vector3, Vector3) {
        let mut lo = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &self.pos {
            lo = lo.inf(p);
            hi = hi.sup(p);
        }
        (lo, hi)
    }

    /// Refresh positions and box sides from a trajectory frame.
    pub fn set_positions(&mut self, frame: &Frame) -> Result<()> {
        if frame.natoms() != self.pos.len() {
            return Err(Error::validation(format!(
                "number of atoms in structure ({}) does not match trajectory frame ({})",
                self.pos.len(),
                frame.natoms(),
            )));
        }
        const ANGSTROM_PER_NM: f64 = 10.0;
        for (pos, new_pos) in self.pos.iter_mut().zip(frame.coords()) {
            *pos = ANGSTROM_PER_NM
                * Vector3::new(new_pos[0] as f64, new_pos[1] as f64, new_pos[2] as f64);
        }
        if let Some(sides) = get_side_lengths(frame) {
            self.box_sides = Some(sides);
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
