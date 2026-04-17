// Copyright 2025 Mikael Lund
// PDB parser adapted from voronota-ltr (Copyright 2026 Kliment Olechnovic and
// Mikael Lund, MIT Licence).
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Minimal PDB and XYZ readers. Replaces chemfiles for the subset of formats
//! that pripps needs: positions and per-atom names for lookup in an
//! [`AtomKind`] list.

use crate::error::IoResultExt;
use crate::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A single atom extracted from an input file: species name plus Cartesian
/// position (Å). For PDB files `residue` carries the three-letter residue
/// name (e.g. "ALA", "LYS"); for XYZ files it is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomRecord {
    pub name: String,
    pub residue: Option<String>,
    pub position: [f64; 3],
}

/// Read an atomic structure file and return its atom records.
///
/// Format is chosen from the path extension: `.pdb` / `.ent` / `.pdb1` for
/// PDB, everything else is parsed as XYZ.
pub fn read_structure(path: &Path) -> Result<Vec<AtomRecord>> {
    let file = File::open(path).with_path(path)?;
    let reader = BufReader::new(file);

    let format = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);

    let atoms = match format.as_deref() {
        Some("pdb" | "ent" | "pdb1") => parse_pdb(reader),
        _ => parse_xyz(reader),
    };

    if atoms.is_empty() {
        return Err(Error::validation(format!(
            "no atoms found in {}",
            path.display()
        )));
    }
    Ok(atoms)
}

/// Extract a substring from fixed-width PDB columns (1-indexed, inclusive).
fn extract_column(line: &str, start: usize, end: usize) -> &str {
    let len = line.len();
    let start_idx = start.saturating_sub(1);
    let end_idx = end.min(len);
    if start_idx >= len {
        return "";
    }
    line.get(start_idx..end_idx).unwrap_or("").trim()
}

fn parse_column_f64(line: &str, start: usize, end: usize) -> Option<f64> {
    let s = extract_column(line, start, end);
    if s.is_empty() { None } else { s.parse().ok() }
}

/// Move leading digits to the end of an atom name (e.g. "1CA" → "CA1"),
/// matching the convention used by PDB-reading tools.
fn normalize_atom_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let first_letter = name.find(|c: char| !c.is_ascii_digit());
    match first_letter {
        Some(0) | None => name.to_string(),
        Some(pos) => format!("{}{}", &name[pos..], &name[..pos]),
    }
}

/// Parse ATOM / HETATM records from a PDB stream.
///
/// Alternate locations other than `A`, `1`, `.` or empty are dropped. Multi-
/// model ensembles stop at the first `ENDMDL`; `END` terminates the file.
fn parse_pdb<R: BufRead>(reader: R) -> Vec<AtomRecord> {
    let mut atoms = Vec::new();
    for line in reader.lines().map_while(|r| r.ok()) {
        let record = extract_column(&line, 1, 6);
        match record {
            "ATOM" | "HETATM" => {
                let alt_loc = extract_column(&line, 17, 17);
                if !alt_loc.is_empty() && alt_loc != "A" && alt_loc != "1" && alt_loc != "." {
                    continue;
                }
                let Some(x) = parse_column_f64(&line, 31, 38) else {
                    continue;
                };
                let Some(y) = parse_column_f64(&line, 39, 46) else {
                    continue;
                };
                let Some(z) = parse_column_f64(&line, 47, 54) else {
                    continue;
                };
                let raw_name = extract_column(&line, 13, 16);
                if raw_name.is_empty() {
                    continue;
                }
                let residue = {
                    let r = extract_column(&line, 18, 20);
                    if r.is_empty() {
                        None
                    } else {
                        Some(r.to_string())
                    }
                };
                atoms.push(AtomRecord {
                    name: normalize_atom_name(raw_name),
                    residue,
                    position: [x, y, z],
                });
            }
            "ENDMDL" | "END" => break,
            _ => {}
        }
    }
    atoms
}

/// Parse a lenient XYZ stream. Blank lines, lines starting with `;` or `#`,
/// and header lines that don't tokenise as `name x y z` are skipped, so both
/// strict XYZ (`count`, `comment`, `name x y z`…) and loose variants (semicolon
/// comment header, no count) parse without configuration.
fn parse_xyz<R: BufRead>(reader: R) -> Vec<AtomRecord> {
    let mut atoms = Vec::new();
    for line in reader.lines().map_while(|r| r.ok()) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(name) = parts.next() else { continue };
        let Some(x) = parts.next().and_then(|s| s.parse::<f64>().ok()) else {
            continue;
        };
        let Some(y) = parts.next().and_then(|s| s.parse::<f64>().ok()) else {
            continue;
        };
        let Some(z) = parts.next().and_then(|s| s.parse::<f64>().ok()) else {
            continue;
        };
        atoms.push(AtomRecord {
            name: name.to_string(),
            residue: None,
            position: [x, y, z],
        });
    }
    atoms
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PDB: &str = "\
ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C
HETATM   99  ZN  ZN  A 100       5.000   5.000   5.000  1.00  0.00          ZN
END
";

    #[test]
    fn pdb_basic() {
        let atoms = parse_pdb(SAMPLE_PDB.as_bytes());
        assert_eq!(atoms.len(), 4);
        assert_eq!(atoms[0].name, "N");
        assert_eq!(atoms[1].name, "CA");
        assert_eq!(atoms[1].position, [1.458, 0.0, 0.0]);
        assert_eq!(atoms[3].name, "ZN");
    }

    #[test]
    fn pdb_normalizes_numbered_atom_names() {
        let line =
            "ATOM      1 1HG2 LEU A   1       1.000   2.000   3.000  1.00  0.00           H\n";
        let atoms = parse_pdb(line.as_bytes());
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].name, "HG21");
    }

    #[test]
    fn xyz_standard_header() {
        let sample = "2\ncomment line\nH 0.0 0.0 0.0\nO 1.5 0.0 0.0\n";
        let atoms = parse_xyz(sample.as_bytes());
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].name, "H");
        assert_eq!(atoms[1].position, [1.5, 0.0, 0.0]);
    }

    #[test]
    fn xyz_loose_header_with_semicolon_comment() {
        let sample = "; lysozyme\nLYS 1.0 2.0 3.0\nTYR 4.0 5.0 6.0\n";
        let atoms = parse_xyz(sample.as_bytes());
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].name, "LYS");
        assert_eq!(atoms[1].position, [4.0, 5.0, 6.0]);
    }
}
