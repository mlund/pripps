// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Per-kind excluded-volume dummies from the atom form-factor table.
//!
//! For each structure atom/residue, this generator copies the per-kind
//! excluded-solvent form factor from the same YAML entry as the atomistic
//! form factor.

use crate::formfactor::FormFactorMap;
use crate::solvent::DummySet;
use crate::{Error, Result, Structure};

pub(super) fn dummies(structure: &Structure, formfactors: &FormFactorMap) -> Result<DummySet> {
    let mut positions = Vec::with_capacity(structure.pos.len());
    let mut form_factors = Vec::with_capacity(structure.pos.len());

    for (&id, &pos) in structure.ids.iter().zip(&structure.pos) {
        let ff = formfactors
            .excluded_solvent(id)
            .ok_or_else(|| {
                Error::validation(format!(
                    "atom kind {} has no `excluded_solvent`; required by ExcludedVolume::PerKind",
                    formfactors.kind(id).name()
                ))
            })?
            .clone();
        positions.push(pos);
        form_factors.push(ff);
    }

    log::info!(
        "Per-kind excluded volume: {} dummies from form-factor table",
        positions.len()
    );
    DummySet::per_dummy(positions, form_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formfactor::FormFactor;
    use crate::{AtomKind, Vector3};

    fn polynomial(coeff0: f64) -> FormFactor {
        FormFactor::Polynomial {
            qrange: (0.0, 1.0),
            coeff: vec![coeff0],
        }
    }

    fn map_with(name: &str, atom_coeff0: f64, excluded_coeff0: Option<f64>) -> FormFactorMap {
        let kind = AtomKind::new(name);
        FormFactorMap::from_entries_with_excluded(vec![(
            kind,
            polynomial(atom_coeff0),
            excluded_coeff0.map(polynomial),
        )])
    }

    #[test]
    fn copies_excluded_solvent_from_atom_map() {
        let atom_map = map_with("ALA", 10.0, Some(3.5));
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };

        let d = dummies(&structure, &atom_map).unwrap();
        let mut f = Vec::new();
        d.form_factors_at(0.2, &mut f).unwrap();
        assert_eq!(f, vec![3.5]);
    }

    #[test]
    fn missing_excluded_solvent_errors() {
        let atom_map = map_with("ALA", 10.0, None);
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };

        let err = dummies(&structure, &atom_map).unwrap_err();
        assert!(format!("{err}").contains("excluded_solvent"));
    }
}
