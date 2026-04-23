// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Polynomial excluded-volume dummies from a separate form-factor table.
//!
//! For each structure atom/residue, this generator copies a polynomial
//! excluded-volume form factor from a dedicated YAML table (typically
//! `assets/poly_atev.yaml`) by matching on kind name.

use crate::formfactor::FormFactorMap;
use crate::solvent::DummySet;
use crate::{Error, Result, Structure};

pub(super) fn dummies(
    structure: &Structure,
    atom_formfactors: &FormFactorMap,
    excluded_formfactors: &FormFactorMap,
) -> Result<DummySet> {
    let mut positions = Vec::with_capacity(structure.pos.len());
    let mut form_factors = Vec::with_capacity(structure.pos.len());

    for (&id, &pos) in structure.ids.iter().zip(&structure.pos) {
        let kind_name = atom_formfactors.kind(id).name();
        let ff = excluded_formfactors
            .form_factor_by_name(kind_name)
            .ok_or_else(|| {
                Error::validation(format!(
                    "atom kind {kind_name} is missing from the polynomial excluded-volume table"
                ))
            })?
            .clone();
        positions.push(pos);
        form_factors.push(ff);
    }

    log::info!(
        "Polynomial excluded volume: {} dummies from separate table",
        positions.len()
    );
    DummySet::per_dummy(positions, form_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formfactor::FormFactor;
    use crate::{AtomKind, Vector3};

    fn map_with(name: &str, coeff0: f64) -> FormFactorMap {
        let kind = AtomKind::new(name);
        let ff = FormFactor::Polynomial {
            qrange: (0.0, 1.0),
            coeff: vec![coeff0],
        };
        FormFactorMap::from_entries(vec![(kind, ff)])
    }

    #[test]
    fn copies_form_factors_by_kind_name() {
        let atom_map = map_with("ALA", 10.0);
        let excluded_map = map_with("ALA", 3.5);
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };

        let d = dummies(&structure, &atom_map, &excluded_map).unwrap();
        let mut f = Vec::new();
        d.form_factors_at(0.2, &mut f).unwrap();
        assert_eq!(f, vec![3.5]);
    }

    #[test]
    fn missing_kind_in_excluded_table_errors() {
        let atom_map = map_with("ALA", 10.0);
        let excluded_map = map_with("GLY", 3.5);
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };

        let err = dummies(&structure, &atom_map, &excluded_map).unwrap_err();
        assert!(format!("{err}").contains("ALA"));
    }
}
