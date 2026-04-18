// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Voronoi solvent-sphere hydration shell.
//!
//! Places explicit water-sized spheres on the solvent-accessible
//! surface using voronota-ltr's `compute_solvent_spheres`. Each
//! sphere sits at an actual surface position (not at atom centers
//! like SASA) and carries a weight proportional to its Voronoi
//! cell volume — larger weight means more accessible space.
//!
//! The two-stage algorithm:
//! 1. Tessellate atoms at probe radius, sample icosahedron
//!    vertices on each atom's probe-expanded surface, keep points
//!    in the atom's Voronoi region.
//! 2. Re-tessellate the combined atom+sphere system at probe=0
//!    to assign Voronoi cell volumes as weights.

use voronota_ltr::{SubdivisionDepth, compute_solvent_spheres};

use crate::formfactor::FormFactorMap;
use crate::solvent::{DummySet, atom_radii, balls_from_radii, water_form_factor};
use crate::{Error, Result, Structure, Vector3};

/// Build Voronoi solvent-sphere hydration dummies.
pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    probe: f64,
) -> Result<DummySet> {
    let atoms = atom_radii(structure, formfactors, "Hydration::Voronoi")?;
    let balls = balls_from_radii(&atoms);

    // Depth0 (12 icosahedron vertices per atom) is sufficient for
    // SAXS resolution — finer subdivisions add spheres without
    // improving the fit but increase the O(N·Q·L²) expansion cost.
    let spheres =
        compute_solvent_spheres(&balls, probe, None, Some(SubdivisionDepth::Depth0), None)
            .map_err(|e| Error::internal(format!("Voronoi solvent spheres: {e}")))?;

    let positions: Vec<Vector3> = spheres
        .iter()
        .map(|s| Vector3::new(s.x, s.y, s.z))
        .collect();

    // Normalize per parent atom so each atom's spheres sum to 1.
    // Without this, the ~12 spheres per atom with raw Voronoi
    // cell volumes (ų) make the hydration intensity ~100× larger
    // than SASA's, pushing c₂ to ~0.0003 (unmeaningful). Per-atom
    // normalization keeps c₂ in the ~0.02 range and makes the
    // model probe-insensitive.
    let mut parent_weight_sum = vec![0.0_f64; atoms.len()];
    for s in &spheres {
        parent_weight_sum[s.parent_index] += s.weight;
    }
    let weights: Vec<f64> = spheres
        .iter()
        .map(|s| {
            let total = parent_weight_sum[s.parent_index];
            if total > 0.0 { s.weight / total } else { 0.0 }
        })
        .collect();

    log::info!(
        "Voronoi hydration shell: {} solvent spheres at probe={probe:.1} Å",
        positions.len()
    );

    let water = water_form_factor(formfactors, "Hydration::Voronoi")?;
    DummySet::weighted(positions, water, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomKind;
    use crate::formfactor::FormFactor;

    fn two_atom_map() -> FormFactorMap {
        let mut kind = AtomKind::new("C");
        kind.sigma = Some(2.0 * 1.7);
        FormFactorMap::from_entries(vec![
            (kind, FormFactor::Constant(6.0)),
            (AtomKind::new("H2O"), FormFactor::Constant(10.0)),
        ])
    }

    #[test]
    fn produces_dummies() {
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 0.0, 0.0)],
            ids: vec![0, 0],
            box_sides: None,
        };
        let ff = two_atom_map();
        let d = dummies(&structure, &ff, 1.4).unwrap();
        let n = d.positions().len();
        assert!(n > 10, "expected many solvent spheres, got {n}");

        // All spheres should be outside the atom envelopes
        for p in d.positions() {
            for &atom_pos in &structure.pos {
                let dist = (p - atom_pos).norm();
                assert!(
                    dist > 1.5,
                    "solvent sphere at {p:?} too close to atom at {atom_pos:?}: d={dist:.2}"
                );
            }
        }
    }

    #[test]
    fn weights_are_positive() {
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 0.0, 0.0)],
            ids: vec![0, 0],
            box_sides: None,
        };
        let ff = two_atom_map();
        let d = dummies(&structure, &ff, 1.4).unwrap();
        let mut out = Vec::new();
        d.form_factors_at(0.0, &mut out).unwrap();
        for (i, f) in out.iter().enumerate() {
            assert!(*f > 0.0, "sphere {i} has non-positive F(0) = {f}");
        }
    }

    #[test]
    fn empty_structure() {
        let structure = Structure {
            pos: vec![],
            ids: vec![],
            box_sides: None,
        };
        let ff = two_atom_map();
        let d = dummies(&structure, &ff, 1.4).unwrap();
        assert_eq!(d.positions().len(), 0);
    }
}
