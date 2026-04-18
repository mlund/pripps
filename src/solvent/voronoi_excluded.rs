// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Voronoi-based excluded volume.
//!
//! Each atom gets a Fraser-style Gaussian form factor sized by its
//! individual Voronoi cell volume from the radical tessellation:
//!
//! ```text
//! F_i(q) = ρ_s · V_i · exp(−V_i^(2/3) · q² / (4π))
//! ```
//!
//! Unlike per-atom Fraser (`ExcludedVolume::Fraser`), this derives
//! displaced volumes geometrically from the structure via Voronoi
//! tessellation — no tabulated `displaced_volume` entries needed,
//! only atomic radii. This makes it suitable for coarse-grained
//! or non-standard structures where empirical volume tables are
//! unavailable.

use std::f64::consts::PI;

use voronota_ltr::{Results, compute_tessellation};

use crate::formfactor::FormFactorMap;
use crate::solvent::fraser::{PEPSI_DENOMINATOR, fraser_form_factor};
use crate::solvent::{DummySet, atom_radii, balls_from_radii};
use crate::{Result, Structure};

/// Build Voronoi excluded-volume dummies: one per atom, each with
/// a Fraser Gaussian sized by its Voronoi cell volume.
pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    bulk_electron_density: f64,
) -> Result<DummySet> {
    let atoms = atom_radii(structure, formfactors, "ExcludedVolume::Voronoi")?;
    let n = atoms.len();
    if n == 0 {
        return DummySet::per_dummy(vec![], vec![]);
    }

    let balls = balls_from_radii(&atoms);

    // Probe=0 gives the radical tessellation: space is partitioned
    // by the power distance to each sphere, so every point in the
    // molecular interior is assigned to exactly one atom. No SAS
    // expansion avoids probe-dependent volume inflation that makes
    // the fit erratic.
    let tessellation = compute_tessellation(&balls, 0.0, None, None, false);
    let volumes = tessellation.volumes();

    let n_none = volumes.iter().filter(|v| v.is_none()).count();
    if n_none > 0 {
        log::warn!(
            "Voronoi excluded volume: {n_none}/{n} atoms have no Voronoi cell \
             (lonely atoms); using sphere volume fallback"
        );
    }

    let per_atom_volumes: Vec<f64> = volumes
        .iter()
        .zip(&balls)
        // None = lonely atom with no Voronoi neighbours (e.g. an
        // isolated HETATM metal ion). Fall back to the bare sphere
        // volume since there is no cell to measure.
        .map(|(vol, ball)| vol.unwrap_or_else(|| 4.0 / 3.0 * PI * ball.r.powi(3)))
        .collect();

    let total_volume: f64 = per_atom_volumes.iter().sum();

    log::info!(
        "Voronoi excluded volume: {n} atoms, V_total={total_volume:.1} Å³, \
         V_avg={:.2} Å³",
        total_volume / n as f64
    );

    // Per-atom Fraser Gaussian sized by each atom's Voronoi cell
    // volume — same formula as Fraser but with geometry-derived
    // volumes instead of tabulated displaced_volume values.
    let form_factors: Vec<_> = per_atom_volumes
        .iter()
        .map(|&v| fraser_form_factor(v, bulk_electron_density, PEPSI_DENOMINATOR))
        .collect();
    DummySet::per_dummy(structure.pos.clone(), form_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomKind;
    use crate::Vector3;
    use crate::formfactor::FormFactor;

    fn one_atom_map() -> FormFactorMap {
        let mut kind = AtomKind::new("C");
        kind.sigma = Some(2.0 * 1.7);
        FormFactorMap::from_entries(vec![(kind, FormFactor::Constant(6.0))])
    }

    #[test]
    fn one_dummy_per_atom() {
        let structure = Structure {
            pos: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(3.0, 0.0, 0.0),
                Vector3::new(0.0, 4.0, 0.0),
            ],
            ids: vec![0, 0, 0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let d = dummies(&structure, &ff, 0.334).unwrap();
        assert_eq!(d.positions().len(), 3);
        assert_eq!(d.positions()[1], Vector3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn form_factor_decays_with_q() {
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 0.0, 0.0)],
            ids: vec![0, 0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let d = dummies(&structure, &ff, 0.334).unwrap();
        let mut f0 = Vec::new();
        let mut f1 = Vec::new();
        d.form_factors_at(0.0, &mut f0).unwrap();
        d.form_factors_at(0.5, &mut f1).unwrap();
        assert!(f1[0] < f0[0], "Fraser Gaussian should decay with q");
        assert!(f1[0] > 0.0);
    }

    #[test]
    fn form_factor_at_q_zero_is_positive() {
        let rho: f64 = 0.334;
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 0.0, 0.0)],
            ids: vec![0, 0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let d = dummies(&structure, &ff, rho).unwrap();
        let mut out = Vec::new();
        d.form_factors_at(0.0, &mut out).unwrap();
        assert!(out[0] > 0.0);
        assert!(out[1] > 0.0);
    }

    #[test]
    fn empty_structure() {
        let structure = Structure {
            pos: vec![],
            ids: vec![],
            box_sides: None,
        };
        let ff = FormFactorMap::from_entries(vec![]);
        let d = dummies(&structure, &ff, 0.334).unwrap();
        assert_eq!(d.positions().len(), 0);
    }
}
