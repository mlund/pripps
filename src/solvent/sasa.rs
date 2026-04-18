// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! SASA-based hydration shell via voronota-ltr.
//!
//! FoXS-style (<https://doi.org/10.1016/j.bpj.2013.07.020>): each
//! protein atom `i` gets a dimensionless accessibility fraction
//! `sᵢ ∈ [0, 1]` and scatters through a shared water form factor as
//!
//! ```text
//! f^wat_i(q) = sᵢ · f_water(q)
//! ```
//!
//! There are no explicit water coordinates — the shell is an
//! SASA-weighted skin colocated with the protein atoms. The fit
//! knob `contrast_density` (FoXS `c₂`) scales the whole hydration
//! contribution at the coefficient-level combine step, not here —
//! so the generator depends only on atom positions, radii, and the
//! probe and can run once per structure.

use std::f64::consts::PI;

use voronota_ltr::{Results, compute_tessellation};

use crate::formfactor::FormFactorMap;
use crate::solvent::{DummySet, atom_radii, balls_from_radii, water_form_factor};
use crate::{Result, Structure};

pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    probe: f64,
) -> Result<DummySet> {
    let atoms = atom_radii(structure, formfactors, "Hydration::Sasa")?;
    let balls = balls_from_radii(&atoms);

    // One tessellation pass gives per-atom accessible surface area.
    // `None` means the atom's Voronoi cell is unbounded — i.e. it
    // has no occluding neighbors, which physically means fully
    // exposed, so the accessibility fraction is 1.
    let tessellation = compute_tessellation(&balls, probe, None, None, false);
    let areas = tessellation.sas_areas();

    let weights: Vec<f64> = balls
        .iter()
        .zip(areas.iter())
        .map(|(ball, area)| {
            let full_sphere = 4.0 * PI * (ball.r + probe).powi(2);
            // `clamp`: Voronoi areas can slightly exceed the flat
            // sphere area because cell boundaries track tangent
            // planes, not curved surfaces. Keeps the weight
            // meaningful.
            area.as_ref()
                .map_or(1.0, |a| (a / full_sphere).clamp(0.0, 1.0))
        })
        .collect();

    log::info!(
        "SASA hydration: {} dummies at probe={probe:.1} Å",
        weights.len()
    );
    let water = water_form_factor(formfactors, "Hydration::Sasa")?;
    DummySet::weighted(structure.pos.clone(), water, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomKind;
    use crate::formfactor::FormFactor;

    fn carbon_kind(radius: Option<f64>) -> AtomKind {
        let mut k = AtomKind::new("C");
        k.sigma = radius.map(|r| 2.0 * r);
        k
    }

    fn water_kind() -> (AtomKind, FormFactor) {
        (
            AtomKind::new("H2O"),
            // Trivial water form factor for tests — a constant so
            // `water.eval(q) = 10.0` at every q.
            FormFactor::Constant(10.0),
        )
    }

    fn ff_with_water(carbon_radius: Option<f64>) -> FormFactorMap {
        FormFactorMap::from_entries(vec![
            (carbon_kind(carbon_radius), FormFactor::Constant(6.0)),
            water_kind(),
        ])
    }

    #[test]
    fn lonely_atom_has_full_weight() {
        // Single atom has no neighbors → voronota returns None →
        // fully exposed → weight = 1.0 exactly.
        let ff = ff_with_water(Some(1.7));
        let structure = Structure {
            pos: vec![crate::Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let dummies = dummies(&structure, &ff, 1.4).unwrap();
        let mut out = Vec::new();
        dummies.form_factors_at(0.0, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 10.0, "weight 1.0 × water 10.0 = 10.0");
    }

    #[test]
    fn two_nearby_atoms_have_reduced_weight() {
        // Two atoms 1.0 Å apart, each radius 1.7. At probe 1.4 the
        // probe sphere can't fit between them, so each atom's
        // accessibility drops below 1.
        let ff = ff_with_water(Some(1.7));
        let structure = Structure {
            pos: vec![
                crate::Vector3::new(0.0, 0.0, 0.0),
                crate::Vector3::new(1.0, 0.0, 0.0),
            ],
            ids: vec![0, 0],
            box_sides: None,
        };
        let dummies = dummies(&structure, &ff, 1.4).unwrap();
        let mut out = Vec::new();
        dummies.form_factors_at(0.0, &mut out).unwrap();
        assert_eq!(out.len(), 2);
        for f in out {
            assert!(f > 0.0, "surface still partially accessible");
            assert!(f < 10.0, "weight must drop below full (1.0 × 10.0)");
        }
    }

    #[test]
    fn missing_radius_errors_with_kind_name() {
        let ff = ff_with_water(None);
        let structure = Structure {
            pos: vec![crate::Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let err = dummies(&structure, &ff, 1.4).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("atom kind C"), "kind name missing: {msg}");
        assert!(msg.contains("radius"), "radius mention missing: {msg}");
    }

    #[test]
    fn missing_h2o_errors() {
        // Form-factor map without an H2O entry.
        let ff =
            FormFactorMap::from_entries(vec![(carbon_kind(Some(1.7)), FormFactor::Constant(6.0))]);
        let structure = Structure {
            pos: vec![crate::Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let err = dummies(&structure, &ff, 1.4).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("H2O"), "water name missing: {msg}");
    }

    #[test]
    fn weighted_shape_single_eval_per_q() {
        // Directly construct a DummySet::weighted and verify that
        // `form_factors_at(q)` gives `weights[i] × shape.eval(q)`
        // for every atom — the single-eval property the SASA
        // dispatch path relies on.
        let positions = vec![
            crate::Vector3::new(0.0, 0.0, 0.0),
            crate::Vector3::new(1.0, 0.0, 0.0),
            crate::Vector3::new(0.0, 1.0, 0.0),
        ];
        let weights = vec![0.25, 0.5, 1.0];
        let dummies =
            DummySet::weighted(positions, FormFactor::Constant(8.0), weights.clone()).unwrap();
        let mut out = Vec::new();
        dummies.form_factors_at(0.3, &mut out).unwrap();
        for (f, &w) in out.iter().zip(&weights) {
            assert_eq!(*f, w * 8.0);
        }
    }
}
