// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Grid-based excluded volume.
//!
//! Fills the molecular envelope with uniform grid cells, each
//! carrying a Fraser-style Gaussian form factor based on cell
//! volume. This is the approach used by AUSAXS
//! (`ExvFormFactor.h`): all cells share
//!
//! ```text
//! F(q) = ρ_s · V_cell · exp(−V_cell^(2/3) · q² / (4π))
//! ```
//!
//! where `V_cell = spacing³`. Unlike per-atom Fraser dummies, this
//! needs only atomic radii (not displaced volumes) and works for
//! any geometry. The `volume_scale` fit knob is applied at the
//! coefficient-level combine step, not here.
//!
//! Reference: Fraser, MacRae & Suzuki,
//! <https://doi.org/10.1107/S0021889878014296>.

use crate::formfactor::FormFactorMap;
use crate::solvent::fraser::{PEPSI_DENOMINATOR, fraser_form_factor};
use crate::solvent::grid::{grid_steps, nearest_surface_distance};
use crate::solvent::{DummySet, atom_radii};
use crate::{Result, Structure, Vector3};

/// Build grid-based excluded-volume dummies inside the molecular
/// envelope.
pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    bulk_electron_density: f64,
    spacing: f64,
) -> Result<DummySet> {
    let atoms = atom_radii(structure, formfactors, "ExcludedVolume::Grid")?;

    let max_radius = atoms.iter().map(|(_, r)| *r).fold(0.0_f64, f64::max);
    let margin = max_radius + spacing;

    let (bb_lo, bb_hi) = structure.bounding_box();
    let lo = bb_lo - Vector3::new(margin, margin, margin);
    let hi = bb_hi + Vector3::new(margin, margin, margin);

    let nx = grid_steps(lo.x, hi.x, spacing);
    let ny = grid_steps(lo.y, hi.y, spacing);
    let nz = grid_steps(lo.z, hi.z, spacing);

    let mut positions = Vec::with_capacity(nx * ny * nz / 2);
    for ix in 0..nx {
        let x = lo.x + ix as f64 * spacing;
        for iy in 0..ny {
            let y = lo.y + iy as f64 * spacing;
            for iz in 0..nz {
                let z = lo.z + iz as f64 * spacing;
                let point = Vector3::new(x, y, z);
                if nearest_surface_distance(&point, &atoms) <= 0.0 {
                    positions.push(point);
                }
            }
        }
    }

    log::info!(
        "Grid excluded volume: {} points at {spacing:.2} Å spacing",
        positions.len()
    );

    let cell_volume = spacing.powi(3);
    let form_factor = fraser_form_factor(cell_volume, bulk_electron_density, PEPSI_DENOMINATOR);
    let weights = vec![1.0; positions.len()];
    DummySet::weighted(positions, form_factor, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    use crate::AtomKind;
    use crate::formfactor::FormFactor;

    fn one_atom_map() -> FormFactorMap {
        let mut kind = AtomKind::new("C");
        kind.sigma = Some(2.0 * 1.7);
        FormFactorMap::from_entries(vec![(kind, FormFactor::Constant(6.0))])
    }

    #[test]
    fn single_atom_fills_sphere() {
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let d = dummies(&structure, &ff, 0.334, 0.5).unwrap();
        let n = d.positions().len();
        assert!(n > 10, "expected many envelope points, got {n}");

        // All points must be inside the atom (d ≤ 0)
        let atoms = vec![(Vector3::new(0.0, 0.0, 0.0), 1.7)];
        for p in d.positions() {
            let dist = nearest_surface_distance(p, &atoms);
            assert!(dist <= 0.0, "point outside envelope: d={dist:.3}");
        }
    }

    #[test]
    fn volume_approximates_sphere() {
        let radius: f64 = 1.7;
        let spacing: f64 = 0.3;
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let d = dummies(&structure, &ff, 0.334, spacing).unwrap();
        let grid_volume = d.positions().len() as f64 * spacing.powi(3);
        let sphere_volume = 4.0 / 3.0 * PI * radius.powi(3);
        let ratio = grid_volume / sphere_volume;
        // Grid approximation should be within ~15% of the sphere
        assert!(
            (0.85..1.15).contains(&ratio),
            "grid volume ratio {ratio:.3} too far from 1.0"
        );
    }

    #[test]
    fn form_factor_at_q_zero() {
        let rho = 0.334;
        let spacing = 1.0;
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let d = dummies(&structure, &ff, rho, spacing).unwrap();
        let mut out = Vec::new();
        d.form_factors_at(0.0, &mut out).unwrap();
        let cell_volume = spacing.powi(3);
        for f in &out {
            assert!(
                (*f - rho * cell_volume).abs() < 1e-10,
                "F(0) = {f}, expected {}",
                rho * cell_volume
            );
        }
    }

    #[test]
    fn no_points_for_empty_structure() {
        let structure = Structure {
            pos: vec![],
            ids: vec![],
            box_sides: None,
        };
        let ff = FormFactorMap::from_entries(vec![]);
        let d = dummies(&structure, &ff, 0.334, 1.0).unwrap();
        assert_eq!(d.positions().len(), 0);
    }
}
