// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Grid-based hydration shell.
//!
//! Places water dummy scatterers on a Cartesian grid in a shell
//! around the molecular envelope, matching PepsiSAXS's approach
//! (`cGrid.cpp:559-681`). Each grid point in the shell shares a
//! single water form factor (H₂O), weighted uniformly — the
//! `contrast_density` fit knob is applied at the coefficient-level
//! combine step, not here.
//!
//! Shell criterion: a grid point at distance `d` from the nearest
//! atom surface (= center-to-center minus atom radius) is kept when
//! `probe < d < probe + thickness`. Points inside the protein
//! envelope (`d ≤ probe`) or in bulk solvent (`d > probe + thickness`)
//! are discarded.

use crate::formfactor::FormFactorMap;
use crate::solvent::{DummySet, atom_radii, water_form_factor};
use crate::{Result, Structure, Vector3};

/// Distance from `point` to the nearest atom surface: the minimum
/// of `|point − atom_center| − atom_radius` over all atoms.
pub(super) fn nearest_surface_distance(point: &Vector3, atoms: &[(Vector3, f64)]) -> f64 {
    atoms
        .iter()
        .map(|(center, radius)| (point - center).norm() - radius)
        .fold(f64::INFINITY, f64::min)
}

/// Number of integer steps to cover `[lo, hi]` at the given spacing.
pub(super) fn grid_steps(lo: f64, hi: f64, spacing: f64) -> usize {
    ((hi - lo) / spacing).ceil() as usize + 1
}

/// Build a grid-based hydration shell around the structure.
pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    spacing: f64,
    thickness: f64,
    probe: f64,
) -> Result<DummySet> {
    let atoms = atom_radii(structure, formfactors, "Hydration::Grid")?;

    let max_radius = atoms.iter().map(|(_, r)| *r).fold(0.0_f64, f64::max);
    let margin = max_radius + probe + thickness + spacing;

    let (bb_lo, bb_hi) = structure.bounding_box();
    let lo = bb_lo - Vector3::new(margin, margin, margin);
    let hi = bb_hi + Vector3::new(margin, margin, margin);

    let nx = grid_steps(lo.x, hi.x, spacing);
    let ny = grid_steps(lo.y, hi.y, spacing);
    let nz = grid_steps(lo.z, hi.z, spacing);

    let outer = probe + thickness;
    let mut positions = Vec::with_capacity(nx * ny * nz / 8);
    for ix in 0..nx {
        let x = lo.x + ix as f64 * spacing;
        for iy in 0..ny {
            let y = lo.y + iy as f64 * spacing;
            for iz in 0..nz {
                let z = lo.z + iz as f64 * spacing;
                let point = Vector3::new(x, y, z);
                let d = nearest_surface_distance(&point, &atoms);
                if d > probe && d < outer {
                    positions.push(point);
                }
            }
        }
    }

    log::info!(
        "Grid hydration shell: {} points at {spacing:.2} Å spacing, \
         {thickness:.1} Å thickness, {probe:.1} Å probe",
        positions.len()
    );

    let water = water_form_factor(formfactors, "Hydration::Grid")?;

    // Each grid cell contains `(spacing / WATER_SPACING)³` water
    // molecules, where WATER_SPACING ≈ 3.105 Å is the cube root
    // of the volume per water molecule at bulk density (30 Å³/molecule).
    // The water form factor gives scattering per molecule, so the
    // weight is the number of molecules per cell. This matches
    // PepsiSAXS's `ratioLinDistances` convention (cSAS.cpp:3872).
    const WATER_SPACING: f64 = 3.10516;
    let molecules_per_cell = (spacing / WATER_SPACING).powi(3);
    let weights = vec![molecules_per_cell; positions.len()];
    DummySet::weighted(positions, water, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomKind;
    use crate::formfactor::FormFactor;

    fn one_atom_map() -> FormFactorMap {
        let mut kind = AtomKind::new("C");
        kind.sigma = Some(2.0 * 1.7);
        FormFactorMap::from_entries(vec![
            (kind, FormFactor::Constant(6.0)),
            (AtomKind::new("H2O"), FormFactor::Constant(10.0)),
        ])
    }

    #[test]
    fn single_atom_shell_is_spherical() {
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let shell = dummies(&structure, &ff, 1.5, 3.0, 1.4).unwrap();
        let n = shell.positions().len();
        assert!(n > 50, "expected many shell points, got {n}");

        // r_inner = atom_radius + probe = 1.7 + 1.4 = 3.1
        // r_outer = atom_radius + probe + thickness = 1.7 + 1.4 + 3.0 = 6.1
        for p in shell.positions() {
            let r = p.norm();
            assert!(r > 3.0, "point too close: r={r:.2}");
            assert!(r < 6.2, "point too far: r={r:.2}");
        }
    }

    #[test]
    fn no_shell_points_inside_protein() {
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0)],
            ids: vec![0, 0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let shell = dummies(&structure, &ff, 1.5, 3.0, 1.4).unwrap();

        let atoms: Vec<(Vector3, f64)> = structure.pos.iter().map(|&p| (p, 1.7)).collect();

        for p in shell.positions() {
            let d = nearest_surface_distance(p, &atoms);
            assert!(
                d > 1.39,
                "grid point at {p:?} is inside protein envelope: d={d:.3}"
            );
        }
    }
}
