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
use crate::solvent::DummySet;
use crate::{Error, Result, Structure, Vector3};

const WATER_KIND_NAME: &str = "H2O";

/// Distance from `point` to the nearest atom surface: the minimum
/// of `|point − atom_center| − atom_radius` over all atoms.
fn nearest_surface_distance(point: &Vector3, atoms: &[(Vector3, f64)]) -> f64 {
    atoms
        .iter()
        .map(|(center, radius)| (point - center).norm() - radius)
        .fold(f64::INFINITY, f64::min)
}

/// Build a grid-based hydration shell around the structure.
pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    spacing: f64,
    thickness: f64,
    probe: f64,
) -> Result<DummySet> {
    // Collect (position, radius) per atom. Radius = sigma / 2.
    let atoms: Vec<(Vector3, f64)> = structure
        .ids
        .iter()
        .zip(&structure.pos)
        .map(|(&id, &pos)| -> Result<(Vector3, f64)> {
            let kind = formfactors.kind(id);
            let sigma = kind.sigma().ok_or_else(|| {
                Error::validation(format!(
                    "atom kind {} has no `radius`; required by Hydration::Grid",
                    kind.name()
                ))
            })?;
            Ok((pos, sigma * 0.5))
        })
        .collect::<Result<_>>()?;

    let max_radius = atoms.iter().map(|(_, r)| *r).fold(0.0_f64, f64::max);
    let margin = max_radius + probe + thickness + spacing;

    let (bb_lo, bb_hi) = structure.bounding_box();
    let lo = bb_lo - Vector3::new(margin, margin, margin);
    let hi = bb_hi + Vector3::new(margin, margin, margin);

    // Iterate grid points and keep those in the hydration shell.
    let mut positions = Vec::new();
    let mut x = lo.x;
    while x <= hi.x {
        let mut y = lo.y;
        while y <= hi.y {
            let mut z = lo.z;
            while z <= hi.z {
                let point = Vector3::new(x, y, z);
                let d = nearest_surface_distance(&point, &atoms);
                if d > probe && d < probe + thickness {
                    positions.push(point);
                }
                z += spacing;
            }
            y += spacing;
        }
        x += spacing;
    }

    log::info!(
        "Grid hydration shell: {} points at {spacing:.2} Å spacing, \
         {thickness:.1} Å thickness, {probe:.1} Å probe",
        positions.len()
    );

    let water = formfactors
        .form_factor_by_name(WATER_KIND_NAME)
        .ok_or_else(|| {
            Error::validation(format!(
                "Hydration::Grid requires an `{WATER_KIND_NAME}` entry in the form-factor YAML"
            ))
        })?
        .clone();

    // Each grid cell contains `(spacing / WATER_SPACING)³` water
    // molecules, where WATER_SPACING ≈ 3.105 Å is the cube root
    // of the volume per water molecule at bulk density (30 ų/molecule).
    // The water form factor gives scattering PER MOLECULE, so the
    // weight is the number of molecules per cell. This matches
    // PepsiSAXS's `ratioLinDistances` convention (cSAS.cpp:3872).
    const WATER_SPACING: f64 = 3.10516;
    let ratio = spacing / WATER_SPACING;
    let molecules_per_cell = ratio * ratio * ratio;
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
        // One atom at origin. Shell should form a hollow sphere.
        let structure = Structure {
            pos: vec![Vector3::new(0.0, 0.0, 0.0)],
            ids: vec![0],
            box_sides: None,
        };
        let ff = one_atom_map();
        let shell = dummies(&structure, &ff, 1.5, 3.0, 1.4).unwrap();
        let n = shell.positions().len();
        assert!(n > 50, "expected many shell points, got {n}");

        // All shell points should be between r_inner and r_outer
        // from the origin, where:
        //   r_inner = atom_radius + probe = 1.7 + 1.4 = 3.1
        //   r_outer = atom_radius + probe + thickness = 1.7 + 1.4 + 3.0 = 6.1
        for p in shell.positions() {
            let r = p.norm();
            assert!(r > 3.0, "point too close: r={r:.2}");
            assert!(r < 6.2, "point too far: r={r:.2}");
        }
    }

    #[test]
    fn no_shell_points_inside_protein() {
        // Two overlapping atoms — no grid point should be within
        // probe distance of any atom surface.
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
