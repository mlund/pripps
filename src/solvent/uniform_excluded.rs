// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Uniform excluded volume.
//!
//! Each atom displaces the same average volume `V_avg = V_total / N`,
//! computed from the total molecular volume (Voronoi tessellation).
//! All dummies share a single Fraser-style Gaussian:
//!
//! ```text
//! F(q) = ρ_s · V_avg · exp(−V_avg^(2/3) · q² / (4π))
//! ```
//!
//! Unlike per-atom Fraser, this needs only atomic radii (not
//! displaced volumes). The `volume_scale` fit knob modulates the
//! contribution at combine time via `|A − c₁·C + c₂·B|²`.
//!
//! Inspired by AUSAXS's "Simple" model.

use std::f64::consts::PI;

use voronota_ltr::{Ball, Results, compute_tessellation};

use crate::formfactor::FormFactorMap;
use crate::solvent::fraser::{PEPSI_DENOMINATOR, fraser_form_factor};
use crate::solvent::{DummySet, atom_radii};
use crate::{Result, Structure};

/// Build uniform excluded-volume dummies: one per atom, all
/// sharing a Fraser Gaussian sized by the average Voronoi cell
/// volume.
pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    bulk_electron_density: f64,
) -> Result<DummySet> {
    let atoms = atom_radii(structure, formfactors, "ExcludedVolume::Uniform")?;
    let n = atoms.len();
    if n == 0 {
        return DummySet::weighted(
            vec![],
            fraser_form_factor(0.0, bulk_electron_density, PEPSI_DENOMINATOR),
            vec![],
        );
    }

    let balls: Vec<Ball> = atoms
        .iter()
        .map(|(pos, r)| Ball::new(pos.x, pos.y, pos.z, *r))
        .collect();

    // Small probe (0.5 Å) so Voronoi cells approximate the bare
    // atomic volume without inflating by the full solvent probe.
    // Probe=0 leaves interstitial gaps unassigned; larger probes
    // make the total volume probe-sensitive and erratic.
    let tessellation = compute_tessellation(&balls, 0.5, None, None, false);
    let volumes = tessellation.volumes();

    let n_none = volumes.iter().filter(|v| v.is_none()).count();
    if n_none > 0 {
        log::warn!(
            "Uniform excluded volume: {n_none}/{n} atoms have no Voronoi cell \
             (lonely atoms); using sphere volume fallback"
        );
    }

    let total_volume: f64 = volumes
        .iter()
        .zip(&balls)
        // None = lonely atom (no Voronoi neighbours); fall back to
        // the bare sphere volume since there's no cell to measure.
        .map(|(vol, ball)| vol.unwrap_or_else(|| 4.0 / 3.0 * PI * ball.r.powi(3)))
        .sum();

    let volume_per_atom = total_volume / n as f64;

    log::info!(
        "Uniform excluded volume: {n} atoms, V_total={total_volume:.1} Å³, \
         V_avg={volume_per_atom:.2} Å³"
    );

    let form_factor = fraser_form_factor(volume_per_atom, bulk_electron_density, PEPSI_DENOMINATOR);
    // Dummies sit at atom positions (not grid cells) so the EV
    // multipole expansion shares the same SphericalBasis as the
    // atoms. Uniform weights — all differentiation is in the
    // shared form factor.
    let weights = vec![1.0; n];
    DummySet::weighted(structure.pos.clone(), form_factor, weights)
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
        // Fraser Gaussian should decay: F(0.5) < F(0)
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
        assert!(f1[0] < f0[0], "F(0.5) should be less than F(0)");
        assert!(f1[0] > 0.0, "F(0.5) should be positive");
    }

    #[test]
    fn form_factor_at_q_zero() {
        // F(0) = ρ × V_avg for the Fraser Gaussian
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
        // F(0) should be positive and proportional to ρ
        assert!(out[0] > 0.0);
        // Both atoms should have identical F(0) (uniform model)
        assert_eq!(out[0], out[1]);
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
