// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Fraser-MacRae-Suzuki excluded-volume dummy atoms.
//!
//! For each protein atom the generator emits one Gaussian dummy atom
//! at the atom's position, sized by that atom's `displaced_volume`
//! and scaled by the global `volume_scale` fit knob. The Gaussian
//! form factor is
//!
//! ```text
//! F_dummy(q) = ρ_s · V · exp(−V^(2/3) · q² / (4π))
//! ```
//!
//! where `V = volume_scale · displaced_volume` and `ρ_s` is the
//! bulk solvent electron density. Reference: Fraser, MacRae & Suzuki,
//! <https://doi.org/10.1107/S0021889878014296>.

use crate::formfactor::{FormFactor, FormFactorMap};
use crate::solvent::DummySet;
use crate::{Error, Result, Structure};
use std::f64::consts::PI;

/// PepsiSAXS Fraser denominator: 4π.
pub(super) const PEPSI_DENOMINATOR: f64 = 4.0 * PI;

/// FoXS Fraser denominator: 16π. Produces a 4× sharper Gaussian
/// that subtracts less excluded volume at high q.
// FoXS `FormFactorTable.cpp:285`: `volr_coeff = -pow(V, 2/3) / (16*PI)`
pub(super) const FOXS_DENOMINATOR: f64 = 16.0 * PI;

/// Build the Fraser-MacRae-Suzuki Gaussian dummy form factor.
/// `denominator` controls the Gaussian width: 4π for PepsiSAXS,
/// 16π for FoXS.
///
/// ```text
/// F(q) = ρ_s · V · exp(−V^(2/3) · q² / denominator)
/// ```
pub(super) fn fraser_form_factor(
    volume: f64,
    bulk_electron_density: f64,
    denominator: f64,
) -> FormFactor {
    FormFactor::single_gaussian(
        bulk_electron_density * volume,
        volume.powf(2.0 / 3.0) / denominator,
    )
}

/// One Gaussian dummy per protein atom, per Fraser-MacRae-Suzuki.
/// `denominator` selects the formula variant (see
/// [`PEPSI_DENOMINATOR`] / [`FOXS_DENOMINATOR`]). Errors if any
/// atom kind referenced by the structure is missing
/// `displaced_volume`.
pub(super) fn dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    bulk_electron_density: f64,
    volume_scale: f64,
    denominator: f64,
) -> Result<DummySet> {
    let mut positions = Vec::with_capacity(structure.pos.len());
    let mut form_factors = Vec::with_capacity(structure.pos.len());

    // Validate per atom rather than up front over all kinds in the
    // map: coarse-grained YAML files (e.g. `single_bead.yaml`) may
    // legally leave some residue entries without a `displaced_volume`,
    // and only kinds the loaded structure references need one.
    for (&id, &pos) in structure.ids.iter().zip(&structure.pos) {
        let kind = formfactors.kind(id);
        let base_volume = kind.displaced_volume().ok_or_else(|| {
            Error::validation(format!(
                "atom kind {} has no `displaced_volume`; required by ExcludedVolume::Fraser",
                kind.name()
            ))
        })?;
        let volume = volume_scale * base_volume;
        positions.push(pos);
        form_factors.push(fraser_form_factor(
            volume,
            bulk_electron_density,
            denominator,
        ));
    }

    log::info!(
        "Fraser excluded volume: {} dummies, ρ={bulk_electron_density:.3} e/ų",
        positions.len()
    );
    DummySet::per_dummy(positions, form_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formfactor::FormFactor;
    use crate::{AtomKind, ExcludedVolume, SolventConfig, Vector3};

    fn one_kind_map(displaced_volume: Option<f64>) -> FormFactorMap {
        let mut kind = AtomKind::new("C");
        kind.displaced_volume = displaced_volume;
        FormFactorMap::from_entries(vec![(kind, FormFactor::Constant(6.0))])
    }

    fn three_atom_structure() -> Structure {
        Structure {
            pos: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 2.0, 0.0),
            ],
            ids: vec![0, 0, 0],
            box_sides: None,
        }
    }

    fn eval_all(dummies: &crate::solvent::DummySet, q: f64) -> Vec<f64> {
        let mut out = Vec::new();
        dummies.form_factors_at(q, &mut out).unwrap();
        out
    }

    #[test]
    fn generates_one_dummy_per_atom() {
        let ff = one_kind_map(Some(16.43));
        let s = three_atom_structure();
        let dummies = dummies(&s, &ff, 0.334, 1.0, PEPSI_DENOMINATOR).unwrap();
        assert_eq!(dummies.positions().len(), 3);
        assert_eq!(eval_all(&dummies, 0.0).len(), 3);
        assert_eq!(dummies.positions()[1], Vector3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn form_factor_at_q_zero_equals_rho_times_volume() {
        let rho = 0.334;
        let v = 16.43;
        let ff = one_kind_map(Some(v));
        let s = three_atom_structure();
        let dummies = dummies(&s, &ff, rho, 1.0, PEPSI_DENOMINATOR).unwrap();
        for f in eval_all(&dummies, 0.0) {
            assert_eq!(f, rho * v);
        }
    }

    #[test]
    fn volume_scale_multiplies_volume_linearly() {
        let rho = 0.334;
        let v = 16.43;
        let ff = one_kind_map(Some(v));
        let s = three_atom_structure();
        let d1 = dummies(&s, &ff, rho, 1.0, PEPSI_DENOMINATOR).unwrap();
        let d2 = dummies(&s, &ff, rho, 2.0, PEPSI_DENOMINATOR).unwrap();
        assert_eq!(eval_all(&d2, 0.0)[0], 2.0 * eval_all(&d1, 0.0)[0]);
    }

    #[test]
    fn missing_displaced_volume_errors_with_kind_name() {
        let ff = one_kind_map(None);
        let s = three_atom_structure();
        let err = dummies(&s, &ff, 0.334, 1.0, PEPSI_DENOMINATOR).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("atom kind C"), "expected kind name in: {msg}");
    }

    #[test]
    fn wired_into_excluded_volume_dummies() {
        // End-to-end through the public-ish enum method.
        let cfg = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        let ff = one_kind_map(Some(16.43));
        let s = three_atom_structure();
        let dummies = cfg
            .excluded
            .dummies(&s, &ff, cfg.bulk_electron_density)
            .unwrap();
        assert_eq!(dummies.positions().len(), 3);
    }
}
