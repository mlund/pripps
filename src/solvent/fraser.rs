// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Fraser-MacRae-Suzuki excluded-volume dummy atoms.
//!
//! For each protein atom the generator emits one Gaussian dummy atom
//! at the atom's position, sized by that atom's `displaced_volume`.
//! The exact Gaussian form factor at volume scale `c1` is
//!
//! ```text
//! F_dummy(q; c1) = ρ_s · (c1·V0) · exp(−(c1·V0)^(2/3) · q² / D)
//!               = c1 · ρ_s·V0 · exp(−c1^(2/3) · V0^(2/3) q²/D)
//! ```
//!
//! where `V0 = displaced_volume`, `ρ_s` is the bulk solvent electron
//! density, and `D` is the denominator (4π for PepsiSAXS, 16π for FoXS).
//! Reference: Fraser, MacRae & Suzuki,
//! <https://doi.org/10.1107/S0021889878014296>.
//!
//! Note that `c1` enters both the linear prefactor *and* the Gaussian
//! width via `c1^(2/3)`. The cached solvent models factor `c1` out of
//! the O(N²) sums so refits are cheap, which the width dependence would
//! forbid. Rather than freeze the width at `c1=1` (which drops the
//! `∝q²` response — a modeling error), the cached path keeps the exact
//! linear prefactor and expands only the width to first order about
//! `c1=1`:
//!
//! ```text
//! F_dummy(q; c1) = c1 · A · exp(−c1^(2/3)·b q²)          (A = ρ_s·V0, b = V0^(2/3)/D)
//!               ≈ c1 · A · exp(−b q²) · [1 − (2/3)(c1−1)·b q²]
//!                = f0·C0(q) + f1·C1(q),   f0 = c1,  f1 = −(2/3)·c1·(c1−1)
//!   C0(q) = A·exp(−b q²)             (base, [`dummies`])
//!   C1(q) = A·b·q²·exp(−b q²) = b q²·C0(q)   ([`derivative_dummies`])
//! ```
//!
//! The `c1` in `f1` is *not* the naive gradient `−(2/3)(c1−1)`: it comes
//! from carrying the exact prefactor `c1` through the width expansion.
//! This is deliberate — it keeps `f1(0)=0` (so `c1=0` gives exactly
//! atoms-only intensity) and tracks the exact form factor more closely
//! than the fully-linearised Taylor across the whole fit range (verified
//! by `debye::tests::first_order_beats_constant_width_and_converges_quadratically`).
//! Both bases are built at unit scale; the scale knob lives only in the
//! `(f0, f1)` recombination. This mirrors PepsiSAXS' `Clm1` second basis
//! (Grudinin 2017, <https://doi.org/10.1107/S1600576717007257>).

use crate::formfactor::{FormFactor, FormFactorMap};
use crate::solvent::DummySet;
use crate::{Error, Result, Structure, Vector3};
use std::f64::consts::PI;

/// PepsiSAXS / CRYSOL Fraser denominator: 4π. Wider q-space
/// Gaussian → faster decay with q → smaller excluded-volume
/// subtraction at high q than the FoXS convention.
pub(super) const PEPSI_DENOMINATOR: f64 = 4.0 * PI;

/// FoXS Fraser denominator: 16π. The 4× larger denominator makes
/// the q-space Gaussian 4× narrower in its exponent coefficient,
/// which means it decays *more slowly* with q than the PepsiSAXS
/// variant — the dummy form factor stays closer to its `q=0`
/// value at high q, so the excluded-volume subtraction is larger
/// there. (In real space this corresponds to a sharper, more
/// concentrated dummy-density blob, the Fourier dual.)
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

/// Build the first-order Taylor derivative (`C1`) form factor,
///
/// ```text
/// C1(q) = ρ_s · V · b · q² · exp(−b q²),   b = V^(2/3) / denominator
/// ```
///
/// the `q²`-weighted sibling of [`fraser_form_factor`] (`= b q² · C0`).
/// See the module docs for how `(f0, f1)` recombine `C0`/`C1`.
pub(super) fn fraser_derivative_form_factor(
    volume: f64,
    bulk_electron_density: f64,
    denominator: f64,
) -> FormFactor {
    let b = volume.powf(2.0 / 3.0) / denominator;
    FormFactor::single_gaussian_q2(bulk_electron_density * volume * b, b)
}

/// Collect `(position, displaced_volume)` for every atom, validating
/// per atom that the referenced kind carries a `displaced_volume`.
/// Shared by the base ([`dummies`]) and derivative
/// ([`derivative_dummies`]) generators.
//
// Validate per atom rather than up front over all kinds in the map:
// coarse-grained YAML files (e.g. `poly_amino_acid.yaml`) may legally
// leave some residue entries without a `displaced_volume`, and only
// kinds the loaded structure references need one.
fn atom_volumes(structure: &Structure, formfactors: &FormFactorMap) -> Result<Vec<(Vector3, f64)>> {
    structure
        .ids
        .iter()
        .zip(&structure.pos)
        .map(|(&id, &pos)| -> Result<(Vector3, f64)> {
            let kind = formfactors.kind(id);
            let volume = kind.displaced_volume().ok_or_else(|| {
                Error::validation(format!(
                    "atom kind {} has no `displaced_volume`; required by ExcludedVolume::Fraser",
                    kind.name()
                ))
            })?;
            Ok((pos, volume))
        })
        .collect()
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
    let atoms = atom_volumes(structure, formfactors)?;
    let (positions, form_factors): (Vec<Vector3>, Vec<FormFactor>) = atoms
        .into_iter()
        .map(|(pos, base_volume)| {
            let volume = volume_scale * base_volume;
            (
                pos,
                fraser_form_factor(volume, bulk_electron_density, denominator),
            )
        })
        .unzip();

    log::info!(
        "Fraser excluded volume: {} dummies, ρ={bulk_electron_density:.3} e/ų",
        positions.len()
    );
    DummySet::per_dummy(positions, form_factors)
}

/// One `q²`-weighted Gaussian dummy per atom — the first-order Taylor
/// derivative (`C1`) basis, built at unit volume scale. The volume
/// scale is applied later during recombination via the `(f0, f1)`
/// coefficients (see module docs), so this builder takes no
/// `volume_scale`. `denominator` selects the variant as in
/// [`dummies`].
pub(super) fn derivative_dummies(
    structure: &Structure,
    formfactors: &FormFactorMap,
    bulk_electron_density: f64,
    denominator: f64,
) -> Result<DummySet> {
    let atoms = atom_volumes(structure, formfactors)?;
    let (positions, form_factors): (Vec<Vector3>, Vec<FormFactor>) = atoms
        .into_iter()
        .map(|(pos, base_volume)| {
            (
                pos,
                fraser_derivative_form_factor(base_volume, bulk_electron_density, denominator),
            )
        })
        .unzip();
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
    fn derivative_vanishes_at_q_zero() {
        let ff = one_kind_map(Some(16.43));
        let s = three_atom_structure();
        let d = derivative_dummies(&s, &ff, 0.334, PEPSI_DENOMINATOR).unwrap();
        for f in eval_all(&d, 0.0) {
            assert_eq!(f, 0.0);
        }
    }

    #[test]
    fn derivative_equals_b_q2_times_base() {
        let rho = 0.334;
        let v = 16.43;
        let ff = one_kind_map(Some(v));
        let s = three_atom_structure();
        let base = dummies(&s, &ff, rho, 1.0, PEPSI_DENOMINATOR).unwrap();
        let deriv = derivative_dummies(&s, &ff, rho, PEPSI_DENOMINATOR).unwrap();
        let b = v.powf(2.0 / 3.0) / PEPSI_DENOMINATOR;
        for q in [0.05, 0.2, 0.5] {
            let c0 = eval_all(&base, q)[0];
            let c1 = eval_all(&deriv, q)[0];
            assert!(
                (c1 - b * q * q * c0).abs() < 1e-12,
                "C1 != b·q²·C0 at q={q}: {c1} vs {}",
                b * q * q * c0
            );
        }
    }

    #[test]
    fn derivative_uses_foxs_denominator() {
        let rho = 0.334;
        let v = 16.43;
        let ff = one_kind_map(Some(v));
        let s = three_atom_structure();
        let deriv = derivative_dummies(&s, &ff, rho, FOXS_DENOMINATOR).unwrap();
        // C1(q) = ρ·V·b·q²·exp(−b q²) with b = V^(2/3)/16π.
        let b = v.powf(2.0 / 3.0) / FOXS_DENOMINATOR;
        let q = 0.3;
        let expected = rho * v * b * q * q * (-b * q * q).exp();
        assert!((eval_all(&deriv, q)[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn derivative_missing_displaced_volume_errors() {
        let ff = one_kind_map(None);
        let s = three_atom_structure();
        let err = derivative_dummies(&s, &ff, 0.334, PEPSI_DENOMINATOR).unwrap_err();
        assert!(format!("{err}").contains("atom kind C"));
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
