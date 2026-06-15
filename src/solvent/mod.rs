// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Solvent modelling for scattering intensity fits.
//!
//! pripps's vacuum intensity (atoms only) is only half the story for
//! real SAXS data. Two extra contributions have to be combined with
//! the atomic scattering to match experiment:
//!
//! - **Excluded volume** — the bulk solvent that the protein
//!   displaces. PepsiSAXS calls this `C` (`C_l^m` in the multipole
//!   expansion, `C` in `|A - C + dp·B|²`).
//! - **Hydration shell** — the first water layer, which has a
//!   different electron density than bulk solvent (≈0.03 e/Å³ above
//!   bulk for most proteins). PepsiSAXS calls this `B` (`B_l^m`).
//!
//! Both are combined coherently with the protein contribution (`A`)
//! in the Multipole expansion
//! (<https://doi.org/10.1107/S0021889895007047>, also the scheme used
//! in PepsiSAXS, <https://doi.org/10.1107/S1600576717007257>). The
//! orientation-averaged total intensity is the squared magnitude of
//! `A - volume_scale·C + contrast_density·B`, summed over all
//! multipole coefficients.
//!
//! The config types are here; the real dummy-atom generators live in
//! sibling files (`fraser.rs` for excluded volume, `sasa.rs` for
//! SASA-weighted hydration, and the Phase 2c `grid.rs` / Phase 2d
//! Voronoi variants).

mod fraser;
mod grid;
mod grid_excluded;
mod polynomial_excluded;
mod sasa;
mod voronoi_excluded;
mod voronoi_hydration;

use crate::formfactor::{FormFactor, FormFactorMap};
use crate::{Error, Result, Structure, Vector3};

pub(super) const WATER_KIND_NAME: &str = "H2O";

/// Bulk water molecular volume in Å³ (ρ_water = 0.334 e/Å³ → ≈ 30 Å³/molecule).
pub(super) const WATER_VOLUME: f64 = 30.0;

/// Convert `(position, radius)` pairs to voronota-ltr `Ball`s.
fn balls_from_radii(atoms: &[(Vector3, f64)]) -> Vec<voronota_ltr::Ball> {
    atoms
        .iter()
        .map(|(pos, r)| voronota_ltr::Ball::new(pos.x, pos.y, pos.z, *r))
        .collect()
}

/// Per-atom `(position, radius)` extracted from the structure and
/// form-factor map. Radius = `sigma / 2` (YAML `radius` is stored
/// doubled as `sigma` at parse time).
fn atom_radii(
    structure: &Structure,
    formfactors: &FormFactorMap,
    caller: &str,
) -> Result<Vec<(Vector3, f64)>> {
    structure
        .ids
        .iter()
        .zip(&structure.pos)
        .map(|(&id, &pos)| -> Result<(Vector3, f64)> {
            let kind = formfactors.kind(id);
            let sigma = kind.sigma().ok_or_else(|| {
                Error::validation(format!(
                    "atom kind {} has no `radius`; required by {caller}",
                    kind.name()
                ))
            })?;
            Ok((pos, sigma * 0.5))
        })
        .collect()
}

/// Look up the shared water form factor from the form-factor map.
fn water_form_factor(formfactors: &FormFactorMap, caller: &str) -> Result<FormFactor> {
    formfactors
        .form_factor_by_name(WATER_KIND_NAME)
        .ok_or_else(|| {
            Error::validation(format!(
                "{caller} requires an `{WATER_KIND_NAME}` entry in the form-factor YAML"
            ))
        })
        .cloned()
}

/// Top-level solvent model passed to [`crate::Pripps::intensity`].
#[derive(Debug, Clone, PartialEq)]
pub struct SolventConfig {
    pub excluded: ExcludedVolume,
    pub hydration: Hydration,
    /// Bulk solvent electron density, in e/Å³. The `Default` value is
    /// 0.334, which is pure water at 4 °C and the standard reference
    /// density used by CRYSOL and PepsiSAXS.
    pub bulk_electron_density: f64,
}

impl Default for SolventConfig {
    fn default() -> Self {
        Self {
            excluded: ExcludedVolume::Disabled,
            hydration: Hydration::Disabled,
            bulk_electron_density: 0.334,
        }
    }
}

impl SolventConfig {
    /// `true` when at least one contribution is active. The multipole method's
    /// three-contribution path is skipped entirely for the all-`None`
    /// config so vacuum callers pay nothing.
    pub(crate) const fn is_active(&self) -> bool {
        !matches!(self.excluded, ExcludedVolume::Disabled)
            || !matches!(self.hydration, Hydration::Disabled)
    }
}

impl ExcludedVolume {
    /// The dimensionless excluded-volume scale. Returns `0.0` for
    /// `None`, which lets `combine_contributions` treat the excluded
    /// term as absent without a separate branch.
    // PepsiSAXS fit parameter `r0` in cSAS.cpp.
    pub(crate) const fn volume_scale(&self) -> f64 {
        match self {
            Self::Disabled => 0.0,
            Self::Fraser { volume_scale }
            | Self::FraserFoxs { volume_scale }
            | Self::Grid { volume_scale, .. }
            | Self::Voronoi { volume_scale }
            | Self::Polynomial { volume_scale, .. } => *volume_scale,
        }
    }
}

impl Hydration {
    /// The contrast electron density (e/Å³) of the hydration layer
    /// over bulk solvent. Returns `0.0` for `None`, which zeros the
    /// hydration term in `combine_contributions` without a branch.
    //
    // For SASA-weighted hydration this knob is dimensionally a
    // dimensionless scale of the water form factor (matches FoXS
    // `c₂`) rather than literally an electron density. Phase 2c
    // grid-based variants will use it closer to the literal
    // density interpretation.
    pub(crate) const fn contrast_density(&self) -> f64 {
        match self {
            Self::Disabled => 0.0,
            Self::Grid {
                contrast_density, ..
            }
            | Self::Sasa {
                contrast_density, ..
            }
            | Self::Voronoi {
                contrast_density, ..
            } => *contrast_density,
        }
    }
}

/// How the excluded-volume contribution is represented. Each variant
/// generates its own set of dummy scatterers via
/// [`ExcludedVolume::dummies`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExcludedVolume {
    Disabled,
    /// One Gaussian dummy atom at each protein atom center, with
    /// displaced volume taken from `AtomKind.dsv`. `volume_scale`
    /// globally scales every dummy's volume and is the primary fit
    /// knob (PepsiSAXS calls this `r0`, typical range 0.95–1.05).
    ///
    /// Named after Fraser, MacRae & Suzuki,
    /// <https://doi.org/10.1107/S0021889878014296> — the
    /// Gaussian-dummy excluded-volume model that CRYSOL and PepsiSAXS
    /// both build on.
    Fraser {
        volume_scale: f64,
    },
    /// FoXS-style Fraser dummies: same per-atom Gaussians as
    /// [`Fraser`] but with a 4× sharper exponential
    /// (`V^(2/3)/(16π)` instead of `V^(2/3)/(4π)`). Subtracts less
    /// excluded volume at high q, matching the FoXS/IMP-SAXS
    /// convention (`FormFactorTable.cpp:285`).
    FraserFoxs {
        volume_scale: f64,
    },
    /// Cartesian grid inside the molecular envelope. Alternative to
    /// Fraser for structures where the Gaussian-dummy approximation
    /// of the displaced volume is too coarse.
    Grid {
        spacing: f64,
        volume_scale: f64,
    },
    /// Per-atom excluded volume from radical Voronoi tessellation.
    /// Each atom's displaced volume is its Voronoi cell volume
    /// (probe=0), used to build a Fraser-style Gaussian. Needs
    /// only atomic radii, not tabulated displaced volumes.
    Voronoi {
        volume_scale: f64,
    },
    /// Per-kind excluded-solvent form factors stored alongside the
    /// atom/residue form factors in the same YAML file.
    Polynomial {
        volume_scale: f64,
    },
}

/// How the hydration shell is represented.
#[derive(Debug, Clone, PartialEq)]
pub enum Hydration {
    Disabled,
    /// Grid points in a shell of `thickness` around the molecular
    /// envelope. `contrast_density` is the excess electron density
    /// of the hydration layer over bulk solvent, in e/Å³.
    Grid {
        spacing: f64,
        thickness: f64,
        probe: f64,
        contrast_density: f64,
    },
    /// FoXS-style SASA-weighted hydration shell — see
    /// [`crate::solvent::sasa`] for the physics writeup.
    Sasa {
        probe: f64,
        contrast_density: f64,
    },
    /// Explicit solvent spheres on the molecular surface, placed
    /// and weighted by Voronoi tessellation via voronota-ltr's
    /// `compute_solvent_spheres`. More physical than SASA: water
    /// dummies sit at actual surface positions, not at atom centers.
    Voronoi {
        probe: f64,
        contrast_density: f64,
    },
}

/// A set of dummy scatterers that feed into the Multipole
/// kernel. Positions are per-dummy; form factors come in two
/// flavours via [`DummyFormFactors`] — per-dummy (Fraser) or one
/// shared factor with per-dummy scalar weights (SASA).
#[derive(Debug, Clone, Default)]
pub(crate) struct DummySet {
    positions: Vec<Vector3>,
    form_factors: DummyFormFactors,
}

/// How the form factors of a [`DummySet`] are stored.
//
// Two variants because Fraser needs one Gaussian per atom (each
// sized by the atom's `displaced_volume`) while SASA hydration has
// one shared water form factor that every atom scatters through,
// weighted by its accessibility fraction. Forcing SASA into a
// per-dummy `Vec<FormFactor>` would clone the water Table N times
// and call `eval(q)` N times per q — both avoidable by factoring
// the shared form factor out.
#[derive(Debug, Clone)]
enum DummyFormFactors {
    PerDummy(Vec<FormFactor>),
    Weighted {
        form_factor: FormFactor,
        weights: Vec<f64>,
    },
}

impl Default for DummyFormFactors {
    // `PerDummy(vec![])` over `Weighted { ... }` because the empty
    // default has zero dummies and therefore no shared form factor
    // to choose — `PerDummy` is the variant that degenerates to
    // "nothing to evaluate" without requiring a placeholder
    // `FormFactor`.
    fn default() -> Self {
        Self::PerDummy(Vec::new())
    }
}

impl DummySet {
    // `Result` instead of `debug_assert`: the positions/form-factors
    // parity is a data-shape contract every generator must uphold,
    // and a silent mismatch in release would corrupt the Multipole
    // per-atom iteration without an explicit panic site.
    pub(super) fn per_dummy(
        positions: Vec<Vector3>,
        form_factors: Vec<FormFactor>,
    ) -> Result<Self> {
        if positions.len() != form_factors.len() {
            return Err(crate::Error::internal(format!(
                "DummySet: positions ({}) and form_factors ({}) length mismatch",
                positions.len(),
                form_factors.len(),
            )));
        }
        Ok(Self {
            positions,
            form_factors: DummyFormFactors::PerDummy(form_factors),
        })
    }

    pub(super) fn weighted(
        positions: Vec<Vector3>,
        form_factor: FormFactor,
        weights: Vec<f64>,
    ) -> Result<Self> {
        if positions.len() != weights.len() {
            return Err(crate::Error::internal(format!(
                "DummySet: positions ({}) and weights ({}) length mismatch",
                positions.len(),
                weights.len(),
            )));
        }
        Ok(Self {
            positions,
            form_factors: DummyFormFactors::Weighted {
                form_factor,
                weights,
            },
        })
    }

    pub(crate) fn positions(&self) -> &[Vector3] {
        &self.positions
    }

    /// Evaluate every dummy's form factor at a single q into
    /// `out`. `out` is cleared first. Errors (without touching
    /// `out`) if any dummy rejects the q value.
    //
    // Out-parameter rather than returning `Vec<f64>` so the rayon
    // worker's per-q scratch (`PerQBuffers::form_factors` in
    // `multipole::mod`) can be reused across every q iteration. A
    // returned `Vec` would allocate once per q per contribution —
    // three allocations per inner-loop iteration in the
    // three-contribution path.
    pub(crate) fn form_factors_at(&self, q: f64, out: &mut Vec<f64>) -> Result<()> {
        out.clear();
        match &self.form_factors {
            DummyFormFactors::PerDummy(ffs) => {
                for f in ffs {
                    out.push(f.eval(q)?);
                }
            }
            DummyFormFactors::Weighted {
                form_factor,
                weights,
            } => {
                let shared = form_factor.eval(q)?;
                out.extend(weights.iter().map(|&w| w * shared));
            }
        }
        Ok(())
    }
}

impl ExcludedVolume {
    pub(crate) fn dummies(
        &self,
        structure: &Structure,
        formfactors: &FormFactorMap,
        bulk_electron_density: f64,
    ) -> Result<DummySet> {
        match self {
            Self::Disabled => Ok(DummySet::default()),
            Self::Fraser { volume_scale } => fraser::dummies(
                structure,
                formfactors,
                bulk_electron_density,
                *volume_scale,
                fraser::PEPSI_DENOMINATOR,
            ),
            Self::FraserFoxs { volume_scale } => fraser::dummies(
                structure,
                formfactors,
                bulk_electron_density,
                *volume_scale,
                fraser::FOXS_DENOMINATOR,
            ),
            Self::Grid { spacing, .. } => {
                grid_excluded::dummies(structure, formfactors, bulk_electron_density, *spacing)
            }
            Self::Voronoi { .. } => {
                voronoi_excluded::dummies(structure, formfactors, bulk_electron_density)
            }
            Self::Polynomial { .. } => polynomial_excluded::dummies(structure, formfactors),
        }
    }
}

impl Hydration {
    pub(crate) fn dummies(
        &self,
        structure: &Structure,
        formfactors: &FormFactorMap,
    ) -> Result<DummySet> {
        match self {
            Self::Disabled => Ok(DummySet::default()),
            Self::Sasa { probe, .. } => sasa::dummies(structure, formfactors, *probe),
            Self::Grid {
                spacing,
                thickness,
                probe,
                ..
            } => grid::dummies(structure, formfactors, *spacing, *thickness, *probe),
            Self::Voronoi { probe, .. } => {
                voronoi_hydration::dummies(structure, formfactors, *probe)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_inert() {
        let cfg = SolventConfig::default();
        assert!(matches!(cfg.excluded, ExcludedVolume::Disabled));
        assert!(matches!(cfg.hydration, Hydration::Disabled));
        assert!(!cfg.is_active());
    }

    #[test]
    fn any_non_none_variant_is_active() {
        let cfg = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        assert!(cfg.is_active());
    }
}
