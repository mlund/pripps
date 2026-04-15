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
//! in the Svergun multipole expansion
//! (<https://doi.org/10.1107/S0021889895007047>, also the scheme used
//! in PepsiSAXS, <https://doi.org/10.1107/S1600576717007257>). The
//! orientation-averaged total intensity is the squared magnitude of
//! `A - volume_scale·C + contrast_density·B`, summed over all
//! multipole coefficients.
//!
//! This module is the scaffold: the config types are here and the
//! `dummies` methods on each variant produce an empty set. Real
//! dummy-atom generators land in sibling files (`fraser.rs`,
//! `sasa.rs`, `grid.rs`, …) in the follow-up phases.

use crate::formfactor::{FormFactor, FormFactorMap};
use crate::{Result, Structure, Vector3};

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
            excluded: ExcludedVolume::None,
            hydration: Hydration::None,
            bulk_electron_density: 0.334,
        }
    }
}

impl SolventConfig {
    /// `true` when at least one contribution is active. Svergun's
    /// three-contribution path is skipped entirely for the all-`None`
    /// config so vacuum callers pay nothing.
    pub(crate) fn is_active(&self) -> bool {
        !matches!(self.excluded, ExcludedVolume::None) || !matches!(self.hydration, Hydration::None)
    }
}

impl ExcludedVolume {
    /// The dimensionless excluded-volume scale. Returns `0.0` for
    /// `None`, which lets `combine_contributions` treat the excluded
    /// term as absent without a separate branch.
    // PepsiSAXS fit parameter `r0` in cSAS.cpp.
    pub(crate) fn volume_scale(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Fraser { volume_scale } | Self::Grid { volume_scale, .. } => *volume_scale,
        }
    }
}

impl Hydration {
    /// The contrast electron density (e/Å³) of the hydration layer
    /// over bulk solvent. Returns `0.0` for `None`, which zeros the
    /// hydration term in `combine_contributions` without a branch.
    // PepsiSAXS fit parameter `dp` (δρ) in cSAS.cpp.
    pub(crate) fn contrast_density(&self) -> f64 {
        match self {
            Self::None => 0.0,
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
    None,
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
    /// Cartesian grid inside the molecular envelope. Alternative to
    /// Fraser for structures where the Gaussian-dummy approximation
    /// of the displaced volume is too coarse.
    Grid {
        spacing: f64,
        volume_scale: f64,
    },
}

/// How the hydration shell is represented.
#[derive(Debug, Clone, PartialEq)]
pub enum Hydration {
    None,
    /// Grid points in a shell of `thickness` around the molecular
    /// envelope. `contrast_density` is the excess electron density
    /// of the hydration layer over bulk solvent, in e/Å³.
    Grid {
        spacing: f64,
        thickness: f64,
        contrast_density: f64,
    },
    /// One dummy per atom, weighted by solvent-accessible surface
    /// area (voronota-ltr tessellation).
    Sasa {
        probe: f64,
        thickness: f64,
        contrast_density: f64,
    },
    /// One dummy per surface atom, weighted by Voronoi cell contact
    /// area.
    Voronoi {
        thickness: f64,
        contrast_density: f64,
    },
}

/// A set of dummy scatterers that feed into the Svergun multipole
/// kernel. Each dummy carries its own form factor (Fraser Gaussians,
/// water shell, etc.); the kernel doesn't care which contribution it
/// came from.
#[derive(Debug, Clone, Default)]
pub(crate) struct DummySet {
    positions: Vec<Vector3>,
    form_factors: Vec<FormFactor>,
}

impl DummySet {
    pub(crate) fn positions(&self) -> &[Vector3] {
        &self.positions
    }

    /// Evaluate every dummy's form factor at a single q. Errors if
    /// any dummy rejects the q value.
    pub(crate) fn form_factors_at(&self, q: f64) -> Result<Vec<f64>> {
        use itertools::Itertools;
        self.form_factors.iter().map(|f| f.eval(q)).try_collect()
    }
}

impl ExcludedVolume {
    pub(crate) fn dummies(
        &self,
        _structure: &Structure,
        _atomkinds: &FormFactorMap,
        _bulk_electron_density: f64,
    ) -> Result<DummySet> {
        match self {
            // Phase 2 will fill these in — Fraser first, then Grid.
            Self::None | Self::Fraser { .. } | Self::Grid { .. } => Ok(DummySet::default()),
        }
    }
}

impl Hydration {
    pub(crate) fn dummies(&self, _structure: &Structure) -> Result<DummySet> {
        match self {
            // Phase 2 will fill these in — Sasa first, then Grid and
            // Voronoi.
            Self::None | Self::Grid { .. } | Self::Sasa { .. } | Self::Voronoi { .. } => {
                Ok(DummySet::default())
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
        assert!(matches!(cfg.excluded, ExcludedVolume::None));
        assert!(matches!(cfg.hydration, Hydration::None));
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
