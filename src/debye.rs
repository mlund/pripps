// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

use crate::q_sampler::QSampler;
use crate::solvent::SolventConfig;
use crate::{
    Contributions, Intensity, IntensityCalculator, Result, Structure, Vector3,
    formfactor::{self, FormFactorMap},
};
use itertools::Itertools;
use rayon::prelude::*;

/// Returns sinc(x) = sin(x)/x with graceful handling of x -> 0
#[inline]
pub fn sinc(x: f64) -> f64 {
    const NEARLY_ZERO: f64 = 1e-8;
    if x.abs() < NEARLY_ZERO {
        1.0
    } else {
        x.sin() / x
    }
}

pub(crate) struct DebyeIntensity {
    sampler: QSampler,
    solvent: Option<SolventConfig>,
}

impl DebyeIntensity {
    #[cfg(test)]
    pub(crate) fn new(
        formfactors: FormFactorMap,
        q_minmax: (f64, f64),
        num_q: usize,
    ) -> Result<Self> {
        Self::new_with_solvent(formfactors, q_minmax, num_q, None)
    }

    pub(crate) fn new_with_solvent(
        formfactors: FormFactorMap,
        q_minmax: (f64, f64),
        num_q: usize,
        solvent: Option<SolventConfig>,
    ) -> Result<Self> {
        Ok(Self {
            sampler: QSampler::linear(formfactors, q_minmax.0, q_minmax.1, num_q),
            solvent,
        })
    }
}

impl IntensityCalculator for DebyeIntensity {
    /// Calculate I(q) over a range of q values using the Debye formula
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity> {
        match self.solvent.as_ref().filter(|cfg| cfg.is_active()) {
            None => self.sampler.run_parallel(|q| {
                intensity_at_q(&structure.ids, &structure.pos, &self.sampler.formfactors, q)
            }),
            Some(cfg) => {
                let model = build_model(&self.sampler, structure, cfg)?;
                Ok(model.intensity(
                    cfg.excluded.volume_scale(),
                    cfg.hydration.contrast_density(),
                ))
            }
        }
    }
}

/// Precompute Debye pair sums for atoms, excluded volume, and hydration.
///
/// Unlike the vacuum Debye path, fitting needs repeated evaluations at
/// different `(volume_scale, contrast_density)`. The full Debye sum is
/// quadratic in those two scalar knobs, so caching the three self terms
/// and three cross terms per q makes refits cheap.
pub(crate) fn build_model(
    sampler: &QSampler,
    structure: &Structure,
    cfg: &SolventConfig,
) -> Result<DebyeModel> {
    let excluded_dummies = cfg.excluded.unit_scale_dummies(
        structure,
        &sampler.formfactors,
        cfg.bulk_electron_density,
    )?;
    let hydration_dummies = cfg.hydration.dummies(structure, &sampler.formfactors)?;

    let terms = sampler
        .q_values
        .par_iter()
        .map(|&q| {
            let atom_ff = formfactor::formfactor_by_index(&sampler.formfactors, q, &structure.ids)?;

            let mut excluded_ff = Vec::new();
            excluded_dummies.form_factors_at(q, &mut excluded_ff)?;

            let mut hydration_ff = Vec::new();
            hydration_dummies.form_factors_at(q, &mut hydration_ff)?;

            let atom_ff = atom_ff.as_slice();
            Ok(DebyeTerms {
                atoms: self_intensity_at_q(&structure.pos, atom_ff, q),
                excluded: self_intensity_at_q(excluded_dummies.positions(), &excluded_ff, q),
                hydration: self_intensity_at_q(hydration_dummies.positions(), &hydration_ff, q),
                atom_excluded: cross_intensity_at_q(
                    &structure.pos,
                    atom_ff,
                    excluded_dummies.positions(),
                    &excluded_ff,
                    q,
                ),
                atom_hydration: cross_intensity_at_q(
                    &structure.pos,
                    atom_ff,
                    hydration_dummies.positions(),
                    &hydration_ff,
                    q,
                ),
                excluded_hydration: cross_intensity_at_q(
                    excluded_dummies.positions(),
                    &excluded_ff,
                    hydration_dummies.positions(),
                    &hydration_ff,
                    q,
                ),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(DebyeModel {
        q_values: sampler.q_values.clone(),
        terms,
    })
}

/// Cached Debye state for solvent fitting.
pub struct DebyeModel {
    q_values: Vec<f64>,
    terms: Vec<DebyeTerms>,
}

impl DebyeModel {
    /// Compute `I(q)` for the given solvent fit knobs.
    pub fn intensity(&self, volume_scale: f64, contrast_density: f64) -> Intensity {
        let per_q: Vec<(f64, f64, f64, f64)> = self
            .terms
            .iter()
            .map(|terms| {
                let total = terms.combine(volume_scale, contrast_density);
                (total, terms.atoms, terms.excluded, terms.hydration)
            })
            .collect();
        let (total, atoms, excluded, hydration): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            per_q.into_iter().multiunzip();

        Intensity {
            q: self.q_values.clone(),
            total,
            contributions: Some(Contributions {
                atoms,
                excluded,
                hydration,
            }),
            ..Default::default()
        }
    }

    /// The q grid the fit was prepared over.
    pub fn q_values(&self) -> &[f64] {
        &self.q_values
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DebyeTerms {
    atoms: f64,
    excluded: f64,
    hydration: f64,
    atom_excluded: f64,
    atom_hydration: f64,
    excluded_hydration: f64,
}

impl DebyeTerms {
    fn combine(self, volume_scale: f64, contrast_density: f64) -> f64 {
        self.atoms
            + volume_scale * volume_scale * self.excluded
            + contrast_density * contrast_density * self.hydration
            - 2.0 * volume_scale * self.atom_excluded
            + 2.0 * contrast_density * self.atom_hydration
            - 2.0 * volume_scale * contrast_density * self.excluded_hydration
    }
}

/// Calculate I(q) using the Debye formula.
///
/// Distances and form factor products are computed on-the-fly to avoid N×N matrix allocations.
fn intensity_at_q(
    atom_ids: &[usize],
    positions: &[Vector3],
    formfactors: &FormFactorMap,
    q: f64,
) -> Result<f64> {
    log::trace!("Calculating 𝐼(𝑞) for 𝑞 = {q}");

    // TODO: Optimize by storing in static table, see std::sync::OnceLock
    let ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
    Ok(self_intensity_at_q(positions, ff.as_slice(), q))
}

/// Self Debye sum: Σ_i f_i² + 2 Σ_{i<j} f_i f_j sinc(q r_ij). Equals the
/// orientationally averaged form factor ⟨|F(q)|²⟩_Ω of the scatterer set.
pub(crate) fn self_intensity_at_q(positions: &[Vector3], form_factors: &[f64], q: f64) -> f64 {
    let n = form_factors.len();
    let diagonal: f64 = form_factors.iter().map(|f| f * f).sum();
    if n < 2 {
        return diagonal;
    }

    let mut off_diagonal = 0.0;
    for i in 0..n - 1 {
        let fi = form_factors[i];
        let pi = &positions[i];
        for j in i + 1..n {
            let r = (pi - positions[j]).norm();
            off_diagonal += fi * form_factors[j] * sinc(q * r);
        }
    }

    2.0f64.mul_add(off_diagonal, diagonal)
}

/// Cross Debye sum between two disjoint scatterer groups:
/// Σ_i∈a Σ_j∈b f_i f_j sinc(q r_ij).
fn cross_intensity_at_q(
    a_positions: &[Vector3],
    a_form_factors: &[f64],
    b_positions: &[Vector3],
    b_form_factors: &[f64],
    q: f64,
) -> f64 {
    let mut total = 0.0;
    for (pa, &fa) in a_positions.iter().zip(a_form_factors) {
        for (pb, &fb) in b_positions.iter().zip(b_form_factors) {
            total += fa * fb * sinc(q * (pa - pb).norm());
        }
    }
    total
}
