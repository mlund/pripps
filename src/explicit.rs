// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Explicit calculation using discrete q vectors

use crate::solvent::SolventConfig;
use crate::{
    Contributions, Error, Intensity, IntensityCalculator, Result, Structure, Vector3,
    formfactor::FormFactorMap,
};
use itertools::Itertools;
use num_complex::Complex;
use rayon::prelude::*;
use std::ops::Mul;
use std::{cmp::Ordering, f64::consts::PI};

/// Reciprocal-space sampling directions (h, k, l).
///
/// The discrete method estimates the orientationally-averaged scattering
/// intensity I(q) by evaluating it along a fixed set of directions. For a
/// periodic box with side lengths L, only wavevectors commensurate with the
/// box contribute, namely the reciprocal-lattice vectors
///
/// ```text
///     q = 2π · p · (h, k, l) / L,    p = 1, 2, …, p_max
/// ```
///
/// Each (h, k, l) below fixes a direction; the integer multiples p (applied in
/// [`DirectTransform::new`]) step outward along it to span the |q| range.
///
/// These are the thirteen low-index directions of the [100], [110] and [111]
/// families. The cube has 26 such directions, but because the scattering
/// density is real, F(−q) = F(q)\* and hence I(q) = I(−q) (Friedel's law).
/// Antipodal pairs are therefore redundant and only one representative of each
/// is kept, leaving thirteen.
const MILLER_INDEX: [Vector3; 13] = [
    Vector3::new(1.0, 0.0, 0.0),
    Vector3::new(0.0, 1.0, 0.0),
    Vector3::new(0.0, 0.0, 1.0),
    Vector3::new(1.0, 1.0, 0.0),
    Vector3::new(0.0, 1.0, 1.0),
    Vector3::new(1.0, 0.0, 1.0),
    Vector3::new(-1.0, 1.0, 0.0),
    Vector3::new(-1.0, 0.0, 1.0),
    Vector3::new(0.0, -1.0, 1.0),
    Vector3::new(1.0, 1.0, 1.0),
    Vector3::new(-1.0, 1.0, 1.0),
    Vector3::new(1.0, -1.0, 1.0),
    Vector3::new(1.0, 1.0, -1.0),
];

/// Calculate I(q) using discrete q vectors
///
/// References:
///
/// - http://doi.org/d8zgw5
/// - http://doi.org/10.1063/1.449987
#[derive(Debug)]
pub(crate) struct DirectTransform {
    /// q directions; divided by box side lengths to get q
    directions: Vec<Vector3>,
    /// Form factors for each atom kind
    formfactors: FormFactorMap,
    /// Optional solvent correction
    solvent: Option<SolventConfig>,
}

/// Complex amplitude A(q) = Σ f_i · exp(i·q·r_i) for a set of
/// scatterers at given positions with given form factor values.
fn complex_amplitude(q: &Vector3, positions: &[Vector3], form_factors: &[f64]) -> Complex<f64> {
    let (re, im) = positions
        .iter()
        .zip(form_factors)
        .fold((0.0, 0.0), |(re, im), (r, &f)| {
            let qr = q.dot(r);
            (re + f * qr.cos(), im + f * qr.sin())
        });
    Complex::new(re, im)
}

impl IntensityCalculator for DirectTransform {
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity> {
        let box_sides = structure.box_sides.ok_or_else(|| {
            Error::usage("direct Fourier transform requires box side lengths on the structure")
        })?;

        match self.solvent.as_ref().filter(|cfg| cfg.is_active()) {
            None => self.vacuum_intensities(structure, &box_sides),
            Some(cfg) => self.solvent_intensities(structure, &box_sides, cfg),
        }
    }
}

impl DirectTransform {
    pub(crate) fn new(
        formfactors: FormFactorMap,
        p_max: usize,
        solvent: Option<SolventConfig>,
    ) -> Self {
        // Expand each base direction into p_max reciprocal-lattice points,
        // q·L = 2π·p·(h,k,l). Larger p reaches higher |q| (finer real-space
        // detail) along the same direction; the division by the box sides is
        // deferred to the per-structure calculation.
        let directions: Vec<_> = MILLER_INDEX
            .iter()
            .flat_map(|dir| (1..=p_max).map(move |p| 2.0 * PI * p as f64 * dir))
            .collect();

        log::info!(
            "Generated {} q-vectors with p_max = {}",
            directions.len(),
            p_max
        );

        Self {
            formfactors,
            directions,
            solvent,
        }
    }

    fn vacuum_intensities(&self, structure: &Structure, box_sides: &Vector3) -> Result<Intensity> {
        let table = self
            .directions
            .par_iter()
            .map(|dir| dir.component_div(box_sides))
            .map(|q| {
                let ff = eval_atom_form_factors(q.norm(), &structure.ids, &self.formfactors)?;
                let intensity = complex_amplitude(&q, &structure.pos, &ff).norm_sqr();
                Ok((q.norm(), intensity))
            })
            .collect::<Result<Vec<(f64, f64)>>>()?;

        let (q, total): (Vec<f64>, Vec<f64>) = average_duplicates(table).into_iter().unzip();
        Ok(Intensity {
            q,
            total,
            ..Default::default()
        })
    }

    fn solvent_intensities(
        &self,
        structure: &Structure,
        box_sides: &Vector3,
        cfg: &SolventConfig,
    ) -> Result<Intensity> {
        let model = build_model(
            &self.formfactors,
            &self.directions,
            structure,
            box_sides,
            cfg,
        )?;
        Ok(model.intensity(
            cfg.excluded.volume_scale(),
            cfg.hydration.contrast_density(),
        ))
    }
}

pub(crate) fn directions(p_max: usize) -> Vec<Vector3> {
    MILLER_INDEX
        .iter()
        .flat_map(|dir| (1..=p_max).map(move |p| 2.0 * PI * p as f64 * dir))
        .collect()
}

/// Precompute the atomic, excluded-volume, and hydration amplitudes
/// for every explicit q-vector. The cached q-vector rows are combined
/// and only then averaged by q magnitude, matching the one-shot
/// [`DirectTransform`] path.
pub(crate) fn build_model(
    formfactors: &FormFactorMap,
    directions: &[Vector3],
    structure: &Structure,
    box_sides: &Vector3,
    cfg: &SolventConfig,
) -> Result<DirectModel> {
    let excluded_dummies =
        cfg.excluded
            .unit_scale_dummies(structure, formfactors, cfg.bulk_electron_density)?;
    let hydration_dummies = cfg.hydration.dummies(structure, formfactors)?;

    let rows = directions
        .par_iter()
        .map(|dir| dir.component_div(box_sides))
        .map(|q| {
            let qnorm = q.norm();

            let atom_ff = eval_atom_form_factors(qnorm, &structure.ids, formfactors)?;
            let atoms = complex_amplitude(&q, &structure.pos, &atom_ff);

            let mut excluded_ff = Vec::new();
            excluded_dummies.form_factors_at(qnorm, &mut excluded_ff)?;
            let excluded = complex_amplitude(&q, excluded_dummies.positions(), &excluded_ff);

            let mut hydration_ff = Vec::new();
            hydration_dummies.form_factors_at(qnorm, &mut hydration_ff)?;
            let hydration = complex_amplitude(&q, hydration_dummies.positions(), &hydration_ff);

            Ok(DirectTerms {
                qnorm,
                atoms,
                excluded,
                hydration,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let q_values = unique_q_values(&rows);
    Ok(DirectModel { q_values, rows })
}

/// Cached explicit-transform state for solvent fitting.
pub struct DirectModel {
    q_values: Vec<f64>,
    rows: Vec<DirectTerms>,
}

impl DirectModel {
    /// Compute `I(q)` for the given solvent fit knobs.
    pub fn intensity(&self, volume_scale: f64, contrast_density: f64) -> Intensity {
        let table = self
            .rows
            .iter()
            .map(|row| {
                let total = (row.atoms - volume_scale * row.excluded
                    + contrast_density * row.hydration)
                    .norm_sqr();
                (
                    row.qnorm,
                    total,
                    row.atoms.norm_sqr(),
                    row.excluded.norm_sqr(),
                    row.hydration.norm_sqr(),
                )
            })
            .collect();

        intensity_from_5col(table)
    }

    /// The q grid the fit was prepared over after duplicate q-norms are averaged.
    pub fn q_values(&self) -> &[f64] {
        &self.q_values
    }
}

#[derive(Debug, Clone, Copy)]
struct DirectTerms {
    qnorm: f64,
    atoms: Complex<f64>,
    excluded: Complex<f64>,
    hydration: Complex<f64>,
}

/// Evaluate form factors for each atom at a given q.
fn eval_atom_form_factors(
    q: f64,
    atom_ids: &[usize],
    formfactors: &FormFactorMap,
) -> Result<Vec<f64>> {
    let per_kind: Vec<f64> = formfactors
        .form_factors()
        .map(|ff| ff.eval(q))
        .try_collect()?;
    Ok(atom_ids.iter().map(|&id| per_kind[id]).collect())
}

/// Merge duplicate q-values by averaging, for 2-column data.
fn average_duplicates(data: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let round = |x: f64| x.mul(1e6).round() as i64;
    let average = |pair: &[(f64, f64)]| {
        let mean_y = pair.iter().map(|(_, y)| y).sum::<f64>() / pair.len() as f64;
        (pair.first().unwrap().0, mean_y)
    };

    let mut table: Vec<_> = data
        .into_iter()
        .into_group_map_by(|(x, _)| round(*x))
        .values()
        .map(|pair| average(pair))
        .collect();
    table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    table
}

/// Merge duplicate q-values by averaging, for 5-column data.
fn average_duplicates_5col(data: Vec<(f64, f64, f64, f64, f64)>) -> Vec<(f64, f64, f64, f64, f64)> {
    let round = |x: f64| x.mul(1e6).round() as i64;
    data.into_iter()
        .into_group_map_by(|(q, _, _, _, _)| round(*q))
        .into_values()
        .map(|group| {
            let n = group.len() as f64;
            let q = group[0].0;
            let t = group.iter().map(|g| g.1).sum::<f64>() / n;
            let a = group.iter().map(|g| g.2).sum::<f64>() / n;
            let e = group.iter().map(|g| g.3).sum::<f64>() / n;
            let h = group.iter().map(|g| g.4).sum::<f64>() / n;
            (q, t, a, e, h)
        })
        .collect()
}

fn intensity_from_5col(table: Vec<(f64, f64, f64, f64, f64)>) -> Intensity {
    let mut grouped = average_duplicates_5col(table);
    grouped.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut q = Vec::with_capacity(grouped.len());
    let mut total = Vec::with_capacity(grouped.len());
    let mut atoms = Vec::with_capacity(grouped.len());
    let mut excluded = Vec::with_capacity(grouped.len());
    let mut hydration = Vec::with_capacity(grouped.len());
    for (qi, t, a, e, h) in grouped {
        q.push(qi);
        total.push(t);
        atoms.push(a);
        excluded.push(e);
        hydration.push(h);
    }

    Intensity {
        q,
        total,
        contributions: Some(Contributions {
            atoms,
            excluded,
            hydration,
        }),
    }
}

fn unique_q_values(rows: &[DirectTerms]) -> Vec<f64> {
    average_duplicates(rows.iter().map(|row| (row.qnorm, 0.0)).collect())
        .into_iter()
        .map(|(q, _)| q)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_duplicates() {
        let v = vec![(1.0, 1.0), (3.0, 0.1), (2.0, 3.0), (2.0, 4.0), (1.0, 2.0)];
        let result = average_duplicates(v);
        assert_eq!(result, vec![(1.0, 1.5), (2.0, 3.5), (3.0, 0.1)]);
    }
}
