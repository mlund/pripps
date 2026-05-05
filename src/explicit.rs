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

/// Sample directions (h,k,l)
///
/// This is the thirteen permutations of the Miller index [100], [110], [101]
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
        let excluded_dummies =
            cfg.excluded
                .dummies(structure, &self.formfactors, cfg.bulk_electron_density)?;
        let hydration_dummies = cfg.hydration.dummies(structure, &self.formfactors)?;

        let volume_scale = cfg.excluded.volume_scale();
        let contrast_density = cfg.hydration.contrast_density();

        // Per q-vector: compute three amplitudes A, C, B and combine
        // as |A − c₁·C + c₂·B|²
        let table = self
            .directions
            .par_iter()
            .map(|dir| dir.component_div(box_sides))
            .map(|q| {
                let qnorm = q.norm();

                let atom_ff = eval_atom_form_factors(qnorm, &structure.ids, &self.formfactors)?;
                let a = complex_amplitude(&q, &structure.pos, &atom_ff);

                let mut ev_ff = Vec::new();
                excluded_dummies.form_factors_at(qnorm, &mut ev_ff)?;
                let c = complex_amplitude(&q, excluded_dummies.positions(), &ev_ff);

                let mut hyd_ff = Vec::new();
                hydration_dummies.form_factors_at(qnorm, &mut hyd_ff)?;
                let b = complex_amplitude(&q, hydration_dummies.positions(), &hyd_ff);

                let total = (a - volume_scale * c + contrast_density * b).norm_sqr();
                Ok((qnorm, total, a.norm_sqr(), c.norm_sqr(), b.norm_sqr()))
            })
            .collect::<Result<Vec<_>>>()?;

        // Average duplicate q-norms (from different Miller directions)
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

        Ok(Intensity {
            q,
            total,
            contributions: Some(Contributions {
                atoms,
                excluded,
                hydration,
            }),
            beta: None,
        })
    }
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
