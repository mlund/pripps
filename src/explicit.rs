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

/// Apply minimum-image convention to a displacement vector given orthorhombic box sides.
fn minimum_image(delta: &Vector3, box_sides: &Vector3) -> Vector3 {
    delta.zip_map(box_sides, |d, l| d - l * (d / l).round())
}

/// Unwrap PBC-wrapped positions so the whole molecule lies in one image.
///
/// Anchors on the first atom and folds each subsequent atom to the nearest
/// image of the previous one. Returns the atom positions with intra-molecular
/// distances preserved.
fn unwrap_positions(positions: &[Vector3], box_sides: &Vector3) -> Vec<Vector3> {
    if positions.is_empty() {
        return vec![];
    }
    let mut out = Vec::with_capacity(positions.len());
    out.push(positions[0]);
    for r in &positions[1..] {
        let prev = *out.last().unwrap();
        let delta = minimum_image(&(r - prev), box_sides);
        out.push(prev + delta);
    }
    out
}

/// Orientationally averaged amplitude ⟨F(q)⟩_Ω = Σ_i f_i · sin(q|r_i − center|)/(q|r_i − center|).
///
/// When `box_sides` is provided positions are first unwrapped so distances
/// from the centroid are physically correct even if the molecule straddles
/// a periodic boundary.
fn mean_sinc_amplitude(
    q_norm: f64,
    positions: &[Vector3],
    form_factors: &[f64],
    box_sides: Option<&Vector3>,
) -> f64 {
    let (pos, center) = match box_sides {
        Some(b) => {
            let unwrapped = unwrap_positions(positions, b);
            let c = centroid(&unwrapped);
            (unwrapped, c)
        }
        None => (positions.to_vec(), centroid(positions)),
    };
    pos.iter().zip(form_factors).map(|(r, &f)| {
        let qr = q_norm * (r - center).norm();
        let sinc = if qr < 1e-10 { 1.0 } else { qr.sin() / qr };
        f * sinc
    }).sum()
}

/// Geometric centroid of a set of positions.
fn centroid(positions: &[Vector3]) -> Vector3 {
    if positions.is_empty() {
        return Vector3::zeros();
    }
    positions.iter().sum::<Vector3>() / positions.len() as f64
}

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

        let beta = q
            .iter()
            .zip(&total)
            .map(|(&qi, &p)| {
                let ff = eval_atom_form_factors(qi, &structure.ids, &self.formfactors)?;
                let mean_amp = mean_sinc_amplitude(qi, &structure.pos, &ff, Some(box_sides));
                Ok(if p > 0.0 { mean_amp * mean_amp / p } else { 1.0 })
            })
            .collect::<Result<Vec<f64>>>()?;

        Ok(Intensity {
            q,
            total,
            beta: Some(beta),
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
        let mut beta = Vec::with_capacity(grouped.len());

        for (qi, t, a, e, h) in &grouped {
            q.push(*qi);
            total.push(*t);
            atoms.push(*a);
            excluded.push(*e);
            hydration.push(*h);
        }

        for (&qi, &p) in q.iter().zip(&total) {
            let atom_ff = eval_atom_form_factors(qi, &structure.ids, &self.formfactors)?;
            let mean_a = mean_sinc_amplitude(qi, &structure.pos, &atom_ff, Some(box_sides));

            let mut ev_ff = Vec::new();
            excluded_dummies.form_factors_at(qi, &mut ev_ff)?;
            let mean_c = mean_sinc_amplitude(qi, excluded_dummies.positions(), &ev_ff, Some(box_sides));

            let mut hyd_ff = Vec::new();
            hydration_dummies.form_factors_at(qi, &mut hyd_ff)?;
            let mean_b = mean_sinc_amplitude(qi, hydration_dummies.positions(), &hyd_ff, Some(box_sides));

            let mean_amp = mean_a - volume_scale * mean_c + contrast_density * mean_b;
            beta.push(if p > 0.0 { mean_amp * mean_amp / p } else { 1.0 });
        }

        Ok(Intensity {
            q,
            total,
            contributions: Some(Contributions {
                atoms,
                excluded,
                hydration,
            }),
            beta: Some(beta),
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

    #[test]
    fn direct_beta_near_one_at_low_q_and_decays() {
        // Two atoms offset from origin: centroid correction must be applied.
        let positions = vec![
            Vector3::new(10.0, 10.0, 10.0),
            Vector3::new(10.0, 10.0, 14.0),
        ];
        let form_factors = vec![1.0, 1.0];
        let center = centroid(&positions);

        let beta_low = {
            let amp = mean_sinc_amplitude(0.001, &positions, &form_factors, None);
            amp * amp / (2.0 * 2.0) // P(q→0) = (Σ f_i)²
        };
        let beta_high = {
            let amp = mean_sinc_amplitude(5.0, &positions, &form_factors, None);
            let p = 4.0; // |f1|² + |f2|² (incoherent limit)
            amp * amp / p
        };

        assert!((beta_low - 1.0).abs() < 1e-4, "β at q→0 should be ≈1, got {beta_low}");
        assert!(beta_high < 0.5, "β should decrease at high q, got {beta_high}");
    }
}
