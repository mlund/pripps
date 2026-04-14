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

//! Explicit calculation using discrete q vector

use crate::{Error, IntensityCalculator, Result, Structure, Vector3, formfactor::FormFactorMap};
use itertools::Itertools;
use num_complex::Complex;
use rayon::prelude::*;
use std::iter::zip;
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
}

impl IntensityCalculator for DirectTransform {
    /// Calculate I(q) for a given structure inside its cuboidal box.
    fn calc_intensities(&self, structure: &Structure) -> Result<Vec<(f64, f64)>> {
        let box_sides = structure.box_sides.ok_or_else(|| {
            Error::usage("direct Fourier transform requires box side lengths on the structure")
        })?;

        let intensity = |q: Vector3| calc_intensity(&q, structure, &self.formfactors);

        let table = self
            .directions
            .par_iter()
            .map(|dir| dir.component_div(&box_sides))
            .map(|q| intensity(q).map(|i| (q.norm(), i)))
            .collect::<Result<Vec<(f64, f64)>>>()?;

        Ok(average_duplicates(table))
    }
}

impl DirectTransform {
    pub(crate) fn new(formfactors: FormFactorMap, p_max: usize) -> Self {
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
        }
    }
}

/// Calculate intensity for given q vector
///
/// See https://doi.org/10.1103/PhysRevE.62.5409 for a potential optimization
fn calc_intensity(q: &Vector3, structure: &Structure, formfactors: &FormFactorMap) -> Result<f64> {
    let qr = structure.pos.iter().map(|r| q.dot(r));
    let formfactors = convert_formfactors(q.norm(), &structure.ids, formfactors)?;

    // Vector of (𝙦⋅𝙧ᵢ, 𝘧ᵢ) tuples
    let qr_f = zip(qr, formfactors).collect::<Vec<_>>();

    let sum_cos = qr_f.iter().fold(0.0, |acc, (qr, f)| acc + qr.cos().mul(*f));
    let sum_sin = qr_f.iter().fold(0.0, |acc, (qr, f)| acc + qr.sin().mul(*f));

    Ok(Complex::new(sum_cos, sum_sin).norm_sqr())
}

/// Helper function to convert formfactor map to F(q) for each atom in structure
fn convert_formfactors(
    q: f64,
    atom_ids: &[usize],
    formfactors: &FormFactorMap,
) -> Result<impl Iterator<Item = f64>> {
    // F(q) for each atom kind
    let formfactors: Vec<f64> = formfactors.values().map(|ff| ff.eval(q)).try_collect()?;
    // F(q) for each atom in structure
    Ok(atom_ids.iter().map(move |i| formfactors[*i]))
}

/// Helper merge duplicate x values by averaging y; then sort by x
///
/// `[(x1,y1),(x1,y2), (x2,y3),...] -> [(x1,<y1,y2>, (x2,y3),...]`
fn average_duplicates(with_duplicate_x: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    // Round x to integer to be used for grouping: 0.123456 -> 123456
    let round = |x: f64| x.mul(1e6).round() as i64;
    // [(x,y1),(x,y2),...] -> (x, <y>)
    let average_duplicates = |pair: &[(f64, f64)]| {
        let mean_y = pair.iter().map(|(_, y)| y).sum::<f64>() / pair.len() as f64;
        (pair.first().unwrap().0, mean_y)
    };

    let mut table: Vec<_> = with_duplicate_x
        .into_iter()
        .into_group_map_by(|(x, _)| round(*x))
        .values()
        .map(|pair| average_duplicates(pair))
        .collect();
    table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    table
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
