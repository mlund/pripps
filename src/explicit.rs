// Copyright 2025 Isabel Vinterbladh, Rémi Vuillemot, Sergei Grudinin and Mikael Lund
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

use crate::{IntensityCalculator, Structure, Vector3, formfactor::FormFactorMap};
use anyhow::Result;
use itertools::Itertools;
use num_complex::Complex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::iter::zip;
use std::ops::Mul;
use std::{cmp::Ordering, f64::consts::PI};

/// Sample directions (h,k,l)
///
/// This is the thirteen permutations of the Miller index [100], [110], [101]
const DIRECTIONS: [Vector3; 13] = [
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
///
/// @TODO: Add form factors
#[derive(Debug, Clone, PartialEq)]
pub struct DirectTransform {
    /// q directions; divided by box side lengths to get q
    directions: Vec<Vector3>,
    /// Form factors for each atom kind
    formfactors: FormFactorMap,
}

impl IntensityCalculator for DirectTransform {
    /// Calculate I(q) for a given structure inside cuboidal box
    fn calc_intensities(
        &self,
        structure: &Structure,
        side_lengths: Option<Vector3>,
    ) -> Result<Vec<(f64, f64)>> {
        let Some(side_lengths) = side_lengths else {
            anyhow::bail!("Box side lengths must be provided");
        };

        let intensity = |q: Vector3| calc_intensity(&q, structure, &self.formfactors).unwrap();

        let table: Vec<(f64, f64)> = self
            .directions
            .par_iter()
            .map(|dir| dir.component_div(&side_lengths))
            .map(|q| (q.norm(), intensity(q)))
            .collect();

        Ok(average_duplicates(table))
    }
}

impl DirectTransform {
    pub fn new(formfactors: FormFactorMap, p_max: usize) -> Self {
        let directions: Vec<_> = DIRECTIONS
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

/// Convert formfactor map to F(q) for each atom in structure
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

/// Average duplicate q values, then sort by q
fn average_duplicates(v: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut map: HashMap<i64, (f64, usize)> = HashMap::new();

    // Function to round f64 to a certain number of decimal places (e.g., 6 places)
    fn round_to_precision(x: f64, precision: usize) -> i64 {
        (x * 10f64.powi(precision as i32)).round() as i64
    }

    // Iterate over the vector and populate the map with sums and counts
    for (x, y) in v {
        let rounded_x = round_to_precision(x, 6); // Round x to 6 decimal places
        let entry = map.entry(rounded_x).or_insert((0.0, 0));
        entry.0 += y; // Sum of y
        entry.1 += 1; // Count of occurrences
    }

    // Create a new vector with averaged results
    let mut table = map
        .into_iter()
        .map(|(rounded_x, (sum_y, count))| (rounded_x as f64 / 10f64.powi(6), sum_y / count as f64))
        .collect::<Vec<_>>();

    table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    table
}
