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

use crate::Vector3;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;

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
}

impl DirectTransform {
    pub fn new(p_max: usize) -> Self {
        let directions: Vec<_> = DIRECTIONS
            .iter()
            .flat_map(|dir| (1..=p_max).map(move |p| 2.0 * PI * p as f64 * dir))
            .collect();

        log::info!(
            "Generated {} q-vectors with p_max = {}",
            directions.len(),
            p_max
        );

        Self { directions }
    }

    /// Calculate I(q) for a given structure inside cuboidal box
    pub fn eval(&self, positions: &[Vector3], cuboid: &Vector3) -> Vec<(f64, f64)> {
        self.directions
            .par_iter()
            .map(|dir| dir.component_div(cuboid))
            .map(|q| (q.norm(), calc_intensity(&q, positions)))
            .collect()
    }
}

/// Calculate intensity for given q vector
fn calc_intensity(q: &Vector3, positions: &[Vector3]) -> f64 {
    let qr: Vec<f64> = positions.iter().map(|r| q.dot(r)).collect();
    let sum_cos = qr.iter().fold(0.0, |acc, qr| acc + qr.cos());
    let sum_sin = qr.iter().fold(0.0, |acc, qr| acc + qr.sin());
    Complex::new(sum_cos, sum_sin).norm_sqr() / positions.len() as f64
}
