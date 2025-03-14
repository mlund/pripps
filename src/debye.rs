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

use crate::{AtomKind, FormFactor, Structure, Vector3, formfactor};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Calculate distance matrix and store in N x N matrix
fn distance_matrix(positions: &[Vector3]) -> nalgebra::DMatrix<f64> {
    let n = positions.len();
    let mut matrix = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n - 1 {
        for j in i + 1..n {
            let r = (positions[i] - positions[j]).norm();
            matrix[(i, j)] = r;
            matrix[(j, i)] = r;
        }
    }
    matrix
}

/// Calculate I(q) over a range of q values using the Debye formula
pub fn calc_intensity(
    structure: &Structure,
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    q_range: (f64, f64),
    nq: usize,
) -> Vec<(f64, f64)> {
    let q_values: Vec<_> = (0..nq)
        .map(|i| q_range.0 + (q_range.1 - q_range.0) * i as f64 / (nq - 1) as f64)
        .collect();

    let distance_matrix = distance_matrix(&structure.pos);
    q_values
        .par_iter()
        .map(|q| {
            (
                *q,
                intensity_at_q(structure, formfactors, &distance_matrix, *q),
            )
        })
        .collect()
}

/// Calculate I(q) for a given q value
fn intensity_at_q(
    structure: &Structure,
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    distance_matrix: &nalgebra::DMatrix<f64>,
    q: f64,
) -> f64 {
    let ff_product = formfactor::formfactor_product_at_q(formfactors, q);
    let n = structure.pos.len();
    let mut intensity = 0.0;
    for i in 0..n - 1 {
        for j in i + 1..n {
            let qr = q * distance_matrix[(i, j)];
            let ff = ff_product[(structure.ids[i], structure.ids[j])];
            intensity += ff * qr.sin() / qr;
        }
    }
    intensity
}
