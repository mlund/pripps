// Copyright 2025 Isabel Vinterbladh, Sergei Grudinin and Mikael Lund
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
use anyhow::Result;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Calculate N x N distance matrix
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
) -> Result<Vec<(f64, f64)>> {
    let q_values: Vec<_> = (0..nq)
        .map(|i| q_range.0 + (q_range.1 - q_range.0) * i as f64 / (nq - 1) as f64)
        .collect();

    let distance_matrix = distance_matrix(&structure.pos);

    let intensities = q_values
        .par_iter()
        .map(|q| intensity_at_q(&structure.ids, formfactors, &distance_matrix, *q))
        .collect::<Result<Vec<f64>>>()?;

    Ok(q_values.into_iter().zip(intensities).collect())
}

/// Calculate I(q) for a given q value
pub fn intensity_at_q(
    atom_ids: &[usize],
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    distance_matrix: &nalgebra::DMatrix<f64>,
    q: f64,
) -> Result<f64> {
    if q.abs() < 1e-10 {
        // avoid division by zero: sin(x)/x -> 1 as x -> 0
        intensity_at_q_with(atom_ids, formfactors, distance_matrix, q, |_| 1.0)
    } else {
        intensity_at_q_with(atom_ids, formfactors, distance_matrix, q, |qr| {
            qr.sin() / qr
        })
    }
}

fn intensity_at_q_with(
    atom_ids: &[usize],
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    distance_matrix: &nalgebra::DMatrix<f64>,
    q: f64,
    func: impl Fn(f64) -> f64,
) -> Result<f64> {
    let ff_products = formfactor::formfactor_product_at_q(formfactors, q)?;
    let mut intensity = 0.0;
    let n_atoms = atom_ids.len();
    for i in 0..n_atoms - 1 {
        for j in i + 1..n_atoms {
            let qr = q * distance_matrix[(i, j)];
            let ff = ff_products[(atom_ids[i], atom_ids[j])];
            intensity += ff * func(qr);
        }
    }
    Ok(intensity)
}
