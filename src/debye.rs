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

use crate::{AtomKind, FormFactor, Structure, formfactor};
use anyhow::Result;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::{collections::BTreeMap, ops::Div};

/// Calculate I(q) over a range of q values using the Debye formula
pub fn calc_intensity(
    structure: &Structure,
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    q_values: &[f64],
) -> Result<Vec<(f64, f64)>> {
    let distance_matrix = structure.distance_matrix();
    let intensities = q_values
        .par_iter()
        .map(|q| intensity_at_q(&structure.ids, formfactors, &distance_matrix, *q))
        .collect::<Result<Vec<f64>>>()?;

    Ok(q_values.iter().cloned().zip(intensities).collect())
}

/// Calculate I(q) using the Debye formula
fn intensity_at_q(
    atom_ids: &[usize],
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    distance_matrix: &DMatrix<f64>,
    q: f64,
) -> Result<f64> {
    log::trace!("Calculating I(q) for q = {}", q);
    let formfactor_product = formfactor::formfactor_product(formfactors, q)?;

    // avoid division by zero: sin(x)/x -> 1 as x -> 0
    if q.abs() < 1e-10 {
        intensity_sum(atom_ids, &formfactor_product, distance_matrix, q, |_| 1.0)
    } else {
        intensity_sum(atom_ids, &formfactor_product, distance_matrix, q, |qr| {
            qr.sin() / qr
        })
    }
}

/// Calculate I(q) using a generic function f(qr)
fn intensity_sum(
    atom_ids: &[usize],
    formfactor_product: &DMatrix<f64>,
    distance_matrix: &DMatrix<f64>,
    q: f64,
    func: impl Fn(f64) -> f64,
) -> Result<f64> {
    let n_atoms = atom_ids.len();

    // half off-diagonal contribution (i < j)
    let mut off_diagonal = 0.0;
    for i in 0..n_atoms - 1 {
        for j in i + 1..n_atoms {
            let qr = q * distance_matrix[(i, j)];
            let id1 = atom_ids[i];
            let id2 = atom_ids[j];
            off_diagonal += formfactor_product[(id1, id2)] * func(qr);
        }
    }

    // diagonal contribution (i = j)
    let trace: f64 = atom_ids
        .iter()
        .map(|&id| formfactor_product[(id, id)])
        .sum();

    Ok((2.0 * off_diagonal + trace).div(n_atoms as f64))
}
