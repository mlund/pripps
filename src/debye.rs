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

use crate::{
    IntensityCalculator, Structure,
    formfactor::{self, FormFactorMap},
};
use anyhow::Result;
use nalgebra::DMatrix;
use rayon::prelude::*;

pub struct DebyeIntensity {
    /// List of q values
    q_values: Vec<f64>,
    /// Form factors for each atom kind
    formfactors: FormFactorMap,
}

impl DebyeIntensity {
    pub fn new(formfactors: FormFactorMap, q_minmax: (f64, f64), num_q: usize) -> Result<Self> {
        Ok(Self {
            q_values: make_q_values(q_minmax.0, q_minmax.1, num_q),
            formfactors,
        })
    }
}

impl IntensityCalculator for DebyeIntensity {
    /// Calculate I(q) over a range of q values using the Debye formula
    fn calc_intensities(&self, structure: &Structure) -> Result<Vec<(f64, f64)>> {
        let distance_matrix = structure.distance_matrix();
        let intensities = self
            .q_values
            .par_iter()
            .map(|q| {
                intensity_at_q(
                    &structure.ids,
                    &self.formfactors,
                    &distance_matrix,
                    *q,
                    None,
                )
            })
            .collect::<Result<Vec<f64>>>()?;

        Ok(self.q_values.iter().cloned().zip(intensities).collect())
    }
    fn set_side_lengths(&self, _side_lengths: crate::Vector3) {
        // The Debye formula does not depend on box side lengths
    }
}

/// Generate a list of q values from qmin to qmax
fn make_q_values(qmin: f64, qmax: f64, nq: usize) -> Vec<f64> {
    (0..nq)
        .map(|i| qmin + (qmax - qmin) * i as f64 / (nq - 1) as f64)
        .collect()
}

/// Calculate I(q) using the Debye formula
///
/// If given, the solvent correction is a list of corrections for each atom that is added to the form factor.
fn intensity_at_q(
    atom_ids: &[usize],
    formfactors: &FormFactorMap,
    distance_matrix: &DMatrix<f64>,
    q: f64,
    solvent_correction: Option<Vec<f64>>,
) -> Result<f64> {
    log::trace!("Calculating I(q) for q = {}", q);
    const NEARLY_ZERO: f64 = 1e-10;

    let mut formfactors = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
    if let Some(solvent) = solvent_correction {
        for (i, q) in formfactors.iter_mut().enumerate() {
            *q += solvent[i];
        }
    }
    let formfactor_product = &formfactors * formfactors.transpose();

    // avoid division by zero: sin(𝑥)/𝑥 -> 1 as 𝑥 -> 0
    // Taylor expansion: sin(𝑥)/𝑥 ≈ 1 - 𝑥²/6 + 𝑥⁴/120 - 𝑥⁶/5040 ...
    if q.abs() < NEARLY_ZERO {
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

    let mut off_diagonal = 0.0;
    for i in 0..n_atoms - 1 {
        for j in i + 1..n_atoms {
            let qr = q * distance_matrix[(i, j)];
            off_diagonal += formfactor_product[(i, j)] * func(qr);
        }
    }

    let diagonal: f64 = atom_ids
        .iter()
        .map(|&id| formfactor_product[(id, id)])
        .sum();

    Ok(2.0 * off_diagonal + diagonal)
}
