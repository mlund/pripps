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
    IntensityCalculator, Result, Structure, Vector3,
    formfactor::{self, FormFactorMap},
};
use rayon::prelude::*;

/// Returns sinc(x) = sin(x)/x with graceful handling of x -> 0
#[inline]
pub fn sinc(x: f64) -> f64 {
    const NEARLY_ZERO: f64 = 1e-8;
    if x.abs() < NEARLY_ZERO {
        1.0
    } else {
        x.sin() / x
    }
}

pub(crate) struct DebyeIntensity {
    /// List of q values - must be evenly spaced and sorted in ascending order
    q_values: Vec<f64>,
    /// Form factors for each atom kind
    formfactors: FormFactorMap,
}

impl DebyeIntensity {
    pub(crate) fn new(
        formfactors: FormFactorMap,
        q_minmax: (f64, f64),
        num_q: usize,
    ) -> Result<Self> {
        Ok(Self {
            q_values: make_q_values(q_minmax.0, q_minmax.1, num_q),
            formfactors,
        })
    }
}

impl IntensityCalculator for DebyeIntensity {
    /// Calculate I(q) over a range of q values using the Debye formula
    fn calc_intensities(&self, structure: &Structure) -> Result<Vec<(f64, f64)>> {
        let intensities = self
            .q_values
            .par_iter()
            .map(|q| intensity_at_q(&structure.ids, &structure.pos, &self.formfactors, *q))
            .collect::<Result<Vec<f64>>>()?;

        Ok(self.q_values.iter().cloned().zip(intensities).collect())
    }
}

/// Generate a list of q values from qmin to qmax
pub fn make_q_values(qmin: f64, qmax: f64, nq: usize) -> Vec<f64> {
    (0..nq)
        .map(|i| qmin + (qmax - qmin) * i as f64 / (nq - 1) as f64)
        .collect()
}

/// Calculate I(q) using the Debye formula.
///
/// Distances and form factor products are computed on-the-fly to avoid N×N matrix allocations.
fn intensity_at_q(
    atom_ids: &[usize],
    positions: &[Vector3],
    formfactors: &FormFactorMap,
    q: f64,
) -> Result<f64> {
    log::trace!("Calculating 𝐼(𝑞) for 𝑞 = {q}");

    // TODO: Optimize by storing in static table, see std::sync::OnceLock
    let ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
    let n = ff.len();
    let ff = ff.as_slice();

    // Compute I(q) = Σ_i f_i² + 2 Σ_{i<j} f_i * f_j * sinc(q * r_ij)
    let diagonal: f64 = ff.iter().map(|f| f * f).sum();

    let mut off_diagonal = 0.0;
    for i in 0..n - 1 {
        let fi = ff[i];
        let pi = &positions[i];
        for j in i + 1..n {
            let r = (pi - positions[j]).norm();
            off_diagonal += fi * ff[j] * sinc(q * r);
        }
    }

    Ok(2.0 * off_diagonal + diagonal)
}
