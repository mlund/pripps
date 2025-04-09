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

use std::{fs::File, io::Read, path::Path, str::FromStr};

use crate::{
    IntensityCalculator, Structure,
    formfactor::{self, FormFactorMap},
};
use anyhow::{Result, anyhow};
use itertools::Itertools;
use nalgebra::DMatrix;
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
        let sasa = structure.calc_sasa(1.4);
        log::info!("Total SASA = {:.1} Å²", sasa.iter().sum::<f64>());
        let intensities = self
            .q_values
            .par_iter()
            .map(|q| {
                intensity_at_q(
                    &structure.ids,
                    &self.formfactors,
                    &distance_matrix,
                    *q,
                    Some(sasa.clone()),
                )
            })
            .collect::<Result<Vec<f64>>>()?;

        Ok(self.q_values.iter().cloned().zip(intensities).collect())
    }

    #[allow(unused_variables)]
    fn set_side_lengths(&self, side_lengths: crate::Vector3) {
        // The Debye formula does not depend on box side lengths
    }
}

/// Generate a list of q values from qmin to qmax
pub fn make_q_values(qmin: f64, qmax: f64, nq: usize) -> Vec<f64> {
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
    sasa: Option<Vec<f64>>,
) -> Result<f64> {
    log::trace!("Calculating 𝐼(𝑞) for 𝑞 = {}", q);
    let water_ff = formfactors
        .iter()
        .find(|(id, _)| id.name() == "H2O")
        .map(|(_, ff)| ff)
        .ok_or_else(|| anyhow!("no water form factor 'H2O' needed for SASA calculation"))?;

    // TODO: Optimize by storing in static table, see std::sync::LazyLock
    let mut formfactors = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;

    // Solvent correction using SASA
    if let Some(sasa) = sasa {
        debug_assert_eq!(sasa.len(), formfactors.len());
        // let sasa = load_vector_from_file::<f64>(Path::new("sasa.dat"))?;

        log::trace!("SASA: {sasa:?}");
        let offset = 0.0; // Å
        let scale = 0.0;
        let max_sasa = sasa
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let water_ff = scale * water_ff.eval(q)? * sinc(q * offset);
        for (ff, sasa) in formfactors.iter_mut().zip(&sasa) {
            *ff += sasa / max_sasa * water_ff;
        }
    }

    // N x N matrix of formfactor products (N = number of atoms)
    let formfactor_product = &formfactors * formfactors.transpose();

    intensity_sum(&formfactor_product, distance_matrix, q)
}

/// Helper function to load space-separated values from a file to a vector
#[allow(dead_code)]
fn load_vector_from_file<T: FromStr>(path: &Path) -> Result<Vec<T>> {
    let mut contents = String::new();
    File::open(path)?.read_to_string(&mut contents)?;
    contents
        .split_whitespace()
        .map(|s| s.parse())
        .try_collect()
        .map_err(|_| anyhow!("parse error while loading values from {}", path.display()))
}

/// Calculate I(q)
///
/// The two input matrices must be N x N (N = number of atoms)
fn intensity_sum(
    formfactor_product: &DMatrix<f64>,
    distance_matrix: &DMatrix<f64>,
    q: f64,
) -> Result<f64> {
    assert!(distance_matrix.is_square());
    assert_eq!(distance_matrix.shape(), formfactor_product.shape());

    let n_atoms = distance_matrix.ncols();

    let mut off_diagonal = 0.0;
    for i in 0..n_atoms - 1 {
        for j in i + 1..n_atoms {
            let qr = q * distance_matrix[(i, j)];
            off_diagonal += formfactor_product[(i, j)] * sinc(qr);
        }
    }

    let diagonal = formfactor_product.diagonal().sum();

    Ok(2.0 * off_diagonal + diagonal)
}
