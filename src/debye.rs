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

use std::{env, fs::File, io::Read, path::Path, str::FromStr};

use crate::{
    IntensityCalculator, Structure, Vector3,
    formfactor::{self, FormFactorMap},
};
use anyhow::{Result, anyhow};
use itertools::Itertools;
use log::info;
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
    /// List of q values - must be evenly spaced and sorted in ascending order
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
        let sasa = if env::var("SASA_DISABLE").is_ok() {
            info!("SASA calculation disabled");
            None
        } else if let Ok(path) = env::var("SASA_PATH") {
            info!("Loading SASA from {path}");
            Some(vector_from_file(Path::new(&path))?)
        } else {
            let probe_radius = env::var("SASA_RADIUS")
                .unwrap_or("1.4".to_owned())
                .parse()?;
            Some(structure.calc_sasa(probe_radius))
        };
        if let Some(ref sasa) = sasa {
            info!("Total SASA = {:.1} Å²", sasa.iter().sum::<f64>());
        }

        let intensities = self
            .q_values
            .par_iter()
            .map(|q| {
                intensity_at_q(
                    &structure.ids,
                    &structure.pos,
                    &self.formfactors,
                    *q,
                    sasa.as_deref(),
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
/// Distances and form factor products are computed on-the-fly to avoid N×N matrix allocations.
/// If given, the solvent correction is a list of corrections for each atom that is added to the form factor.
fn intensity_at_q(
    atom_ids: &[usize],
    positions: &[Vector3],
    formfactors: &FormFactorMap,
    q: f64,
    sasa: Option<&[f64]>,
) -> Result<f64> {
    log::trace!("Calculating 𝐼(𝑞) for 𝑞 = {q}");
    let water_ff = formfactors
        .iter()
        .find(|(id, _)| id.name() == "H2O")
        .map(|(_, ff)| ff)
        .ok_or_else(|| anyhow!("no water form factor 'H2O' needed for SASA calculation"))?;

    // TODO: Optimize by storing in static table, see std::sync::OnceLock
    let mut ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;

    // Solvent correction using SASA
    if let Some(sasa) = sasa {
        log::trace!("SASA: {sasa:?}");
        assert_eq!(sasa.len(), ff.len());

        let offset = 0.0; // Å
        let scale = 0.0;
        let max_sasa = sasa
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let water_ff = scale * water_ff.eval(q)? * sinc(q * offset);
        for (f, sasa) in ff.iter_mut().zip(sasa) {
            *f += sasa / max_sasa * water_ff;
        }
    }

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

/// Helper function to load values from a file to a vector
///
/// - Value type `T` must implement `FromStr`
/// - One value per line
/// - Lines starting with `#` are ignored
#[allow(dead_code)]
fn vector_from_file<T: FromStr>(path: &Path) -> Result<Vec<T>> {
    let mut contents = String::new();
    File::open(path)?.read_to_string(&mut contents)?;
    contents
        .lines()
        .filter(|s| !s.starts_with("#"))
        .map(|s| s.trim().parse())
        .try_collect()
        .map_err(|_| anyhow!("parse error while loading values from {}", path.display()))
}

